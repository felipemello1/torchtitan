# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DPPO loss: policy gradient inside a probability-divergence trust region."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss, compute_logprobs
from torchtitan.config import CompileConfig
from torchtitan.experiments.rl.losses.ops import (
    aggregate,
    compute_ratio,
    DivergenceType,
    normalize_metrics,
    pg_truncated_reinforce,
    policy_stats_metrics,
    trust_region_mask,
    valid_response_mask,
)


class DPPOLoss(BaseLoss):
    """DPPO (Divergence PPO): replace PPO's ratio clip with a mask on how far each
    sampled token's probability moved away from the generator's.

    Reference: Qi et al., "Rethinking the Trust Region in LLM Reinforcement Learning"
    (https://arxiv.org/abs/2602.04879), Eq. 12-14 and Eq. 23.

        keep       = not (moving away from p_gen in A's direction and D(p_gen, p) > delta)
        token_loss = -A * sg(min(r, max_ratio)) * log p * keep
        loss       = sum(token_loss * mask) / global_valid_tokens

    PPO's ratio bound ``|r - 1| <= eps`` means ``|p - p_gen| <= eps * p_gen``: a tiny
    budget for rare tokens (the digits and "Wait"s that get clipped most) and a loose
    one for confident tokens, whose large moves are what destabilizes training. The
    divergence budget is the same for every token. The trust region is anchored on the
    generator's logprobs, which the paper shows is required for stability.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        divergence_type: DivergenceType = "binary_tv"
        """``binary_tv`` bounds ``|p - p_gen|``; ``binary_kl`` bounds the Bernoulli
        KL(p_gen || p) of the sampled token."""

        divergence_threshold: float | None = None
        """The paper's ``delta``. None picks its per-variant default: 0.2 for
        ``binary_tv`` (robust in [0.1, 0.2]), 0.05 for ``binary_kl``."""

        max_ratio: float = 5.0
        """Truncate the detached importance weight ``p / p_gen`` at this value
        (paper's scaling runs: 5). Truncation, unlike clipping, keeps the gradient."""

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ) -> None:
        del compile_config
        self.divergence_type = config.divergence_type
        if config.divergence_threshold is not None:
            self.divergence_threshold = config.divergence_threshold
        else:
            self.divergence_threshold = (
                0.2 if config.divergence_type == "binary_tv" else 0.05
            )
        self.max_ratio = config.max_ratio

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        *,
        generator_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
        num_global_training_samples: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the trust-region-masked policy gradient loss.

        Args:
            logits: [T, V] current-policy output.
            labels: [T] pre-shifted target token ids.
            generator_logprobs: [T] logprobs from the sampling policy.
            loss_mask: [T] bool mask; True for response tokens.
            advantages: [T] per-token advantages (0.0 for prompt/padding).
            global_valid_tokens: total response tokens with finite generator logprobs
                across all microbatches and DP ranks; the loss denominator.
            num_global_training_samples: unused; every RL loss receives it.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        del num_global_training_samples
        trainer_logprobs, token_entropy = compute_logprobs(
            logits, labels, return_entropy=True
        )
        mask = valid_response_mask(loss_mask, generator_logprobs)
        ratio, raw_log_ratio = compute_ratio(trainer_logprobs, generator_logprobs, mask)

        keep_mask = trust_region_mask(
            trainer_logprobs,
            generator_logprobs,
            advantages,
            divergence_type=self.divergence_type,
            divergence_threshold=self.divergence_threshold,
        )
        token_loss, trust_region_metrics = pg_truncated_reinforce(
            ratio,
            trainer_logprobs,
            advantages,
            keep_mask,
            mask,
            max_ratio=self.max_ratio,
        )
        loss = aggregate(token_loss, mask, global_valid_tokens)

        metrics = normalize_metrics(
            {
                **trust_region_metrics,
                **policy_stats_metrics(ratio, raw_log_ratio, token_entropy, mask),
            },
            global_valid_tokens,
        )
        metrics["loss/mean"] = loss.detach()
        return loss, metrics
