# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GRPO loss, implemented as Dr. GRPO: symmetric PPO clip with a fixed token normalizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss, compute_logprobs
from torchtitan.config import CompileConfig
from torchtitan.experiments.rl.losses.ops import (
    aggregate,
    compute_ratio,
    normalize_metrics,
    pg_ppo_clip,
    policy_stats_metrics,
    valid_response_mask,
)


class GRPOLoss(BaseLoss):
    """GRPO's clipped surrogate without its two biases ("Dr. GRPO", GRPO Done Right).

    Reference: Liu et al., "Understanding R1-Zero-Like Training: A Critical
    Perspective" (https://arxiv.org/abs/2503.20783), §3.1-3.2.

        token_loss = -min(r * A, clip(r, 1 - eps, 1 + eps) * A)
        loss       = sum(token_loss * mask) / (num_global_training_samples * max_response_tokens)

    1. Length bias: GRPO averages each response's token losses by ``1/|o_i|``, which
       favors short correct and long incorrect responses. Here token losses are summed
       and divided by a configured token normalizer rather than observed response lengths, so
       response length does not rescale a token's gradient weight.
    2. Difficulty bias: GRPO divides advantages by the group reward std. That is an
       advantage-estimator choice, ``AdvantageEstimator.Config(should_std_normalize)``,
       whose default (False) is the Dr. GRPO one.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        clip_eps: float = 0.2
        """Symmetric PPO clip: the ratio is clamped to ``[1 - clip_eps, 1 + clip_eps]``."""

        max_response_tokens: int
        """Token normalizer assigned to each training sample. Current GRPO recipes set it
        to ``generator.sampling.max_tokens``. Multi-turn samples may contain more response
        tokens; observed length does not change the loss denominator."""

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ) -> None:
        del compile_config
        self.clip_eps = config.clip_eps
        self.max_response_tokens = config.max_response_tokens

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
        """Compute the fixed-normalizer clipped surrogate loss.

        Args:
            logits: [T, V] current-policy output.
            labels: [T] pre-shifted target token ids.
            generator_logprobs: [T] logprobs from the sampling policy.
            loss_mask: [T] bool mask; True for response tokens.
            advantages: [T] per-token advantages (0.0 for prompt/padding).
            global_valid_tokens: total response tokens with finite generator logprobs
                across all microbatches and DP ranks; normalizes the metrics only.
            num_global_training_samples: training samples across all microbatches and
                DP ranks this step; with ``max_response_tokens`` it forms the loss
                denominator.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        trainer_logprobs, token_entropy = compute_logprobs(
            logits, labels, return_entropy=True
        )
        mask = valid_response_mask(loss_mask, generator_logprobs)
        ratio, raw_log_ratio = compute_ratio(trainer_logprobs, generator_logprobs, mask)

        token_loss, clip_metrics = pg_ppo_clip(
            ratio, advantages, mask, clip_low=self.clip_eps, clip_high=self.clip_eps
        )
        loss = aggregate(
            token_loss, mask, num_global_training_samples * self.max_response_tokens
        )

        metrics = normalize_metrics(
            {
                **clip_metrics,
                **policy_stats_metrics(ratio, raw_log_ratio, token_entropy, mask),
            },
            global_valid_tokens,
        )
        metrics["loss/mean"] = loss.detach()
        return loss, metrics
