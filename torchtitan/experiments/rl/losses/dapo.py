# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO loss: per-token clipped surrogate with asymmetric "clip-higher" bounds."""

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


class DAPOLoss(BaseLoss):
    """DAPO: PPO clip with independent lower/upper bounds, token-level aggregation.

    Reference: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at
    Scale" (https://arxiv.org/abs/2503.14476), Eq. 8.

        token_loss = -min(r * A, clip(r, 1 - clip_low, 1 + clip_high) * A)
        loss       = sum(token_loss * mask) / global_valid_tokens

    A larger upper bound keeps more probability mass on up-weighted tokens, countering
    entropy collapse. Dividing by the batch's response-token count (not per response)
    lets long responses contribute in proportion to their length. Dynamic sampling and
    overlong reward shaping live in the rollout path, not here.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        ratio_clip_low: float = 0.2
        """Lower clip: the importance ratio is clamped to ``>= 1 - ratio_clip_low``."""

        ratio_clip_high: float = 0.2
        """Upper clip: the ratio is clamped to ``<= 1 + ratio_clip_high``. Set larger
        than ``ratio_clip_low`` for DAPO "clip-higher" (e.g. 0.28)."""

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ) -> None:
        del compile_config
        self.ratio_clip_low = config.ratio_clip_low
        self.ratio_clip_high = config.ratio_clip_high

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
        """Compute the per-token clip-higher surrogate loss.

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

        token_loss, clip_metrics = pg_ppo_clip(
            ratio,
            advantages,
            mask,
            clip_low=self.ratio_clip_low,
            clip_high=self.ratio_clip_high,
        )
        loss = aggregate(token_loss, mask, global_valid_tokens)

        metrics = normalize_metrics(
            {
                **clip_metrics,
                **policy_stats_metrics(ratio, raw_log_ratio, token_entropy, mask),
            },
            global_valid_tokens,
        )
        metrics["loss/mean"] = loss.detach()
        return loss, metrics
