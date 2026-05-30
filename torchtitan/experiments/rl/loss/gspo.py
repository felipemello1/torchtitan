# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.config import Configurable
from torchtitan.experiments.rl.loss.ops import (
    aggregate_loss,
    compute_sequence_ratio,
    entropy_metrics,
    pg_ppo_clip,
    ppo_clip_metrics,
    ratio_metrics,
)
from torchtitan.experiments.rl.loss.types import AggType, LossNormalization, LossOutput


class GSPOLoss(Configurable):
    """GSPO: Group Sequence Policy Optimization.

    Reference: Zheng et al., "Group Sequence Policy Optimization" (2025).
    https://arxiv.org/abs/2507.18071

    Per-token: L_t = max(-s*A, -clip(s, max=1+ε)*A)
    Aggregated: L = mean_i(sum_t(L_t * mask) / sum_t(mask))

    where:
        s = exp(mean_t(log π_θ - log π_old))    — sequence-level ratio
        s_t = sg(s) * π_θ(y_t) / sg(π_θ(y_t))   — reparameterized for token gradients
        A = caller-provided advantages
        sg(·) = stop gradient (detach)

    Note: s_t has same VALUE as s in forward pass, but gradient flows through π_θ(y_t).

    GSPO computes one importance ratio per sequence instead of per token. This
    matches how rewards are actually assigned (per-response, not per-token),
    which reduces variance, especially for long sequences and MoE models.

    Differences from GRPO:
        1. Sequence-level ratio: one ratio per sequence (mean of token log-ratios)
           instead of per-token. Reduces variance for long sequences.

    NOTE: Both the sequence ratio and 'sequence_mean' aggregation operate per
    source episode via segment_ids (multiple episodes may be packed into one row),
    so segment_ids is required.

    Args:
        clip_low (float): Lower clip bound offset (default 0.2).
        clip_high (float): Upper clip bound offset (default 0.2).
        agg_type (AggType): Aggregation method (default "sequence_mean").
        log_entropy (bool): Emit loss/entropy/mean (needs logits; default True).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_low: float = 0.2
        clip_high: float = 0.2
        agg_type: AggType = "sequence_mean"
        log_entropy: bool = True

    def __init__(self, config: Config):
        self.clip_low = config.clip_low
        self.clip_high = config.clip_high
        self.agg_type = config.agg_type
        self.log_entropy = config.log_entropy

    def __call__(
        self,
        *,
        policy_logprobs: torch.Tensor,  # (B, S)
        generator_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        advantages: torch.Tensor,  # (B, S)
        normalization: LossNormalization,
        segment_ids: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        ref_logprobs: torch.Tensor | None = None,
    ) -> LossOutput:
        if segment_ids is None:
            raise ValueError(
                "GSPOLoss requires segment_ids (sequence ratio + sequence_mean)."
            )
        ratio, log_ratio = compute_sequence_ratio(
            policy_logprobs, generator_logprobs, loss_mask, segment_ids
        )
        pg_loss, clipped_ratio = pg_ppo_clip(
            ratio, advantages, clip_low=self.clip_low, clip_high=self.clip_high
        )
        loss = aggregate_loss(
            pg_loss,
            loss_mask,
            agg_type=self.agg_type,
            normalization=normalization,
            segment_ids=segment_ids,
        )
        with torch.no_grad():
            sum_metrics = {
                "loss/mean": loss.detach(),
                **ratio_metrics(ratio, log_ratio, loss_mask, normalization),
                **ppo_clip_metrics(
                    ratio, clipped_ratio, advantages, loss_mask, normalization
                ),
                **entropy_metrics(self.log_entropy, logits, loss_mask, normalization),
            }
        return LossOutput(loss=loss, sum_metrics=sum_metrics)
