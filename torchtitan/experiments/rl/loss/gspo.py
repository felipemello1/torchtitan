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
    compute_entropy,
    compute_logprobs,
    compute_sequence_ratio,
    logprob_drift_metrics,
    pg_ppo_clip,
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
    source episode via sample_ids (multiple episodes may be packed into one row),
    so sample_ids is required.

    Args:
        clip_low (float): Lower clip bound offset (default 0.2).
        clip_high (float): Upper clip bound offset (default 0.2).
        agg_type (AggType): Aggregation method (default "sequence_mean").
        log_entropy (bool): Emit loss/entropy/mean (default True).
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
        logits: torch.Tensor,  # (B, S, V)
        target_ids: torch.Tensor,  # (B, S)
        generator_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        advantages: torch.Tensor,  # (B, S)
        normalization: LossNormalization,
        sample_ids: torch.Tensor | None = None,
        ref_logprobs: torch.Tensor | None = None,
    ) -> LossOutput:
        if sample_ids is None:
            raise ValueError(
                "GSPOLoss requires sample_ids (sequence ratio + sequence_mean)."
            )
        policy_logprobs = compute_logprobs(logits, target_ids)
        ratio, _log_ratio, m_ratio = compute_sequence_ratio(
            policy_logprobs, generator_logprobs, loss_mask, sample_ids, normalization
        )
        pg_loss, m_clip = pg_ppo_clip(
            ratio,
            advantages,
            loss_mask,
            normalization,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
        )
        loss = aggregate_loss(
            pg_loss,
            loss_mask,
            agg_type=self.agg_type,
            normalization=normalization,
            sample_ids=sample_ids,
        )
        drift_sum, max_metrics = logprob_drift_metrics(
            policy_logprobs, generator_logprobs, loss_mask, normalization
        )
        sum_metrics = {
            "loss/mean": loss.detach(),
            **m_ratio,
            **m_clip,
            **drift_sum,
        }
        if self.log_entropy:
            _entropy, m_entropy = compute_entropy(logits, loss_mask, normalization)
            sum_metrics.update(m_entropy)
        return LossOutput(loss=loss, sum_metrics=sum_metrics, max_metrics=max_metrics)
