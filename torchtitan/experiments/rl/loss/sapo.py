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
    compute_token_ratio,
    logprob_drift_metrics,
    masked_token_mean,
)
from torchtitan.experiments.rl.loss.types import AggType, LossNormalization, LossOutput


def pg_soft_gate(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    normalization: LossNormalization,
    *,
    tau_pos: float = 1.0,
    tau_neg: float = 1.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SAPO's soft sigmoid gating.

    Reference: Gao et al., "Soft Adaptive Policy Optimization" (2025).
    https://arxiv.org/abs/2511.20347

    Formula: gate(r) = (4/τ) * sigmoid(τ * (r - 1))
             L = -gate(r) * A

    Replaces PPO's hard clipping with smooth sigmoid gating. The 4/τ normalization
    ensures the GRADIENT ∂gate/∂r = 1.0 at r=1, matching vanilla policy gradient
    on-policy. As r deviates from 1, that gradient decays smoothly toward 0 (the
    gate value itself saturates at 4/τ), so updates taper instead of being
    hard-clipped.

    Asymmetric temperature: τ_neg > τ_pos makes the gate's gradient decay faster
    for negative advantages. When decreasing a token's probability (negative
    advantage), that probability mass redistributes across the entire vocabulary.
    This one-to-many effect amplifies noise in negative updates. A higher τ_neg
    compensates by applying a tighter trust region for negative advantages.

    Args:
        ratio (torch.Tensor): Importance ratio (B, S).
        advantages (torch.Tensor): Advantage estimates (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        normalization (LossNormalization): Global token denominator for the metric.
        tau_pos (float): Temperature for positive advantages (default 1.0).
        tau_neg (float): Temperature for negative advantages (default 1.05).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: (pg_loss, metrics); pg_loss
            is (B, S).
    """
    pos_gate = (4.0 / tau_pos) * torch.sigmoid(tau_pos * (ratio - 1))
    neg_gate = (4.0 / tau_neg) * torch.sigmoid(tau_neg * (ratio - 1))
    gate = torch.where(advantages > 0, pos_gate, neg_gate)
    pg_loss = -gate * advantages
    with torch.no_grad():
        metrics = {
            "loss/soft_gate/gate/mean": masked_token_mean(
                gate, loss_mask, normalization
            )
        }
    return pg_loss, metrics


class SAPOLoss(Configurable):
    """SAPO: Soft Adaptive Policy Optimization.

    Reference: Gao et al., "Soft Adaptive Policy Optimization" (2025).
    https://arxiv.org/abs/2511.20347

    Per-token: L_t = -gate(r) * A
    Aggregated: L = mean over sequences of (mean over tokens of L_t)

    where:
        gate(r) = (4/τ) * sigmoid(τ * (r - 1))   — soft sigmoid gate
        τ = τ_pos if A > 0, else τ_neg           — asymmetric temperature
        r = π_θ/π_old                            — importance ratio
        A = caller-provided advantages

    SAPO replaces PPO's hard clipping with smooth sigmoid gating. The 4/τ factor
    is chosen so that the effective gradient scaling equals 1.0 at r=1 (on-policy).
    As r deviates from 1, that gradient scaling decays smoothly toward 0 (the gate
    value saturates at 4/τ).

    Asymmetric temperature: τ_neg > τ_pos makes the gate's gradient decay faster
    for negative advantages. When decreasing a token's probability (negative
    advantage), that probability mass redistributes across the entire vocabulary.
    This one-to-many effect amplifies noise in negative updates. A higher τ_neg
    compensates by applying a tighter trust region for negative advantages.

    Differences from GRPO:
        1. Soft gating: No discontinuity at clip boundary. Gradients decay
           smoothly rather than dropping to zero.

    NOTE: The default 'sequence_mean' aggregation operates per source episode via
    sample_ids (multiple episodes may be packed into one row), so sample_ids is
    required unless agg_type is changed to a token-level mode.

    Args:
        tau_pos (float): Temperature for positive advantages (default 1.0).
        tau_neg (float): Temperature for negative advantages (default 1.05).
        agg_type (AggType): Aggregation method (default "sequence_mean").
        log_entropy (bool): Emit loss/entropy/mean (default True).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        tau_pos: float = 1.0
        tau_neg: float = 1.05
        agg_type: AggType = "sequence_mean"
        log_entropy: bool = True

    def __init__(self, config: Config):
        self.tau_pos = config.tau_pos
        self.tau_neg = config.tau_neg
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
        if self.agg_type == "sequence_mean" and sample_ids is None:
            raise ValueError(
                "SAPOLoss with agg_type='sequence_mean' requires sample_ids."
            )
        policy_logprobs = compute_logprobs(logits, target_ids)
        ratio, _log_ratio, m_ratio = compute_token_ratio(
            policy_logprobs, generator_logprobs, loss_mask, normalization
        )
        pg_loss, m_gate = pg_soft_gate(
            ratio,
            advantages,
            loss_mask,
            normalization,
            tau_pos=self.tau_pos,
            tau_neg=self.tau_neg,
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
            **m_gate,
            **drift_sum,
        }
        if self.log_entropy:
            _entropy, m_entropy = compute_entropy(logits, loss_mask, normalization)
            sum_metrics.update(m_entropy)
        return LossOutput(loss=loss, sum_metrics=sum_metrics, max_metrics=max_metrics)
