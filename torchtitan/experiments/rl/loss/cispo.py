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


def pg_cispo(
    ratio: torch.Tensor,
    policy_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    normalization: LossNormalization,
    *,
    clip_low: float = 1.0,
    clip_high: float = 4.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """CISPO: Clipped Importance Sampling Policy Optimization.

    Reference: Chen et al., "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention" (2025).
    https://arxiv.org/abs/2506.13585

    Formula: L = -clip(r, 1-ε_low, 1+ε_high).detach() * A * logprobs

    Unlike PPO which uses the ratio directly in the surrogate objective, CISPO
    uses REINFORCE-style gradients: the ratio is detached and acts as an
    importance weight on -A * log(π). In long reasoning chains, some tokens have
    very high importance ratios because they represent reflective reasoning steps.
    PPO would zero out their gradients entirely, but CISPO preserves them (just
    weighted down by the clipped ratio).

    Paper recommendation: No lower clipping. Use clip_low=1.0 (min=0, no effective
    lower bound).

    Args:
        ratio (torch.Tensor): Importance ratio (B, S).
        policy_logprobs (torch.Tensor): Log probs from current policy (B, S).
        advantages (torch.Tensor): Advantage estimates (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        normalization (LossNormalization): Global token denominator for metrics.
        clip_low (float): Lower clip bound offset (default 1.0, no effective clipping).
        clip_high (float): Upper clip bound offset (default 4.0).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: (pg_loss, metrics); pg_loss
            is (B, S). The high/low fractions are unconditional (no advantage-sign
            filter), unlike PPO's pg clip fractions.
    """
    clipped_ratio = torch.clamp(ratio, min=1 - clip_low, max=1 + clip_high).detach()
    pg_loss = -clipped_ratio * advantages * policy_logprobs
    with torch.no_grad():
        metrics = {
            "loss/clip/clipped_ratio/mean": masked_token_mean(
                clipped_ratio, loss_mask, normalization
            ),
            "loss/clip/high_unconditional/frac": masked_token_mean(
                (ratio > 1 + clip_high).float(), loss_mask, normalization
            ),
            "loss/clip/low_unconditional/frac": masked_token_mean(
                (ratio < 1 - clip_low).float(), loss_mask, normalization
            ),
        }
    return pg_loss, metrics


class CISPOLoss(Configurable):
    """CISPO: Clipped Importance Sampling Policy Optimization.

    Reference: Chen et al., "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention" (2025).
    https://arxiv.org/abs/2506.13585

    Per-token: L_t = -sg(clip(r, 1-ε_low, 1+ε_high)) * A * log π_θ
    Aggregated: L = sum(L_t * mask) / num_global_valid_tokens

    where:
        r = π_θ/π_old                            — importance ratio
        A = caller-provided advantages
        clip(r, 1-ε_low, 1+ε_high)               — clipping bounds
        sg(·) = stop gradient (detach)           — ratio is detached

    CISPO uses REINFORCE-style gradients with a clipped, detached importance
    weight. Unlike PPO where the gradient flows through the ratio, here it flows
    through logprobs. This preserves learning signal for high-ratio "reflective"
    tokens that PPO would completely clip away. In long reasoning chains, some
    tokens have very high importance ratios because they represent reflective
    reasoning steps. PPO would zero out their gradients, but CISPO preserves them
    (just weighted down).

    Paper recommendation: No lower clipping. Use clip_low=1.0 (min=0, no effective
    lower bound since ratio=exp()>=0).

    Differences from GRPO:
        1. REINFORCE-style: Ratio is detached; gradient flows through logprobs.
        2. Upper-only clipping (default): No lower bound, like GSPO.
        3. Token-level aggregation: Divides by total trainable tokens across all sequences.

    Args:
        clip_low (float): Lower clip bound offset (default 1.0,  effectively
            no lower clipping).
        clip_high (float): Upper clip bound offset (default 4.0).
        agg_type (AggType): Aggregation method (default "token_mean").
        log_entropy (bool): Emit loss/entropy/mean (default True).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_low: float = 1.0
        clip_high: float = 4.0
        agg_type: AggType = "token_mean"
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
        policy_logprobs = compute_logprobs(logits, target_ids)
        ratio, _log_ratio, ratio_metrics = compute_token_ratio(
            policy_logprobs, generator_logprobs, loss_mask, normalization
        )
        pg_loss, clip_metrics = pg_cispo(
            ratio,
            policy_logprobs,
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
        drift_sum_metrics, drift_max_metrics = logprob_drift_metrics(
            policy_logprobs, generator_logprobs, loss_mask, normalization
        )
        sum_metrics = {
            "loss/mean": loss.detach(),
            **ratio_metrics,
            **clip_metrics,
            **drift_sum_metrics,
        }
        if self.log_entropy:
            _entropy, entropy_metrics = compute_entropy(
                logits, loss_mask, normalization
            )
            sum_metrics.update(entropy_metrics)
        return LossOutput(
            loss=loss, sum_metrics=sum_metrics, max_metrics=drift_max_metrics
        )
