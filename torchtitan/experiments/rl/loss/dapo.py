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
    compute_token_ratio,
    entropy_metrics,
    masked_token_mean,
    pg_ppo_clip,
    ppo_clip_metrics,
    ratio_metrics,
)
from torchtitan.experiments.rl.loss.types import AggType, LossNormalization, LossOutput


def pg_dual_clip(
    pg_loss: torch.Tensor,
    advantages: torch.Tensor,
    *,
    dual_clip_c: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DAPO's dual-clip for negative advantages.

    Reference: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025).
    https://arxiv.org/abs/2503.14476

    Formula: L = min(L_PPO, -c*A) when A < 0

    Standard PPO clipping can over-penalize bad actions, especially in reasoning
    tasks where some "wrong" tokens are actually productive exploration. Dual-clip
    adds a ceiling: penalties on negative-advantage tokens cannot exceed c times
    the advantage magnitude.

    Args:
        pg_loss (torch.Tensor): Per-token PPO loss from pg_ppo_clip (B, S).
        advantages (torch.Tensor): Advantage estimates (B, S).
        dual_clip_c (float): Dual-clip constant (default 3.0).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (dual-clipped loss, dual_clip_bound),
            both (B, S). The bound is returned so the clip-fraction metric can
            reuse it.
    """
    dual_clip_bound = -dual_clip_c * advantages
    loss = torch.where(advantages < 0, torch.minimum(pg_loss, dual_clip_bound), pg_loss)
    return loss, dual_clip_bound


def dual_clip_metrics(
    pg_loss: torch.Tensor,
    dual_clip_bound: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    normalization: LossNormalization,
) -> dict[str, torch.Tensor]:
    """Fraction of valid negative-advantage tokens whose penalty was dual-clipped.

    `pg_loss` is the pre-dual-clip PPO loss, so the comparison detects where the
    dual-clip ceiling actually bound.
    """
    was_dual_clipped = (pg_loss > dual_clip_bound) & (advantages < 0) & loss_mask.bool()
    return {
        "loss/dual_clip/clip/frac": masked_token_mean(
            was_dual_clipped.float(), loss_mask, normalization
        )
    }


class DAPOLoss(Configurable):
    """DAPO: Decoupled clip + Dynamic sAmpling Policy Optimization.

    Reference: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025).
    https://arxiv.org/abs/2503.14476

    Per-token:
        L_clip = max(-r*A, -clip(r, 1-ε_low, 1+ε_high)*A)
        L_t = min(L_clip, -c*A) when A < 0, else L_clip
    Aggregated: L = sum(L_t * mask) / num_global_valid_tokens

    where:
        r = π_θ/π_old                            — importance ratio
        A = caller-provided advantages
        ε_high > ε_low                           — asymmetric clip (more exploration)
        c = dual-clip cap penalty

    Differences from GRPO:
    - Clip-higher: ε_high > ε_low allows more exploration for low-probability tokens.
    - Dual-clip: Caps penalty on negative advantages to prevent over-penalization.
    - Token-level aggregation: Divides by total trainable tokens across all sequences.

    NOTE: This is the DAPO policy loss only; it consumes caller-provided advantages.
    The DAPO paper's other techniques are data/controller concerns and are NOT
    performed here:
    - Dynamic Sampling: filters groups where all responses have the same reward.
    - Overlong Reward Shaping: filters truncated sequences + soft length penalty.
    - Advantage std-normalization.

    Args:
        clip_low (float): Lower clip bound (default 0.2).
        clip_high (float): Upper clip bound (default 0.28).
        dual_clip_c (float): Dual-clip constant (default 3.0).
        agg_type (AggType): Aggregation method (default "token_mean").
        log_entropy (bool): Emit loss/entropy/mean (needs logits; default True).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_low: float = 0.2
        clip_high: float = 0.28
        dual_clip_c: float = 3.0
        agg_type: AggType = "token_mean"
        log_entropy: bool = True

    def __init__(self, config: Config):
        self.clip_low = config.clip_low
        self.clip_high = config.clip_high
        self.dual_clip_c = config.dual_clip_c
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
        sample_ids: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        ref_logprobs: torch.Tensor | None = None,
    ) -> LossOutput:
        ratio, log_ratio = compute_token_ratio(policy_logprobs, generator_logprobs)
        pg_loss, clipped_ratio = pg_ppo_clip(
            ratio, advantages, clip_low=self.clip_low, clip_high=self.clip_high
        )
        dual_pg_loss, dual_clip_bound = pg_dual_clip(
            pg_loss, advantages, dual_clip_c=self.dual_clip_c
        )
        loss = aggregate_loss(
            dual_pg_loss,
            loss_mask,
            agg_type=self.agg_type,
            normalization=normalization,
            sample_ids=sample_ids,
        )
        with torch.no_grad():
            sum_metrics = {
                "loss/mean": loss.detach(),
                **ratio_metrics(ratio, log_ratio, loss_mask, normalization),
                **ppo_clip_metrics(
                    ratio, clipped_ratio, advantages, loss_mask, normalization
                ),
                **dual_clip_metrics(
                    pg_loss, dual_clip_bound, advantages, loss_mask, normalization
                ),
                **entropy_metrics(self.log_entropy, logits, loss_mask, normalization),
            }
        return LossOutput(loss=loss, sum_metrics=sum_metrics)
