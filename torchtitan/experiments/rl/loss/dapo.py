# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.config import Configurable
from torchtitan.experiments.rl.loss.ops import (
    compute_entropy,
    compute_logprobs,
    compute_token_ratio,
    logprob_drift_metrics,
    masked_token_mean,
    pg_ppo_clip,
)
from torchtitan.experiments.rl.loss.types import LossMetric, LossOutput


def pg_dual_clip(
    pg_loss: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    num_global_valid_tokens: int,
    *,
    dual_clip_c: float = 3.0,
) -> tuple[torch.Tensor, dict[str, LossMetric]]:
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
        loss_mask (torch.Tensor): Valid token mask (B, S).
        num_global_valid_tokens (int): Global token denominator for the metric.
        dual_clip_c (float): Dual-clip constant (default 3.0).

    Returns:
        tuple[torch.Tensor, dict[str, LossMetric]]: (dual-clipped loss, metrics);
            loss is (B, S). The metric is the fraction of valid negative-advantage
            tokens whose penalty was dual-clipped.
    """
    dual_clip_bound = -dual_clip_c * advantages
    loss = torch.where(advantages < 0, torch.minimum(pg_loss, dual_clip_bound), pg_loss)
    with torch.no_grad():
        # Compare against the pre-dual-clip PPO loss to detect where the ceiling bound.
        was_dual_clipped = (
            (pg_loss > dual_clip_bound) & (advantages < 0) & loss_mask.bool()
        )
        metrics = {
            "loss/dual_clip/clip/frac": LossMetric(
                masked_token_mean(
                    was_dual_clipped.float(), loss_mask, num_global_valid_tokens
                )
            )
        }
    return loss, metrics


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
        log_entropy (bool): Emit loss/entropy/mean (default True).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_low: float = 0.2
        clip_high: float = 0.28
        dual_clip_c: float = 3.0
        log_entropy: bool = True

    def __init__(self, config: Config):
        self.clip_low = config.clip_low
        self.clip_high = config.clip_high
        self.dual_clip_c = config.dual_clip_c
        self.log_entropy = config.log_entropy

    def __call__(
        self,
        *,
        logits: torch.Tensor,  # (B, S, V)
        target_ids: torch.Tensor,  # (B, S)
        generator_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        advantages: torch.Tensor,  # (B, S)
        num_global_valid_tokens: int,
        ref_logprobs: torch.Tensor | None = None,
    ) -> LossOutput:
        policy_logprobs = compute_logprobs(logits, target_ids)
        ratio, _log_ratio, ratio_metrics = compute_token_ratio(
            policy_logprobs, generator_logprobs, loss_mask, num_global_valid_tokens
        )
        pg_loss, clip_metrics = pg_ppo_clip(
            ratio,
            advantages,
            loss_mask,
            num_global_valid_tokens,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
        )
        dual_pg_loss, dual_metrics = pg_dual_clip(
            pg_loss,
            advantages,
            loss_mask,
            num_global_valid_tokens,
            dual_clip_c=self.dual_clip_c,
        )
        loss = masked_token_mean(dual_pg_loss, loss_mask, num_global_valid_tokens)
        drift_metrics = logprob_drift_metrics(
            policy_logprobs, generator_logprobs, loss_mask, num_global_valid_tokens
        )
        metrics = {
            "loss/mean": LossMetric(loss.detach()),
            **ratio_metrics,
            **clip_metrics,
            **dual_metrics,
            **drift_metrics,
        }
        if self.log_entropy:
            _entropy, entropy_metrics = compute_entropy(
                logits, loss_mask, num_global_valid_tokens
            )
            metrics.update(entropy_metrics)
        return LossOutput(loss=loss, metrics=metrics)
