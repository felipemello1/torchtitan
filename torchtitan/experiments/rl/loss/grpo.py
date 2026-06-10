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
    compute_kl,
    compute_logprobs,
    compute_token_ratio,
    logprob_drift_metrics,
    masked_token_mean,
    pg_ppo_clip,
)
from torchtitan.experiments.rl.loss.types import KLType, LossMetric, LossOutput


class GRPOLoss(Configurable):
    """GRPO: Group Relative Policy Optimization.

    Reference: Liu et al., "Understanding R1-Zero-Like Training" (2025).
    https://arxiv.org/abs/2503.20783

    Per-token: L_t = max(-r*A, -clip(r, 1-ε_low, 1+ε_high)*A) + beta*KL
    Aggregated: L = sum(L_t * mask) / num_global_valid_tokens

    where:
        r = π_θ(y_t|q,y_<t) / π_old(y_t|q,y_<t)  — importance ratio
        A = R - mean(R)                          — no std norm (caller-provided)
        KL = r_ref - log(r_ref) - 1              — k3 estimator, r_ref = π_ref/π_θ

    GRPO replaces PPO's learned value function with group-relative advantages.
    Sample multiple responses per prompt, compute advantages by comparing rewards
    within each group. This eliminates the need for a separate critic model at
    the cost of sampling more responses.

    NOTE: difficulty bias (vanilla GRPO normalizes advantages by std, over-weighting
    easy low-variance problems) is a property of the caller-provided advantages, not
    this loss. clip_high > clip_low ("clip-higher") is reportedly better, though not
    in the original paper.

    NOTE: beta defaults to 0.0. KL (beta>0) needs ref_logprobs from a reference
    model, which the RL controller does not yet provide, so RLTrainer.Config
    rejects beta>0. The KL path is implemented and unit-testable with explicit
    ref_logprobs.

    Args:
        clip_low (float): Lower clip bound (default 0.2).
        clip_high (float): Upper clip bound (default 0.28).
        beta (float): KL penalty coefficient (default 0.0; >0 requires ref_logprobs).
        kl_type (KLType): KL estimator (default "k3").
        log_entropy (bool): Emit loss/entropy/mean (default True).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_low: float = 0.2
        clip_high: float = 0.28
        beta: float = 0.0
        kl_type: KLType = "k3"
        log_entropy: bool = True

    def __init__(self, config: Config):
        self.clip_low = config.clip_low
        self.clip_high = config.clip_high
        self.beta = config.beta
        self.kl_type = config.kl_type
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

        kl_metrics: dict[str, LossMetric] = {}
        if self.beta > 0:
            if ref_logprobs is None:
                raise ValueError("GRPOLoss.beta>0 requires ref_logprobs")
            kl, kl_metrics = compute_kl(
                policy_logprobs,
                ref_logprobs,
                loss_mask,
                num_global_valid_tokens,
                self.kl_type,
            )
            pg_loss = pg_loss + self.beta * kl

        loss = masked_token_mean(pg_loss, loss_mask, num_global_valid_tokens)
        drift_metrics = logprob_drift_metrics(
            policy_logprobs, generator_logprobs, loss_mask, num_global_valid_tokens
        )
        metrics = {
            "loss/mean": LossMetric(loss.detach()),
            **ratio_metrics,
            **clip_metrics,
            **kl_metrics,
            **drift_metrics,
        }
        if self.log_entropy:
            _entropy, entropy_metrics = compute_entropy(
                logits, loss_mask, num_global_valid_tokens
            )
            metrics.update(entropy_metrics)
        return LossOutput(loss=loss, metrics=metrics)
