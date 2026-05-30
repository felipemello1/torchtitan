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
    compute_kl,
    compute_token_ratio,
    entropy_metrics,
    pg_ppo_clip,
    ppo_clip_metrics,
    ratio_metrics,
)
from torchtitan.experiments.rl.loss.types import (
    AggType,
    KLType,
    LossNormalization,
    LossOutput,
)


class GRPOLoss(Configurable):
    """DR-GRPO: "Done Right" GRPO with unbiased aggregation.

    Reference: Liu et al., "Understanding R1-Zero-Like Training" (2025).
    https://arxiv.org/abs/2503.20783

    Per-token: L_t = max(-r*A, -clip(r, 1-ε_low, 1+ε_high)*A) + beta*KL

    where:
        r = π_θ(y_t|q,y_<t) / π_old(y_t|q,y_<t)  — importance ratio
        A = R - mean(R)                          — no std norm (caller-provided)
        KL = r_ref - log(r_ref) - 1              — k3 estimator, r_ref = π_ref/π_θ

    GRPO replaces PPO's learned value function with group-relative advantages.
    Sample multiple responses per prompt, compute advantages by comparing rewards
    within each group. This eliminates the need for a separate critic model at
    the cost of sampling more responses.

    DR-GRPO fixes two biases in vanilla GRPO:
    1. Length bias: GRPO divides by |o_i| (agg_type='sequence_mean'), rewarding
       shorter correct and longer incorrect sequences. agg_type='fixed_horizon'
       removes this by dividing by a constant horizon instead.
    2. Difficulty bias: GRPO normalizes advantages by std, over-weighting easy
       problems with low variance. DR-GRPO uses mean-only advantages. NOTE: this
       is a property of the caller-provided advantages, not this loss.

    NOTE: Default agg_type is 'token_mean' (matches the controller's global-token
    normalization); 'fixed_horizon' stays selectable for DR-GRPO. clip_high >
    clip_low ("clip-higher") is reportedly better, though not in the original paper.

    NOTE: beta defaults to 0.0. KL (beta>0) needs ref_logprobs from a reference
    model, which the RL controller does not yet provide, so RLTrainer.Config
    rejects beta>0. The KL path is implemented and unit-testable with explicit
    ref_logprobs.

    Args:
        clip_low (float): Lower clip bound (default 0.2).
        clip_high (float): Upper clip bound (default 0.28).
        beta (float): KL penalty coefficient (default 0.0; >0 requires ref_logprobs).
        kl_type (KLType): KL estimator (default "k3").
        agg_type (AggType): Aggregation method (default "token_mean").
        log_entropy (bool): Emit loss/entropy/mean (needs logits; default True).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_low: float = 0.2
        clip_high: float = 0.28
        beta: float = 0.0
        kl_type: KLType = "k3"
        agg_type: AggType = "token_mean"
        log_entropy: bool = True

    def __init__(self, config: Config):
        self.clip_low = config.clip_low
        self.clip_high = config.clip_high
        self.beta = config.beta
        self.kl_type = config.kl_type
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
        ratio, log_ratio = compute_token_ratio(policy_logprobs, generator_logprobs)
        pg_loss, clipped_ratio = pg_ppo_clip(
            ratio, advantages, clip_low=self.clip_low, clip_high=self.clip_high
        )
        sum_metrics = {
            **ratio_metrics(ratio, log_ratio, loss_mask, normalization),
            **ppo_clip_metrics(
                ratio, clipped_ratio, advantages, loss_mask, normalization
            ),
        }

        if self.beta > 0:
            if ref_logprobs is None:
                raise ValueError("GRPOLoss.beta>0 requires ref_logprobs")
            kl, kl_metrics = compute_kl(
                policy_logprobs, ref_logprobs, loss_mask, normalization, self.kl_type
            )
            pg_loss = pg_loss + self.beta * kl
            sum_metrics.update(kl_metrics)

        sum_metrics.update(
            entropy_metrics(self.log_entropy, logits, loss_mask, normalization)
        )
        loss = aggregate_loss(
            pg_loss,
            loss_mask,
            agg_type=self.agg_type,
            normalization=normalization,
            segment_ids=segment_ids,
        )
        sum_metrics["loss/mean"] = loss.detach()
        return LossOutput(loss=loss, sum_metrics=sum_metrics)
