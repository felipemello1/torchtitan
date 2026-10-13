# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""IPO loss: policy gradient with an absolute-probability trust region (Prime-RL's default)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss, compute_logprobs
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.rl.losses.dapo import _MAX_LOG_RATIO, _normalize


class IPOLoss(BaseLoss):
    """Policy gradient that drops tokens whose probability moved more than ``eps``.

    Prime-RL's default RL loss. With ``r = p_y / q_y`` (trainer over generator probability of
    the sampled token) and ``|p_y - q_y| <= eps`` selecting the kept tokens:

        w          = min(r, max_importance_ratio)    # value capped, gradient w * grad log p_y
        token_loss = -adv_tau * A * w                # kept tokens; 0 for dropped ones
        loss       = sum(token_loss) / global_loss_token_counts

    Dropped tokens stay in the denominator. Unlike PPO clipping, the trust region is symmetric
    in probability, not ratio, so a rare token can move freely while a likely one cannot.

    Example (eps=0.3):

        q_y = [0.9, 0.5, 0.01],  p_y = [0.5, 0.6, 0.2]   # |p - q| = [0.4, 0.1, 0.19]
        # token 0 is dropped; tokens 1 and 2 train with weights 1.2 and 20.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        eps: float = 0.3
        """Drop a token when ``|p_y - q_y|`` exceeds this."""

        max_importance_ratio: float = 1e4
        """Cap on the ratio's value; the gradient keeps flowing through the capped weight."""

        adv_tau: float = 1.0
        """Scale on the advantage."""

        global_vocab_size: int | None = None
        """Full vocabulary size; required when TP shards the vocabulary."""

    def __init__(self, config: Config) -> None:
        self.eps = config.eps
        self.max_importance_ratio = config.max_importance_ratio
        self.adv_tau = config.adv_tau
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        *,
        generator_logprobs: torch.Tensor,
        temperature: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the IPO policy-gradient loss.

        Args:
            logits: [T, V] current-policy output.
            labels: [T] pre-shifted target token ids.
            generator_logprobs: [T] logprobs from the sampling policy.
            temperature: [T] temperature each token was sampled at.
            loss_mask: [T] bool mask; True for response tokens.
            advantages: [T] per-token advantages (0.0 for prompt/padding).
            global_loss_token_counts: total response tokens with finite generator logprobs
                across all microbatches and DP ranks; the loss denominator.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        logits = logits / temperature.unsqueeze(-1)
        trainer_logprobs, token_entropy = compute_logprobs(
            logits,
            labels,
            vocab_parallel_group=spmd_mesh_group("tp"),
            return_entropy=True,
            global_vocab_size=self.global_vocab_size,
        )
        # A non-finite generator logprob has no old-policy reference: drop the token.
        effective_loss_mask = loss_mask & torch.isfinite(generator_logprobs)
        safe_generator_logprobs = torch.where(
            effective_loss_mask, generator_logprobs, trainer_logprobs.detach()
        )
        log_ratio = trainer_logprobs - safe_generator_logprobs

        with torch.no_grad():
            larger = torch.maximum(trainer_logprobs, safe_generator_logprobs)
            smaller = torch.minimum(trainer_logprobs, safe_generator_logprobs)
            # |p - q| = e^max * (1 - e^(min - max)); expm1 avoids cancellation near p == q.
            abs_prob_diff = torch.exp(larger) * -torch.expm1(smaller - larger)
            is_dropped = (abs_prob_diff > self.eps) & effective_loss_mask
            keep_mask = effective_loss_mask & ~is_dropped
            capped_log_ratio = log_ratio.clamp(
                max=math.log(self.max_importance_ratio)
            )

        # Value min(r, cap); gradient (value) * grad log p, as Prime-RL's capped ratio.
        importance_weight = torch.exp(
            capped_log_ratio + (log_ratio - log_ratio.detach())
        )
        token_loss = -self.adv_tau * advantages * importance_weight
        loss = _normalize(
            torch.where(keep_mask, token_loss, 0.0).sum(), global_loss_token_counts
        )

        with torch.no_grad():
            masked_log_ratio = torch.where(effective_loss_mask, log_ratio, 0.0)
            # k3 = r - 1 - log r, a nonnegative estimate of KL(q || p) (Prime-RL's mismatch KL).
            clamped_log_ratio = masked_log_ratio.clamp(-_MAX_LOG_RATIO, _MAX_LOG_RATIO)
            kl_k3 = torch.expm1(clamped_log_ratio) - clamped_log_ratio
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": _normalize(
                    torch.where(keep_mask, importance_weight, 0.0).sum(),
                    global_loss_token_counts,
                ),
                # Prime-RL's is_masked: share of loss tokens outside the trust region.
                "loss/ipo_dropped_frac": _normalize(
                    is_dropped.float().sum(), global_loss_token_counts
                ),
                "bit_wise/logprob_diff/mean": _normalize(
                    masked_log_ratio.float().sum(), global_loss_token_counts
                ),
                "bit_wise/logprob_diff/max": masked_log_ratio.abs().max(),
                "bit_wise/logprob_diff_abs/mean": _normalize(
                    masked_log_ratio.float().abs().sum(), global_loss_token_counts
                ),
                "bit_wise/kl_k3/mean": _normalize(
                    kl_k3.float().sum(), global_loss_token_counts
                ),
                "trainer/entropy/mean": _normalize(
                    (token_entropy * effective_loss_mask).sum(),
                    global_loss_token_counts,
                ),
            }

        return loss, metrics
