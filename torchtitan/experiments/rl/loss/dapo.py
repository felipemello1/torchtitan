# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO: Decoupled clip + Dynamic sAmpling Policy Optimization.

Reference: Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning
System at Scale" (2025), https://arxiv.org/abs/2503.14476.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.config import Configurable
from torchtitan.experiments.rl.loss.ops import aggregate, compute_ratio, pg_ppo_clip
from torchtitan.experiments.rl.loss.types import LossOutput

__all__ = ["DAPOLoss", "pg_dual_clip"]


def pg_dual_clip(
    pg_loss: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_global_valid_tokens: torch.Tensor,
    c: float = 3.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DAPO's dual-clip for negative advantages.

    Standard PPO clipping over-penalizes "wrong" tokens that may actually
    be productive exploration. Dual-clip adds a ceiling so penalties on
    negative-advantage tokens can't exceed `c * |A|`::

        L = min(L_PPO, -c * A)   when A < 0
        L = L_PPO                otherwise

    Returns `(per_token_loss, sum_metrics)`. The metric is pre-normalized
    for SUM reduction across the loss mesh.

    Example::

        # Heavy positive PPO loss on a negative-advantage token would
        # over-penalize; dual_clip caps it at c * |A| = 3.0 * 0.5 = 1.5.
        pg_loss    = torch.tensor([[5.00]])
        advantages = torch.tensor([[-0.5]])
        mask       = torch.tensor([[1.00]])
        N = torch.tensor(1.0)

        loss, m = pg_dual_clip(pg_loss, advantages, mask,
                               num_global_valid_tokens=N, c=3.0)
        # dual_clip_bound = -c * A = 1.5; loss = min(5.0, 1.5) = 1.5
        # m["loss/dual_clip/clip_fraction"] = 1.0
    """
    dual_clip_bound = -c * advantages
    loss = torch.where(
        advantages < 0,
        torch.minimum(pg_loss, dual_clip_bound),
        pg_loss,
    )

    with torch.no_grad():
        denom = num_global_valid_tokens.clamp(min=1.0)
        neg_mask = (advantages < 0) & mask.bool()
        was_dual_clipped = (pg_loss > dual_clip_bound) & neg_mask
        sum_metrics = {
            "loss/dual_clip/clip_fraction": (was_dual_clipped.float() * mask).sum()
            / denom,
        }
    return loss, sum_metrics


class DAPOLoss(Configurable):
    """DAPO loss: asymmetric clip + dual clip, no reference KL.

    Per-token::

        L_clip = max(-r * A, -clip(r, 1 - clip_low, 1 + clip_high) * A)
        L_t    = min(L_clip, -c * A)   when A < 0
        L_t    = L_clip                otherwise

    Aggregated as `sum(L_t * mask) / num_global_valid_tokens`, where the
    denominator is the global loss-mask sum across all DP ranks for the
    optimizer step. This makes the gradient shard-invariant: accumulating
    gradients across microbatches and DP ranks reproduces a single
    large-batch step.

    Differences from vanilla GRPO:

    - **Clip-higher** (`clip_high > clip_low`): asymmetric trust region;
      more headroom upward for low-probability tokens.
    - **Dual-clip**: `-c * A` ceiling on negative-advantage tokens.
    - **Global-denominator aggregation**: shard-invariant gradient via the
      global `num_global_valid_tokens`.

    Preprocessing concerns the orchestrator owns and that are intentionally
    not in this loss: dynamic sampling, overlong-reward shaping.

    Example::

        loss_fn = DAPOLoss(DAPOLoss.Config(
            clip_low=0.2, clip_high=0.28, dual_clip_c=3.0,
        ))
        # B=1, L=4, only tokens 2,3 are trainable.
        out = loss_fn(
            policy_logprobs=torch.tensor([[-1.0, -1.0, -0.30, -0.50]]),
            ref_logprobs=torch.tensor([[0.0, 0.0, -0.40, -0.55]]),
            loss_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            advantages=torch.tensor([[0.0, 0.0, 0.5, -0.2]]),
            num_global_valid_tokens=torch.tensor(2.0),
        )
        out.loss.backward()
        # out.sum_metrics["loss/ratio/mean"] ~= 1.08
        # out.max_metrics["loss/ratio/max_abs"] ~= 1.105
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_low: float = 0.2
        """Lower clip bound (DAPO paper default 0.2)."""

        clip_high: float = 0.28
        """Upper clip bound (DAPO paper default 0.28). Must be > clip_low."""

        dual_clip_c: float = 3.0
        """Dual-clip cap for negative advantages (DAPO paper default 3.0)."""

    def __init__(self, config: Config) -> None:
        if config.clip_high <= config.clip_low:
            raise ValueError(
                "DAPOLoss requires clip_high > clip_low; got "
                f"clip_high={config.clip_high}, clip_low={config.clip_low}"
            )
        if config.dual_clip_c <= 1.0:
            raise ValueError(
                "DAPOLoss requires dual_clip_c > 1.0 to be a meaningful ceiling; "
                f"got {config.dual_clip_c}"
            )
        self.clip_low = config.clip_low
        self.clip_high = config.clip_high
        self.dual_clip_c = config.dual_clip_c

    def __call__(
        self,
        *,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        num_global_valid_tokens: torch.Tensor,
    ) -> LossOutput:
        ratio, _, ratio_sum = compute_ratio(
            policy_logprobs,
            ref_logprobs,
            loss_mask,
            num_global_valid_tokens=num_global_valid_tokens,
        )
        pg_loss, clip_sum = pg_ppo_clip(
            ratio,
            advantages,
            loss_mask,
            num_global_valid_tokens=num_global_valid_tokens,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
        )
        pg_loss, dual_sum = pg_dual_clip(
            pg_loss,
            advantages,
            loss_mask,
            num_global_valid_tokens=num_global_valid_tokens,
            c=self.dual_clip_c,
        )
        loss = aggregate(pg_loss, loss_mask, loss_scale=num_global_valid_tokens)

        with torch.no_grad():
            ratio_masked = ratio * loss_mask
            sum_metrics: dict[str, torch.Tensor] = {
                "loss/mean": loss.detach(),
                **ratio_sum,
                **clip_sum,
                **dual_sum,
            }
            max_metrics: dict[str, torch.Tensor] = {
                "loss/ratio/max_abs": ratio_masked.abs().max(),
            }
        return LossOutput(loss=loss, sum_metrics=sum_metrics, max_metrics=max_metrics)
