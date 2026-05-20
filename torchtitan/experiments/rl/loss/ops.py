# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-token loss primitives shared across DAPO/GRPO/GSPO/etc.

Ported from ``forge/src/forge/rl/loss/ops.py``. The metric dicts use
the SUM/MAX split described in :mod:`loss.types`: each op returns a
``(tensor, sum_metrics)`` pair (and the loss caller adds ``max_metrics``)
so the caller can stitch them into the final :class:`LossOutput`.
"""

from __future__ import annotations

import torch

__all__ = ["aggregate", "compute_ratio", "masked_mean", "pg_ppo_clip"]


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    loss_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """``sum(values * mask) / divisor`` with optional explicit divisor.

    In distributed settings pass ``loss_scale = global_loss_mask_sum``
    so the local masked-sum / global-N produces the correct global
    mean once SUM-reduced across DP ranks.
    """
    masked_sum = (values * mask).sum()
    if loss_scale is not None:
        divisor = loss_scale.clamp(min=1.0)
    else:
        divisor = mask.sum().clamp(min=1.0)
    return masked_sum / divisor


def compute_ratio(
    policy_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    mask: torch.Tensor,
    *,
    log_ratio_clamp: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Token-level importance ratio ``r_t = exp(policy - behavior)``.

    The log-ratio is computed only at ``mask=1`` positions (off-mask
    positions get ``log_ratio = 0`` so a stale ``behavior_logprob = 0``
    on a prompt position can't contribute a spurious ratio), then
    sanitized to handle three numerical-edge cases that vLLM bf16
    sampling can produce on low-probability tokens:

    - ``NaN`` (from non-finite input logits or bf16 propagation) →
      ``0`` (ratio = 1, no gradient contribution at that token).
    - ``+inf`` (rare; bf16 overflow) → clamped to ``+log_ratio_clamp``.
    - ``-inf`` (most common; sampled token has probability < 1e-38
      under the generator) → clamped to ``-log_ratio_clamp``.

    Without ``nan_to_num`` a single NaN propagates through ``exp``,
    the masked sum, and the optimizer step — that was the root cause
    of the ``loss/mean=nan`` we observed when running with
    ``top_p=1.0`` on the smaller models.

    Returns:
        (ratio, log_ratio, sum_metrics)
    """
    mask_bool = mask > 0.5
    raw_log_ratio = torch.where(
        mask_bool,
        policy_logprobs - behavior_logprobs,
        torch.zeros_like(policy_logprobs),
    )
    sanitized = torch.nan_to_num(
        raw_log_ratio,
        nan=0.0,
        posinf=log_ratio_clamp,
        neginf=-log_ratio_clamp,
    )
    log_ratio = torch.clamp(sanitized, min=-log_ratio_clamp, max=log_ratio_clamp)
    ratio = torch.exp(log_ratio)

    with torch.no_grad():
        denom = mask.sum().clamp(min=1.0)
        policy_nonfinite = (~torch.isfinite(policy_logprobs)).float()
        behavior_nonfinite = (~torch.isfinite(behavior_logprobs)).float()
        ratio_nonfinite = (~torch.isfinite(ratio)).float()
        sum_metrics = {
            "loss/ratio/mean": (ratio * mask).sum() / denom,
            "loss/kl_policy/mean": (-log_ratio * mask).sum() / denom,
            "loss/logprob/policy_nonfinite_frac": (policy_nonfinite * mask).sum()
            / denom,
            "loss/logprob/behavior_nonfinite_frac": (behavior_nonfinite * mask).sum()
            / denom,
            "loss/ratio/nonfinite_frac": (ratio_nonfinite * mask).sum() / denom,
        }
    return ratio, log_ratio, sum_metrics


def aggregate(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_scale: torch.Tensor,
) -> torch.Tensor:
    """DAPO token-mean reduction: ``sum(loss * mask) / loss_scale``.

    Pass ``loss_scale = global_loss_mask_sum`` so the local masked-sum
    / global-N produces the correct global mean once SUM-reduced
    across DP ranks.
    """
    return masked_mean(per_token_loss, mask, loss_scale)


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO clipped surrogate (Schulman et al. 2017, arXiv:1707.06347).

    ``L = max(-r * A, -clip(r, 1 - clip_low, 1 + clip_high) * A)``.

    Asymmetric clip (``clip_high > clip_low``) is the DAPO modification
    — allows more upward exploration on low-probability tokens.

    Returns:
        (per_token_loss, sum_metrics)
    """
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    unclipped_loss = -ratio * advantages
    clipped_loss = -clipped_ratio * advantages
    pg_loss = torch.maximum(unclipped_loss, clipped_loss)

    with torch.no_grad():
        denom = mask.sum().clamp(min=1.0)
        mask_b = mask.bool()
        clipped_high = (ratio > 1 + clip_high) & mask_b
        clipped_low = (ratio < 1 - clip_low) & mask_b
        pos_adv = advantages > 0
        neg_adv = advantages < 0
        sum_metrics = {
            "loss/clip/clipped_ratio/mean": (clipped_ratio * mask).sum() / denom,
            "loss/clip/high_fraction": ((clipped_high & pos_adv).float() * mask).sum()
            / denom,
            "loss/clip/low_fraction": ((clipped_low & neg_adv).float() * mask).sum()
            / denom,
        }
    return pg_loss, sum_metrics
