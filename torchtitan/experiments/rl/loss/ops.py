# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-token loss primitives shared across DAPO / GRPO / GSPO / etc.

Each op denominates its metrics by the global valid-token count for the
optimizer step. Summing across microbatches and DP ranks then reproduces
the global mean (rather than `num_microbatches * local_mean`).
"""

from __future__ import annotations

import torch

__all__ = ["aggregate", "compute_ratio", "masked_mean", "pg_ppo_clip"]


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    loss_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """`sum(values * mask) / divisor` with an optional explicit divisor.

    Pass `loss_scale = num_global_valid_tokens` in distributed settings:
    each rank's local masked-sum divided by the global token count gives
    a SUM-reducible per-rank contribution to the global mean.

    Example::

        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask   = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        # No loss_scale: local mean over mask=1 positions.
        masked_mean(values, mask)
        # tensor(2.5)  # (2.0 + 3.0) / 2

        # With global N=10 (e.g. summed across DP ranks), the per-rank
        # contribution is (2.0 + 3.0) / 10 = 0.5.
        masked_mean(values, mask, loss_scale=torch.tensor(10.0))
        # tensor(0.5)
    """
    masked_sum = (values * mask).sum()
    if loss_scale is not None:
        divisor = loss_scale.clamp(min=1.0)
    else:
        divisor = mask.sum().clamp(min=1.0)
    return masked_sum / divisor


def compute_ratio(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_global_valid_tokens: torch.Tensor,
    log_ratio_clamp: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Token-level importance ratio `r_t = exp(policy - ref)`.

    The log-ratio is computed only at `mask=1` positions (off-mask
    positions get `log_ratio = 0`, so a stale `ref_logprob = 0` on a
    prompt position can't contribute a spurious ratio), then sanitized
    for the three numerical edge cases vLLM bf16 sampling can produce on
    low-probability tokens:

    - `NaN` (non-finite input logits, bf16 propagation) becomes `0`
      (ratio = 1, no gradient contribution at that token).
    - `+inf` (rare; bf16 overflow) is clamped to `+log_ratio_clamp`.
    - `-inf` (most common; sampled token has probability `< 1e-38`
      under the generator) is clamped to `-log_ratio_clamp`.

    Without `nan_to_num` a single NaN propagates through `exp`, the
    masked sum, and the optimizer step.

    Returns `(ratio, log_ratio, sum_metrics)`. Metrics are pre-normalized
    for SUM reduction across the loss mesh.

    Example::

        # 1 sample, 4 tokens; first 2 are prompt (mask=0).
        policy = torch.tensor([[-1.0, -1.0, -0.30, -0.50]])
        ref    = torch.tensor([[ 0.0,  0.0, -0.40, -0.55]])
        mask   = torch.tensor([[ 0.0,  0.0,  1.00,  1.00]])
        N = torch.tensor(2.0)  # global valid-token count

        ratio, log_ratio, m = compute_ratio(
            policy, ref, mask, num_global_valid_tokens=N
        )
        # log_ratio ~= [[0.0, 0.0, 0.10, 0.05]]
        # ratio     ~= [[1.0, 1.0, 1.105, 1.051]]
        # m["loss/ratio/mean"] ~= (1.105 + 1.051) / 2 ~= 1.078
    """
    mask_bool = mask > 0.5
    raw_log_ratio = torch.where(
        mask_bool,
        policy_logprobs - ref_logprobs,
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
        denom = num_global_valid_tokens.clamp(min=1.0)
        policy_nonfinite = (~torch.isfinite(policy_logprobs)).float()
        ref_nonfinite = (~torch.isfinite(ref_logprobs)).float()
        sum_metrics = {
            "loss/ratio/mean": (ratio * mask).sum() / denom,
            "loss/kl/ref_to_policy_mean": (-log_ratio * mask).sum() / denom,
            "health/loss/policy_logprob_nonfinite_frac": (
                (policy_nonfinite * mask).sum() / denom
            ),
            "health/loss/ref_logprob_nonfinite_frac": (
                (ref_nonfinite * mask).sum() / denom
            ),
        }
    return ratio, log_ratio, sum_metrics


def aggregate(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_scale: torch.Tensor,
) -> torch.Tensor:
    """DAPO token-mean reduction: `sum(loss * mask) / loss_scale`.

    Pass `loss_scale = num_global_valid_tokens` so the local masked-sum
    divided by the global token count produces a SUM-reducible per-rank
    contribution that reconstructs the global mean.

    Example::

        # 1 sample x 4 tokens; only tokens 2,3 are trainable.
        per_token_loss = torch.tensor([[1.0, 1.0, 0.2, 0.4]])
        mask           = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        loss_scale     = torch.tensor(2.0)  # = sum of mask across DP

        aggregate(per_token_loss, mask, loss_scale)
        # tensor(0.30)  # (0.2 + 0.4) / 2
    """
    return masked_mean(per_token_loss, mask, loss_scale)


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_global_valid_tokens: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO clipped surrogate (Schulman et al. 2017, arXiv:1707.06347).

    `L = max(-r * A, -clip(r, 1 - clip_low, 1 + clip_high) * A)`.

    Asymmetric clip (`clip_high > clip_low`) is the DAPO modification:
    more headroom upward for low-probability tokens.

    Returns `(per_token_loss, sum_metrics)`. Metrics are pre-normalized for
    SUM reduction across the loss mesh.

    Example::

        # Token with high upward ratio and positive advantage gets clipped.
        ratio      = torch.tensor([[1.50]])
        advantages = torch.tensor([[1.00]])
        mask       = torch.tensor([[1.00]])
        N = torch.tensor(1.0)

        pg_loss, m = pg_ppo_clip(
            ratio, advantages, mask, num_global_valid_tokens=N,
            clip_low=0.2, clip_high=0.28,
        )
        # clipped_ratio = clamp(1.50, 0.8, 1.28) = 1.28
        # pg_loss = max(-1.50 * 1.0, -1.28 * 1.0) = -1.28
        # m["loss/clip/high_fraction"] = 1.0 (this token was clipped high)
    """
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    unclipped_loss = -ratio * advantages
    clipped_loss = -clipped_ratio * advantages
    pg_loss = torch.maximum(unclipped_loss, clipped_loss)

    with torch.no_grad():
        denom = num_global_valid_tokens.clamp(min=1.0)
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
