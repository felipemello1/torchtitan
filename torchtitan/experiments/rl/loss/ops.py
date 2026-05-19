# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared policy-gradient loss operations for token-selected RL batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RatioOutput:
    """Importance-ratio tensors used by clipped policy-gradient losses."""

    raw_log_ratio: torch.Tensor
    log_ratio: torch.Tensor
    ratio: torch.Tensor


def validate_clip_bound(name: str, value: float) -> None:
    """Validate a PPO/DAPO clip bound."""
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def validate_max_log_ratio(max_log_ratio: float) -> None:
    """Validate the log-ratio clamp used before exponentiation."""
    if max_log_ratio <= 0:
        raise ValueError(f"max_log_ratio must be positive, got {max_log_ratio}")


def compute_ratio(
    policy_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    *,
    max_log_ratio: float,
) -> RatioOutput:
    """Compute finite importance ratios from selected token logprobs."""
    raw_log_ratio = policy_logprobs.float() - behavior_logprobs.detach().float()
    sanitized_log_ratio = torch.nan_to_num(
        raw_log_ratio,
        nan=0.0,
        posinf=max_log_ratio,
        neginf=-max_log_ratio,
    )
    log_ratio = torch.clamp(
        sanitized_log_ratio,
        min=-max_log_ratio,
        max=max_log_ratio,
    )
    return RatioOutput(
        raw_log_ratio=raw_log_ratio,
        log_ratio=log_ratio,
        ratio=torch.exp(log_ratio),
    )


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO clipped surrogate objective adapted from Forge's ``pg_ppo_clip``.

    The returned clip masks use Forge/DAPO's active-advantage semantics: upper
    clipping is counted for positive advantages and lower clipping for negative
    advantages.
    """
    advantages = advantages.float()
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    unclipped_loss = -ratio * advantages
    clipped_loss = -clipped_ratio * advantages
    pg_loss = torch.maximum(unclipped_loss, clipped_loss)

    with torch.no_grad():
        clipped_high = (ratio > 1 + clip_high) & (advantages > 0)
        clipped_low = (ratio < 1 - clip_low) & (advantages < 0)

    return pg_loss, clipped_low, clipped_high


def pg_dual_clip(
    pg_loss: torch.Tensor,
    advantages: torch.Tensor,
    *,
    dual_clip_c: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DAPO dual-clip for negative advantages, adapted from Forge."""
    advantages = advantages.float()
    dual_clip_bound = -dual_clip_c * advantages
    dual_clipped = (advantages < 0) & (pg_loss > dual_clip_bound)
    loss = torch.where(
        advantages < 0,
        torch.minimum(pg_loss, dual_clip_bound),
        pg_loss,
    )
    return loss, dual_clipped


def clipped_policy_gradient_loss(
    *,
    policy_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    num_global_valid_tokens: torch.Tensor,
    clip_low: float,
    clip_high: float,
    max_log_ratio: float,
    dual_clip_c: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a token-mean clipped policy-gradient loss and metric shares.

    Metric values are divided by ``num_global_valid_tokens`` before return so
    the trainer can SUM-reduce them across microbatches and loss-parallel ranks.
    """
    ratio_output = compute_ratio(
        policy_logprobs=policy_logprobs,
        behavior_logprobs=behavior_logprobs,
        max_log_ratio=max_log_ratio,
    )
    advantages = advantages.float()
    token_pg_losses, clipped_low, clipped_high = pg_ppo_clip(
        ratio_output.ratio,
        advantages,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    if dual_clip_c is None:
        dual_clipped = torch.zeros_like(clipped_low, dtype=torch.bool)
    else:
        token_pg_losses, dual_clipped = pg_dual_clip(
            token_pg_losses,
            advantages,
            dual_clip_c=dual_clip_c,
        )

    pg_loss = token_pg_losses.sum() / num_global_valid_tokens

    with torch.no_grad():
        clipped_frac = clipped_low | clipped_high
        nonfinite_log_ratio = ~torch.isfinite(ratio_output.raw_log_ratio)
        policy_nonfinite = ~torch.isfinite(policy_logprobs)
        behavior_nonfinite = ~torch.isfinite(behavior_logprobs)
        log_ratio_clipped = ratio_output.raw_log_ratio != ratio_output.log_ratio
        metrics = {
            "loss/mean": pg_loss.detach(),
            "loss/ratio/mean": ratio_output.ratio.sum() / num_global_valid_tokens,
            "loss/ratio/clipped_frac": clipped_frac.sum() / num_global_valid_tokens,
            "loss/ratio/clipped_low_frac": clipped_low.sum() / num_global_valid_tokens,
            "loss/ratio/clipped_high_frac": clipped_high.sum()
            / num_global_valid_tokens,
            "loss/ratio/log_clipped_frac": log_ratio_clipped.sum()
            / num_global_valid_tokens,
            "loss/ratio/nonfinite_frac": nonfinite_log_ratio.sum()
            / num_global_valid_tokens,
            "loss/logprob/policy_nonfinite_frac": policy_nonfinite.sum()
            / num_global_valid_tokens,
            "loss/logprob/behavior_nonfinite_frac": behavior_nonfinite.sum()
            / num_global_valid_tokens,
        }
        if dual_clip_c is not None:
            metrics["loss/dual_clip/clipped_frac"] = (
                dual_clipped.sum() / num_global_valid_tokens
            )

    return pg_loss, metrics
