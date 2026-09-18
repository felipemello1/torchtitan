# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-token primitives shared by the RL losses.

Every primitive takes flattened ``[T]`` tensors (the batch dim is folded) and returns
per-token tensors plus *metric sums* over the masked tokens. A loss composes them, then
calls ``normalize_metrics`` once so every ``/mean`` and ``/frac`` metric is divided by
``global_valid_tokens`` and can be SUM-reduced across microbatches and DP ranks. The loss
itself goes through ``aggregate`` with whatever denominator the algorithm prescribes.

Example (DAPO):

    mask = valid_response_mask(loss_mask, generator_logprobs)
    ratio, raw_log_ratio = compute_ratio(trainer_logprobs, generator_logprobs, mask)
    token_loss, clip_metrics = pg_ppo_clip(ratio, advantages, mask, clip_low=0.2, clip_high=0.28)
    loss = aggregate(token_loss, mask, global_valid_tokens)
"""

from __future__ import annotations

from typing import Literal

import torch

DivergenceType = Literal["binary_tv", "binary_kl"]

# Clamp |log(pi_theta/pi_gen)| before exp() so a large generator/trainer
# logprob mismatch cannot overflow exp() to inf/NaN.
_MAX_LOG_RATIO = 10.0


def valid_response_mask(
    loss_mask: torch.Tensor, generator_logprobs: torch.Tensor
) -> torch.Tensor:
    """Response tokens with a finite generator logprob; the tokens every loss trains on.

    A non-finite generator logprob (vLLM under CUDA graph) has no valid old-policy
    reference, so the token contributes neither per-token loss nor token-normalized
    metrics, rather than being trained as if it were on-policy.

    Example:

        loss_mask          = [False, True, True, True]
        generator_logprobs = [-0.5, -1.0, nan, -2.0]
        valid_response_mask(...)  # -> [False, True, False, True]
    """
    return loss_mask & torch.isfinite(generator_logprobs)


def compute_ratio(
    trainer_logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token importance ratio ``pi_theta / pi_gen`` and its raw log for metrics.

    Args:
        trainer_logprobs: ``[T]`` current-policy logprobs of the sampled tokens.
        generator_logprobs: ``[T]`` logprobs from the sampling policy.
        mask: ``[T]`` bool; masked-out tokens get ratio 1 so they never produce NaN.

    Returns:
        ``(ratio, raw_log_ratio)``, both ``[T]``. ``ratio`` is ``exp(clamp(log_ratio))``
        and carries gradient; ``raw_log_ratio`` is unclamped and unmasked, for metrics.

    Example:

        trainer_logprobs   = [-1.0, -1.0, -3.0]
        generator_logprobs = [-1.0, -2.0, -1.0]
        mask               = [True, True, False]
        ratio          # -> [1.0, e^1 ~ 2.72, 1.0]   (masked token forced to 1)
        raw_log_ratio  # -> [0.0, 1.0, -2.0]
    """
    raw_log_ratio = trainer_logprobs - generator_logprobs
    masked_log_ratio = torch.where(mask, raw_log_ratio, torch.zeros_like(raw_log_ratio))
    log_ratio = torch.clamp(masked_log_ratio, -_MAX_LOG_RATIO, _MAX_LOG_RATIO)
    return torch.exp(log_ratio), raw_log_ratio


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_low: float,
    clip_high: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO clipped surrogate: ``-min(ratio * A, clip(ratio, 1 - low, 1 + high) * A)``.

    Args:
        ratio: ``[T]`` importance ratio from ``compute_ratio``.
        advantages: ``[T]`` per-token advantages.
        mask: ``[T]`` bool; only for the metric count.
        clip_low: the ratio is clamped to ``>= 1 - clip_low``.
        clip_high: the ratio is clamped to ``<= 1 + clip_high``; larger than
            ``clip_low`` gives DAPO "clip-higher" (https://arxiv.org/abs/2503.14476).

    Returns:
        ``(token_loss [T], {"loss/ratio_clipped_frac": count})``; the count is the
        number of valid tokens whose ratio hit a clip bound.

    Example:

        ratio = [1.5, 0.5, 1.0];  advantages = [1.0, -1.0, 1.0];  clip 0.2 / 0.2
        token_loss                # -> [-1.2, 0.8, -1.0]  (first two clipped)
        loss/ratio_clipped_frac   # -> 2
    """
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    token_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    with torch.no_grad():
        was_clipped = (torch.abs(ratio - clipped_ratio) > 1e-6).float() * mask
        metrics = {"loss/ratio_clipped_frac": was_clipped.sum()}
    return token_loss, metrics


def trust_region_mask(
    trainer_logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    divergence_type: DivergenceType,
    divergence_threshold: float,
) -> torch.Tensor:
    """DPPO trust region: keep a token unless its probability already moved past the
    divergence threshold in the direction its advantage pushes it.

    Both variants collapse the vocabulary into a Bernoulli over "the sampled token vs.
    everything else" (https://arxiv.org/abs/2602.04879, Eq. 12-14):

        binary_tv:  |p_theta - p_gen|
        binary_kl:  p_gen * log(p_gen / p_theta) + (1 - p_gen) * log((1 - p_gen) / (1 - p_theta))

    Unlike a ratio bound this is the same absolute budget for rare and common tokens.
    Zero-advantage tokens are always kept (they contribute no loss either way).

    Args:
        trainer_logprobs: ``[T]`` current-policy logprobs of the sampled tokens.
        generator_logprobs: ``[T]`` logprobs from the sampling policy (the anchor).
        advantages: ``[T]`` per-token advantages; sign selects the direction that counts.
        divergence_type: which Bernoulli divergence to bound.
        divergence_threshold: the paper's ``delta``; 0.2 for TV, 0.05 for KL.

    Example:

        p_theta = [0.50, 0.05, 0.10, 0.30];  p_gen = [0.20, 0.01, 0.40, 0.01]
        advantages = [1.0, 1.0, -1.0, 0.0];  binary_tv, threshold 0.2
        keep  # -> [False, True, False, True]
        # token 0: +0.30 > 0.2, dropped.  token 1: ratio 5x but only +0.04, kept.
        # token 2: -0.30 < -0.2, dropped.  token 3: A == 0, kept.
    """
    with torch.no_grad():
        trainer_prob = torch.exp(trainer_logprobs)
        generator_prob = torch.exp(generator_logprobs)
        prob_delta = trainer_prob - generator_prob
        if divergence_type == "binary_tv":
            divergence = prob_delta.abs()
        elif divergence_type == "binary_kl":
            # Clamp away from 1 so log(1 - p) stays finite; the KL is only compared
            # against the threshold, never differentiated.
            one_minus_eps = 1 - torch.finfo(trainer_prob.dtype).eps
            trainer_prob = trainer_prob.clamp(max=one_minus_eps)
            generator_prob = generator_prob.clamp(max=one_minus_eps)
            divergence = generator_prob * (generator_logprobs - trainer_logprobs) + (
                1 - generator_prob
            ) * (torch.log1p(-generator_prob) - torch.log1p(-trainer_prob))
        else:
            raise ValueError(f"Unknown divergence_type: {divergence_type}")
        moving_away = ((advantages > 0) & (prob_delta > 0)) | (
            (advantages < 0) & (prob_delta < 0)
        )
        return ~(moving_away & (divergence > divergence_threshold))


def pg_truncated_reinforce(
    ratio: torch.Tensor,
    trainer_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    keep_mask: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_ratio: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """REINFORCE surrogate with a detached, truncated importance weight:
    ``-A * sg(min(ratio, max_ratio)) * log pi_theta`` on kept tokens.

    Its gradient equals PPO's unclipped ``-A * ratio`` whenever ``ratio <= max_ratio``,
    since ``d ratio = ratio * d log pi``. Truncation bounds the weight of very
    off-policy tokens instead of zeroing them (https://arxiv.org/abs/2602.04879, §5.4).

    Args:
        ratio: ``[T]`` importance ratio from ``compute_ratio``.
        trainer_logprobs: ``[T]`` current-policy logprobs; carries the gradient.
        advantages: ``[T]`` per-token advantages.
        keep_mask: ``[T]`` bool trust-region mask from ``trust_region_mask``.
        mask: ``[T]`` bool valid-token mask; only for the metric counts.
        max_ratio: truncation point for the importance weight.

    Returns:
        ``(token_loss [T], {"loss/trust_region_dropped_frac": count,
        "loss/ratio_truncated_frac": count})``.
    """
    truncated_ratio = torch.clamp(ratio, max=max_ratio).detach()
    token_loss = -advantages * truncated_ratio * trainer_logprobs * keep_mask
    with torch.no_grad():
        metrics = {
            "loss/trust_region_dropped_frac": (~keep_mask & mask).float().sum(),
            "loss/ratio_truncated_frac": ((ratio > max_ratio) & keep_mask & mask)
            .float()
            .sum(),
        }
    return token_loss, metrics


def aggregate(
    token_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_denominator: float | None,
) -> torch.Tensor:
    """Sum the masked per-token losses and divide by one batch-wide denominator.

    The denominator is the same on every microbatch and DP rank (FSDP gradient division
    is off), so gradient accumulation matches one large batch and no per-sequence
    ``1/|o_i|`` weighting is applied. DAPO passes the valid response-token count
    (https://arxiv.org/abs/2503.14476, Eq. 8); GRPO passes
    ``num_global_training_samples * max_response_tokens`` (https://arxiv.org/abs/2503.20783, §3.2).
    ``None`` means an unnormalized sum, like ``BaseLoss``.
    """
    denominator = max(loss_denominator, 1) if loss_denominator is not None else 1
    return (token_loss * mask).sum() / denominator


def policy_stats_metrics(
    ratio: torch.Tensor,
    raw_log_ratio: torch.Tensor,
    token_entropy: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Trainer-vs-generator mismatch and entropy metric sums shared by every loss.

    Returns sums over ``mask`` (to be passed through ``normalize_metrics``) plus one
    ``/max`` key: ``loss/ratio_mean``, ``bit_wise/logprob_diff/mean`` (k1 estimate of
    ``-KL(gen || trainer)``), ``bit_wise/ratio_tokens_different/mean``,
    ``bit_wise/logprob_diff/max``, ``trainer/entropy/mean``.
    """
    with torch.no_grad():
        diff_for_metrics = torch.where(
            mask, raw_log_ratio, torch.zeros_like(raw_log_ratio)
        )
        return {
            "loss/ratio_mean": (ratio * mask).sum(),
            "bit_wise/logprob_diff/mean": diff_for_metrics.float().sum(),
            "bit_wise/ratio_tokens_different/mean": (
                (diff_for_metrics.abs() > 1e-6).float() * mask
            ).sum(),
            "bit_wise/logprob_diff/max": diff_for_metrics.abs().max(),
            "trainer/entropy/mean": (token_entropy * mask).sum(),
        }


def normalize_metrics(
    metric_sums: dict[str, torch.Tensor], global_valid_tokens: float | None
) -> dict[str, torch.Tensor]:
    """Divide every ``/mean`` and ``/frac`` sum by ``global_valid_tokens``; ``/max`` keys
    pass through. The result is what the trainer SUM-reduces (MAX for ``/max``).

    Example:

        normalize_metrics({"loss/ratio_clipped_frac": tensor(3.), "x/max": tensor(2.)}, 10)
        # -> {"loss/ratio_clipped_frac": 0.3, "x/max": 2.0}
    """
    loss_denominator = (
        max(global_valid_tokens, 1) if global_valid_tokens is not None else 1
    )
    return {
        key: value if key.endswith("/max") else value / loss_denominator
        for key, value in metric_sums.items()
    }
