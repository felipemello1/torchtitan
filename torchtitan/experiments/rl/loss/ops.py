# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared tensor primitives for the RL policy-gradient losses.

Each op computes its result first, then assembles its own metrics in a trailing
`torch.no_grad()` block and returns them alongside the result (`result, metrics
= op(...)`). Each metric is a `LossMetric` that carries its own reduction. Metrics
and the loss are normalized by the global response-token count
(`num_global_valid_tokens`) so summing the per-rank / per-microbatch shares
reconstructs the exact averaged global value.
"""

import torch
import torch.nn.functional as F

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.experiments.rl.loss.types import KLType, LossMetric


def masked_token_mean(
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    num_global_valid_tokens: int,
) -> torch.Tensor:
    """Per-rank share of a global token mean: sum(values * mask) / global_tokens.

    Uses the global token count (not the local mask sum) as the divisor, so that
    SUM-reducing this across DP ranks and gradient-accumulation microbatches
    reconstructs the exact global mean (the denominator is the same constant
    everywhere). This is also the loss reduction (token-level mean).

    Args:
        values (torch.Tensor): Per-token values (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        num_global_valid_tokens (int): Total response tokens across the global batch.

    Returns:
        torch.Tensor: Scalar pre-normalized contribution.
    """
    return (values * loss_mask).sum() / max(num_global_valid_tokens, 1)


def compute_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token log probs for the target tokens via negative cross-entropy.

    Casts to fp32 before the temperature division to preserve precision under
    bf16/fp16 training. A vocab-replicated DTensor is converted to local first,
    mirroring the model's logit layout under tensor parallelism.

    Args:
        logits (torch.Tensor): Model output logits (B, S, V).
        target_ids (torch.Tensor): Target token ids (B, S); pre-shifted per
            episode by the batcher (target_ids[i] = raw_token_ids[i+1]).
        temperature (float): Softmax temperature (default 1.0).

    Returns:
        torch.Tensor: Per-token log probs (B, S).
    """
    from torch.distributed.tensor import DTensor

    if isinstance(logits, DTensor):
        logits = logits.to_local()
    logits_fp32 = logits.float() / temperature
    B, S, V = logits_fp32.shape
    return -F.cross_entropy(
        logits_fp32.reshape(B * S, V),
        target_ids.reshape(B * S).long(),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).reshape(B, S)


def compute_token_ratio(
    policy_logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    num_global_valid_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, LossMetric]]:
    """Per-token importance ratio for off-policy correction.

    The ratio r = π_θ/π_old measures how much the current policy differs from
    the policy that generated the samples, enabling reuse of old samples while
    adjusting for distribution shift.

    - ratio = 1.0: on-policy
    - ratio > 1.0 / < 1.0: current policy assigns higher / lower probability

    Args:
        policy_logprobs (torch.Tensor): Log probs from current policy (B, S).
        generator_logprobs (torch.Tensor): Log probs from sampling policy (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        num_global_valid_tokens (int): Global token denominator for metrics.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, LossMetric]]: (ratio,
            log_ratio, metrics); ratio and log_ratio are (B, S). Metrics: mean
            ratio (≈1 on-policy) and the k1 policy/old KL proxy mean(-log_ratio).
    """
    log_ratio = policy_logprobs - generator_logprobs.detach()
    ratio = torch.exp(log_ratio)
    with torch.no_grad():
        metrics = {
            "loss/ratio/mean": LossMetric(
                masked_token_mean(ratio, loss_mask, num_global_valid_tokens)
            ),
            "loss/kl_policy/mean": LossMetric(
                masked_token_mean(-log_ratio, loss_mask, num_global_valid_tokens)
            ),
        }
    return ratio, log_ratio, metrics


def compute_entropy(
    logits: torch.Tensor,
    loss_mask: torch.Tensor,
    num_global_valid_tokens: int,
) -> tuple[torch.Tensor, dict[str, LossMetric]]:
    """Compute per-token entropy (logging only).

    Formula: H = logsumexp(logits) - sum(softmax(logits) * logits)
        This is equivalent to -sum(p * log(p)) but numerically stable.

    Converts a vocab-replicated DTensor to local first, mirroring the logprob
    path.

    Args:
        logits (torch.Tensor): Model output logits (B, S, V).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        num_global_valid_tokens (int): Global token denominator for the metric.

    Returns:
        tuple[torch.Tensor, dict[str, LossMetric]]: (entropy, metrics); entropy
            is (B, S), metrics is {"loss/entropy/mean": ...}.
    """
    from torch.distributed.tensor import DTensor

    if isinstance(logits, DTensor):
        logits = logits.to_local()
    logits_fp32 = logits.float()
    probs = F.softmax(logits_fp32, dim=-1)
    entropy = torch.logsumexp(logits_fp32, dim=-1) - (probs * logits_fp32).sum(dim=-1)
    with torch.no_grad():
        metrics = {
            "loss/entropy/mean": LossMetric(
                masked_token_mean(entropy, loss_mask, num_global_valid_tokens)
            )
        }
    return entropy, metrics


def compute_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    num_global_valid_tokens: int,
    kl_type: KLType = "k3",
) -> tuple[torch.Tensor, dict[str, LossMetric]]:
    """Compute per-token KL divergence using Schulman's estimators.

    Reference: Schulman's blog post (http://joschu.net/blog/kl-approx.html).

    KL divergence measures how much the current policy differs from a reference
    policy. In RLHF, this prevents the model from straying too far from its
    pretrained behavior.

    Estimator properties (for KL[policy, ref]):
    - k1: Unbiased KL estimate, but E[grad k1] = 0 (useless for optimization).
    - k2: Biased KL estimate, but E[grad k2] = grad KL (unbiased gradient).
    - k3: Unbiased KL estimate with low variance. E[grad k3] = grad KL[ref, policy].

    k3 is preferred for monitoring KL value. k2 is preferred when using KL as a
    regularizer (gradient flows correctly). k1 is rarely used in practice.

    Args:
        policy_logprobs (torch.Tensor): Log probs from current policy (B, S).
        ref_logprobs (torch.Tensor): Log probs from reference policy (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        num_global_valid_tokens (int): Global token denominator for the metric.
        kl_type (KLType): KL estimator type: "k1", "k2", or "k3" (default: "k3").

    Returns:
        tuple[torch.Tensor, dict[str, LossMetric]]: (kl, metrics); kl is (B, S),
            metrics is {"loss/kl_ref/mean": ...}.
    """
    log_ratio = policy_logprobs - ref_logprobs.detach()  # log(π_θ / π_ref)

    if kl_type == "k1":
        kl = log_ratio
    elif kl_type == "k2":
        kl = 0.5 * log_ratio.square()
    elif kl_type == "k3":
        neg_log_ratio = torch.clamp(-log_ratio, min=-10.0, max=10.0)
        ratio = torch.exp(neg_log_ratio)  # π_ref / π_θ
        kl = ratio - neg_log_ratio - 1
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    with torch.no_grad():
        metrics = {
            "loss/kl_ref/mean": LossMetric(
                masked_token_mean(kl, loss_mask, num_global_valid_tokens)
            )
        }
    return kl, metrics


def logprob_drift_metrics(
    policy_logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    num_global_valid_tokens: int,
) -> dict[str, LossMetric]:
    """Generator-vs-policy logprob drift, a bitwise on-policy diagnostic.

    The mean drift and the fraction of tokens that differ are pre-normalized by
    the global token count (`reduce="sum"`); the max absolute drift is
    `reduce="max"`.

    Args:
        policy_logprobs (torch.Tensor): Trainer-recomputed log probs (B, S).
        generator_logprobs (torch.Tensor): Generator (sampling) log probs (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        num_global_valid_tokens (int): Global token denominator for the means.

    Returns:
        dict[str, LossMetric]: drift mean / ratio-different (sum) and max (max).
    """
    with torch.no_grad():
        valid = loss_mask.bool()
        ref = generator_logprobs[valid].float()
        policy = policy_logprobs[valid].float()
        if ref.numel() == 0:
            zero = torch.zeros(
                (), dtype=torch.float32, device=generator_logprobs.device
            )
            return {
                "bit_wise/logprob_diff/mean": LossMetric(zero),
                "bit_wise/ratio_tokens_different/mean": LossMetric(zero),
                "bit_wise/logprob_diff/max": LossMetric(zero, "max"),
            }
        denom = max(num_global_valid_tokens, 1)
        diff = policy - ref
        return {
            "bit_wise/logprob_diff/mean": LossMetric(diff.sum() / denom),
            "bit_wise/ratio_tokens_different/mean": LossMetric(
                (diff.abs() > 1e-6).sum() / denom
            ),
            "bit_wise/logprob_diff/max": LossMetric(diff.abs().max(), "max"),
        }


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    num_global_valid_tokens: int,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> tuple[torch.Tensor, dict[str, LossMetric]]:
    """PPO clipped surrogate objective.

    Reference: Schulman et al., "Proximal Policy Optimization" (2017).
    https://arxiv.org/abs/1707.06347

    Clips the importance ratio to prevent the policy from changing too much in
    a single update. The max() operator creates a "pessimistic" bound: we only
    take credit for improvement up to the clip boundary. This keeps updates
    within a trust region around the old policy.

    Formula: L = max(-r*A, -clip(r, 1-ε_low, 1+ε_high)*A)

    Args:
        ratio (torch.Tensor): Importance ratio π_θ/π_old (B, S).
        advantages (torch.Tensor): Advantage estimates (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        num_global_valid_tokens (int): Global token denominator for metrics.
        clip_low (float): Lower bound offset. Ratio is clamped to min of (1 - clip_low).
            E.g., clip_low=0.2 means ratio >= 0.8. Default: 0.2.
        clip_high (float): Upper bound offset. Ratio is clamped to max of (1 + clip_high).
            E.g., clip_high=0.2 means ratio <= 1.2. Default: 0.2.

    Returns:
        tuple[torch.Tensor, dict[str, LossMetric]]: (pg_loss, metrics); pg_loss
            is (B, S). `high/frac` / `low/frac` are the fractions of valid tokens
            clipped against a positive / negative advantage respectively.
    """
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    pg_loss = torch.maximum(-ratio * advantages, -clipped_ratio * advantages)
    with torch.no_grad():
        clipped_high = ratio > clipped_ratio  # ratio above 1 + clip_high
        clipped_low = ratio < clipped_ratio  # ratio below 1 - clip_low
        metrics = {
            "loss/clip/clipped_ratio/mean": LossMetric(
                masked_token_mean(clipped_ratio, loss_mask, num_global_valid_tokens)
            ),
            "loss/clip/high/frac": LossMetric(
                masked_token_mean(
                    (clipped_high & (advantages > 0)).float(),
                    loss_mask,
                    num_global_valid_tokens,
                )
            ),
            "loss/clip/low/frac": LossMetric(
                masked_token_mean(
                    (clipped_low & (advantages < 0)).float(),
                    loss_mask,
                    num_global_valid_tokens,
                )
            ),
        }
    return pg_loss, metrics
