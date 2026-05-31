# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared tensor primitives for the RL policy-gradient losses.

Each op computes its result first, then assembles its own metrics in a trailing
`torch.no_grad()` block and returns them alongside the result (`result, metrics
= op(...)`). Metrics are pre-normalized scalars (sum / global_denominator) so
SUM-reduction across the loss mesh and gradient-accumulation microbatches
reconstructs the exact global value (see `LossNormalization`).
"""

import torch
import torch.nn.functional as F

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.experiments.rl.loss.types import AggType, KLType, LossNormalization


def masked_token_mean(
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    normalization: LossNormalization,
) -> torch.Tensor:
    """Per-rank share of a global token mean: sum(values * mask) / global_tokens.

    Uses the global token count (not the local mask sum) as the divisor, so that
    SUM-reducing this across DP ranks and gradient-accumulation microbatches
    reconstructs the exact global mean (the denominator is the same constant
    everywhere).

    Args:
        values (torch.Tensor): Per-token values (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        normalization (LossNormalization): Carries `num_global_valid_tokens`.

    Returns:
        torch.Tensor: Scalar pre-normalized contribution.
    """
    return (values * loss_mask).sum() / max(normalization.num_global_valid_tokens, 1)


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
    normalization: LossNormalization,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
        normalization (LossNormalization): Global token denominator for metrics.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: (ratio,
            log_ratio, metrics); ratio and log_ratio are (B, S). Metrics: mean
            ratio (≈1 on-policy) and the k1 policy/old KL proxy mean(-log_ratio).
    """
    log_ratio = policy_logprobs - generator_logprobs.detach()
    ratio = torch.exp(log_ratio)
    with torch.no_grad():
        metrics = {
            "loss/ratio/mean": masked_token_mean(ratio, loss_mask, normalization),
            "loss/kl_policy/mean": masked_token_mean(
                -log_ratio, loss_mask, normalization
            ),
        }
    return ratio, log_ratio, metrics


def compute_sequence_ratio(
    policy_logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_ids: torch.Tensor,
    normalization: LossNormalization,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Sequence-level importance ratio, one per source episode (sample).

    One ratio per response (not per token) matches how rewards are assigned and
    lowers variance for long sequences. Multiple episodes may be packed into one
    (B=1, S) row, so the per-episode mean is taken over `sample_ids` rather
    than the whole row. A reparameterization keeps per-token gradient flow: the
    forward value is the episode ratio, but gradients flow through each token's
    current-policy logprob.

    Reference: Zheng et al., "GSPO" (arXiv:2507.18071, 2025).

    Args:
        policy_logprobs (torch.Tensor): Log probs from current policy (B, S).
        generator_logprobs (torch.Tensor): Log probs from sampling policy (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        sample_ids (torch.Tensor): Source-episode id per token (B, S), -1 for padding.
        normalization (LossNormalization): Global token denominator for metrics.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: (ratio,
            log_ratio, metrics); ratio and log_ratio are (B, S), 1.0 / 0.0 at
            non-trained positions.
    """
    valid = loss_mask.bool() & (sample_ids >= 0)
    if not bool(valid.any()):
        log_ratio = torch.zeros_like(policy_logprobs)
        ratio = torch.ones_like(policy_logprobs)
    else:
        token_log_ratio = policy_logprobs - generator_logprobs.detach()
        _unique, inverse = torch.unique(
            sample_ids[valid], sorted=True, return_inverse=True
        )
        sums = torch.zeros(
            int(inverse.max()) + 1,
            device=policy_logprobs.device,
            dtype=policy_logprobs.dtype,
        )
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, inverse, token_log_ratio[valid])
        counts.scatter_add_(0, inverse, torch.ones_like(token_log_ratio[valid]))
        seq_log_ratio = sums / counts.clamp_min(1)  # [num_local_samples]

        # Reparameterization: forward = sequence ratio, backward = per-token grads.
        log_ratio = torch.zeros_like(policy_logprobs)
        log_ratio[valid] = (
            policy_logprobs[valid]
            - policy_logprobs[valid].detach()
            + seq_log_ratio[inverse].detach()
        )
        ratio = torch.exp(log_ratio)
    with torch.no_grad():
        metrics = {
            "loss/ratio/mean": masked_token_mean(ratio, loss_mask, normalization),
            "loss/kl_policy/mean": masked_token_mean(
                -log_ratio, loss_mask, normalization
            ),
        }
    return ratio, log_ratio, metrics


def compute_entropy(
    logits: torch.Tensor,
    loss_mask: torch.Tensor,
    normalization: LossNormalization,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute per-token entropy (logging only).

    Formula: H = logsumexp(logits) - sum(softmax(logits) * logits)
        This is equivalent to -sum(p * log(p)) but numerically stable.

    Converts a vocab-replicated DTensor to local first, mirroring the logprob
    path.

    Args:
        logits (torch.Tensor): Model output logits (B, S, V).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        normalization (LossNormalization): Global token denominator for the metric.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: (entropy, metrics); entropy
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
            "loss/entropy/mean": masked_token_mean(entropy, loss_mask, normalization)
        }
    return entropy, metrics


def compute_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    normalization: LossNormalization,
    kl_type: KLType = "k3",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        normalization (LossNormalization): Global token denominator for the metric.
        kl_type (KLType): KL estimator type: "k1", "k2", or "k3" (default: "k3").

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: (kl, metrics); kl is (B, S),
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
        metrics = {"loss/kl_ref/mean": masked_token_mean(kl, loss_mask, normalization)}
    return kl, metrics


def logprob_drift_metrics(
    policy_logprobs: torch.Tensor,
    generator_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    normalization: LossNormalization,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Generator-vs-policy logprob drift, a bitwise on-policy diagnostic.

    Returns (sum_metrics, max_metrics): the mean drift and the fraction of tokens
    that differ are pre-normalized by the global token count (SUM-folded); the max
    absolute drift is MAX-folded.

    Args:
        policy_logprobs (torch.Tensor): Trainer-recomputed log probs (B, S).
        generator_logprobs (torch.Tensor): Generator (sampling) log probs (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        normalization (LossNormalization): Global token denominator for the means.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: (sum_metrics,
            max_metrics).
    """
    with torch.no_grad():
        valid = loss_mask.bool()
        ref = generator_logprobs[valid].float()
        policy = policy_logprobs[valid].float()
        if ref.numel() == 0:
            zero = torch.zeros(
                (), dtype=torch.float32, device=generator_logprobs.device
            )
            return (
                {
                    "bit_wise/logprob_diff/mean": zero,
                    "bit_wise/ratio_tokens_different/mean": zero,
                },
                {"bit_wise/logprob_diff/max": zero},
            )
        denom = max(normalization.num_global_valid_tokens, 1)
        diff = policy - ref
        sum_metrics = {
            "bit_wise/logprob_diff/mean": diff.sum() / denom,
            "bit_wise/ratio_tokens_different/mean": (diff.abs() > 1e-6).sum() / denom,
        }
        max_metrics = {"bit_wise/logprob_diff/max": diff.abs().max()}
    return sum_metrics, max_metrics


def aggregate_loss(
    per_token_loss: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    agg_type: AggType,
    normalization: LossNormalization,
    sample_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Aggregate per-token loss to a scalar using a global denominator.

    Different aggregation strategies have different bias properties that affect
    training dynamics:

    token_mean: sum(loss*mask) / num_global_valid_tokens
        Pre-normalized by the global token count so gradient accumulation and DP
        reproduce a single large-batch step.

    fixed_horizon: sum(loss*mask) / num_global_fixed_horizon_tokens
        Constant denominator (num_global_sequences * seq_len) removes length
        bias. Each token contributes equally regardless of sequence length.

    sequence_mean: mean per source episode, then sum / num_global_sequences
        Episode boundaries come from sample_ids (multiple episodes may be packed
        into one row). NOTE: per-sequence averaging introduces a length bias, as
        discussed in the DR-GRPO paper.

    Args:
        per_token_loss (torch.Tensor): Per-token loss (B, S).
        loss_mask (torch.Tensor): Valid token mask (B, S).
        agg_type (AggType): Aggregation strategy.
        normalization (LossNormalization): Global denominators.
        sample_ids (torch.Tensor | None): Source-episode ids (B, S); required for
            sequence_mean.

    Returns:
        torch.Tensor: Scalar loss.
    """
    if agg_type == "token_mean":
        return (per_token_loss * loss_mask).sum() / max(
            normalization.num_global_valid_tokens, 1
        )
    if agg_type == "fixed_horizon":
        return (per_token_loss * loss_mask).sum() / max(
            normalization.num_global_fixed_horizon_tokens, 1
        )
    if agg_type == "sequence_mean":
        if sample_ids is None:
            raise ValueError("sample_ids is required for sequence_mean aggregation")
        return _sequence_mean_loss(per_token_loss, loss_mask, sample_ids, normalization)
    raise ValueError(f"Unknown agg_type: {agg_type}")


def _sequence_mean_loss(
    per_token_loss: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_ids: torch.Tensor,
    normalization: LossNormalization,
) -> torch.Tensor:
    """Mean each source episode, then sum(per-episode means) / num_global_sequences."""
    valid = loss_mask.bool() & (sample_ids >= 0)
    if not bool(valid.any()):
        return per_token_loss.sum() * 0.0  # finite 0 that keeps the autograd graph

    _unique, inverse = torch.unique(sample_ids[valid], sorted=True, return_inverse=True)
    sums = torch.zeros(
        int(inverse.max()) + 1,
        device=per_token_loss.device,
        dtype=per_token_loss.dtype,
    )
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, inverse, per_token_loss[valid])
    counts.scatter_add_(0, inverse, torch.ones_like(per_token_loss[valid]))
    per_sequence = sums / counts.clamp_min(1)
    return per_sequence.sum() / max(normalization.num_global_sequences, 1)


def pg_ppo_clip(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    normalization: LossNormalization,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        normalization (LossNormalization): Global token denominator for metrics.
        clip_low (float): Lower bound offset. Ratio is clamped to min of (1 - clip_low).
            E.g., clip_low=0.2 means ratio >= 0.8. Default: 0.2.
        clip_high (float): Upper bound offset. Ratio is clamped to max of (1 + clip_high).
            E.g., clip_high=0.2 means ratio <= 1.2. Default: 0.2.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: (pg_loss, metrics); pg_loss
            is (B, S). `high/frac` / `low/frac` are the fractions of valid tokens
            clipped against a positive / negative advantage respectively.
    """
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    pg_loss = torch.maximum(-ratio * advantages, -clipped_ratio * advantages)
    with torch.no_grad():
        clipped_high = ratio > clipped_ratio  # ratio above 1 + clip_high
        clipped_low = ratio < clipped_ratio  # ratio below 1 - clip_low
        metrics = {
            "loss/clip/clipped_ratio/mean": masked_token_mean(
                clipped_ratio, loss_mask, normalization
            ),
            "loss/clip/high/frac": masked_token_mean(
                (clipped_high & (advantages > 0)).float(), loss_mask, normalization
            ),
            "loss/clip/low/frac": masked_token_mean(
                (clipped_low & (advantages < 0)).float(), loss_mask, normalization
            ),
        }
    return pg_loss, metrics
