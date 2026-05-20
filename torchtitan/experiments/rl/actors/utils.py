# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer-side helpers: per-token logprob computation, memory stats,
drift verification.

All extraction is mask-based: the trainer reads ``loss_mask`` (1 on
tokens to learn from, 0 elsewhere) instead of doing arithmetic on
``(prompt_lens, response_lens)``. This unblocks multi-turn rollouts
where there's no single prompt/response boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.observability import structured_logger as sl

_GIB = 1024**3


def _bytes_to_gib(value: float) -> float:
    return float(value) / _GIB


def reset_cuda_peak_memory_stats(device: torch.device | None = None) -> None:
    """Reset CUDA allocator peak stats when CUDA is available."""
    if not torch.cuda.is_available():
        return
    if device is not None and device.type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    """Return current and peak CUDA memory stats for the visible device."""
    if not torch.cuda.is_available():
        return {}
    if device is not None and device.type != "cuda":
        return {}

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)
    capacity = float(props.total_memory)
    free, total = torch.cuda.mem_get_info(device)
    driver_used = float(total - free)
    stats = torch.cuda.memory_stats(device)
    active_peak = float(
        stats.get("active_bytes.all.peak", torch.cuda.max_memory_allocated(device))
    )
    reserved_peak = float(
        stats.get("reserved_bytes.all.peak", torch.cuda.max_memory_reserved(device))
    )
    allocated = float(torch.cuda.memory_allocated(device))
    reserved = float(torch.cuda.memory_reserved(device))
    return {
        "allocated_gib": _bytes_to_gib(allocated),
        "allocated_pct": 100.0 * allocated / capacity,
        "reserved_gib": _bytes_to_gib(reserved),
        "reserved_pct": 100.0 * reserved / capacity,
        "peak_allocated_gib": _bytes_to_gib(active_peak),
        "peak_allocated_pct": 100.0 * active_peak / capacity,
        "peak_reserved_gib": _bytes_to_gib(reserved_peak),
        "peak_reserved_pct": 100.0 * reserved_peak / capacity,
        "driver_used_gib": _bytes_to_gib(driver_used),
        "driver_used_pct": 100.0 * driver_used / capacity,
        "driver_free_gib": _bytes_to_gib(float(free)),
        "driver_free_pct": 100.0 * float(free) / capacity,
        "alloc_retries": float(stats.get("num_alloc_retries", 0)),
        "ooms": float(stats.get("num_ooms", 0)),
    }


@sl.log_trace_span("compute_logprobs")
def compute_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token log p(next | <=t) from logits + targets.

    Args:
        logits: ``[B, T, V]`` next-token logits (typically B=1 for varlen).
        token_ids: ``[B, T]`` token sequence.
        temperature: divide logits by this before ``log_softmax``. Must
            match the sampling temperature the generator used so the
            trainer's logprobs are computed on the *same* distribution
            the rollouts came from — without it, the importance ratio
            ``exp(policy_lp - behavior_lp)`` is biased even at step 0
            with identical weights.

    Returns:
        ``[B, T - 1]`` per-token logprobs. Position ``t`` is
        ``log p(token_{t+1} | token_{<=t})``. Cross-sample-boundary
        positions in a packed batch are present but unused; the loss
        mask zeros them out.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    from torch.distributed.tensor import DTensor

    # Config-based TP can return logits as a Replicate DTensor. Downstream
    # code does plain-tensor indexing; materialize once.
    if isinstance(logits, DTensor):
        # TODO: pass `grad_placements=[Replicate(), ...]` to make the autograd
        # contract explicit (see .claude/rules/distributed.md).
        logits = logits.to_local()
    # fp32 BEFORE temperature division to preserve precision under bf16.
    shift_logits = logits[:, :-1, :].float() / temperature
    shift_targets = token_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    return logprobs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True, slots=True)
class PartialLogprobDrift:
    """Per-rank generator-vs-trainer drift; ready for cross-rank reduction.

    Args:
        logprob_diff_mean: Scalar; to be SUM-reduced across DP.
        logprob_diff_max: Scalar; to be MAX-reduced across DP.
        ratio_tokens_different: Scalar; to be SUM-reduced across DP.
    """

    logprob_diff_mean: torch.Tensor
    logprob_diff_max: torch.Tensor
    ratio_tokens_different: torch.Tensor


@torch.no_grad()
@sl.log_trace_span("verify_logprob_identity")
def verify_logprob_identity(
    *,
    behavior_logprobs: torch.Tensor,
    policy_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    num_global_valid_tokens: torch.Tensor,
) -> PartialLogprobDrift:
    """Mask-aware drift between rollout-time and trainer-time logprobs.

    Args:
        behavior_logprobs: ``[1, T - 1]`` rollout-time logprobs (shifted
            to align with the model's predicted positions; 0 where the
            mask is 0).
        policy_logprobs: ``[1, T - 1]`` trainer-time logprobs.
        loss_mask: ``[1, T - 1]`` 1 on loss positions.
        num_global_valid_tokens: Scalar; sum of ``loss_mask`` across all
            DP ranks. Used to normalize the SUM-reduced metrics.

    Returns:
        :class:`PartialLogprobDrift` with scalars on the trainer device.
    """
    diff = (policy_logprobs - behavior_logprobs) * loss_mask
    abs_diff = diff.abs()
    above_eps = (abs_diff > 1e-6).float() * loss_mask
    return PartialLogprobDrift(
        logprob_diff_mean=diff.sum() / num_global_valid_tokens,
        logprob_diff_max=abs_diff.max(),
        ratio_tokens_different=above_eps.sum() / num_global_valid_tokens,
    )
