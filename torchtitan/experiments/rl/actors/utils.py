# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    """Compute per-token logprobs from logits.

    Returns logprobs for positions 1..N (the predicted tokens).
    Output shape is ``[batch, seq_len - 1]``.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    from torch.distributed.tensor import DTensor

    # Config-based TP returns logits as a Replicate DTensor. Downstream RL
    # code (gather with plain-tensor indices, slicing per-sample) expects a
    # plain tensor - materialize once here.
    if isinstance(logits, DTensor):
        # TODO: pass `grad_placements=[Replicate(), ...]` to make the autograd
        # contract explicit (see .claude/rules/distributed.md).
        logits = logits.to_local()
    shift_logits = logits[:, :-1, :].float() / temperature
    shift_targets = token_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    return logprobs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True, slots=True)
class MaskedLogprobs:
    """Token-selected tensors consumed by GRPO loss and drift metrics."""

    policy_logprobs: torch.Tensor
    behavior_logprobs: torch.Tensor
    advantages: torch.Tensor


@sl.log_trace_span("extract_masked_logprobs")
def extract_masked_logprobs(
    packed_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> MaskedLogprobs:
    """Select loss-token logprobs from packed next-token predictions.

    ``packed_logprobs[:, i]`` predicts ``token_ids[:, i + 1]``. The inputs
    are token-aligned, so we drop position 0 from masks and auxiliary tensors
    before selecting loss tokens.

    Example::

        token_ids          = [[10, 11, 20, 21]]
        loss_mask          = [[ 0,  0,  1,  1]]
        packed_logprobs    = [[p11, p20, p21]]
        selected policy    = [p20, p21]
    """
    shifted_mask = loss_mask[:, 1:].bool()
    return MaskedLogprobs(
        policy_logprobs=packed_logprobs[shifted_mask],
        behavior_logprobs=behavior_logprobs[:, 1:][shifted_mask],
        advantages=advantages[:, 1:][shifted_mask],
    )


@dataclass(frozen=True, slots=True)
class LogprobDrift:
    """Per-rank generator-vs-trainer logprob drift awaiting reduction across the loss-mesh.

    Args:
        logprob_diff_mean: Scalar tensor; To be sum-reduced.
        logprob_diff_max: Scalar tensor; To be max-reduced.
        ratio_tokens_different: Scalar tensor; To be sum-reduced.
        nonfinite_logprob_frac: Scalar tensor; To be sum-reduced.
    """

    logprob_diff_mean: torch.Tensor
    logprob_diff_max: torch.Tensor
    ratio_tokens_different: torch.Tensor
    nonfinite_logprob_frac: torch.Tensor


@torch.no_grad()
@sl.log_trace_span("compute_logprob_drift")
def compute_logprob_drift(
    generator_token_logprobs: torch.Tensor,
    trainer_token_logprobs: torch.Tensor,
    *,
    num_global_valid_tokens: torch.Tensor,
    device: torch.device,
) -> LogprobDrift:
    """Compute per-rank drift between generator and trainer logprobs.

    Args:
        generator_token_logprobs: Generator-side per-token logprobs, shaped
            ``[num_loss_tokens]``.
        trainer_token_logprobs: Trainer-side per-token logprobs, shaped
            ``[num_loss_tokens]``.
        num_global_valid_tokens (torch.Tensor): Scalar tensor holding global token count
             across DP ranks. Used to normalize the output metrics.
        device: Device to use for tensor allocation, so metrics are ready for
            reduction across loss_mesh.

    Returns:
        LogprobDrift.
    """
    generator_flat = generator_token_logprobs.to(device=device, dtype=torch.float32)
    trainer_flat = trainer_token_logprobs.to(device=device, dtype=torch.float32)

    if generator_flat.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return LogprobDrift(zero, zero, zero, zero)

    # 1e-6 threshold ignores bf16-quantization-level diffs
    finite_mask = torch.isfinite(generator_flat) & torch.isfinite(trainer_flat)
    diff = torch.where(
        finite_mask,
        trainer_flat - generator_flat,
        torch.zeros_like(trainer_flat),
    )
    abs_diff = diff.abs()
    logprob_diff_max = (
        abs_diff[finite_mask].max()
        if bool(finite_mask.any().item())
        else torch.zeros((), dtype=torch.float32, device=device)
    )
    return LogprobDrift(
        logprob_diff_mean=diff.sum() / num_global_valid_tokens,
        logprob_diff_max=logprob_diff_max,
        ratio_tokens_different=(diff.abs() > 1e-6).sum() / num_global_valid_tokens,
        nonfinite_logprob_frac=(~finite_mask).sum() / num_global_valid_tokens,
    )
