# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl
from torch import nn


_MAX_PROGRAMS = 1024


@triton.jit
def _accumulate_tensor_statistics_triton(
    value_ptr,
    counts_ptr,
    sums_ptr,
    maximum_ptr,
    enabled_ptr,
    value_count,
    BLOCK_SIZE: tl.constexpr,
    NEEDS_LOOP: tl.constexpr,
):
    if tl.load(enabled_ptr) == 0:
        return

    present_count = tl.zeros((), dtype=tl.int64)
    nonfinite_count = tl.zeros((), dtype=tl.int64)
    zero_count = tl.zeros((), dtype=tl.int64)
    absolute_sum = tl.zeros((), dtype=tl.float32)
    square_sum = tl.zeros((), dtype=tl.float32)
    fourth_moment_sum = tl.zeros((), dtype=tl.float32)
    absolute_maximum = tl.full((), -float("inf"), dtype=tl.float32)

    program_start = tl.program_id(0) * BLOCK_SIZE
    if NEEDS_LOOP:
        program_stride = tl.num_programs(0) * BLOCK_SIZE
        for block_start in tl.range(
            program_start,
            value_count,
            program_stride,
            num_stages=3,
        ):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            present = offsets < value_count
            value = tl.load(value_ptr + offsets, mask=present, other=0.0).to(tl.float32)
            finite = (
                present
                & (value == value)
                & (value != float("inf"))
                & (value != -float("inf"))
            )
            finite_value = tl.where(finite, value, 0.0)
            absolute = tl.abs(finite_value)
            square = finite_value * finite_value
            present_i64 = present.to(tl.int64)  # pyrefly: ignore [missing-attribute]
            nonfinite = present & ~finite  # pyrefly: ignore [deprecated]
            nonfinite_i64 = nonfinite.to(  # pyrefly: ignore [missing-attribute]
                tl.int64
            )
            zero_i64 = (  # pyrefly: ignore [missing-attribute]
                finite & (value == 0.0)
            ).to(tl.int64)
            present_count += tl.sum(present_i64)
            nonfinite_count += tl.sum(nonfinite_i64)
            zero_count += tl.sum(zero_i64)
            absolute_sum += tl.sum(absolute)
            square_sum += tl.sum(square)
            fourth_moment_sum += tl.sum(square * square)
            absolute_maximum = tl.maximum(
                absolute_maximum,
                tl.max(tl.where(finite, absolute, -float("inf"))),
            )
    else:
        offsets = program_start + tl.arange(0, BLOCK_SIZE)
        present = offsets < value_count
        value = tl.load(value_ptr + offsets, mask=present, other=0.0).to(tl.float32)
        finite = (
            present
            & (value == value)
            & (value != float("inf"))
            & (value != -float("inf"))
        )
        finite_value = tl.where(finite, value, 0.0)
        absolute = tl.abs(finite_value)
        square = finite_value * finite_value
        present_i64 = present.to(tl.int64)  # pyrefly: ignore [missing-attribute]
        nonfinite = present & ~finite  # pyrefly: ignore [deprecated]
        nonfinite_i64 = nonfinite.to(tl.int64)  # pyrefly: ignore [missing-attribute]
        zero_i64 = (finite & (value == 0.0)).to(  # pyrefly: ignore [missing-attribute]
            tl.int64
        )
        present_count = tl.sum(present_i64)
        nonfinite_count = tl.sum(nonfinite_i64)
        zero_count = tl.sum(zero_i64)
        absolute_sum = tl.sum(absolute)
        square_sum = tl.sum(square)
        fourth_moment_sum = tl.sum(square * square)
        absolute_maximum = tl.max(tl.where(finite, absolute, -float("inf")))

    tl.atomic_add(counts_ptr, present_count)
    tl.atomic_add(counts_ptr + 1, nonfinite_count)
    tl.atomic_add(counts_ptr + 2, zero_count)
    if tl.program_id(0) == 0:
        tl.atomic_add(counts_ptr + 3, 1)
    tl.atomic_add(sums_ptr, absolute_sum)
    tl.atomic_add(sums_ptr + 1, square_sum)
    tl.atomic_add(sums_ptr + 2, fourth_moment_sum)
    tl.atomic_max(maximum_ptr, absolute_maximum)


class StatisticBuffers(nn.Module):
    """Mutable sufficient-statistic buffers shared by every registered key."""

    counts: torch.Tensor
    sums: torch.Tensor
    maxima: torch.Tensor
    enabled: torch.Tensor

    def __init__(
        self,
        key_count: int,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "counts",
            torch.zeros((key_count, 4), dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "sums",
            torch.zeros((key_count, 3), dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "maxima",
            torch.full(
                (key_count,),
                -torch.inf,
                dtype=torch.float32,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "enabled",
            torch.zeros((), dtype=torch.int32, device=device),
            persistent=False,
        )

    def clear(self) -> None:
        self.counts.zero_()
        self.sums.zero_()
        self.maxima.fill_(-torch.inf)


def _normalize_tensor_layout(value: torch.Tensor) -> torch.Tensor:
    """Return a memory-ordered view, collapsing contiguous dimensions."""

    if value.ndim <= 1:
        return value

    dimension_order = sorted(
        range(value.ndim),
        key=lambda dimension: value.stride()[dimension],
        reverse=True,
    )
    value = value.permute(dimension_order)
    if value.is_contiguous() or value.ndim <= 2:
        return value

    shape_stride = [
        (size, stride)
        for size, stride in zip(value.shape, value.stride(), strict=True)
        if size != 1
    ]
    if len(shape_stride) <= 1:
        return value.reshape(value.numel())

    collapsed_shape = [shape_stride[0][0]]
    collapsed_stride = [shape_stride[0][1]]
    for size, stride in shape_stride[1:]:
        if collapsed_stride[-1] == stride * size:
            collapsed_shape[-1] *= size
            collapsed_stride[-1] = stride
        else:
            collapsed_shape.append(size)
            collapsed_stride.append(stride)

    if len(collapsed_shape) < len(shape_stride):
        value = value.as_strided(
            collapsed_shape,
            collapsed_stride,
            value.storage_offset(),
        )
    return value


def _accumulate_contiguous_tensor_statistics_triton(
    value: torch.Tensor,
    counts: torch.Tensor,
    sums: torch.Tensor,
    maximum: torch.Tensor,
    enabled: torch.Tensor,
) -> None:
    """Launch the Triton accumulator for one contiguous tensor."""

    block_size = 4096
    ideal_program_count = max(
        1,
        (value.numel() + block_size - 1) // block_size,
    )
    program_count = min(_MAX_PROGRAMS, ideal_program_count)
    needs_loop = ideal_program_count > program_count
    _accumulate_tensor_statistics_triton[(program_count,)](
        value,
        counts,
        sums,
        maximum,
        enabled,
        value.numel(),
        BLOCK_SIZE=block_size,  # pyrefly: ignore [bad-argument-type]
        NEEDS_LOOP=needs_loop,  # pyrefly: ignore [bad-argument-type]
    )


def _accumulate_normalized_tensor_statistics_triton(
    value: torch.Tensor,
    counts: torch.Tensor,
    sums: torch.Tensor,
    maximum: torch.Tensor,
    enabled: torch.Tensor,
) -> None:
    """Normalize one GPU tensor and accumulate it with Triton."""

    value = _normalize_tensor_layout(value)
    if not value.is_contiguous():
        value = value.contiguous()
    _accumulate_contiguous_tensor_statistics_triton(
        value,
        counts,
        sums,
        maximum,
        enabled,
    )


@torch.library.custom_op(
    "torchtitan::accumulate_tensor_statistics",
    mutates_args={"counts", "sums", "maximum"},
)
def accumulate_tensor_statistics(
    value: torch.Tensor,
    counts: torch.Tensor,
    sums: torch.Tensor,
    maximum: torch.Tensor,
    enabled: torch.Tensor,
) -> None:
    """Accumulate one tensor through an opaque, compile-safe operation."""

    if value.is_cuda:
        _accumulate_normalized_tensor_statistics_triton(
            value,
            counts,
            sums,
            maximum,
            enabled,
        )
        return

    with torch.no_grad():
        value = value.detach()
        enabled_i64 = enabled.to(torch.int64)
        enabled_fp32 = enabled.to(torch.float32)
        counts[3].add_(enabled_i64)
        if value.numel() == 0:
            return

        finite = torch.isfinite(value)
        value_fp32 = value.to(torch.float32)
        finite_value = torch.where(finite, value_fp32, 0.0)
        absolute = finite_value.abs()
        square = finite_value.square()

        counts[0].add_(value.numel() * enabled_i64)
        counts[1].add_(torch.count_nonzero(~finite) * enabled_i64)
        counts[2].add_(torch.count_nonzero(finite & (value == 0)) * enabled_i64)

        sums[0].add_(absolute.sum() * enabled_fp32)
        sums[1].add_(square.sum() * enabled_fp32)
        sums[2].add_(square.square().sum() * enabled_fp32)

        finite_absolute = torch.where(finite, value_fp32.abs(), -torch.inf)
        updated_maximum = torch.maximum(maximum, finite_absolute.amax())
        maximum.copy_(torch.where(enabled.bool(), updated_maximum, maximum))


@accumulate_tensor_statistics.register_fake
def _(
    value: torch.Tensor,
    counts: torch.Tensor,
    sums: torch.Tensor,
    maximum: torch.Tensor,
    enabled: torch.Tensor,
) -> None:
    return None
