# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _accumulate_tensor_statistics_cuda(
    value_ptr,
    counts_ptr,
    sums_ptr,
    maximum_ptr,
    enabled_ptr,
    value_count,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.load(enabled_ptr) == 0:
        return

    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    present = offsets < value_count
    value = tl.load(value_ptr + offsets, mask=present, other=0.0).to(tl.float32)
    finite = (
        present & (value == value) & (value != float("inf")) & (value != -float("inf"))
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
    tl.atomic_add(counts_ptr, tl.sum(present_i64))
    tl.atomic_add(counts_ptr + 1, tl.sum(nonfinite_i64))
    tl.atomic_add(counts_ptr + 2, tl.sum(zero_i64))
    if tl.program_id(0) == 0:
        tl.atomic_add(counts_ptr + 3, 1)

    tl.atomic_add(sums_ptr, tl.sum(absolute))
    tl.atomic_add(sums_ptr + 1, tl.sum(square))
    tl.atomic_add(sums_ptr + 2, tl.sum(square * square))
    tl.atomic_max(
        maximum_ptr,
        tl.max(tl.where(finite, absolute, -float("inf"))),
    )


class _StatisticSlot(nn.Module):
    """Independent cotangent buffers for one registered key."""

    counts: torch.Tensor
    sums: torch.Tensor
    maximum: torch.Tensor

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.register_buffer(
            "counts",
            torch.zeros(4, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "sums",
            torch.zeros(3, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "maximum",
            torch.zeros((), dtype=torch.float32, device=device),
            persistent=False,
        )


class StatisticBuffers(nn.Module):
    """Mutable sufficient-statistic buffers shared by every registered key."""

    counts: torch.Tensor
    sums: torch.Tensor
    maxima: torch.Tensor
    enabled: torch.Tensor
    suppressed: torch.Tensor
    cotangents: nn.ModuleList

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
        self.cotangents = nn.ModuleList(
            [_StatisticSlot(device=device) for _ in range(key_count)]
        )
        self.register_buffer(
            "enabled",
            torch.zeros((), dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "suppressed",
            torch.zeros((), dtype=torch.int32, device=device),
            persistent=False,
        )

    def clear(self) -> None:
        self.counts.zero_()
        self.sums.zero_()
        self.maxima.fill_(-torch.inf)
        if self.cotangents:
            slots = [cast(_StatisticSlot, slot) for slot in self.cotangents]
            torch._foreach_zero_([slot.counts for slot in slots])
            torch._foreach_zero_([slot.sums for slot in slots])
            torch._foreach_zero_([slot.maximum for slot in slots])

    def stack_cotangents(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = [cast(_StatisticSlot, slot) for slot in self.cotangents]
        return (
            torch.stack([slot.counts for slot in slots]),
            torch.stack([slot.sums for slot in slots]),
            torch.stack([slot.maximum for slot in slots]),
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

    if value.is_cuda and value.is_contiguous():
        block_size = 4096
        program_count = max(1, (value.numel() + block_size - 1) // block_size)
        _accumulate_tensor_statistics_cuda[(program_count,)](
            value,
            counts,
            sums,
            maximum,
            enabled,
            value.numel(),
            BLOCK_SIZE=block_size,  # pyrefly: ignore [bad-argument-type]
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
