# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


@dataclass(frozen=True, slots=True)
class FiniteStatistics:
    counts: torch.Tensor
    sums: torch.Tensor
    abs_max: torch.Tensor


def finite_statistics(value: torch.Tensor) -> FiniteStatistics:
    """Build fixed-shape statistics from local floating-point tensor storage."""
    if isinstance(value, DTensor):
        raise TypeError("finite_statistics expects local tensor storage, not a DTensor")
    if not value.is_floating_point():
        raise TypeError("finite_statistics expects a floating-point tensor")

    value_fp32 = value.detach().float()
    finite = torch.isfinite(value_fp32)
    finite_value = torch.where(finite, value_fp32, 0.0)
    finite_abs = torch.abs(finite_value)

    counts = torch.stack(
        (
            torch.full((), value.numel(), dtype=torch.int64, device=value.device),
            torch.count_nonzero(~finite),
            torch.count_nonzero(finite & (value_fp32 == 0)),
        )
    )
    sums = torch.stack((finite_abs.sum(), torch.square(finite_value).sum()))
    abs_max = (
        finite_abs.amax().reshape(1)
        if value.numel() > 0
        else torch.zeros(1, dtype=torch.float32, device=value.device)
    )
    return FiniteStatistics(counts=counts, sums=sums, abs_max=abs_max)


def derive_finite_statistics(statistics: FiniteStatistics) -> dict[str, int | float]:
    """Derive scalar metrics from already-completed CPU statistics."""
    if any(
        value.device.type != "cpu"
        for value in (statistics.counts, statistics.sums, statistics.abs_max)
    ):
        raise ValueError("derive_finite_statistics expects completed CPU statistics")

    numel, nonfinite_count, zero_count = (
        int(value) for value in statistics.counts.tolist()
    )
    result: dict[str, int | float] = {
        "numel": numel,
        "nonfinite_count": nonfinite_count,
        "zero_count": zero_count,
    }
    finite_count = numel - nonfinite_count
    if finite_count == 0:
        return result

    abs_sum, square_sum = (float(value) for value in statistics.sums.tolist())
    square_mean = square_sum / finite_count
    result.update(
        {
            "zero_fraction": zero_count / finite_count,
            "abs_mean": abs_sum / finite_count,
            "square_mean": square_mean,
            "rms": math.sqrt(square_mean),
            "abs_max": float(statistics.abs_max.item()),
        }
    )
    return result


def reduce_sum(value: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
    """Sum a device tensor over one family-resolved mesh."""
    return funcol.wait_tensor(funcol.all_reduce(value, reduceOp="sum", group=mesh))


def reduce_max(value: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
    """Take a device-tensor maximum over one family-resolved mesh."""
    return funcol.wait_tensor(funcol.all_reduce(value, reduceOp="max", group=mesh))


def reduce_finite_statistics(
    statistics: FiniteStatistics,
    owner_meshes: tuple[DeviceMesh, ...],
) -> FiniteStatistics:
    """Reduce one owned snapshot over an ordered sequence of owner meshes."""
    counts = statistics.counts.clone()
    sums = statistics.sums.clone()
    abs_max = statistics.abs_max.clone()
    for mesh in owner_meshes:
        counts = reduce_sum(counts, mesh)
        sums = reduce_sum(sums, mesh)
        abs_max = reduce_max(abs_max, mesh)
    return FiniteStatistics(counts=counts, sums=sums, abs_max=abs_max)
