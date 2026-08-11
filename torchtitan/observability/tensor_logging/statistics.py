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


def finite_statistics(
    value: torch.Tensor,
    *,
    max_chunk_elements: int | None = None,
) -> FiniteStatistics:
    """Build fixed-shape statistics from local floating-point tensor storage.

    Args:
        value: Local tensor storage. DTensor inputs must be unwrapped first.
        max_chunk_elements: Maximum elements converted to FP32 at once, or all
            elements when omitted.

    Example:

        stats = finite_statistics(
            torch.tensor([0.0, -2.0, float("nan")]),
            max_chunk_elements=2,
        )
        # stats.counts == [3, 1, 1]
        # stats.sums == [2.0, 4.0]
    """
    if isinstance(value, DTensor):
        raise TypeError("finite_statistics expects local tensor storage, not a DTensor")
    if not value.is_floating_point():
        raise TypeError("finite_statistics expects a floating-point tensor")
    if max_chunk_elements is not None and max_chunk_elements <= 0:
        raise ValueError("max_chunk_elements must be positive")

    counts = torch.zeros(3, dtype=torch.int64, device=value.device)
    sums = torch.zeros(2, dtype=torch.float32, device=value.device)
    abs_max = torch.zeros(1, dtype=torch.float32, device=value.device)
    if value.numel() == 0:
        return FiniteStatistics(counts=counts, sums=sums, abs_max=abs_max)

    # Split tensor views before FP32 conversion so strided inputs stay bounded.
    chunks = [value.detach()]
    if max_chunk_elements is not None:
        for dimension in range(value.ndim):
            split_chunks = []
            for chunk in chunks:
                if chunk.numel() <= max_chunk_elements:
                    split_chunks.append(chunk)
                    continue
                elements_per_index = chunk.numel() // chunk.shape[dimension]
                indices_per_chunk = max(
                    1,
                    max_chunk_elements // elements_per_index,
                )
                split_chunks.extend(chunk.split(indices_per_chunk, dim=dimension))
            chunks = split_chunks

    for chunk in chunks:
        finite = torch.isfinite(chunk)
        finite_value = torch.where(finite, chunk, 0.0).float()
        finite_abs = torch.abs(finite_value)

        counts[0].add_(chunk.numel())
        counts[1].add_(torch.count_nonzero(~finite))
        counts[2].add_(torch.count_nonzero(finite & (chunk == 0)))
        sums[0].add_(finite_abs.sum())
        sums[1].add_(torch.square(finite_value).sum())
        abs_max.copy_(torch.maximum(abs_max, finite_abs.amax().reshape(1)))
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

    result["zero_fraction"] = zero_count / finite_count
    result["abs_max"] = float(statistics.abs_max.item())

    abs_sum, square_sum = (float(value) for value in statistics.sums.tolist())
    if math.isfinite(abs_sum):
        result["abs_mean"] = abs_sum / finite_count
    if math.isfinite(square_sum):
        square_mean = square_sum / finite_count
        result["square_mean"] = square_mean
        result["rms"] = math.sqrt(square_mean)
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
