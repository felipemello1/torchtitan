# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Placement

from torchtitan.distributed.utils import check_dtensor_placements_match


@dataclass(frozen=True, slots=True)
class FiniteStatistics:
    counts: torch.Tensor
    sums: torch.Tensor
    abs_max: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ReductionRequest:
    value: torch.Tensor
    meshes: tuple[DeviceMesh, ...]
    operation: str


def _accumulate_finite_statistics(
    value: torch.Tensor,
    counts: torch.Tensor,
    sums: torch.Tensor,
    abs_max: torch.Tensor,
) -> None:
    finite = torch.isfinite(value)
    finite_value = torch.where(finite, value, 0.0).float()
    finite_abs = torch.abs(finite_value)

    counts[0].add_(value.numel())
    counts[1].add_(torch.count_nonzero(~finite))
    counts[2].add_(torch.count_nonzero(finite & (value == 0)))
    sums[0].add_(finite_abs.sum())
    sums[1].add_(torch.square(finite_value).sum())
    abs_max.copy_(torch.maximum(abs_max, finite_abs.amax().reshape(1)))


class ReductionBatch:
    """Pack tensor telemetry reductions by mesh, dtype, and operation."""

    def __init__(self) -> None:
        self._requests: list[_ReductionRequest] = []

    def sum(
        self,
        value: torch.Tensor,
        meshes: tuple[DeviceMesh, ...],
    ) -> torch.Tensor:
        self._requests.append(_ReductionRequest(value, meshes, "sum"))
        return value

    def max(
        self,
        value: torch.Tensor,
        meshes: tuple[DeviceMesh, ...],
    ) -> torch.Tensor:
        self._requests.append(_ReductionRequest(value, meshes, "max"))
        return value

    def reduce(self) -> None:
        """Execute at most one collective per stage, mesh, dtype, and operation."""
        stages = max((len(request.meshes) for request in self._requests), default=0)
        for stage in range(stages):
            groups: dict[
                tuple[object, ...],
                tuple[DeviceMesh, str, list[torch.Tensor]],
            ] = {}
            for request in self._requests:
                if stage >= len(request.meshes):
                    continue
                mesh = request.meshes[stage]
                mesh_key = (
                    mesh.device_type,
                    mesh.mesh_dim_names,
                    tuple(mesh.mesh.reshape(-1).tolist()),
                )
                key = (
                    mesh_key,
                    request.value.device,
                    request.value.dtype,
                    request.operation,
                )
                if key not in groups:
                    groups[key] = (mesh, request.operation, [])
                groups[key][2].append(request.value)

            for mesh, operation, values in groups.values():
                packed = torch.cat([value.reshape(-1) for value in values])
                packed = funcol.wait_tensor(
                    funcol.all_reduce(packed, reduceOp=operation, group=mesh)
                )
                offset = 0
                for value in values:
                    next_offset = offset + value.numel()
                    value.copy_(packed[offset:next_offset].view_as(value))
                    offset = next_offset


def validate_tp_tensor(
    value: torch.Tensor,
    *,
    tp_mesh: DeviceMesh | None,
    expected_placements: tuple[Placement, ...],
    label: str,
) -> None:
    """Require local TP=1 storage or one exact ParallelDims TP DTensor."""
    if tp_mesh is None:
        if isinstance(value, DTensor):
            raise ValueError(f"TP=1 {label} must be a local tensor")
        return
    if not isinstance(value, DTensor):
        raise ValueError(f"TP>1 {label} must be a DTensor")
    if (
        value.device_mesh.device_type != tp_mesh.device_type
        or value.device_mesh.mesh_dim_names != tp_mesh.mesh_dim_names
        or not torch.equal(value.device_mesh.mesh, tp_mesh.mesh)
    ):
        raise ValueError(f"{label} must use the ParallelDims TP mesh")
    if not check_dtensor_placements_match(
        value.placements,
        expected_placements,
        value.ndim,
    ):
        raise ValueError(
            f"expected placements {expected_placements}, got {value.placements}"
        )


def bounded_tensor_views(
    value: torch.Tensor,
    *,
    max_chunk_elements: int,
) -> tuple[torch.Tensor, ...]:
    """Split local storage into bounded views without copying it.

    Args:
        value: Local tensor storage.
        max_chunk_elements: Maximum elements in each returned view.

    Example:

        views = bounded_tensor_views(torch.ones(4, 8), max_chunk_elements=10)
        assert max(view.numel() for view in views) <= 10
    """
    if max_chunk_elements <= 0:
        raise ValueError("max_chunk_elements must be positive")
    chunks = [value.detach()]
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
    return tuple(chunks)


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
    chunks = (
        (value.detach(),)
        if max_chunk_elements is None
        else bounded_tensor_views(value, max_chunk_elements=max_chunk_elements)
    )

    for chunk in chunks:
        _accumulate_finite_statistics(chunk, counts, sums, abs_max)
    return FiniteStatistics(counts=counts, sums=sums, abs_max=abs_max)


def derive_finite_statistics(statistics: FiniteStatistics) -> dict[str, int | float]:
    """Derive scalar metrics from already-completed CPU statistics."""
    if any(
        value.device.type != "cpu"
        for value in (statistics.counts, statistics.sums, statistics.abs_max)
    ):
        raise ValueError("derive_finite_statistics expects completed CPU statistics")

    return derive_finite_statistics_values(
        statistics.counts.tolist(),
        statistics.sums.tolist(),
        statistics.abs_max.item(),
    )


def derive_finite_statistics_values(
    counts: Sequence[int],
    sums: Sequence[float],
    abs_max: float,
) -> dict[str, int | float]:
    """Derive one row after its packed tensors have moved to the host."""
    numel, nonfinite_count, zero_count = counts
    result: dict[str, int | float] = {
        "numel": numel,
        "nonfinite_count": nonfinite_count,
        "zero_count": zero_count,
    }
    finite_count = numel - nonfinite_count
    if finite_count == 0:
        return result

    result["zero_fraction"] = zero_count / finite_count
    result["abs_max"] = abs_max

    abs_sum, square_sum = sums
    if math.isfinite(abs_sum):
        result["abs_mean"] = abs_sum / finite_count
    if math.isfinite(square_sum):
        square_mean = square_sum / finite_count
        result["square_mean"] = square_mean
        result["rms"] = math.sqrt(square_mean)
    return result


def reduce_sum(
    value: torch.Tensor,
    mesh: DeviceMesh,
    *,
    batch: ReductionBatch | None = None,
) -> torch.Tensor:
    """Sum a device tensor over one family-resolved mesh."""
    if batch is not None:
        return batch.sum(value, (mesh,))
    return funcol.wait_tensor(funcol.all_reduce(value, reduceOp="sum", group=mesh))


def reduce_max(
    value: torch.Tensor,
    mesh: DeviceMesh,
    *,
    batch: ReductionBatch | None = None,
) -> torch.Tensor:
    """Take a device-tensor maximum over one family-resolved mesh."""
    if batch is not None:
        return batch.max(value, (mesh,))
    return funcol.wait_tensor(funcol.all_reduce(value, reduceOp="max", group=mesh))


def reduce_finite_statistics(
    statistics: FiniteStatistics,
    owner_meshes: tuple[DeviceMesh, ...],
    *,
    batch: ReductionBatch | None = None,
) -> FiniteStatistics:
    """Reduce one owned snapshot over an ordered sequence of owner meshes."""
    counts = statistics.counts.clone()
    sums = statistics.sums.clone()
    abs_max = statistics.abs_max.clone()
    if batch is not None:
        return FiniteStatistics(
            counts=batch.sum(counts, owner_meshes),
            sums=batch.sum(sums, owner_meshes),
            abs_max=batch.max(abs_max, owner_meshes),
        )
    for mesh in owner_meshes:
        counts = reduce_sum(counts, mesh)
        sums = reduce_sum(sums, mesh)
        abs_max = reduce_max(abs_max, mesh)
    return FiniteStatistics(counts=counts, sums=sums, abs_max=abs_max)
