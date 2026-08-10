# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.recorders import FiniteStatistics


def resolve_rowwise_parameter_owner_meshes(
    value: torch.Tensor,
    *,
    parallel_dims: ParallelDims,
) -> tuple[DeviceMesh, ...]:
    """Validate a rowwise parameter and return its non-replica owner meshes."""
    if parallel_dims.spmd_backend != "default":
        raise ValueError("rowwise parameter statistics require spmd_backend='default'")
    if not isinstance(value, DTensor):
        raise ValueError("rowwise parameter statistics require a DTensor")
    if any(placement.is_partial() for placement in value.placements):
        raise ValueError(
            "rowwise parameter statistics do not accept Partial placements"
        )

    expected_mesh_axis_names = (
        *(("dp_replicate",) if parallel_dims.dp_replicate > 1 else ()),
        "fsdp",
        *(("tp",) if parallel_dims.tp > 1 else ()),
    )
    expected_placements = (
        *((Replicate(),) if parallel_dims.dp_replicate > 1 else ()),
        Shard(0),
        *((Shard(1),) if parallel_dims.tp > 1 else ()),
    )
    mesh = value.device_mesh
    if mesh.mesh_dim_names != expected_mesh_axis_names:
        raise ValueError(
            "rowwise parameter statistics expected mesh axes "
            f"{expected_mesh_axis_names}, got {mesh.mesh_dim_names}"
        )
    if value.placements != expected_placements:
        raise ValueError(
            "rowwise parameter statistics expected placements "
            f"{expected_placements}, got {value.placements}"
        )

    expected_mesh = parallel_dims.get_mesh(list(expected_mesh_axis_names))
    if mesh.device_type != expected_mesh.device_type:
        raise ValueError(
            "rowwise parameter statistics require the ParallelDims device type"
        )
    if not torch.equal(mesh.mesh, expected_mesh.mesh):
        raise ValueError(
            "rowwise parameter statistics require the ParallelDims rank grid"
        )

    owner_meshes = []
    fsdp_mesh = parallel_dims.get_mesh("fsdp")
    if parallel_dims.fsdp_enabled:
        owner_meshes.append(fsdp_mesh)
    if parallel_dims.tp_enabled:
        owner_meshes.append(parallel_dims.get_mesh("tp"))
    return tuple(owner_meshes)


def validate_reduced_parameter_numel(
    statistics: FiniteStatistics,
    expected_numel: int,
) -> None:
    """Require a completed host snapshot to cover one logical parameter."""
    if statistics.counts.device.type != "cpu":
        raise ValueError("parameter numel validation requires completed CPU statistics")
    reduced_numel = int(statistics.counts[0].item())
    if reduced_numel != expected_numel:
        raise ValueError(
            f"reduced parameter numel is {reduced_numel}, expected {expected_numel}"
        )
