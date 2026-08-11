# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.distributed.parallel_dims import ParallelDims


def resolve_parameter_owner_meshes(
    value: torch.Tensor,
    *,
    parallel_dims: ParallelDims,
) -> tuple[DeviceMesh, ...]:
    """Validate a parameter and return its non-replica owner meshes."""
    if parallel_dims.spmd_backend != "default":
        raise ValueError("parameter statistics require spmd_backend='default'")
    if not isinstance(value, DTensor):
        raise ValueError("parameter statistics require a DTensor")
    if any(placement.is_partial() for placement in value.placements):
        raise ValueError("parameter statistics do not accept Partial placements")

    dense_mesh_axis_names = (
        *(("dp_replicate",) if parallel_dims.dp_replicate > 1 else ()),
        "fsdp",
        *(("tp",) if parallel_dims.tp > 1 else ()),
    )
    sparse_mesh_axis_names = (
        *(("dp_replicate",) if parallel_dims.dp_replicate > 1 else ()),
        "efsdp",
        "ep",
    )
    mesh = value.device_mesh
    mesh_axis_names = mesh.mesh_dim_names
    if mesh_axis_names == dense_mesh_axis_names:
        resolved_mesh_axis_names = dense_mesh_axis_names
        storage_axis = "fsdp"
        parallel_axis = "tp" if parallel_dims.tp > 1 else None
    elif parallel_dims.ep > 1 and mesh_axis_names == sparse_mesh_axis_names:
        resolved_mesh_axis_names = sparse_mesh_axis_names
        storage_axis = "efsdp"
        parallel_axis = "ep"
    else:
        raise ValueError(
            "parameter statistics expected dense mesh axes "
            f"{dense_mesh_axis_names} or sparse mesh axes {sparse_mesh_axis_names}, "
            f"got {mesh_axis_names}"
        )
    placement_by_axis = dict(
        zip(resolved_mesh_axis_names, value.placements, strict=True)
    )
    if (
        parallel_dims.dp_replicate > 1
        and placement_by_axis["dp_replicate"] != Replicate()
    ):
        raise ValueError(
            "parameter statistics require Replicate on the dp_replicate axis"
        )
    storage_placement = placement_by_axis[storage_axis]
    if not isinstance(storage_placement, (Shard, _StridedShard)):
        raise ValueError(
            f"parameter statistics require a shard on the {storage_axis} axis"
        )
    if parallel_axis == "tp":
        tp_placement = placement_by_axis[parallel_axis]
        if not (
            tp_placement.is_replicate()
            or isinstance(tp_placement, (Shard, _StridedShard))
        ):
            raise ValueError(
                "parameter statistics require Replicate or Shard on the TP axis"
            )
    elif parallel_axis == "ep":
        ep_placement = placement_by_axis[parallel_axis]
        if not (
            isinstance(ep_placement, (Shard, _StridedShard)) and ep_placement.dim == 0
        ):
            raise ValueError(
                "parameter statistics require a dim-0 shard on the EP axis"
            )

    expected_mesh = parallel_dims.get_mesh(list(resolved_mesh_axis_names))
    if mesh.device_type != expected_mesh.device_type:
        raise ValueError("parameter statistics require the ParallelDims device type")
    if not torch.equal(mesh.mesh, expected_mesh.mesh):
        raise ValueError("parameter statistics require the ParallelDims rank grid")

    owner_meshes = []
    for axis_name, placement in placement_by_axis.items():
        if isinstance(placement, (Shard, _StridedShard)):
            axis_mesh = parallel_dims.get_mesh(axis_name)
            if axis_mesh.size() > 1:
                owner_meshes.append(axis_mesh)
    return tuple(owner_meshes)
