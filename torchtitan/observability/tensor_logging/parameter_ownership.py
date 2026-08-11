# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.distributed.parallel_dims import ParallelDims


@dataclass(frozen=True, slots=True)
class BoundParameter:
    fqn: str
    aliases: tuple[str, ...]
    value: torch.Tensor
    numel: int


@dataclass(frozen=True, slots=True)
class ParameterOwnerGroup:
    parameters: tuple[BoundParameter, ...]
    owner_meshes: tuple[DeviceMesh, ...]
    reduction_meshes: tuple[DeviceMesh, ...]
    expected_contributors: int


def bind_parameters(
    model: nn.Module,
    *,
    layer_ids: tuple[int, ...] | None,
) -> tuple[BoundParameter, ...]:
    """Bind each trainable floating parameter once under its canonical name."""
    layer_prefixes = (
        tuple(f"layers.{layer_id}." for layer_id in layer_ids)
        if layer_ids is not None
        else None
    )
    parameters_by_identity: dict[int, torch.Tensor] = {}
    names_by_identity: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        fqn = canonical_fqn(name)
        if layer_prefixes is not None and not any(
            fqn.startswith(prefix) for prefix in layer_prefixes
        ):
            continue
        if not parameter.requires_grad or not parameter.is_floating_point():
            continue
        identity = id(parameter)
        parameters_by_identity[identity] = parameter
        names_by_identity.setdefault(identity, []).append(fqn)

    parameters = tuple(
        sorted(
            (
                BoundParameter(
                    fqn=min(names),
                    aliases=tuple(sorted(names)),
                    value=parameters_by_identity[identity],
                    numel=parameters_by_identity[identity].numel(),
                )
                for identity, names in names_by_identity.items()
            ),
            key=lambda parameter: parameter.fqn,
        )
    )
    fqns = tuple(parameter.fqn for parameter in parameters)
    if len(fqns) != len(set(fqns)):
        raise ValueError("tensor logging found duplicate canonical parameter FQNs")
    return parameters


def group_parameters_by_owner(
    parameters: tuple[BoundParameter, ...],
    *,
    parallel_dims: ParallelDims,
) -> tuple[ParameterOwnerGroup, ...]:
    """Group parameters that share one literal owner/reduction cohort."""
    parameters_by_owner: dict[tuple[str, ...], list[BoundParameter]] = {}
    owner_meshes_by_key: dict[tuple[str, ...], tuple[DeviceMesh, ...]] = {}
    for parameter in parameters:
        owner_meshes = resolve_parameter_owner_meshes(
            parameter.value,
            parallel_dims=parallel_dims,
        )
        owner_key = tuple(
            mesh.mesh_dim_names[0] for mesh in owner_meshes if mesh.mesh_dim_names
        )
        parameters_by_owner.setdefault(owner_key, []).append(parameter)
        owner_meshes_by_key[owner_key] = owner_meshes

    groups = []
    for owner_key, owned_parameters in parameters_by_owner.items():
        owner_meshes = owner_meshes_by_key[owner_key]
        expected_contributors = math.prod(mesh.size() for mesh in owner_meshes)
        reduction_meshes = (
            (parallel_dims.world_mesh,)
            if len(owner_meshes) > 1
            and expected_contributors == parallel_dims.world_mesh.size()
            else owner_meshes
        )
        groups.append(
            ParameterOwnerGroup(
                parameters=tuple(owned_parameters),
                owner_meshes=owner_meshes,
                reduction_meshes=reduction_meshes,
                expected_contributors=expected_contributors,
            )
        )
    return tuple(groups)


def local_value_for_owner_group(
    value: torch.Tensor,
    *,
    owner_meshes: tuple[DeviceMesh, ...],
    parallel_dims: ParallelDims,
    label: str,
) -> torch.Tensor:
    """Return local storage after matching a parameter's owner cohort."""
    actual_owner_meshes = resolve_parameter_owner_meshes(
        value,
        parallel_dims=parallel_dims,
    )
    if len(actual_owner_meshes) != len(owner_meshes) or any(
        actual is not expected
        for actual, expected in zip(
            actual_owner_meshes,
            owner_meshes,
            strict=True,
        )
    ):
        raise ValueError(f"{label} owner cohort differs from its parameter")
    if not isinstance(value, DTensor):
        raise ValueError(f"{label} is not a DTensor")
    return value.to_local()


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
        "fsdp",
        *(("tp",) if parallel_dims.tp > 1 else ()),
    )
    sparse_mesh_axis_names = ("efsdp", "ep")
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
