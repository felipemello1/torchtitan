# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial, Replicate

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.utils import check_dtensor_placements_match
from torchtitan.models.common.moe import MoE
from torchtitan.observability.tensor_logging.statistics import reduce_sum


@dataclass(frozen=True, slots=True)
class OfferedAssignmentsSnapshot:
    """Reduced expert-assignment counts and any recorder error."""

    values: torch.Tensor
    local_error: Exception | None


class OfferedAssignmentsRecorder:
    """Interval counts from selected Qwen3 MoE routing sites."""

    def __init__(
        self,
        *,
        model: nn.Module,
        parallel_dims: ParallelDims,
        layer_ids: tuple[int, ...],
        device: torch.device,
    ) -> None:
        modules = []
        num_experts: int | None = None
        top_k = []
        for layer_id in layer_ids:
            fqn = f"layers.{layer_id}.moe"
            module = model.get_submodule(fqn)
            if type(module) is not MoE:
                raise ValueError(f"tensor logging requires an ordinary MoE at {fqn!r}")
            if num_experts is None:
                num_experts = module.router.num_experts
            elif num_experts != module.router.num_experts:
                raise ValueError(
                    "tensor logging requires selected MoE layers to have the same "
                    "number of experts"
                )
            modules.append(module)
            top_k.append(module.router.top_k)

        assert num_experts is not None
        if num_experts % parallel_dims.ep != 0:
            raise ValueError(
                f"tensor logging requires {num_experts} experts to be divisible by "
                f"ep ({parallel_dims.ep})"
            )

        self._layer_ids = layer_ids
        self._modules = tuple(modules)
        self._num_experts = num_experts
        self._top_k = tuple(top_k)
        self._ep = parallel_dims.ep
        self._tp_mesh = parallel_dims.get_optional_mesh("tp")
        self._expected_placements = (
            (Partial(),) if parallel_dims.ep > 1 else (Replicate(),)
        )
        self._contributes_counts = (
            parallel_dims.ep > 1
            or self._tp_mesh is None
            or self._tp_mesh.get_local_rank() == 0
        )
        self._contributes_observations = (
            self._tp_mesh is None or self._tp_mesh.get_local_rank() == 0
        )
        self._world_mesh = (
            parallel_dims.world_mesh if parallel_dims.world_size > 1 else None
        )
        self._interval = torch.zeros(
            (len(layer_ids), num_experts + 1), dtype=torch.int64, device=device
        )
        self._validated = [False] * len(layer_ids)
        self._failed = [False] * len(layer_ids)
        self._local_error: Exception | None = None

        for row_index, module in enumerate(self._modules):
            module.offered_assignments_recorder = partial(self._record, row_index)

    def collect(self) -> OfferedAssignmentsSnapshot:
        """Reduce and reset one publication interval."""
        values = self._interval.clone()
        self._interval.zero_()
        if self._world_mesh is not None:
            values = reduce_sum(values, self._world_mesh)
        snapshot = OfferedAssignmentsSnapshot(
            values=values,
            local_error=self._local_error,
        )
        self._local_error = None
        return snapshot

    def derive_metrics(
        self,
        snapshot: OfferedAssignmentsSnapshot,
        *,
        window_steps: int,
    ) -> dict[str, int | float]:
        """Expand completed CPU expert counts into stable scalar keys."""
        values = snapshot.values.cpu()
        metrics: dict[str, int | float] = {}
        experts_per_owner = self._num_experts // self._ep
        for row_index, layer_id in enumerate(self._layer_ids):
            counts = values[row_index, : self._num_experts]
            observation_count = int(values[row_index, self._num_experts])
            if observation_count == 0:
                continue

            prefix = f"tensor_metrics/layers.{layer_id}"
            for expert_id, count in enumerate(counts.tolist()):
                metrics[f"{prefix}.experts.{expert_id}.offered_count"] = int(count)

            total_assignments = int(counts.sum())
            if total_assignments % self._top_k[row_index] != 0:
                raise RuntimeError(
                    f"offered assignment total {total_assignments} at layer "
                    f"{layer_id} is not divisible by top_k {self._top_k[row_index]}"
                )
            metrics[f"{prefix}.moe.offered_assignments.routed_position_count"] = (
                total_assignments // self._top_k[row_index]
            )
            metrics[
                f"{prefix}.moe.offered_assignments.observation_count"
            ] = observation_count
            metrics[f"{prefix}.moe.offered_assignments.window_steps"] = window_steps
            if total_assignments == 0:
                continue

            mean_count = total_assignments / self._num_experts
            loads = counts.to(torch.float64) / mean_count
            for expert_id, load in enumerate(loads.tolist()):
                metrics[f"{prefix}.experts.{expert_id}.offered_load"] = load
            metrics[f"{prefix}.moe.offered_maximum_violation"] = (
                float(loads.max()) - 1.0
            )
            if self._ep > 1:
                owner_counts = counts.reshape(self._ep, experts_per_owner).sum(dim=1)
                metrics[f"{prefix}.moe.offered_ep_shard_imbalance"] = (
                    int(owner_counts.max()) * self._ep / total_assignments
                )
        return metrics

    def close(self) -> None:
        """Unbind recorders from their owning MoE modules."""
        for module in self._modules:
            module.offered_assignments_recorder = None

    def _record(self, row_index: int, value: torch.Tensor) -> None:
        if self._failed[row_index]:
            return
        try:
            if not self._validated[row_index]:
                self._validate(value)
                self._validated[row_index] = True
            local_value = value.to_local() if isinstance(value, DTensor) else value
            if self._contributes_counts:
                self._interval[row_index, : self._num_experts].add_(local_value)
            if self._contributes_observations:
                self._interval[row_index, self._num_experts].add_(1)
        except Exception as error:
            self._failed[row_index] = True
            if self._local_error is None:
                layer_id = self._layer_ids[row_index]
                self._local_error = ValueError(
                    f"invalid offered assignments at layers.{layer_id}.moe: {error}"
                )

    def _validate(self, value: torch.Tensor) -> None:
        if value.dtype is not torch.int64:
            raise ValueError(f"expected int64 counts, got {value.dtype}")
        if value.shape != (self._num_experts,):
            raise ValueError(
                f"expected shape ({self._num_experts},), got {tuple(value.shape)}"
            )
        if self._tp_mesh is None:
            if isinstance(value, DTensor):
                raise ValueError("TP=1 counts must be a local tensor")
            return
        if not isinstance(value, DTensor):
            raise ValueError("TP>1 counts must be a DTensor")
        if (
            value.device_mesh.device_type != self._tp_mesh.device_type
            or value.device_mesh.mesh_dim_names != self._tp_mesh.mesh_dim_names
            or not torch.equal(value.device_mesh.mesh, self._tp_mesh.mesh)
        ):
            raise ValueError("counts must use the ParallelDims TP mesh")
        if not check_dtensor_placements_match(
            value.placements,
            self._expected_placements,
            value.ndim,
        ):
            raise ValueError(
                f"expected placements {self._expected_placements}, "
                f"got {value.placements}"
            )
