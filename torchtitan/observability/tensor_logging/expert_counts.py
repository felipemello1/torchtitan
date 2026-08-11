# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packed exact-count telemetry for selected MoE producer sites."""

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial, Replicate

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.moe import MoE
from torchtitan.observability.tensor_logging.families import TensorMetricFamily
from torchtitan.observability.tensor_logging.statistics import (
    reduce_sum,
    validate_tp_tensor,
)


@dataclass(frozen=True, slots=True)
class ExpertCountSnapshot:
    """Reduced expert counts and any recorder error."""

    values: torch.Tensor
    local_error: Exception | None


class ExpertCountRecorder:
    """Packed interval counts from selected Qwen3 MoE sites."""

    def __init__(
        self,
        *,
        model: nn.Module,
        parallel_dims: ParallelDims,
        layer_ids: tuple[int, ...],
        families: tuple[TensorMetricFamily, ...],
        device: torch.device,
    ) -> None:
        offered_selected = TensorMetricFamily.OFFERED_ASSIGNMENTS in families
        compute_selected = TensorMetricFamily.EXPERT_COMPUTE_ROWS in families
        if not offered_selected and not compute_selected:
            raise ValueError("expert count recorder requires an internal family")
        if compute_selected and parallel_dims.ep == 1:
            raise ValueError("expert compute rows require expert parallelism")

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
        self._experts_per_owner = num_experts // parallel_dims.ep
        self._top_k = tuple(top_k)
        self._ep = parallel_dims.ep
        self._offered_row_offset = 0 if offered_selected else None
        self._compute_row_offset: int | None = (
            (len(layer_ids) if offered_selected else 0) if compute_selected else None
        )

        self._tp_mesh = parallel_dims.get_optional_mesh("tp")
        self._ep_mesh = parallel_dims.get_optional_mesh("ep")
        self._expected_offered_placements = (
            (Partial(),) if parallel_dims.ep > 1 else (Replicate(),)
        )
        self._contributes_offered_counts = (
            parallel_dims.ep > 1
            or self._tp_mesh is None
            or self._tp_mesh.get_local_rank() == 0
        )
        self._contributes_observations = (
            self._tp_mesh is None or self._tp_mesh.get_local_rank() == 0
        )
        self._expert_owner = (
            self._ep_mesh.get_local_rank() if self._ep_mesh is not None else 0
        )
        self._world_mesh = (
            parallel_dims.world_mesh if parallel_dims.world_size > 1 else None
        )

        selected_family_count = int(offered_selected) + int(compute_selected)
        row_count = len(layer_ids) * selected_family_count
        self._interval = torch.zeros(
            (row_count, num_experts + 1), dtype=torch.int64, device=device
        )
        self._validated = [False] * row_count
        self._failed = [False] * row_count
        self._local_error: Exception | None = None

        for row_index, module in enumerate(self._modules):
            if self._offered_row_offset is not None:
                module.offered_assignments_recorder = partial(
                    self._record_offered_assignments,
                    self._offered_row_offset + row_index,
                    layer_id=layer_ids[row_index],
                )
            if self._compute_row_offset is not None:
                module.routed_experts.expert_compute_rows_recorder = partial(
                    self._record_expert_compute_rows,
                    self._compute_row_offset + row_index,
                    layer_id=layer_ids[row_index],
                )

    def collect(self) -> ExpertCountSnapshot:
        """Reduce and reset one publication interval."""
        values = self._interval.clone()
        self._interval.zero_()
        if self._world_mesh is not None:
            values = reduce_sum(values, self._world_mesh)
        snapshot = ExpertCountSnapshot(values=values, local_error=self._local_error)
        self._local_error = None
        return snapshot

    def derive_metrics(
        self,
        snapshot: ExpertCountSnapshot,
        *,
        window_steps: int,
    ) -> dict[str, int | float]:
        """Expand completed CPU expert counts into stable scalar keys."""
        values = snapshot.values.cpu()
        metrics: dict[str, int | float] = {}
        if self._offered_row_offset is not None:
            for row_index, layer_id in enumerate(self._layer_ids):
                counts = values[
                    self._offered_row_offset + row_index, : self._num_experts
                ]
                observation_count = int(
                    values[
                        self._offered_row_offset + row_index,
                        self._num_experts,
                    ]
                )
                if observation_count == 0:
                    continue
                self._derive_offered_metrics(
                    metrics,
                    counts=counts,
                    observation_count=observation_count,
                    layer_id=layer_id,
                    row_index=row_index,
                    window_steps=window_steps,
                )

        if self._compute_row_offset is not None:
            for row_index, layer_id in enumerate(self._layer_ids):
                counts = values[
                    self._compute_row_offset + row_index, : self._num_experts
                ]
                observation_count = int(
                    values[
                        self._compute_row_offset + row_index,
                        self._num_experts,
                    ]
                )
                if observation_count == 0:
                    continue
                prefix = f"tensor_metrics/layers.{layer_id}"
                for expert_id, count in enumerate(counts.tolist()):
                    metrics[
                        f"{prefix}.experts.{expert_id}."
                        "expert_compute_rows.standard_dropless"
                    ] = int(count)
                metrics[
                    f"{prefix}.moe.expert_compute_rows.standard_dropless."
                    "observation_count"
                ] = observation_count
                metrics[
                    f"{prefix}.moe.expert_compute_rows.standard_dropless.window_steps"
                ] = window_steps
        return metrics

    def close(self) -> None:
        """Unbind recorders from their owning MoE modules."""
        for module in self._modules:
            if self._offered_row_offset is not None:
                module.offered_assignments_recorder = None
            if self._compute_row_offset is not None:
                module.routed_experts.expert_compute_rows_recorder = None

    def _derive_offered_metrics(
        self,
        metrics: dict[str, int | float],
        *,
        counts: torch.Tensor,
        observation_count: int,
        layer_id: int,
        row_index: int,
        window_steps: int,
    ) -> None:
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
            return

        mean_count = total_assignments / self._num_experts
        loads = counts.to(torch.float64) / mean_count
        for expert_id, load in enumerate(loads.tolist()):
            metrics[f"{prefix}.experts.{expert_id}.offered_load"] = load
        metrics[f"{prefix}.moe.offered_maximum_violation"] = float(loads.max()) - 1.0
        if self._ep > 1:
            owner_counts = counts.reshape(self._ep, self._experts_per_owner).sum(dim=1)
            metrics[f"{prefix}.moe.offered_ep_shard_imbalance"] = (
                int(owner_counts.max()) * self._ep / total_assignments
            )

    def _record_offered_assignments(
        self,
        row_index: int,
        value: torch.Tensor,
        *,
        layer_id: int,
    ) -> None:
        if self._failed[row_index]:
            return
        try:
            if not self._validated[row_index]:
                self._validate_offered_assignments(value)
                self._validated[row_index] = True
            local_value = value.to_local() if isinstance(value, DTensor) else value
            if self._contributes_offered_counts:
                self._interval[row_index, : self._num_experts].add_(local_value)
            if self._contributes_observations:
                self._interval[row_index, self._num_experts].add_(1)
        except Exception as error:
            self._failed[row_index] = True
            if self._local_error is None:
                self._local_error = ValueError(
                    f"invalid offered assignments at layers.{layer_id}.moe: {error}"
                )

    def _record_expert_compute_rows(
        self,
        row_index: int,
        value: torch.Tensor,
        *,
        layer_id: int,
    ) -> None:
        if self._failed[row_index]:
            return
        try:
            if not self._validated[row_index]:
                self._validate_expert_compute_rows(value)
                self._validated[row_index] = True
            start = self._expert_owner * self._experts_per_owner
            self._interval[
                row_index,
                start : start + self._experts_per_owner,
            ].add_(value)
            if self._contributes_observations:
                self._interval[row_index, self._num_experts].add_(1)
        except Exception as error:
            self._failed[row_index] = True
            if self._local_error is None:
                self._local_error = ValueError(
                    f"invalid expert compute rows at layers.{layer_id}.moe: {error}"
                )

    def _validate_offered_assignments(self, value: torch.Tensor) -> None:
        if value.dtype is not torch.int64:
            raise ValueError(f"expected int64 counts, got {value.dtype}")
        if value.shape != (self._num_experts,):
            raise ValueError(
                f"expected shape ({self._num_experts},), got {tuple(value.shape)}"
            )
        validate_tp_tensor(
            value,
            tp_mesh=self._tp_mesh,
            expected_placements=self._expected_offered_placements,
            label="counts",
        )

    def _validate_expert_compute_rows(self, value: torch.Tensor) -> None:
        if isinstance(value, DTensor):
            raise ValueError("expert compute rows must be a local tensor")
        if value.dtype is not torch.int64:
            raise ValueError(f"expected int64 counts, got {value.dtype}")
        if value.shape != (self._experts_per_owner,):
            raise ValueError(
                f"expected shape ({self._experts_per_owner},), "
                f"got {tuple(value.shape)}"
            )
