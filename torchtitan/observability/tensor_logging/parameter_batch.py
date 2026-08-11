# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.families import TensorMetricFamily
from torchtitan.observability.tensor_logging.parameter_ownership import (
    resolve_rowwise_parameter_owner_meshes,
)
from torchtitan.observability.tensor_logging.statistics import (
    derive_finite_statistics,
    finite_statistics,
    FiniteStatistics,
    reduce_finite_statistics,
)


@dataclass(frozen=True, slots=True)
class _BoundParameter:
    fqn: str
    value: torch.Tensor
    numel: int


@dataclass(frozen=True, slots=True)
class _ParameterRow:
    parameter: _BoundParameter
    family: TensorMetricFamily

    @property
    def metric_prefix(self) -> str:
        suffix = "w" if self.family is TensorMetricFamily.PARAMETER else "dw_preclip"
        return f"tensor_metrics/{self.parameter.fqn}.{suffix}"


@dataclass(frozen=True, slots=True)
class ParameterStatisticsSnapshot:
    statistics: FiniteStatistics
    local_error: Exception | None


class ParameterStatisticsBatch:
    """One fixed batch for selected rowwise attention parameters."""

    def __init__(
        self,
        *,
        model: nn.Module,
        parallel_dims: ParallelDims,
        layer_ids: tuple[int, ...],
        families: tuple[TensorMetricFamily, ...],
    ) -> None:
        expected_fqns = tuple(
            f"layers.{layer_id}.attention.wo.weight" for layer_id in layer_ids
        )
        matches: dict[str, list[torch.Tensor]] = {fqn: [] for fqn in expected_fqns}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            fqn = canonical_fqn(name)
            if fqn in matches:
                matches[fqn].append(parameter)

        bound_parameters = []
        resolved_owner_meshes: list[tuple[DeviceMesh, ...]] = []
        for fqn in expected_fqns:
            parameters = matches[fqn]
            if len(parameters) != 1:
                raise ValueError(
                    f"tensor logging expected exactly one parameter {fqn!r}, "
                    f"found {len(parameters)}"
                )
            parameter = parameters[0]
            if parameter.ndim != 2:
                raise ValueError(
                    f"tensor logging expected {fqn!r} to be two-dimensional"
                )
            if parameter.shape[0] % parallel_dims.dp_shard != 0:
                raise ValueError(
                    f"tensor logging requires {fqn!r} dim 0 ({parameter.shape[0]}) "
                    f"to be divisible by dp_shard ({parallel_dims.dp_shard})"
                )
            if parameter.shape[1] % parallel_dims.tp != 0:
                raise ValueError(
                    f"tensor logging requires {fqn!r} dim 1 ({parameter.shape[1]}) "
                    f"to be divisible by tp ({parallel_dims.tp})"
                )
            owner_meshes = resolve_rowwise_parameter_owner_meshes(
                parameter,
                parallel_dims=parallel_dims,
            )
            resolved_owner_meshes.append(owner_meshes)
            bound_parameters.append(
                _BoundParameter(
                    fqn=fqn,
                    value=parameter,
                    numel=parameter.numel(),
                )
            )

        owner_meshes = resolved_owner_meshes[0]
        use_world_mesh = (
            len(owner_meshes) == 2
            and parallel_dims.dp_replicate == 1
            and parallel_dims.cp == 1
            and parallel_dims.pp == 1
            and parallel_dims.ep == 1
        )
        if use_world_mesh:
            owner_count = owner_meshes[0].size() * owner_meshes[1].size()
            if owner_count != parallel_dims.world_mesh.size():
                raise ValueError(
                    "tensor logging compound parameter owners must span WORLD"
                )
            reduction_meshes: tuple[DeviceMesh, ...] = (parallel_dims.world_mesh,)
        else:
            reduction_meshes = owner_meshes
        self._reduction_meshes = reduction_meshes
        self._local_device = bound_parameters[0].value.device
        self._parallel_dims = parallel_dims
        self._rows = tuple(
            _ParameterRow(parameter=parameter, family=family)
            for parameter in bound_parameters
            for family in families
        )

    def collect(self, *, step: int) -> ParameterStatisticsSnapshot:
        """Build and synchronously reduce one logging-step parameter snapshot."""
        row_counts = []
        row_sums = []
        row_maxima = []
        local_error: Exception | None = None
        for row in self._rows:
            parameter = row.parameter
            value = (
                parameter.value
                if row.family is TensorMetricFamily.PARAMETER
                else parameter.value.grad
            )
            try:
                if value is None:
                    raise ValueError(
                        f"tensor logging sample {row.metric_prefix!r} is absent"
                    )
                if row.family is TensorMetricFamily.PRECLIP_GRADIENT:
                    resolve_rowwise_parameter_owner_meshes(
                        value,
                        parallel_dims=self._parallel_dims,
                    )
                if not isinstance(value, DTensor):
                    raise ValueError("value is not a DTensor")
                local_value = value.to_local()
                statistics = finite_statistics(local_value)
                present = statistics.counts.new_ones(1)
            except Exception as error:
                is_optional_absence = (
                    value is None and row.family is TensorMetricFamily.PRECLIP_GRADIENT
                )
                if not is_optional_absence and local_error is None:
                    local_error = ValueError(
                        f"invalid tensor logging sample {row.metric_prefix!r} "
                        f"at step {step}: {error}"
                    )
                statistics = FiniteStatistics(
                    counts=torch.zeros(3, dtype=torch.int64, device=self._local_device),
                    sums=torch.zeros(2, dtype=torch.float32, device=self._local_device),
                    abs_max=torch.zeros(
                        1, dtype=torch.float32, device=self._local_device
                    ),
                )
                present = statistics.counts.new_zeros(1)

            row_counts.append(torch.cat((statistics.counts, present)))
            row_sums.append(statistics.sums)
            row_maxima.append(statistics.abs_max)

        reduced = reduce_finite_statistics(
            FiniteStatistics(
                counts=torch.stack(row_counts),
                sums=torch.stack(row_sums),
                abs_max=torch.stack(row_maxima),
            ),
            self._reduction_meshes,
        )
        return ParameterStatisticsSnapshot(
            statistics=reduced,
            local_error=local_error,
        )

    def derive_metrics(
        self,
        snapshot: ParameterStatisticsSnapshot,
        *,
        expected_contributors: int,
        window_steps: int,
    ) -> dict[str, int | float]:
        """Copy two packed matrices to CPU and derive writer-side scalars."""
        host_counts = snapshot.statistics.counts.cpu()
        host_floats = torch.cat(
            (snapshot.statistics.sums, snapshot.statistics.abs_max), dim=1
        ).cpu()

        metrics: dict[str, int | float] = {}
        for index, row in enumerate(self._rows):
            present = int(host_counts[index, 3])
            if present == 0:
                if row.family is TensorMetricFamily.PARAMETER:
                    raise RuntimeError(
                        f"required tensor logging parameter {row.parameter.fqn!r} "
                        "was absent on every owner"
                    )
                continue
            if present != expected_contributors:
                raise RuntimeError(
                    f"tensor logging sample {row.metric_prefix!r} was present on "
                    f"{present} of {expected_contributors} expected owners"
                )

            statistics = FiniteStatistics(
                counts=host_counts[index, :3],
                sums=host_floats[index, :2],
                abs_max=host_floats[index, 2:3],
            )
            reduced_numel = int(statistics.counts[0])
            if reduced_numel != row.parameter.numel:
                raise ValueError(
                    f"reduced parameter numel is {reduced_numel}, "
                    f"expected {row.parameter.numel}"
                )
            derived = derive_finite_statistics(statistics)
            derived["observation_count"] = 1
            derived["window_steps"] = window_steps
            metrics.update(
                {
                    f"{row.metric_prefix}.{name}": value
                    for name, value in derived.items()
                }
            )
        return metrics
