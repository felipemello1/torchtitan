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

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.families import TensorMetricFamily
from torchtitan.observability.tensor_logging.parameter_ownership import (
    bind_parameters,
    BoundParameter,
    group_parameters_by_owner,
    local_value_for_owner_group,
)
from torchtitan.observability.tensor_logging.statistics import (
    derive_finite_statistics,
    finite_statistics,
    FiniteStatistics,
    reduce_finite_statistics,
    ReductionBatch,
)


@dataclass(frozen=True, slots=True)
class _ParameterRow:
    parameter: BoundParameter
    family: TensorMetricFamily

    @property
    def metric_prefix(self) -> str:
        suffix = "w" if self.family is TensorMetricFamily.PARAMETER else "dw_preclip"
        return f"tensor_metrics/{self.parameter.fqn}.{suffix}"


@dataclass(frozen=True, slots=True)
class _ParameterGroup:
    rows: tuple[_ParameterRow, ...]
    owner_meshes: tuple[DeviceMesh, ...]
    reduction_meshes: tuple[DeviceMesh, ...]
    expected_contributors: int


@dataclass(frozen=True, slots=True)
class ParameterStatisticsSnapshot:
    statistics: tuple[FiniteStatistics, ...]
    local_error: Exception | None


class ParameterStatisticsBatch:
    """Fixed packed cohorts for selected-layer parameters."""

    def __init__(
        self,
        *,
        model: nn.Module,
        parallel_dims: ParallelDims,
        layer_ids: tuple[int, ...],
        families: tuple[TensorMetricFamily, ...],
    ) -> None:
        bound_parameters = bind_parameters(model, layer_ids=layer_ids)
        if not bound_parameters:
            raise ValueError("tensor logging found no selected-layer parameters")

        groups = []
        for owner_group in group_parameters_by_owner(
            bound_parameters,
            parallel_dims=parallel_dims,
        ):
            groups.append(
                _ParameterGroup(
                    rows=tuple(
                        _ParameterRow(parameter=parameter, family=family)
                        for parameter in owner_group.parameters
                        for family in families
                    ),
                    owner_meshes=owner_group.owner_meshes,
                    reduction_meshes=owner_group.reduction_meshes,
                    expected_contributors=owner_group.expected_contributors,
                )
            )

        self._parallel_dims = parallel_dims
        self._groups = tuple(groups)

    def collect(
        self,
        *,
        step: int,
        batch: ReductionBatch | None = None,
    ) -> ParameterStatisticsSnapshot:
        """Build and synchronously reduce one logging-step parameter snapshot."""
        reduced_groups = []
        local_error: Exception | None = None
        for group in self._groups:
            row_counts = []
            row_sums = []
            row_maxima = []
            for row in group.rows:
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
                        local_value = local_value_for_owner_group(
                            value,
                            owner_meshes=group.owner_meshes,
                            parallel_dims=self._parallel_dims,
                            label="gradient",
                        )
                    else:
                        if not isinstance(value, DTensor):
                            raise ValueError("value is not a DTensor")
                        local_value = value.to_local()
                    statistics = finite_statistics(local_value)
                    present = statistics.counts.new_ones(1)
                except Exception as error:
                    is_optional_absence = (
                        value is None
                        and row.family is TensorMetricFamily.PRECLIP_GRADIENT
                    )
                    if not is_optional_absence and local_error is None:
                        local_error = ValueError(
                            f"invalid tensor logging sample {row.metric_prefix!r} "
                            f"at step {step}: {error}"
                        )
                    statistics = FiniteStatistics(
                        counts=torch.zeros(
                            3, dtype=torch.int64, device=parameter.value.device
                        ),
                        sums=torch.zeros(
                            2, dtype=torch.float32, device=parameter.value.device
                        ),
                        abs_max=torch.zeros(
                            1, dtype=torch.float32, device=parameter.value.device
                        ),
                    )
                    present = statistics.counts.new_zeros(1)

                row_counts.append(torch.cat((statistics.counts, present)))
                row_sums.append(statistics.sums)
                row_maxima.append(statistics.abs_max)

            reduced_groups.append(
                reduce_finite_statistics(
                    FiniteStatistics(
                        counts=torch.stack(row_counts),
                        sums=torch.stack(row_sums),
                        abs_max=torch.stack(row_maxima),
                    ),
                    group.reduction_meshes,
                    batch=batch,
                )
            )
        return ParameterStatisticsSnapshot(
            statistics=tuple(reduced_groups),
            local_error=local_error,
        )

    def derive_metrics(
        self,
        snapshot: ParameterStatisticsSnapshot,
        *,
        window_steps: int,
    ) -> dict[str, int | float]:
        """Copy two packed matrices to CPU and derive writer-side scalars."""
        metrics: dict[str, int | float] = {}
        for group, statistics in zip(
            self._groups,
            snapshot.statistics,
            strict=True,
        ):
            host_counts = statistics.counts.cpu()
            host_floats = torch.cat((statistics.sums, statistics.abs_max), dim=1).cpu()
            for index, row in enumerate(group.rows):
                present = int(host_counts[index, 3])
                if present == 0:
                    if row.family is TensorMetricFamily.PARAMETER:
                        raise RuntimeError(
                            f"required tensor logging parameter "
                            f"{row.parameter.fqn!r} was absent on every owner"
                        )
                    continue
                if present != group.expected_contributors:
                    raise RuntimeError(
                        f"tensor logging sample {row.metric_prefix!r} was present on "
                        f"{present} of {group.expected_contributors} expected owners"
                    )

                row_statistics = FiniteStatistics(
                    counts=host_counts[index, :3],
                    sums=host_floats[index, :2],
                    abs_max=host_floats[index, 2:3],
                )
                reduced_numel = int(row_statistics.counts[0])
                if reduced_numel != row.parameter.numel:
                    raise ValueError(
                        f"reduced parameter numel is {reduced_numel}, "
                        f"expected {row.parameter.numel}"
                    )
                derived = derive_finite_statistics(row_statistics)
                derived["observation_count"] = 1
                derived["window_steps"] = window_steps
                metrics.update(
                    {
                        f"{row.metric_prefix}.{name}": value
                        for name, value in derived.items()
                    }
                )
        return metrics
