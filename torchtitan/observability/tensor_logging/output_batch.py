# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement
from torch.utils.hooks import RemovableHandle

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.observability.tensor_logging.families import TensorMetricFamily
from torchtitan.observability.tensor_logging.statistics import (
    derive_finite_statistics,
    finite_statistics,
    FiniteStatistics,
    reduce_finite_statistics,
    validate_tp_tensor,
)
from torchtitan.protocols.sharding import resolve_placements


_MAX_CHUNK_ELEMENTS = 1_048_576
_CALL_COUNT = 3
_EXPECTED_NUMEL = 4
_EXPECTED_CALL_COUNT = 5
_PRESENT = 6


@dataclass(frozen=True, slots=True)
class _OutputRow:
    """One output statistic row and its expected TP layout."""

    metric_prefix: str
    expected_mesh: DeviceMesh | None
    expected_placements: tuple[Placement, ...]


@dataclass(frozen=True, slots=True)
class _BoundOutput:
    """One hooked module and its optional forward and cotangent rows."""

    fqn: str
    expected_mesh: DeviceMesh | None
    expected_placements: tuple[Placement, ...]
    output_row: int | None
    cotangent_row: int | None


@dataclass(frozen=True, slots=True)
class OutputStatisticsSnapshot:
    """Reduced output statistics and any error latched inside a hook."""

    statistics: FiniteStatistics
    local_error: Exception | None


class OutputStatisticsBatch:
    """Logging-step statistics for selected attention and dense-FFN outputs."""

    def __init__(
        self,
        *,
        model: nn.Module,
        parallel_dims: ParallelDims,
        layer_ids: tuple[int, ...],
        families: tuple[TensorMetricFamily, ...],
        device: torch.device,
    ) -> None:
        record_output = TensorMetricFamily.BOUNDARY_OUTPUT in families
        record_cotangent = TensorMetricFamily.BOUNDARY_OUTPUT_COTANGENT in families
        if not record_output and not record_cotangent:
            raise ValueError("output statistics require an output metric family")

        tp_mesh = parallel_dims.get_mesh("tp") if parallel_dims.tp > 1 else None
        rows: list[_OutputRow] = []
        bindings: list[_BoundOutput] = []
        sequence_sharded: bool | None = None
        for layer_id in layer_ids:
            modules = (
                (f"layers.{layer_id}.attention", "wo", GQAttention),
                (f"layers.{layer_id}.feed_forward", "w2", FeedForward),
            )
            for fqn, output_projection_name, expected_type in modules:
                module = model.get_submodule(fqn)
                if type(module) is not expected_type:
                    raise ValueError(
                        f"tensor logging requires an ordinary {expected_type.__name__} "
                        f"at {fqn!r}"
                    )
                output_projection = getattr(module, output_projection_name)
                sharding_config = output_projection._sharding_config
                if sharding_config is None or sharding_config.out_dst_shardings is None:
                    raise ValueError(
                        f"tensor logging requires output sharding at "
                        f"{fqn}.{output_projection_name!s}"
                    )

                expected_placements: tuple[Placement, ...] = ()
                module_sequence_sharded = False
                if tp_mesh is not None:
                    expected_placements = resolve_placements(
                        sharding_config.out_dst_shardings,
                        tp_mesh,
                    )
                    if expected_placements == (Replicate(),):
                        module_sequence_sharded = False
                    elif expected_placements == (Shard(1),):
                        module_sequence_sharded = True
                    else:
                        raise ValueError(
                            f"tensor logging does not support output placements "
                            f"{expected_placements} at {fqn!r}"
                        )
                if sequence_sharded is None:
                    sequence_sharded = module_sequence_sharded
                elif sequence_sharded != module_sequence_sharded:
                    raise ValueError(
                        "tensor logging requires matching attention and FFN "
                        "output placements"
                    )

                output_row = None
                if record_output:
                    output_row = len(rows)
                    rows.append(
                        _OutputRow(
                            metric_prefix=f"tensor_metrics/{fqn}.output.value",
                            expected_mesh=tp_mesh,
                            expected_placements=expected_placements,
                        )
                    )
                cotangent_row = None
                if record_cotangent:
                    cotangent_row = len(rows)
                    rows.append(
                        _OutputRow(
                            metric_prefix=f"tensor_metrics/{fqn}.output.cotangent",
                            expected_mesh=tp_mesh,
                            expected_placements=expected_placements,
                        )
                    )
                bindings.append(
                    _BoundOutput(
                        fqn=fqn,
                        expected_mesh=tp_mesh,
                        expected_placements=expected_placements,
                        output_row=output_row,
                        cotangent_row=cotangent_row,
                    )
                )

        assert sequence_sharded is not None
        tp_rank = tp_mesh.get_local_rank() if tp_mesh is not None else 0
        self._contributes_logical_counts = tp_rank == 0
        self._contributes_values = sequence_sharded or self._contributes_logical_counts
        self._expected_contributors = (
            parallel_dims.world_size
            if sequence_sharded
            else parallel_dims.dp_replicate * parallel_dims.dp_shard * parallel_dims.cp
        )
        self._reduction_meshes: tuple[DeviceMesh, ...]
        if sequence_sharded and parallel_dims.world_size > 1:
            self._reduction_meshes = (parallel_dims.world_mesh,)
        elif self._contributes_values:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            self._reduction_meshes = () if loss_mesh is None else (loss_mesh,)
        else:
            self._reduction_meshes = ()

        self._rows = tuple(rows)
        self._bindings = tuple(bindings)
        self._counts = torch.zeros((len(rows), 7), dtype=torch.int64, device=device)
        self._sums = torch.zeros((len(rows), 2), dtype=torch.float32, device=device)
        self._abs_max = torch.zeros((len(rows), 1), dtype=torch.float32, device=device)
        self._failed_rows = [False] * len(rows)
        self._validated_forward = [False] * len(bindings)
        self._validated_cotangent = [False] * len(rows)
        self._active = False
        self._local_error: Exception | None = None
        self._gradient_hook_handles: list[RemovableHandle] = []
        self._forward_hook_handles = tuple(
            model.get_submodule(binding.fqn).register_forward_hook(
                partial(self._observe_output, binding_index)
            )
            for binding_index, binding in enumerate(self._bindings)
        )

    def begin_step(self, *, should_log: bool) -> None:
        """Clear and enable output rows for one logging step."""
        self._remove_gradient_hooks()
        self._active = should_log
        if not should_log:
            return
        self._counts.zero_()
        self._sums.zero_()
        self._abs_max.zero_()
        self._failed_rows = [False] * len(self._rows)
        self._local_error = None

    def collect(self) -> OutputStatisticsSnapshot:
        """Disable hooks and synchronously reduce the logging-step state."""
        self._active = False
        self._remove_gradient_hooks()
        statistics = FiniteStatistics(
            counts=self._counts.clone(),
            sums=self._sums.clone(),
            abs_max=self._abs_max.clone(),
        )
        if self._contributes_values:
            statistics = reduce_finite_statistics(
                statistics,
                self._reduction_meshes,
            )
        return OutputStatisticsSnapshot(
            statistics=statistics,
            local_error=self._local_error,
        )

    def derive_metrics(
        self,
        snapshot: OutputStatisticsSnapshot,
        *,
        window_steps: int,
    ) -> dict[str, int | float]:
        """Copy two packed matrices to CPU and derive writer-side scalars."""
        host_counts = snapshot.statistics.counts.cpu()
        host_floats = torch.cat(
            (snapshot.statistics.sums, snapshot.statistics.abs_max), dim=1
        ).cpu()

        for binding in self._bindings:
            if binding.output_row is None or binding.cotangent_row is None:
                continue
            output_counts = host_counts[binding.output_row]
            cotangent_counts = host_counts[binding.cotangent_row]
            if not torch.equal(output_counts[3:6], cotangent_counts[3:6]):
                raise RuntimeError(
                    f"tensor logging output/cotangent counts differ at {binding.fqn!r}"
                )

        metrics: dict[str, int | float] = {}
        for index, row in enumerate(self._rows):
            counts = host_counts[index]
            present = int(counts[_PRESENT])
            if present != self._expected_contributors:
                raise RuntimeError(
                    f"tensor logging sample {row.metric_prefix!r} was present on "
                    f"{present} of {self._expected_contributors} expected contributors"
                )
            call_count = int(counts[_CALL_COUNT])
            expected_call_count = int(counts[_EXPECTED_CALL_COUNT])
            if call_count != expected_call_count:
                raise RuntimeError(
                    f"tensor logging sample {row.metric_prefix!r} observed "
                    f"{call_count} calls, expected {expected_call_count}"
                )
            physical_numel = int(counts[0])
            expected_numel = int(counts[_EXPECTED_NUMEL])
            if physical_numel != expected_numel:
                raise RuntimeError(
                    f"tensor logging sample {row.metric_prefix!r} contains "
                    f"{physical_numel} elements, expected {expected_numel}"
                )

            derived = derive_finite_statistics(
                FiniteStatistics(
                    counts=counts[:3],
                    sums=host_floats[index, :2],
                    abs_max=host_floats[index, 2:3],
                )
            )
            derived["observation_count"] = call_count
            derived["window_steps"] = window_steps
            metrics.update(
                {
                    f"{row.metric_prefix}.{name}": value
                    for name, value in derived.items()
                }
            )
        return metrics

    def close(self) -> None:
        """Remove all hooks owned by this batch."""
        self._active = False
        self._remove_gradient_hooks()
        for handle in self._forward_hook_handles:
            handle.remove()

    def _observe_output(
        self,
        binding_index: int,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> None:
        if not self._active or not self._contributes_values:
            return
        binding = self._bindings[binding_index]
        try:
            if not self._validated_forward[binding_index]:
                self._validate_tensor(
                    output,
                    binding.expected_mesh,
                    binding.expected_placements,
                )
                self._validated_forward[binding_index] = True
            if (
                binding.output_row is not None
                and not self._failed_rows[binding.output_row]
            ):
                self._record_statistics(
                    binding.output_row,
                    output,
                    records_expected=True,
                )
            if (
                binding.cotangent_row is not None
                and not self._failed_rows[binding.cotangent_row]
            ):
                if self._contributes_logical_counts:
                    self._counts[binding.cotangent_row, _EXPECTED_CALL_COUNT].add_(1)
                    self._counts[binding.cotangent_row, _EXPECTED_NUMEL].add_(
                        output.numel()
                    )
                handle = output.register_hook(
                    partial(self._observe_cotangent, binding.cotangent_row)
                )
                self._gradient_hook_handles.append(handle)
        except Exception as error:
            for row_index in (binding.output_row, binding.cotangent_row):
                if row_index is not None:
                    self._latch_error(row_index, binding.fqn, error)

    def _observe_cotangent(
        self,
        row_index: int,
        cotangent: torch.Tensor,
    ) -> None:
        try:
            if not self._validated_cotangent[row_index]:
                self._validate_tensor(
                    cotangent,
                    self._rows[row_index].expected_mesh,
                    self._rows[row_index].expected_placements,
                )
                self._validated_cotangent[row_index] = True
            self._record_statistics(
                row_index,
                cotangent,
                records_expected=False,
            )
        except Exception as error:
            self._latch_error(row_index, self._rows[row_index].metric_prefix, error)
        return None

    def _record_statistics(
        self,
        row_index: int,
        value: torch.Tensor,
        *,
        records_expected: bool,
    ) -> None:
        if self._failed_rows[row_index]:
            return
        local_value = value.to_local() if isinstance(value, DTensor) else value
        if isinstance(local_value, AsyncCollectiveTensor):
            local_value = funcol.wait_tensor(local_value)
        statistics = finite_statistics(
            local_value,
            max_chunk_elements=_MAX_CHUNK_ELEMENTS,
        )
        self._counts[row_index, :3].add_(statistics.counts)
        self._sums[row_index].add_(statistics.sums)
        self._abs_max[row_index].copy_(
            torch.maximum(self._abs_max[row_index], statistics.abs_max)
        )
        self._counts[row_index, _PRESENT].fill_(1)
        if not self._contributes_logical_counts:
            return
        self._counts[row_index, _CALL_COUNT].add_(1)
        if records_expected:
            self._counts[row_index, _EXPECTED_CALL_COUNT].add_(1)
            self._counts[row_index, _EXPECTED_NUMEL].add_(value.numel())

    @staticmethod
    def _validate_tensor(
        value: torch.Tensor,
        expected_mesh: DeviceMesh | None,
        expected_placements: tuple[Placement, ...],
    ) -> None:
        if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported dtype {value.dtype}")
        if value.ndim != 3:
            raise ValueError(f"expected a three-dimensional output, got {value.ndim}")
        validate_tp_tensor(
            value,
            tp_mesh=expected_mesh,
            expected_placements=expected_placements,
            label="output",
        )

    def _latch_error(
        self,
        row_index: int,
        fqn: str,
        error: Exception,
    ) -> None:
        if self._failed_rows[row_index]:
            return
        self._failed_rows[row_index] = True
        self._counts[row_index, _PRESENT].zero_()
        if self._local_error is None:
            self._local_error = ValueError(
                f"invalid tensor logging output {fqn!r}: {error}"
            )

    def _remove_gradient_hooks(self) -> None:
        for handle in self._gradient_hook_handles:
            handle.remove()
        self._gradient_hook_handles.clear()
