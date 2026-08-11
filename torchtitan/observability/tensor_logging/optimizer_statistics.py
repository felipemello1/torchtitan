# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Selected-parameter AdamW state and update statistics."""

import math
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.families import TensorMetricFamily
from torchtitan.observability.tensor_logging.parameter_ownership import (
    bind_parameters,
    BoundParameter,
    group_parameters_by_owner,
    local_value_for_owner_group,
)
from torchtitan.observability.tensor_logging.statistics import (
    bounded_tensor_views,
    derive_finite_statistics_values,
    finite_statistics,
    reduce_max,
    reduce_sum,
    ReductionBatch,
)


_DISTRIBUTION_NAMES = (
    "numerator",
    "denominator",
    "preconditioned_gradient",
    "update_pre_apply",
)
_MAX_CHUNK_ELEMENTS = 1_048_576


@dataclass(frozen=True, slots=True)
class OptimizerStatisticsSnapshot:
    counts: tuple[torch.Tensor, ...]
    sums: tuple[torch.Tensor, ...]
    maxima: tuple[torch.Tensor, ...]
    local_error: Exception | None


class AdamWStatisticsRecorder:
    """Records exact public AdamW equations from authoritative post-step state."""

    def __init__(
        self,
        *,
        model: nn.Module,
        parallel_dims: ParallelDims,
        layer_ids: tuple[int, ...],
        families: tuple[TensorMetricFamily, ...],
    ) -> None:
        parameters = bind_parameters(model, layer_ids=layer_ids)
        if not parameters:
            raise ValueError(
                "tensor logging found no selected-layer optimizer parameters"
            )

        self._groups = group_parameters_by_owner(
            parameters,
            parallel_dims=parallel_dims,
        )
        self._locations = {
            id(parameter.value): (group_index, parameter_index)
            for group_index, group in enumerate(self._groups)
            for parameter_index, parameter in enumerate(group.parameters)
        }
        self._parallel_dims = parallel_dims
        self._record_distributions = (
            TensorMetricFamily.OPTIMIZER_DISTRIBUTION in families
        )
        self._record_cosine = TensorMetricFamily.MOMENTUM_GRADIENT_COSINE in families
        self._distribution_rows = tuple(
            len(group.parameters) * len(_DISTRIBUTION_NAMES)
            if self._record_distributions
            else 0
            for group in self._groups
        )
        device = parameters[0].value.device
        self._counts = tuple(
            torch.zeros(
                (
                    distribution_rows
                    + (len(group.parameters) if self._record_cosine else 0),
                    4,
                ),
                dtype=torch.int64,
                device=device,
            )
            for group, distribution_rows in zip(
                self._groups, self._distribution_rows, strict=True
            )
        )
        self._sums = tuple(
            torch.zeros(
                (
                    distribution_rows
                    + (len(group.parameters) if self._record_cosine else 0),
                    3,
                ),
                dtype=torch.float32,
                device=device,
            )
            for group, distribution_rows in zip(
                self._groups, self._distribution_rows, strict=True
            )
        )
        self._maxima = tuple(
            torch.zeros(
                (distribution_rows, 1),
                dtype=torch.float32,
                device=device,
            )
            for distribution_rows in self._distribution_rows
        )
        self._record_next_step = False
        self._local_error: Exception | None = None
        self._hook_handles: list[RemovableHandle] = []

    def bind_optimizer(self, optimizers: OptimizersContainer[Optimizer]) -> None:
        """Bind selected parameters to their named public AdamW instance."""
        if self._hook_handles:
            raise RuntimeError("AdamW statistics recorder is already bound")
        found: set[int] = set()
        for optimizer in optimizers.optimizers:
            selected_groups: list[tuple[int, tuple[BoundParameter, ...]]] = []
            for optimizer_group_index, parameter_group in enumerate(
                optimizer.param_groups
            ):
                selected_parameters = []
                parameter_names = parameter_group["param_names"]
                if len(parameter_names) != len(parameter_group["params"]):
                    raise ValueError("AdamW param_names are not aligned with params")
                for optimizer_parameter, optimizer_name in zip(
                    parameter_group["params"], parameter_names, strict=True
                ):
                    location = self._locations.get(id(optimizer_parameter))
                    if location is not None:
                        parameter = self._groups[location[0]].parameters[location[1]]
                        if canonical_fqn(optimizer_name) != parameter.fqn:
                            raise ValueError(
                                f"AdamW names {optimizer_name!r} and "
                                f"{parameter.fqn!r} do not match"
                            )
                        selected_parameters.append(parameter)
                if not selected_parameters:
                    continue
                if type(optimizer) is not torch.optim.AdamW:
                    raise ValueError(
                        "optimizer tensor logging currently requires public AdamW"
                    )
                self._validate_parameter_group(parameter_group)
                found.update(id(parameter.value) for parameter in selected_parameters)
                selected_groups.append(
                    (optimizer_group_index, tuple(selected_parameters))
                )
            if selected_groups:
                self._hook_handles.append(
                    optimizer.register_step_post_hook(
                        partial(
                            self._record_optimizer_state,
                            selected_groups=tuple(selected_groups),
                        )
                    )
                )

        expected = set(self._locations)
        if found != expected:
            missing = sorted(
                self._groups[group_index].parameters[parameter_index].fqn
                for identity, (group_index, parameter_index) in self._locations.items()
                if identity not in found
            )
            raise ValueError(
                f"optimizer tensor logging did not find selected parameters {missing}"
            )

    def begin_step(self, *, should_log: bool) -> None:
        """Arm one optimizer point sample and clear its fixed local state."""
        if should_log:
            for values in (*self._counts, *self._sums, *self._maxima):
                values.zero_()
            self._local_error = None
        self._record_next_step = should_log

    def collect(
        self,
        *,
        batch: ReductionBatch | None = None,
    ) -> OptimizerStatisticsSnapshot:
        """Reduce the optimizer sample after every inner AdamW has stepped."""
        counts = []
        sums = []
        maxima = []
        for group_index, group in enumerate(self._groups):
            reduced_counts = self._counts[group_index].clone()
            reduced_sums = self._sums[group_index].clone()
            reduced_maxima = self._maxima[group_index].clone()
            if batch is None:
                for mesh in group.reduction_meshes:
                    reduced_counts = reduce_sum(reduced_counts, mesh)
                    reduced_sums = reduce_sum(reduced_sums, mesh)
                    if self._record_distributions:
                        reduced_maxima = reduce_max(reduced_maxima, mesh)
            else:
                batch.sum(reduced_counts, group.reduction_meshes)
                batch.sum(reduced_sums, group.reduction_meshes)
                if self._record_distributions:
                    batch.max(reduced_maxima, group.reduction_meshes)
            counts.append(reduced_counts)
            sums.append(reduced_sums)
            maxima.append(reduced_maxima)

        self._record_next_step = False
        return OptimizerStatisticsSnapshot(
            counts=tuple(counts),
            sums=tuple(sums),
            maxima=tuple(maxima),
            local_error=self._local_error,
        )

    def derive_metrics(
        self,
        snapshot: OptimizerStatisticsSnapshot,
        *,
        window_steps: int,
    ) -> dict[str, int | float]:
        """Validate owner completeness and derive selected AdamW metrics."""
        metrics: dict[str, int | float] = {}
        for group_index, group in enumerate(self._groups):
            host_counts = snapshot.counts[group_index].cpu().tolist()
            reduced_sums = snapshot.sums[group_index]
            reduced_maxima = snapshot.maxima[group_index]
            packed_floats = torch.cat(
                (reduced_sums.flatten(), reduced_maxima.flatten())
            ).cpu()
            sums_elements = reduced_sums.numel()
            host_sums = packed_floats[:sums_elements].view(reduced_sums.shape).tolist()
            host_maxima = (
                packed_floats[sums_elements:].view(reduced_maxima.shape).tolist()
            )

            if self._record_distributions:
                for parameter_index, parameter in enumerate(group.parameters):
                    for distribution_index, distribution_name in enumerate(
                        _DISTRIBUTION_NAMES
                    ):
                        row_index = (
                            parameter_index * len(_DISTRIBUTION_NAMES)
                            + distribution_index
                        )
                        counts = host_counts[row_index]
                        present = counts[3]
                        self._validate_presence(
                            parameter.fqn,
                            present,
                            group.expected_contributors,
                        )
                        if counts[0] != parameter.numel:
                            raise ValueError(
                                f"reduced optimizer tensor numel for {parameter.fqn!r} "
                                f"is {counts[0]}, expected "
                                f"{parameter.numel}"
                            )
                        derived = derive_finite_statistics_values(
                            counts[:3],
                            host_sums[row_index][:2],
                            host_maxima[row_index][0],
                        )
                        derived["observation_count"] = 1
                        derived["window_steps"] = window_steps
                        prefix = (
                            f"tensor_metrics/{parameter.fqn}.optimizer."
                            f"{distribution_name}"
                        )
                        metrics.update(
                            {
                                f"{prefix}.{name}": value
                                for name, value in derived.items()
                            }
                        )

            if self._record_cosine:
                for parameter_index, parameter in enumerate(group.parameters):
                    row_index = self._distribution_rows[group_index] + parameter_index
                    self._validate_presence(
                        parameter.fqn,
                        host_counts[row_index][3],
                        group.expected_contributors,
                    )
                    dot, momentum_square, gradient_square = host_sums[row_index]
                    prefix = f"tensor_metrics/{parameter.fqn}.optimizer.cosine"
                    metrics[f"{prefix}.observation_count"] = 1
                    metrics[f"{prefix}.window_steps"] = window_steps
                    if (
                        math.isfinite(dot)
                        and math.isfinite(momentum_square)
                        and math.isfinite(gradient_square)
                        and momentum_square > 0
                        and gradient_square > 0
                    ):
                        cosine = dot / math.sqrt(momentum_square * gradient_square)
                        metrics[
                            f"tensor_metrics/{parameter.fqn}.optimizer."
                            "momentum_gradient_cosine"
                        ] = max(-1.0, min(1.0, cosine))
        return metrics

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    @staticmethod
    def _validate_parameter_group(parameter_group: dict) -> None:
        unsupported = {
            "maximize": parameter_group["maximize"],
            "amsgrad": parameter_group["amsgrad"],
            "differentiable": parameter_group["differentiable"],
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError(
                "optimizer tensor logging does not support AdamW options " f"{enabled}"
            )
        lr = parameter_group["lr"]
        beta1, beta2 = parameter_group["betas"]
        if any(isinstance(value, torch.Tensor) for value in (lr, beta1, beta2)):
            raise ValueError(
                "optimizer tensor logging requires scalar AdamW lr and betas"
            )

    @staticmethod
    def _validate_presence(fqn: str, present: int, expected: int) -> None:
        if present != expected:
            raise RuntimeError(
                f"optimizer tensor sample {fqn!r} was present on {present} of "
                f"{expected} expected owners"
            )

    def _record_optimizer_state(
        self,
        optimizer: Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
        *,
        selected_groups: tuple[tuple[int, tuple[BoundParameter, ...]], ...],
    ) -> None:
        if not self._record_next_step:
            return
        for optimizer_group_index, parameters in selected_groups:
            parameter_group = optimizer.param_groups[optimizer_group_index]
            self._validate_parameter_group(parameter_group)
            lr = parameter_group["lr"]
            beta1, beta2 = parameter_group["betas"]
            eps = parameter_group["eps"]
            for parameter in parameters:
                group_index, parameter_index = self._locations[id(parameter.value)]
                group = self._groups[group_index]
                try:
                    gradient = parameter.value.grad
                    if gradient is None:
                        raise ValueError("gradient is absent")
                    state = optimizer.state[parameter.value]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step = state["step"]
                    if not isinstance(step, torch.Tensor) or step.numel() != 1:
                        raise ValueError("AdamW step state is not a scalar tensor")
                    local_gradient = local_value_for_owner_group(
                        gradient,
                        owner_meshes=group.owner_meshes,
                        parallel_dims=self._parallel_dims,
                        label="optimizer tensor",
                    )
                    local_exp_avg = local_value_for_owner_group(
                        exp_avg,
                        owner_meshes=group.owner_meshes,
                        parallel_dims=self._parallel_dims,
                        label="optimizer tensor",
                    )
                    local_exp_avg_sq = local_value_for_owner_group(
                        exp_avg_sq,
                        owner_meshes=group.owner_meshes,
                        parallel_dims=self._parallel_dims,
                        label="optimizer tensor",
                    )
                    if (
                        local_gradient.shape != local_exp_avg.shape
                        or local_gradient.shape != local_exp_avg_sq.shape
                    ):
                        raise ValueError("AdamW gradient and state shapes differ")
                    if local_exp_avg.dtype is not local_exp_avg_sq.dtype:
                        raise ValueError("AdamW moment state dtypes differ")
                    if local_exp_avg.dtype is not torch.float32:
                        raise ValueError(
                            "optimizer tensor logging requires FP32 AdamW states"
                        )

                    if self._record_distributions:
                        bias_correction1 = 1 - beta1**step
                        bias_correction2 = 1 - beta2**step
                        for moment, variance, gradient_chunk in zip(
                            bounded_tensor_views(
                                local_exp_avg,
                                max_chunk_elements=_MAX_CHUNK_ELEMENTS,
                            ),
                            bounded_tensor_views(
                                local_exp_avg_sq,
                                max_chunk_elements=_MAX_CHUNK_ELEMENTS,
                            ),
                            bounded_tensor_views(
                                local_gradient,
                                max_chunk_elements=_MAX_CHUNK_ELEMENTS,
                            ),
                            strict=True,
                        ):
                            self._add_distribution_statistics(
                                group_index, parameter_index, 0, moment
                            )
                            denominator = (
                                variance.sqrt() / bias_correction2.sqrt()
                            ).add(eps)
                            self._add_distribution_statistics(
                                group_index, parameter_index, 1, denominator
                            )
                            self._add_distribution_statistics(
                                group_index,
                                parameter_index,
                                2,
                                gradient_chunk / denominator,
                            )
                            self._add_distribution_statistics(
                                group_index,
                                parameter_index,
                                3,
                                -(lr / bias_correction1) * moment / denominator,
                            )
                        first_row = parameter_index * len(_DISTRIBUTION_NAMES)
                        self._counts[group_index][
                            first_row : first_row + len(_DISTRIBUTION_NAMES), 3
                        ] = 1

                    if self._record_cosine:
                        cosine_row = (
                            self._distribution_rows[group_index] + parameter_index
                        )
                        for moment, gradient_chunk in zip(
                            bounded_tensor_views(
                                local_exp_avg,
                                max_chunk_elements=_MAX_CHUNK_ELEMENTS,
                            ),
                            bounded_tensor_views(
                                local_gradient,
                                max_chunk_elements=_MAX_CHUNK_ELEMENTS,
                            ),
                            strict=True,
                        ):
                            gradient_float = gradient_chunk.float()
                            self._sums[group_index][cosine_row].add_(
                                torch.stack(
                                    (
                                        torch.sum(moment * gradient_float),
                                        torch.sum(torch.square(moment)),
                                        torch.sum(torch.square(gradient_float)),
                                    )
                                )
                            )
                        self._counts[group_index][cosine_row, 3] = 1
                except Exception as error:
                    if self._local_error is None:
                        self._local_error = ValueError(
                            f"invalid AdamW tensor sample {parameter.fqn!r}: {error}"
                        )

    def _add_distribution_statistics(
        self,
        group_index: int,
        parameter_index: int,
        distribution_index: int,
        value: torch.Tensor,
    ) -> None:
        statistics = finite_statistics(value)
        row_index = parameter_index * len(_DISTRIBUTION_NAMES) + distribution_index
        self._counts[group_index][row_index, :3].add_(statistics.counts)
        self._sums[group_index][row_index, :2].add_(statistics.sums)
        self._maxima[group_index][row_index].copy_(
            torch.maximum(
                self._maxima[group_index][row_index],
                statistics.abs_max,
            )
        )
