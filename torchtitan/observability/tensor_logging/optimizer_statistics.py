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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.families import TensorMetricFamily
from torchtitan.observability.tensor_logging.parameter_ownership import (
    resolve_parameter_owner_meshes,
)
from torchtitan.observability.tensor_logging.statistics import (
    bounded_tensor_views,
    derive_finite_statistics,
    finite_statistics,
    FiniteStatistics,
    reduce_finite_statistics,
    reduce_sum,
)


_DISTRIBUTION_NAMES = (
    "numerator",
    "denominator",
    "preconditioned_gradient",
    "update_pre_apply",
)
_MAX_CHUNK_ELEMENTS = 1_048_576


@dataclass(frozen=True, slots=True)
class _BoundParameter:
    fqn: str
    value: torch.Tensor
    numel: int


@dataclass(frozen=True, slots=True)
class _ParameterGroup:
    parameters: tuple[_BoundParameter, ...]
    owner_meshes: tuple[DeviceMesh, ...]
    reduction_meshes: tuple[DeviceMesh, ...]
    expected_contributors: int


@dataclass(frozen=True, slots=True)
class OptimizerStatisticsSnapshot:
    distributions: tuple[FiniteStatistics, ...]
    cosine_sums: tuple[torch.Tensor, ...]
    cosine_present: tuple[torch.Tensor, ...]
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
        layer_prefixes = tuple(f"layers.{layer_id}." for layer_id in layer_ids)
        parameters_by_identity: dict[int, torch.Tensor] = {}
        names_by_identity: dict[int, list[str]] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            fqn = canonical_fqn(name)
            if not any(fqn.startswith(prefix) for prefix in layer_prefixes):
                continue
            if not parameter.requires_grad or not parameter.is_floating_point():
                continue
            parameters_by_identity[id(parameter)] = parameter
            names_by_identity.setdefault(id(parameter), []).append(fqn)

        parameters = tuple(
            sorted(
                (
                    _BoundParameter(
                        fqn=min(names),
                        value=parameters_by_identity[identity],
                        numel=parameters_by_identity[identity].numel(),
                    )
                    for identity, names in names_by_identity.items()
                ),
                key=lambda parameter: parameter.fqn,
            )
        )
        if not parameters:
            raise ValueError(
                "tensor logging found no selected-layer optimizer parameters"
            )

        parameters_by_owner: dict[tuple[str, ...], list[_BoundParameter]] = {}
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
                _ParameterGroup(
                    parameters=tuple(owned_parameters),
                    owner_meshes=owner_meshes,
                    reduction_meshes=reduction_meshes,
                    expected_contributors=expected_contributors,
                )
            )

        self._groups = tuple(groups)
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
        device = parameters[0].value.device
        self._counts = tuple(
            torch.zeros(
                (len(group.parameters) * len(_DISTRIBUTION_NAMES), 4),
                dtype=torch.int64,
                device=device,
            )
            for group in self._groups
        )
        self._sums = tuple(
            torch.zeros(
                (len(group.parameters) * len(_DISTRIBUTION_NAMES), 2),
                dtype=torch.float32,
                device=device,
            )
            for group in self._groups
        )
        self._maxima = tuple(
            torch.zeros(
                (len(group.parameters) * len(_DISTRIBUTION_NAMES), 1),
                dtype=torch.float32,
                device=device,
            )
            for group in self._groups
        )
        self._cosine_sums = tuple(
            torch.zeros((len(group.parameters), 3), dtype=torch.float32, device=device)
            for group in self._groups
        )
        self._cosine_present = tuple(
            torch.zeros(len(group.parameters), dtype=torch.int64, device=device)
            for group in self._groups
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
            selected_groups: list[tuple[dict, tuple[_BoundParameter, ...]]] = []
            for parameter_group in optimizer.param_groups:
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
                selected_groups.append((parameter_group, tuple(selected_parameters)))
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
        for values in (
            *self._counts,
            *self._sums,
            *self._maxima,
            *self._cosine_sums,
            *self._cosine_present,
        ):
            values.zero_()
        self._local_error = None
        self._record_next_step = should_log

    def collect(self) -> OptimizerStatisticsSnapshot:
        """Reduce the optimizer sample after every inner AdamW has stepped."""
        distributions = []
        cosine_sums = []
        cosine_present = []
        for group_index, group in enumerate(self._groups):
            if self._record_distributions:
                distributions.append(
                    reduce_finite_statistics(
                        FiniteStatistics(
                            counts=self._counts[group_index],
                            sums=self._sums[group_index],
                            abs_max=self._maxima[group_index],
                        ),
                        group.reduction_meshes,
                    )
                )
            if self._record_cosine:
                reduced_sums = self._cosine_sums[group_index].clone()
                reduced_present = self._cosine_present[group_index].clone()
                for mesh in group.reduction_meshes:
                    reduced_sums = reduce_sum(reduced_sums, mesh)
                    reduced_present = reduce_sum(reduced_present, mesh)
                cosine_sums.append(reduced_sums)
                cosine_present.append(reduced_present)

        self._record_next_step = False
        return OptimizerStatisticsSnapshot(
            distributions=tuple(distributions),
            cosine_sums=tuple(cosine_sums),
            cosine_present=tuple(cosine_present),
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
        distribution_group_index = 0
        cosine_group_index = 0
        for group in self._groups:
            if self._record_distributions:
                statistics = snapshot.distributions[distribution_group_index]
                distribution_group_index += 1
                host_counts = statistics.counts.cpu()
                host_floats = torch.cat(
                    (statistics.sums, statistics.abs_max), dim=1
                ).cpu()
                for parameter_index, parameter in enumerate(group.parameters):
                    for distribution_index, distribution_name in enumerate(
                        _DISTRIBUTION_NAMES
                    ):
                        row_index = (
                            parameter_index * len(_DISTRIBUTION_NAMES)
                            + distribution_index
                        )
                        present = int(host_counts[row_index, 3])
                        self._validate_presence(
                            parameter.fqn,
                            present,
                            group.expected_contributors,
                        )
                        row_statistics = FiniteStatistics(
                            counts=host_counts[row_index, :3],
                            sums=host_floats[row_index, :2],
                            abs_max=host_floats[row_index, 2:3],
                        )
                        if int(row_statistics.counts[0]) != parameter.numel:
                            raise ValueError(
                                f"reduced optimizer tensor numel for {parameter.fqn!r} "
                                f"is {int(row_statistics.counts[0])}, expected "
                                f"{parameter.numel}"
                            )
                        derived = derive_finite_statistics(row_statistics)
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
                host_sums = snapshot.cosine_sums[cosine_group_index].cpu()
                host_present = snapshot.cosine_present[cosine_group_index].cpu()
                cosine_group_index += 1
                for parameter_index, parameter in enumerate(group.parameters):
                    self._validate_presence(
                        parameter.fqn,
                        int(host_present[parameter_index]),
                        group.expected_contributors,
                    )
                    dot, momentum_square, gradient_square = (
                        float(value) for value in host_sums[parameter_index]
                    )
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

    def _local_owned_value(
        self,
        value: torch.Tensor,
        *,
        expected_owner_meshes: tuple[DeviceMesh, ...],
    ) -> torch.Tensor:
        owner_meshes = resolve_parameter_owner_meshes(
            value,
            parallel_dims=self._parallel_dims,
        )
        if len(owner_meshes) != len(expected_owner_meshes) or any(
            actual is not expected
            for actual, expected in zip(
                owner_meshes,
                expected_owner_meshes,
                strict=True,
            )
        ):
            raise ValueError("optimizer tensor owner cohort differs from parameter")
        if not isinstance(value, DTensor):
            raise ValueError("optimizer tensor is not a DTensor")
        return value.to_local()

    def _record_optimizer_state(
        self,
        optimizer: Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
        *,
        selected_groups: tuple[tuple[dict, tuple[_BoundParameter, ...]], ...],
    ) -> None:
        if not self._record_next_step:
            return
        for parameter_group, parameters in selected_groups:
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
                    local_gradient = self._local_owned_value(
                        gradient,
                        expected_owner_meshes=group.owner_meshes,
                    )
                    local_exp_avg = self._local_owned_value(
                        exp_avg,
                        expected_owner_meshes=group.owner_meshes,
                    )
                    local_exp_avg_sq = self._local_owned_value(
                        exp_avg_sq,
                        expected_owner_meshes=group.owner_meshes,
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
                        momentum = local_exp_avg.float()
                        gradient_float = local_gradient.float()
                        self._cosine_sums[group_index][parameter_index].copy_(
                            torch.stack(
                                (
                                    torch.sum(momentum * gradient_float),
                                    torch.sum(torch.square(momentum)),
                                    torch.sum(torch.square(gradient_float)),
                                )
                            )
                        )
                        self._cosine_present[group_index][parameter_index] = 1
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
        self._sums[group_index][row_index].add_(statistics.sums)
        self._maxima[group_index][row_index].copy_(
            torch.maximum(
                self._maxima[group_index][row_index],
                statistics.abs_max,
            )
        )
