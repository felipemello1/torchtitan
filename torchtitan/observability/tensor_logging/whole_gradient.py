# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Logical whole-gradient health over dense and sparse parameter owners."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed.parallel_dims import ParallelDims
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


_CATEGORY_NAMES = ("all", "token_embedding", "moe")


@dataclass(frozen=True, slots=True)
class _BoundGradient:
    parameter: BoundParameter
    category_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _GradientGroup:
    gradients: tuple[_BoundGradient, ...]
    owner_meshes: tuple[DeviceMesh, ...]
    reduction_meshes: tuple[DeviceMesh, ...]
    expected_numel: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WholeGradientSnapshot:
    statistics: tuple[FiniteStatistics, ...]
    local_error: Exception | None


class WholeGradientStatistics:
    """Collects each logical gradient element once across owner cohorts."""

    def __init__(self, *, model: nn.Module, parallel_dims: ParallelDims) -> None:
        parameters = bind_parameters(model, layer_ids=None)
        gradients = []
        for parameter in parameters:
            category_indices = [0]
            if "tok_embeddings.weight" in parameter.aliases:
                category_indices.append(1)
            if any(".moe." in name for name in parameter.aliases):
                category_indices.append(2)
            gradients.append(
                _BoundGradient(
                    parameter=parameter,
                    category_indices=tuple(category_indices),
                )
            )
        if not gradients:
            raise ValueError("tensor logging found no trainable floating parameters")

        gradient_by_parameter_id = {
            id(gradient.parameter.value): gradient for gradient in gradients
        }

        groups = []
        for owner_group in group_parameters_by_owner(
            parameters,
            parallel_dims=parallel_dims,
        ):
            owned_gradients = tuple(
                gradient_by_parameter_id[id(parameter.value)]
                for parameter in owner_group.parameters
            )
            expected_numel = [0] * len(_CATEGORY_NAMES)
            for gradient in owned_gradients:
                for category_index in gradient.category_indices:
                    expected_numel[category_index] += gradient.parameter.numel
            groups.append(
                _GradientGroup(
                    gradients=owned_gradients,
                    owner_meshes=owner_group.owner_meshes,
                    reduction_meshes=owner_group.reduction_meshes,
                    expected_numel=tuple(expected_numel),
                )
            )

        self._groups = tuple(groups)
        self._parallel_dims = parallel_dims
        self._device = gradients[0].parameter.value.device

    def collect(
        self,
        *,
        step: int,
        batch: ReductionBatch | None = None,
    ) -> WholeGradientSnapshot:
        """Read completed preclip gradients and reduce each owner cohort."""
        reduced_groups = []
        local_error: Exception | None = None
        for group in self._groups:
            counts = torch.zeros(
                (len(_CATEGORY_NAMES), 3), dtype=torch.int64, device=self._device
            )
            sums = torch.zeros(
                (len(_CATEGORY_NAMES), 2), dtype=torch.float32, device=self._device
            )
            abs_max = torch.zeros(
                (len(_CATEGORY_NAMES), 1), dtype=torch.float32, device=self._device
            )
            for gradient in group.gradients:
                value = gradient.parameter.value.grad
                try:
                    if value is None:
                        raise ValueError("gradient is absent")
                    local_value = local_value_for_owner_group(
                        value,
                        owner_meshes=group.owner_meshes,
                        parallel_dims=self._parallel_dims,
                        label="gradient",
                    )
                    statistics = finite_statistics(local_value)
                except Exception as error:
                    if local_error is None:
                        local_error = ValueError(
                            f"invalid whole-gradient sample "
                            f"{gradient.parameter.fqn!r} "
                            f"at step {step}: {error}"
                        )
                    continue

                for category_index in gradient.category_indices:
                    counts[category_index].add_(statistics.counts)
                    sums[category_index].add_(statistics.sums)
                    abs_max[category_index].copy_(
                        torch.maximum(
                            abs_max[category_index],
                            statistics.abs_max,
                        )
                    )

            reduced_groups.append(
                reduce_finite_statistics(
                    FiniteStatistics(counts=counts, sums=sums, abs_max=abs_max),
                    group.reduction_meshes,
                    batch=batch,
                )
            )
        return WholeGradientSnapshot(
            statistics=tuple(reduced_groups),
            local_error=local_error,
        )

    def derive_metrics(
        self,
        snapshot: WholeGradientSnapshot,
        *,
        window_steps: int,
    ) -> dict[str, int | float]:
        """Combine completed owner fragments and derive three logical views."""
        combined_counts = torch.zeros((len(_CATEGORY_NAMES), 3), dtype=torch.int64)
        combined_sums = torch.zeros((len(_CATEGORY_NAMES), 2), dtype=torch.float32)
        combined_abs_max = torch.zeros((len(_CATEGORY_NAMES), 1), dtype=torch.float32)
        expected_numel = [0] * len(_CATEGORY_NAMES)

        for group, statistics in zip(
            self._groups,
            snapshot.statistics,
            strict=True,
        ):
            host_counts = statistics.counts.cpu()
            host_sums = statistics.sums.cpu()
            host_abs_max = statistics.abs_max.cpu()
            for category_index, group_numel in enumerate(group.expected_numel):
                if group_numel == 0:
                    continue
                reduced_numel = int(host_counts[category_index, 0])
                if reduced_numel != group_numel:
                    raise ValueError(
                        f"reduced {_CATEGORY_NAMES[category_index]} gradient numel "
                        f"is {reduced_numel}, expected {group_numel}"
                    )
                expected_numel[category_index] += group_numel
                combined_counts[category_index].add_(host_counts[category_index])
                combined_sums[category_index].add_(host_sums[category_index])
                combined_abs_max[category_index].copy_(
                    torch.maximum(
                        combined_abs_max[category_index],
                        host_abs_max[category_index],
                    )
                )

        if expected_numel[1] + expected_numel[2] > expected_numel[0]:
            raise RuntimeError("whole-gradient subsets exceed the all-gradient view")

        metrics: dict[str, int | float] = {}
        for category_index, category_name in enumerate(_CATEGORY_NAMES):
            if expected_numel[category_index] == 0:
                continue
            derived = derive_finite_statistics(
                FiniteStatistics(
                    counts=combined_counts[category_index],
                    sums=combined_sums[category_index],
                    abs_max=combined_abs_max[category_index],
                )
            )
            derived["observation_count"] = 1
            derived["window_steps"] = window_steps
            prefix = f"tensor_metrics/gradients/{category_name}"
            metrics.update(
                {f"{prefix}.{name}": value for name, value in derived.items()}
            )
        return metrics
