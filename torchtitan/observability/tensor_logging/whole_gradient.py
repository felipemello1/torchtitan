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
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.parameter_ownership import (
    resolve_parameter_owner_meshes,
)
from torchtitan.observability.tensor_logging.statistics import (
    derive_finite_statistics,
    finite_statistics,
    FiniteStatistics,
    reduce_finite_statistics,
)


_CATEGORY_NAMES = ("all", "token_embedding", "moe")


@dataclass(frozen=True, slots=True)
class _BoundGradient:
    fqn: str
    parameter: torch.Tensor
    numel: int
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
        parameters_by_identity: dict[int, torch.Tensor] = {}
        names_by_identity: dict[int, list[str]] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if not parameter.requires_grad or not parameter.is_floating_point():
                continue
            identity = id(parameter)
            parameters_by_identity[identity] = parameter
            names_by_identity.setdefault(identity, []).append(canonical_fqn(name))

        gradients = []
        for identity, names in names_by_identity.items():
            fqn = min(names)
            category_indices = [0]
            if "tok_embeddings.weight" in names:
                category_indices.append(1)
            if any(".moe." in name for name in names):
                category_indices.append(2)
            parameter = parameters_by_identity[identity]
            gradients.append(
                _BoundGradient(
                    fqn=fqn,
                    parameter=parameter,
                    numel=parameter.numel(),
                    category_indices=tuple(category_indices),
                )
            )
        gradients.sort(key=lambda gradient: gradient.fqn)
        if not gradients:
            raise ValueError("tensor logging found no trainable floating parameters")

        gradients_by_owner: dict[tuple[str, ...], list[_BoundGradient]] = {}
        owner_meshes_by_key: dict[tuple[str, ...], tuple[DeviceMesh, ...]] = {}
        for gradient in gradients:
            owner_meshes = resolve_parameter_owner_meshes(
                gradient.parameter,
                parallel_dims=parallel_dims,
            )
            owner_key = tuple(
                mesh.mesh_dim_names[0] for mesh in owner_meshes if mesh.mesh_dim_names
            )
            gradients_by_owner.setdefault(owner_key, []).append(gradient)
            owner_meshes_by_key[owner_key] = owner_meshes

        groups = []
        for owner_key, owned_gradients in gradients_by_owner.items():
            owner_meshes = owner_meshes_by_key[owner_key]
            owner_count = 1
            for mesh in owner_meshes:
                owner_count *= mesh.size()
            reduction_meshes = (
                (parallel_dims.world_mesh,)
                if len(owner_meshes) > 1
                and owner_count == parallel_dims.world_mesh.size()
                else owner_meshes
            )
            expected_numel = [0] * len(_CATEGORY_NAMES)
            for gradient in owned_gradients:
                for category_index in gradient.category_indices:
                    expected_numel[category_index] += gradient.numel
            groups.append(
                _GradientGroup(
                    gradients=tuple(owned_gradients),
                    owner_meshes=owner_meshes,
                    reduction_meshes=reduction_meshes,
                    expected_numel=tuple(expected_numel),
                )
            )

        self._groups = tuple(groups)
        self._parallel_dims = parallel_dims
        self._device = gradients[0].parameter.device

    def collect(self, *, step: int) -> WholeGradientSnapshot:
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
                value = gradient.parameter.grad
                try:
                    if value is None:
                        raise ValueError("gradient is absent")
                    gradient_owner_meshes = resolve_parameter_owner_meshes(
                        value,
                        parallel_dims=self._parallel_dims,
                    )
                    if len(gradient_owner_meshes) != len(group.owner_meshes) or any(
                        actual is not expected
                        for actual, expected in zip(
                            gradient_owner_meshes,
                            group.owner_meshes,
                            strict=True,
                        )
                    ):
                        raise ValueError(
                            "gradient owner cohort differs from its parameter"
                        )
                    if not isinstance(value, DTensor):
                        raise ValueError("gradient is not a DTensor")
                    statistics = finite_statistics(value.to_local())
                except Exception as error:
                    if local_error is None:
                        local_error = ValueError(
                            f"invalid whole-gradient sample {gradient.fqn!r} "
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
