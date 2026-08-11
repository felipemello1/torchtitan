# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fixed-state router distribution and per-sequence telemetry."""

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial, Placement, Replicate, Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.utils import check_dtensor_placements_match
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter
from torchtitan.observability.tensor_logging.families import TensorMetricFamily
from torchtitan.observability.tensor_logging.statistics import reduce_max, reduce_sum


@dataclass(frozen=True, slots=True)
class RouterStatisticsSnapshot:
    """Reduced router sufficient statistics and any recorder error."""

    distribution_sums: torch.Tensor | None
    distribution_counts: torch.Tensor | None
    sequence_floats: torch.Tensor | None
    sequence_counts: torch.Tensor | None
    local_error: Exception | None


class RouterStatisticsRecorder:
    """Records selected router statistics without retaining token activations."""

    def __init__(
        self,
        *,
        model: nn.Module,
        parallel_dims: ParallelDims,
        layer_ids: tuple[int, ...],
        families: tuple[TensorMetricFamily, ...],
        local_batch_size: int,
        device: torch.device,
    ) -> None:
        distribution_selected = TensorMetricFamily.ROUTER_DISTRIBUTION in families
        sequence_selected = TensorMetricFamily.PER_SEQUENCE_ROUTING in families
        if not distribution_selected and not sequence_selected:
            raise ValueError("router statistics recorder requires a router family")

        modules = []
        score_functions = []
        num_experts: int | None = None
        for layer_id in layer_ids:
            fqn = f"layers.{layer_id}.moe"
            module = model.get_submodule(fqn)
            if type(module) is not MoE:
                raise ValueError(f"tensor logging requires an ordinary MoE at {fqn!r}")
            if type(module.router) is not TokenChoiceTopKRouter:
                raise ValueError(
                    "tensor logging requires an ordinary token-choice router at "
                    f"{fqn!r}"
                )
            if num_experts is None:
                num_experts = module.router.num_experts
            elif num_experts != module.router.num_experts:
                raise ValueError(
                    "tensor logging requires selected MoE layers to have the same "
                    "number of experts"
                )
            if distribution_selected:
                if module.router.num_expert_groups is not None:
                    raise ValueError(
                        "router distribution logging does not support "
                        "node-limited routing"
                    )
                if module.router._debug_force_load_balance:
                    raise ValueError(
                        "router distribution logging does not support forced routing"
                    )
            modules.append(module)
            score_functions.append(module.router.score_func)

        assert num_experts is not None
        self._layer_ids = layer_ids
        self._modules = tuple(modules)
        self._score_functions = tuple(score_functions)
        self._num_experts = num_experts
        self._distribution_selected = distribution_selected
        self._sequence_selected = sequence_selected

        self._tp_mesh = parallel_dims.get_optional_mesh("tp")
        self._world_mesh = (
            parallel_dims.world_mesh if parallel_dims.world_size > 1 else None
        )
        self._router_is_tp_sharded = parallel_dims.ep > 1
        self._expected_router_placements = (
            (Shard(1),) if self._router_is_tp_sharded else (Replicate(),)
        )
        self._expected_count_placements = (
            (Partial(),) if self._router_is_tp_sharded else (Replicate(),)
        )
        tp_representative = self._tp_mesh is None or self._tp_mesh.get_local_rank() == 0
        self._contributes_distribution = self._router_is_tp_sharded or tp_representative
        self._contributes_observations = tp_representative

        layer_count = len(layer_ids)
        self._distribution_sums = (
            torch.zeros(
                (layer_count, 3, num_experts), dtype=torch.float32, device=device
            )
            if distribution_selected
            else None
        )
        self._distribution_counts = (
            torch.zeros((layer_count, 2), dtype=torch.int64, device=device)
            if distribution_selected
            else None
        )
        self._sequence_assignments = (
            torch.zeros(
                (layer_count, local_batch_size, num_experts),
                dtype=torch.int64,
                device=device,
            )
            if sequence_selected
            else None
        )
        self._sequence_present = (
            torch.zeros(layer_count, dtype=torch.int64, device=device)
            if sequence_selected
            else None
        )
        self._distribution_validated = [False] * layer_count
        self._sequence_validated = [False] * layer_count
        self._distribution_failed = [False] * layer_count
        self._sequence_failed = [False] * layer_count
        self._local_error: Exception | None = None

        for row_index, module in enumerate(self._modules):
            if distribution_selected:
                module.router.statistics_recorder = partial(
                    self._record_distribution,
                    row_index,
                    layer_id=layer_ids[row_index],
                )
            if sequence_selected:
                module.per_sequence_assignments_recorder = partial(
                    self._record_sequence_assignments,
                    row_index,
                    layer_id=layer_ids[row_index],
                )

    def collect(self) -> RouterStatisticsSnapshot:
        """Reduce and reset one publication interval."""
        distribution_sums = None
        distribution_counts = None
        if self._distribution_sums is not None:
            assert self._distribution_counts is not None
            distribution_sums = self._distribution_sums.clone()
            distribution_counts = self._distribution_counts.clone()
            self._distribution_sums.zero_()
            self._distribution_counts.zero_()
            if self._world_mesh is not None:
                distribution_sums = reduce_sum(distribution_sums, self._world_mesh)
                distribution_counts = reduce_sum(distribution_counts, self._world_mesh)

        sequence_floats = None
        sequence_counts = None
        if self._sequence_assignments is not None:
            assert self._sequence_present is not None
            assignments = self._sequence_assignments.clone()
            present = self._sequence_present.clone()
            self._sequence_assignments.zero_()
            self._sequence_present.zero_()
            if self._router_is_tp_sharded:
                assert self._tp_mesh is not None
                assignments = reduce_sum(assignments, self._tp_mesh)
            sequence_sum = torch.zeros(
                len(self._layer_ids), dtype=torch.float32, device=assignments.device
            )
            sequence_max = torch.zeros_like(sequence_sum)
            sequence_counts = torch.zeros(
                (len(self._layer_ids), 3),
                dtype=torch.int64,
                device=assignments.device,
            )
            if self._contributes_observations:
                totals = assignments.sum(dim=2)
                assigned = totals > 0
                safe_totals = torch.where(assigned, totals, 1)
                violations = (
                    assignments.amax(dim=2).float() * self._num_experts / safe_totals
                    - 1.0
                )
                violations = torch.where(assigned, violations, 0.0)
                sequence_sum.copy_(violations.sum(dim=1))
                sequence_max.copy_(violations.amax(dim=1))
                sequence_counts[:, 0].copy_(present * assignments.shape[1])
                sequence_counts[:, 1].copy_(assigned.sum(dim=1) * present)
                sequence_counts[:, 2].copy_(present)
            if self._world_mesh is not None:
                sequence_sum = reduce_sum(sequence_sum, self._world_mesh)
                sequence_max = reduce_max(sequence_max, self._world_mesh)
                sequence_counts = reduce_sum(sequence_counts, self._world_mesh)
            sequence_floats = torch.stack((sequence_sum, sequence_max), dim=1)

        snapshot = RouterStatisticsSnapshot(
            distribution_sums=distribution_sums,
            distribution_counts=distribution_counts,
            sequence_floats=sequence_floats,
            sequence_counts=sequence_counts,
            local_error=self._local_error,
        )
        self._local_error = None
        return snapshot

    def derive_metrics(
        self,
        snapshot: RouterStatisticsSnapshot,
        *,
        window_steps: int,
    ) -> dict[str, int | float]:
        """Derive stable scalar keys from completed router statistics."""
        metrics: dict[str, int | float] = {}
        if snapshot.distribution_sums is not None:
            assert snapshot.distribution_counts is not None
            sums = snapshot.distribution_sums.cpu()
            counts = snapshot.distribution_counts.cpu()
            for row_index, layer_id in enumerate(self._layer_ids):
                position_count = int(counts[row_index, 0])
                if position_count == 0:
                    continue
                observation_count = int(counts[row_index, 1])
                mean_logits = sums[row_index, 0] / position_count
                mean_choice_scores = sums[row_index, 1] / position_count
                mean_bias = sums[row_index, 2] / position_count
                score_function = self._score_functions[row_index]
                if score_function == "softmax":
                    mean_scores = torch.softmax(mean_logits, dim=0)
                else:
                    mean_scores = torch.sigmoid(mean_logits)
                entropy_probabilities = mean_scores + mean_bias - mean_bias.min()
                entropy_probabilities /= entropy_probabilities.sum()
                entropy = -torch.sum(
                    entropy_probabilities * torch.log(entropy_probabilities)
                )

                prefix = f"tensor_metrics/layers.{layer_id}"
                for expert_id in range(self._num_experts):
                    metrics[f"{prefix}.experts.{expert_id}.router_logit_mean"] = float(
                        mean_logits[expert_id]
                    )
                    metrics[
                        f"{prefix}.experts.{expert_id}.router_choice_score_mean"
                    ] = float(mean_choice_scores[expert_id])
                metrics[f"{prefix}.moe.router_choice_entropy"] = float(entropy)
                metrics[f"{prefix}.moe.router.routed_position_count"] = position_count
                metrics[f"{prefix}.moe.router.observation_count"] = observation_count
                metrics[f"{prefix}.moe.router.window_steps"] = window_steps

        if snapshot.sequence_floats is not None:
            assert snapshot.sequence_counts is not None
            floats = snapshot.sequence_floats.cpu()
            counts = snapshot.sequence_counts.cpu()
            for row_index, layer_id in enumerate(self._layer_ids):
                sequence_count = int(counts[row_index, 0])
                if sequence_count == 0:
                    continue
                assigned_count = int(counts[row_index, 1])
                prefix = f"tensor_metrics/layers.{layer_id}.moe.per_sequence"
                metrics[f"{prefix}.sequence_count"] = sequence_count
                metrics[f"{prefix}.assigned_sequence_count"] = assigned_count
                metrics[f"{prefix}.observation_count"] = int(counts[row_index, 2])
                metrics[f"{prefix}.window_steps"] = window_steps
                if assigned_count > 0:
                    metrics[f"{prefix}.maximum_violation_mean"] = (
                        float(floats[row_index, 0]) / assigned_count
                    )
                    metrics[f"{prefix}.maximum_violation_max"] = float(
                        floats[row_index, 1]
                    )
        return metrics

    def close(self) -> None:
        """Unbind recorders from their owning router modules."""
        for module in self._modules:
            if self._distribution_selected:
                module.router.statistics_recorder = None
            if self._sequence_selected:
                module.per_sequence_assignments_recorder = None

    def _record_distribution(
        self,
        row_index: int,
        router_logits_BLE: torch.Tensor,
        choice_scores_BLE: torch.Tensor,
        expert_bias_E: torch.Tensor | None,
        *,
        layer_id: int,
    ) -> None:
        if self._distribution_failed[row_index]:
            return
        try:
            if not self._distribution_validated[row_index]:
                self._validate_distribution(
                    router_logits_BLE,
                    choice_scores_BLE,
                    expert_bias_E,
                )
                self._distribution_validated[row_index] = True
            logits = (
                router_logits_BLE.to_local()
                if isinstance(router_logits_BLE, DTensor)
                else router_logits_BLE
            ).detach()
            choice_scores = (
                choice_scores_BLE.to_local()
                if isinstance(choice_scores_BLE, DTensor)
                else choice_scores_BLE
            ).detach()
            position_count = logits.numel() // self._num_experts
            if self._contributes_distribution:
                assert self._distribution_sums is not None
                assert self._distribution_counts is not None
                self._distribution_sums[row_index, 0].add_(logits.sum(dim=(0, 1)))
                self._distribution_sums[row_index, 1].add_(
                    choice_scores.sum(dim=(0, 1))
                )
                if expert_bias_E is not None:
                    expert_bias = (
                        expert_bias_E.to_local()
                        if isinstance(expert_bias_E, DTensor)
                        else expert_bias_E
                    ).detach()
                    self._distribution_sums[row_index, 2].add_(
                        expert_bias * position_count
                    )
                self._distribution_counts[row_index, 0].add_(position_count)
            if self._contributes_observations:
                assert self._distribution_counts is not None
                self._distribution_counts[row_index, 1].add_(1)
        except Exception as error:
            self._distribution_failed[row_index] = True
            if self._local_error is None:
                self._local_error = ValueError(
                    f"invalid router distribution at layers.{layer_id}.moe: {error}"
                )

    def _record_sequence_assignments(
        self,
        row_index: int,
        value: torch.Tensor,
        *,
        layer_id: int,
    ) -> None:
        if self._sequence_failed[row_index]:
            return
        try:
            if not self._sequence_validated[row_index]:
                self._validate_sequence_assignments(value)
                self._sequence_validated[row_index] = True
            local_value = value.to_local() if isinstance(value, DTensor) else value
            assert self._sequence_assignments is not None
            assert self._sequence_present is not None
            self._sequence_assignments[row_index].copy_(local_value)
            self._sequence_present[row_index].fill_(1)
        except Exception as error:
            self._sequence_failed[row_index] = True
            if self._local_error is None:
                self._local_error = ValueError(
                    f"invalid per-sequence assignments at layers.{layer_id}.moe: "
                    f"{error}"
                )

    def _validate_distribution(
        self,
        router_logits_BLE: torch.Tensor,
        choice_scores_BLE: torch.Tensor,
        expert_bias_E: torch.Tensor | None,
    ) -> None:
        if router_logits_BLE.dtype is not torch.float32:
            raise ValueError(f"expected float32 logits, got {router_logits_BLE.dtype}")
        if choice_scores_BLE.dtype is not torch.float32:
            raise ValueError(
                f"expected float32 choice scores, got {choice_scores_BLE.dtype}"
            )
        if router_logits_BLE.shape != choice_scores_BLE.shape:
            raise ValueError("router logits and choice scores must have the same shape")
        if (
            router_logits_BLE.ndim != 3
            or router_logits_BLE.shape[-1] != self._num_experts
        ):
            raise ValueError(
                f"expected router shape (B, L, {self._num_experts}), "
                f"got {tuple(router_logits_BLE.shape)}"
            )
        self._validate_tp_value(
            router_logits_BLE,
            expected_placements=self._expected_router_placements,
        )
        self._validate_tp_value(
            choice_scores_BLE,
            expected_placements=self._expected_router_placements,
        )
        if expert_bias_E is not None:
            if expert_bias_E.dtype is not torch.float32 or expert_bias_E.shape != (
                self._num_experts,
            ):
                raise ValueError(
                    f"expected float32 expert bias shape ({self._num_experts},), "
                    f"got {expert_bias_E.dtype} {tuple(expert_bias_E.shape)}"
                )
            self._validate_tp_value(
                expert_bias_E,
                expected_placements=(Replicate(),),
            )

    def _validate_sequence_assignments(self, value: torch.Tensor) -> None:
        if value.dtype is not torch.int64:
            raise ValueError(f"expected int64 counts, got {value.dtype}")
        assert self._sequence_assignments is not None
        expected_shape = self._sequence_assignments.shape[1:]
        if value.shape != expected_shape:
            raise ValueError(
                f"expected shape {tuple(expected_shape)}, got {tuple(value.shape)}"
            )
        self._validate_tp_value(
            value,
            expected_placements=self._expected_count_placements,
        )

    def _validate_tp_value(
        self,
        value: torch.Tensor,
        *,
        expected_placements: tuple[Placement, ...],
    ) -> None:
        if self._tp_mesh is None:
            if isinstance(value, DTensor):
                raise ValueError("TP=1 router values must be local tensors")
            return
        if not isinstance(value, DTensor):
            raise ValueError("TP>1 router values must be DTensors")
        if (
            value.device_mesh.device_type != self._tp_mesh.device_type
            or value.device_mesh.mesh_dim_names != self._tp_mesh.mesh_dim_names
            or not torch.equal(value.device_mesh.mesh, self._tp_mesh.mesh)
        ):
            raise ValueError("router values must use the ParallelDims TP mesh")
        if not check_dtensor_placements_match(
            value.placements,
            expected_placements,
            value.ndim,
        ):
            raise ValueError(
                f"expected placements {expected_placements}, got {value.placements}"
            )
