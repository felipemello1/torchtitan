# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter
from torchtitan.observability.tensor_logging import TensorMetricFamily
from torchtitan.observability.tensor_logging.router_statistics import (
    RouterStatisticsRecorder,
)
from torchtitan.observability.tensor_logging.statistics import ReductionBatch


class _Layer(nn.Module):
    def __init__(self, moe: MoE) -> None:
        super().__init__()
        self.moe = moe


class _Model(nn.Module):
    def __init__(self, moe: MoE) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(moe)])


def _build_model(
    *,
    expert_bias: torch.Tensor | None,
    score_func: str = "softmax",
) -> tuple[_Model, MoE]:
    router = object.__new__(TokenChoiceTopKRouter)
    nn.Module.__init__(router)
    router.num_experts = 2
    router.score_func = score_func
    router.num_expert_groups = None
    router._debug_force_load_balance = False
    router.statistics_recorder = None

    moe = object.__new__(MoE)
    nn.Module.__init__(moe)
    moe.router = router
    moe.expert_bias_E = expert_bias
    moe.per_sequence_assignments_recorder = None
    return _Model(moe), moe


def _single_rank_dims(*, ep: int) -> SimpleNamespace:
    return SimpleNamespace(
        ep=ep,
        world_size=1,
        get_optional_mesh=lambda _name: None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestRouterStatisticsFourRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_tp_partials_and_replicas_use_different_contributors(self) -> None:
        device = torch.device(self.device_type)

        ep_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=4,
            pp=1,
            ep=2,
            world_size=self.world_size,
        )
        ep_dims.build_mesh()
        ep_model, ep_moe = _build_model(
            expert_bias=torch.zeros(2, dtype=torch.float32, device=device)
        )
        ep_recorder = RouterStatisticsRecorder(
            model=ep_model,
            parallel_dims=ep_dims,
            layer_ids=(0,),
            families=(
                TensorMetricFamily.ROUTER_DISTRIBUTION,
                TensorMetricFamily.PER_SEQUENCE_ROUTING,
            ),
            local_batch_size=1,
            device=device,
        )
        rank_value = float(self.rank + 1)
        logits = DTensor.from_local(
            torch.tensor([[[rank_value, 0.0]]], device=device),
            ep_dims.get_mesh("tp"),
            (Shard(1),),
            shape=torch.Size((1, 4, 2)),
            stride=(8, 2, 1),
            run_check=False,
        )
        choice_scores = DTensor.from_local(
            torch.tensor([[[rank_value, 0.0]]], device=device),
            ep_dims.get_mesh("tp"),
            (Shard(1),),
            shape=torch.Size((1, 4, 2)),
            stride=(8, 2, 1),
            run_check=False,
        )
        bias = DTensor.from_local(
            torch.zeros(2, dtype=torch.float32, device=device),
            ep_dims.get_mesh("tp"),
            (Replicate(),),
            shape=torch.Size((2,)),
            stride=(1,),
            run_check=False,
        )
        partial_counts = DTensor.from_local(
            torch.tensor(
                [[2, 0] if self.rank % 2 == 0 else [0, 2]],
                dtype=torch.int64,
                device=device,
            ),
            ep_dims.get_mesh("tp"),
            (Partial(),),
            shape=torch.Size((1, 2)),
            stride=(2, 1),
            run_check=False,
        )
        assert ep_moe.router.statistics_recorder is not None
        ep_moe.router.statistics_recorder(logits, choice_scores, bias)
        assert ep_moe.per_sequence_assignments_recorder is not None
        ep_moe.per_sequence_assignments_recorder(partial_counts)

        ep_batch = ReductionBatch()
        ep_snapshot = ep_recorder.collect(batch=ep_batch)
        ep_batch.reduce()
        ep_metrics = ep_recorder.derive_metrics(ep_snapshot, window_steps=1)
        prefix = "tensor_metrics/layers.0"
        assert ep_metrics[f"{prefix}.experts.0.router_logit_mean"] == 2.5
        assert ep_metrics[f"{prefix}.moe.router.routed_position_count"] == 4
        assert ep_metrics[f"{prefix}.moe.router.observation_count"] == 1
        assert ep_metrics[f"{prefix}.moe.per_sequence.maximum_violation_mean"] == 0.0
        assert ep_metrics[f"{prefix}.moe.per_sequence.maximum_violation_max"] == 0.0
        assert ep_metrics[f"{prefix}.moe.per_sequence.sequence_count"] == 1
        ep_recorder.close()

        replica_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        replica_dims.build_mesh()
        replica_model, replica_moe = _build_model(expert_bias=None)
        replica_recorder = RouterStatisticsRecorder(
            model=replica_model,
            parallel_dims=replica_dims,
            layer_ids=(0,),
            families=(TensorMetricFamily.ROUTER_DISTRIBUTION,),
            local_batch_size=1,
            device=device,
        )
        dp_rank = replica_dims.get_mesh("batch").get_local_rank()
        replica_logits = DTensor.from_local(
            torch.tensor([[[float(dp_rank), 0.0]]], device=device),
            replica_dims.get_mesh("tp"),
            (Replicate(),),
            shape=torch.Size((1, 1, 2)),
            stride=(2, 2, 1),
            run_check=False,
        )
        replica_scores = DTensor.from_local(
            torch.tensor([[[0.5, 0.5]]], device=device),
            replica_dims.get_mesh("tp"),
            (Replicate(),),
            shape=torch.Size((1, 1, 2)),
            stride=(2, 2, 1),
            run_check=False,
        )
        assert replica_moe.router.statistics_recorder is not None
        replica_moe.router.statistics_recorder(replica_logits, replica_scores, None)
        replica_batch = ReductionBatch()
        replica_snapshot = replica_recorder.collect(batch=replica_batch)
        replica_batch.reduce()
        replica_metrics = replica_recorder.derive_metrics(
            replica_snapshot, window_steps=1
        )
        assert replica_metrics[f"{prefix}.experts.0.router_logit_mean"] == 0.5
        assert replica_metrics[f"{prefix}.moe.router.routed_position_count"] == 2
        assert replica_metrics[f"{prefix}.moe.router.observation_count"] == 2
        replica_recorder.close()

    @with_comms
    def test_ep_without_tp_uses_distinct_dp_sequences(self) -> None:
        device = torch.device(self.device_type)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=4,
            cp=1,
            tp=1,
            pp=1,
            ep=4,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        model, moe = _build_model(expert_bias=None)
        recorder = RouterStatisticsRecorder(
            model=model,
            parallel_dims=parallel_dims,
            layer_ids=(0,),
            families=(TensorMetricFamily.PER_SEQUENCE_ROUTING,),
            local_batch_size=2,
            device=device,
        )
        assert moe.per_sequence_assignments_recorder is not None
        moe.per_sequence_assignments_recorder(
            torch.tensor([[1, 0], [0, 1]], dtype=torch.int64, device=device)
        )

        batch = ReductionBatch()
        snapshot = recorder.collect(batch=batch)
        batch.reduce()
        metrics = recorder.derive_metrics(snapshot, window_steps=1)
        prefix = "tensor_metrics/layers.0.moe.per_sequence"
        assert metrics[f"{prefix}.maximum_violation_mean"] == 1.0
        assert metrics[f"{prefix}.maximum_violation_max"] == 1.0
        assert metrics[f"{prefix}.sequence_count"] == 8
        assert metrics[f"{prefix}.assigned_sequence_count"] == 8
        assert metrics[f"{prefix}.observation_count"] == 4
        recorder.close()


def test_router_distribution_is_token_weighted_and_resets() -> None:
    model, moe = _build_model(expert_bias=torch.tensor([0.2, -0.2]))
    recorder = RouterStatisticsRecorder(
        model=model,
        parallel_dims=_single_rank_dims(ep=1),
        layer_ids=(0,),
        families=(TensorMetricFamily.ROUTER_DISTRIBUTION,),
        local_batch_size=1,
        device=torch.device("cpu"),
    )
    assert moe.router.statistics_recorder is not None
    moe.router.statistics_recorder(
        torch.tensor([[[0.0, 2.0]]], requires_grad=True),
        torch.tensor([[[0.2, 0.8]]], requires_grad=True),
        torch.tensor([0.0, 0.0]),
    )
    moe.router.statistics_recorder(
        torch.tensor([[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]]),
        torch.tensor([[[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]]),
        torch.tensor([0.2, -0.2]),
    )

    snapshot = recorder.collect()
    assert snapshot.distribution_sums is not None
    assert not snapshot.distribution_sums.requires_grad
    metrics = recorder.derive_metrics(snapshot, window_steps=4)
    prefix = "tensor_metrics/layers.0"
    assert metrics[f"{prefix}.experts.0.router_logit_mean"] == pytest.approx(1.5)
    assert metrics[f"{prefix}.experts.1.router_logit_mean"] == pytest.approx(0.5)
    assert metrics[f"{prefix}.experts.0.router_choice_score_mean"] == pytest.approx(
        0.725
    )
    assert metrics[f"{prefix}.experts.1.router_choice_score_mean"] == pytest.approx(
        0.275
    )

    mean_scores = torch.softmax(torch.tensor([1.5, 0.5]), dim=0)
    entropy_probabilities = mean_scores + torch.tensor([0.3, 0.0])
    entropy_probabilities /= entropy_probabilities.sum()
    expected_entropy = -torch.sum(
        entropy_probabilities * torch.log(entropy_probabilities)
    )
    assert metrics[f"{prefix}.moe.router_choice_entropy"] == pytest.approx(
        float(expected_entropy)
    )
    assert metrics[f"{prefix}.moe.router.routed_position_count"] == 4
    assert metrics[f"{prefix}.moe.router.observation_count"] == 2
    assert metrics[f"{prefix}.moe.router.window_steps"] == 4
    assert recorder.derive_metrics(recorder.collect(), window_steps=1) == {}

    recorder.close()
    assert moe.router.statistics_recorder is None


def test_router_distribution_accepts_bias_free_choice_scores() -> None:
    model, moe = _build_model(expert_bias=None)
    recorder = RouterStatisticsRecorder(
        model=model,
        parallel_dims=_single_rank_dims(ep=1),
        layer_ids=(0,),
        families=(TensorMetricFamily.ROUTER_DISTRIBUTION,),
        local_batch_size=1,
        device=torch.device("cpu"),
    )
    assert moe.router.statistics_recorder is not None
    logits = torch.tensor([[[0.0, 0.0]]])
    scores = torch.tensor([[[0.5, 0.5]]])
    moe.router.statistics_recorder(logits, scores, None)

    metrics = recorder.derive_metrics(recorder.collect(), window_steps=1)
    assert metrics[
        "tensor_metrics/layers.0.moe.router_choice_entropy"
    ] == pytest.approx(float(torch.log(torch.tensor(2.0))))
    recorder.close()


@pytest.mark.parametrize("score_func", ("softmax", "sigmoid"))
def test_router_distribution_entropy_handles_saturated_logits(
    score_func: str,
) -> None:
    model, moe = _build_model(expert_bias=None, score_func=score_func)
    recorder = RouterStatisticsRecorder(
        model=model,
        parallel_dims=_single_rank_dims(ep=1),
        layer_ids=(0,),
        families=(TensorMetricFamily.ROUTER_DISTRIBUTION,),
        local_batch_size=1,
        device=torch.device("cpu"),
    )
    assert moe.router.statistics_recorder is not None
    moe.router.statistics_recorder(
        torch.tensor([[[-1000.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0]]]),
        None,
    )

    metrics = recorder.derive_metrics(recorder.collect(), window_steps=1)
    assert metrics["tensor_metrics/layers.0.moe.router_choice_entropy"] == 0.0
    recorder.close()


def test_per_sequence_routing_overwrites_the_previous_forward() -> None:
    model, moe = _build_model(expert_bias=None)
    recorder = RouterStatisticsRecorder(
        model=model,
        parallel_dims=_single_rank_dims(ep=1),
        layer_ids=(0,),
        families=(TensorMetricFamily.PER_SEQUENCE_ROUTING,),
        local_batch_size=1,
        device=torch.device("cpu"),
    )
    assert moe.per_sequence_assignments_recorder is not None
    moe.per_sequence_assignments_recorder(torch.tensor([[2, 0]], dtype=torch.int64))
    moe.per_sequence_assignments_recorder(torch.tensor([[0, 2]], dtype=torch.int64))

    metrics = recorder.derive_metrics(recorder.collect(), window_steps=5)
    prefix = "tensor_metrics/layers.0.moe.per_sequence"
    assert metrics[f"{prefix}.maximum_violation_mean"] == 1.0
    assert metrics[f"{prefix}.maximum_violation_max"] == 1.0
    assert metrics[f"{prefix}.sequence_count"] == 1
    assert metrics[f"{prefix}.assigned_sequence_count"] == 1
    assert metrics[f"{prefix}.observation_count"] == 1
    assert metrics[f"{prefix}.window_steps"] == 5
    recorder.close()
