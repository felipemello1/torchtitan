# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial, Replicate
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.moe import MoE
from torchtitan.observability.tensor_logging.offered_assignments import (
    OfferedAssignmentsRecorder,
)


class _Layer(nn.Module):
    def __init__(self, moe: MoE) -> None:
        super().__init__()
        self.moe = moe


class _Model(nn.Module):
    def __init__(self, moe: MoE) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(moe)])


def _build_model(*, num_experts: int = 4, top_k: int = 2) -> tuple[_Model, MoE]:
    moe = object.__new__(MoE)
    nn.Module.__init__(moe)
    moe.router = SimpleNamespace(num_experts=num_experts, top_k=top_k)
    moe.offered_assignments_recorder = None
    return _Model(moe), moe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestOfferedAssignmentsFourRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_ep_partials_and_tp_replicas_use_different_contributors(self) -> None:
        device = torch.device(self.device_type)

        ep_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=2,
            world_size=self.world_size,
        )
        ep_dims.build_mesh()
        ep_model, ep_moe = _build_model()
        ep_batch = OfferedAssignmentsRecorder(
            model=ep_model,
            parallel_dims=ep_dims,
            layer_ids=(0,),
            device=device,
        )
        rank_value = self.rank + 1
        ep_counts = DTensor.from_local(
            torch.full((4,), rank_value, dtype=torch.int64, device=device),
            ep_dims.get_mesh("tp"),
            (Partial(),),
            shape=torch.Size((4,)),
            stride=(1,),
            run_check=False,
        )
        assert ep_moe.offered_assignments_recorder is not None
        ep_moe.offered_assignments_recorder(ep_counts)
        ep_snapshot = ep_batch.collect()

        assert ep_snapshot.local_error is None
        assert ep_snapshot.values[0, :4].tolist() == [10, 10, 10, 10]
        assert int(ep_snapshot.values[0, 4]) == 2
        ep_batch.close()

        mismatched_model, mismatched_moe = _build_model()
        mismatched_batch = OfferedAssignmentsRecorder(
            model=mismatched_model,
            parallel_dims=ep_dims,
            layer_ids=(0,),
            device=device,
        )
        mismatched_counts = DTensor.from_local(
            torch.ones(4, dtype=torch.int64, device=device),
            ep_dims.get_mesh("tp"),
            (Replicate(),),
            shape=torch.Size((4,)),
            stride=(1,),
            run_check=False,
        )
        assert mismatched_moe.offered_assignments_recorder is not None
        mismatched_moe.offered_assignments_recorder(mismatched_counts)
        mismatched_snapshot = mismatched_batch.collect()
        assert mismatched_snapshot.local_error is not None
        assert "expected placements" in str(mismatched_snapshot.local_error)
        mismatched_batch.close()

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
        replica_model, replica_moe = _build_model()
        replica_batch = OfferedAssignmentsRecorder(
            model=replica_model,
            parallel_dims=replica_dims,
            layer_ids=(0,),
            device=device,
        )
        dp_rank = replica_dims.get_mesh("batch").get_local_rank()
        replica_counts = DTensor.from_local(
            torch.full((4,), dp_rank + 1, dtype=torch.int64, device=device),
            replica_dims.get_mesh("tp"),
            (Replicate(),),
            shape=torch.Size((4,)),
            stride=(1,),
            run_check=False,
        )
        assert replica_moe.offered_assignments_recorder is not None
        replica_moe.offered_assignments_recorder(replica_counts)
        replica_snapshot = replica_batch.collect()

        assert replica_snapshot.local_error is None
        assert replica_snapshot.values[0, :4].tolist() == [3, 3, 3, 3]
        assert int(replica_snapshot.values[0, 4]) == 2
        replica_batch.close()


def test_host_derivations_and_interval_reset() -> None:
    model, moe = _build_model()
    parallel_dims = SimpleNamespace(
        ep=2,
        world_size=1,
        get_optional_mesh=lambda _name: None,
    )
    batch = OfferedAssignmentsRecorder(
        model=model,
        parallel_dims=parallel_dims,
        layer_ids=(0,),
        device=torch.device("cpu"),
    )
    assert moe.offered_assignments_recorder is not None
    moe.offered_assignments_recorder(torch.tensor([1, 0, 2, 5], dtype=torch.int64))

    snapshot = batch.collect()
    metrics = batch.derive_metrics(snapshot, window_steps=3)
    prefix = "tensor_metrics/layers.0"
    assert metrics[f"{prefix}.experts.0.offered_count"] == 1
    assert metrics[f"{prefix}.experts.3.offered_load"] == 2.5
    assert metrics[f"{prefix}.moe.offered_maximum_violation"] == 1.5
    assert metrics[f"{prefix}.moe.offered_ep_shard_imbalance"] == 1.75
    assert metrics[f"{prefix}.moe.offered_assignments.routed_position_count"] == 4
    assert metrics[f"{prefix}.moe.offered_assignments.observation_count"] == 1
    assert metrics[f"{prefix}.moe.offered_assignments.window_steps"] == 3

    assert moe.offered_assignments_recorder is not None
    moe.offered_assignments_recorder(torch.zeros(4, dtype=torch.int64))
    zero_snapshot = batch.collect()
    zero_metrics = batch.derive_metrics(zero_snapshot, window_steps=1)
    assert zero_metrics[f"{prefix}.experts.0.offered_count"] == 0
    assert zero_metrics[f"{prefix}.moe.offered_assignments.routed_position_count"] == 0
    assert zero_metrics[f"{prefix}.moe.offered_assignments.observation_count"] == 1
    assert zero_metrics[f"{prefix}.moe.offered_assignments.window_steps"] == 1
    assert not any(key.endswith("offered_load") for key in zero_metrics)
    assert f"{prefix}.moe.offered_maximum_violation" not in zero_metrics
    assert f"{prefix}.moe.offered_ep_shard_imbalance" not in zero_metrics

    empty_snapshot = batch.collect()
    assert batch.derive_metrics(empty_snapshot, window_steps=1) == {}
    batch.close()
    assert moe.offered_assignments_recorder is None


def test_invalid_recorder_payload_is_latched() -> None:
    model, moe = _build_model()
    parallel_dims = SimpleNamespace(
        ep=1,
        world_size=1,
        get_optional_mesh=lambda _name: None,
    )
    batch = OfferedAssignmentsRecorder(
        model=model,
        parallel_dims=parallel_dims,
        layer_ids=(0,),
        device=torch.device("cpu"),
    )
    assert moe.offered_assignments_recorder is not None
    moe.offered_assignments_recorder(torch.ones(4))

    snapshot = batch.collect()
    assert snapshot.local_error is not None
    assert "expected int64 counts" in str(snapshot.local_error)
    batch.close()
