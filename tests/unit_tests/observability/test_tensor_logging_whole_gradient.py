# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.testing._internal.distributed.fake_pg  # noqa: F401
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.whole_gradient import (
    WholeGradientStatistics,
)


class _Weight(nn.Module):
    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__()
        self.weight = weight


class _Layer(nn.Module):
    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__()
        self.moe = _Weight(weight)


class _Model(nn.Module):
    def __init__(
        self,
        token_embedding: nn.Parameter,
        moe_weight: nn.Parameter,
    ) -> None:
        super().__init__()
        self.tok_embeddings = _Weight(token_embedding)
        self.output = _Weight(token_embedding)
        self.layers = nn.ModuleList([_Layer(moe_weight)])


@pytest.fixture
def fake_world_one() -> Iterator[None]:
    dist.init_process_group("fake", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _parameter(
    local: torch.Tensor,
    mesh,
    placements,
    *,
    shape: torch.Size | None = None,
) -> nn.Parameter:
    return nn.Parameter(
        DTensor.from_local(
            local,
            mesh,
            placements,
            shape=local.shape if shape is None else shape,
            stride=(local.shape[-1], 1),
            run_check=False,
        )
    )


def test_singleton_derives_unique_logical_subsets(fake_world_one: None) -> None:
    with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )
        parallel_dims.build_mesh()
    mesh = parallel_dims.get_mesh("fsdp")
    token_embedding = _parameter(torch.ones(2, 2), mesh, (Shard(0),))
    moe_weight = _parameter(torch.ones(2, 2), mesh, (Shard(0),))
    model = _Model(token_embedding, moe_weight)
    token_embedding.grad = DTensor.from_local(
        torch.tensor([[1.0, -1.0], [0.0, float("nan")]]),
        mesh,
        (Shard(0),),
        shape=torch.Size((2, 2)),
        stride=(2, 1),
        run_check=False,
    )
    moe_weight.grad = DTensor.from_local(
        torch.tensor([[2.0, -2.0], [0.0, float("inf")]]),
        mesh,
        (Shard(0),),
        shape=torch.Size((2, 2)),
        stride=(2, 1),
        run_check=False,
    )

    statistics = WholeGradientStatistics(model=model, parallel_dims=parallel_dims)
    metrics = statistics.derive_metrics(
        statistics.collect(step=3),
        window_steps=2,
    )

    assert metrics["tensor_metrics/gradients/all.numel"] == 8
    assert metrics["tensor_metrics/gradients/all.nonfinite_count"] == 2
    assert metrics["tensor_metrics/gradients/all.zero_count"] == 2
    assert metrics["tensor_metrics/gradients/all.abs_mean"] == 1.0
    assert metrics["tensor_metrics/gradients/all.square_mean"] == pytest.approx(5 / 3)
    assert metrics["tensor_metrics/gradients/all.abs_max"] == 2.0
    assert metrics["tensor_metrics/gradients/all.observation_count"] == 1
    assert metrics["tensor_metrics/gradients/all.window_steps"] == 2
    assert metrics["tensor_metrics/gradients/token_embedding.numel"] == 4
    assert metrics["tensor_metrics/gradients/moe.numel"] == 4


def test_absent_gradient_is_a_local_error(fake_world_one: None) -> None:
    with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        )
        parallel_dims.build_mesh()
    mesh = parallel_dims.get_mesh("fsdp")
    token_embedding = _parameter(torch.ones(2, 2), mesh, (Shard(0),))
    moe_weight = _parameter(torch.ones(2, 2), mesh, (Shard(0),))
    token_embedding.grad = DTensor.from_local(
        torch.ones(2, 2),
        mesh,
        (Shard(0),),
        shape=torch.Size((2, 2)),
        stride=(2, 1),
        run_check=False,
    )
    statistics = WholeGradientStatistics(
        model=_Model(token_embedding, moe_weight),
        parallel_dims=parallel_dims,
    )

    snapshot = statistics.collect(step=1)
    assert snapshot.local_error is not None
    assert "layers.0.moe.weight" in str(snapshot.local_error)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestWholeGradientFourRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_dense_replica_and_sparse_owners_form_one_logical_view(self) -> None:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=4,
            pp=1,
            ep=2,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        dense_mesh = parallel_dims.get_mesh(["fsdp", "tp"])
        sparse_mesh = parallel_dims.get_mesh(["efsdp", "ep"])
        token_embedding = _parameter(
            torch.full((2, 2), 10.0, device=self.device_type),
            dense_mesh,
            (Shard(0), Replicate()),
        )
        moe_weight = _parameter(
            torch.full(
                (1, 4),
                float(self.rank + 1),
                device=self.device_type,
            ),
            sparse_mesh,
            (Shard(0), Shard(0)),
            shape=torch.Size((4, 4)),
        )
        token_embedding.grad = DTensor.from_local(
            token_embedding.to_local().clone(),
            dense_mesh,
            (Shard(0), Replicate()),
            shape=token_embedding.shape,
            stride=token_embedding.stride(),
            run_check=False,
        )
        moe_weight.grad = DTensor.from_local(
            moe_weight.to_local().clone(),
            sparse_mesh,
            (Shard(0), Shard(0)),
            shape=moe_weight.shape,
            stride=moe_weight.stride(),
            run_check=False,
        )

        statistics = WholeGradientStatistics(
            model=_Model(token_embedding, moe_weight),
            parallel_dims=parallel_dims,
        )
        metrics = statistics.derive_metrics(
            statistics.collect(step=1),
            window_steps=1,
        )

        self.assertEqual(metrics["tensor_metrics/gradients/all.numel"], 20)
        self.assertEqual(metrics["tensor_metrics/gradients/all.abs_mean"], 4.0)
        self.assertEqual(metrics["tensor_metrics/gradients/all.square_mean"], 26.0)
        self.assertEqual(metrics["tensor_metrics/gradients/all.abs_max"], 10.0)
        self.assertEqual(metrics["tensor_metrics/gradients/token_embedding.numel"], 4)
        self.assertEqual(
            metrics["tensor_metrics/gradients/token_embedding.abs_mean"], 10.0
        )
        self.assertEqual(metrics["tensor_metrics/gradients/moe.numel"], 16)
        self.assertEqual(metrics["tensor_metrics/gradients/moe.abs_mean"], 2.5)
        self.assertEqual(metrics["tensor_metrics/gradients/moe.abs_max"], 4.0)
