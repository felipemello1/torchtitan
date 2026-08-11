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

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.moe import MoE
from torchtitan.observability.tensor_logging.expert_bias import ExpertBiasRecorder


class _Block(nn.Module):
    def __init__(self, moe: MoE) -> None:
        super().__init__()
        self.moe_enabled = True
        self.moe = moe


class _Model(nn.Module):
    def __init__(self, moe: MoE) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _Block(moe)})


@pytest.fixture
def fake_world_one() -> Iterator[None]:
    dist.init_process_group("fake", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _build_model() -> tuple[_Model, MoE]:
    moe = object.__new__(MoE)
    nn.Module.__init__(moe)
    moe.load_balance_coeff = 0.1
    moe.register_buffer("expert_bias_E", torch.zeros(2, dtype=torch.float32))
    moe.register_buffer(
        "tokens_per_expert_E",
        torch.tensor([1.0, 3.0], dtype=torch.float32),
    )
    moe.dummy = nn.Parameter(torch.ones(()))
    return _Model(moe), moe


def test_records_after_source_bias_update(fake_world_one: None) -> None:
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
    model, moe = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    register_moe_load_balancing_hook(optimizer, [model], parallel_dims)  # type: ignore[arg-type]
    recorder = ExpertBiasRecorder(
        model=model,
        layer_ids=(0,),
        device=torch.device("cpu"),
    )
    recorder.bind_optimizer(optimizer)
    recorder.begin_step(should_log=True)
    moe.dummy.grad = torch.ones_like(moe.dummy)

    optimizer.step()
    metrics = recorder.derive_metrics(recorder.collect(), window_steps=3)

    prefix = "tensor_metrics/layers.0"
    assert metrics[
        f"{prefix}.experts.0.router_expert_bias_post_update"
    ] == pytest.approx(0.1)
    assert metrics[
        f"{prefix}.experts.1.router_expert_bias_post_update"
    ] == pytest.approx(-0.1)
    assert (
        metrics[f"{prefix}.moe.router_expert_bias_post_update.observation_count"] == 1
    )
    assert metrics[f"{prefix}.moe.router_expert_bias_post_update.window_steps"] == 3
    assert torch.equal(moe.tokens_per_expert_E, torch.zeros(2))
    recorder.close()


def test_missing_optimizer_hook_is_reported(fake_world_one: None) -> None:
    model, _ = _build_model()
    recorder = ExpertBiasRecorder(
        model=model,
        layer_ids=(0,),
        device=torch.device("cpu"),
    )
    recorder.begin_step(should_log=True)

    snapshot = recorder.collect()
    assert snapshot.local_error is not None
    assert "did not observe" in str(snapshot.local_error)
