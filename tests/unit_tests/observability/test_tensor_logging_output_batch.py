# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from typing import Any, cast
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
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.llama3.config_registry import llama3_debugmodel
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.sharding import set_llama3_sharding_config
from torchtitan.observability.tensor_logging.families import TensorMetricFamily
from torchtitan.observability.tensor_logging.output_batch import OutputStatisticsBatch


class _Layer(nn.Module):
    def __init__(
        self,
        attention: GQAttention,
        feed_forward: FeedForward,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward


class _Model(nn.Module):
    def __init__(
        self,
        attention: GQAttention,
        feed_forward: FeedForward,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(attention, feed_forward)])

    @property
    def layer(self) -> _Layer:
        return cast(_Layer, self.layers[0])


@pytest.fixture
def fake_world_one() -> Iterator[None]:
    dist.init_process_group("fake", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _build_model(*, enable_sp: bool = False) -> _Model:
    config = llama3_debugmodel()
    assert config.model_spec is not None
    model_config = cast(Llama3Model.Config, config.model_spec.model)
    set_llama3_sharding_config(model_config, enable_sp=enable_sp)
    layer_config = model_config.layers[0]
    assert layer_config.feed_forward is not None
    attention = layer_config.attention.build()
    feed_forward = layer_config.feed_forward.build()
    cast(Any, attention).forward = lambda value, masks, positions=None: value * 2
    cast(Any, feed_forward).forward = lambda value: value * 3
    return _Model(attention, feed_forward)


def _build_batch(
    model: _Model,
    families: tuple[TensorMetricFamily, ...],
) -> OutputStatisticsBatch:
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
    return OutputStatisticsBatch(
        model=model,
        parallel_dims=parallel_dims,
        layer_ids=(0,),
        families=families,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    ("families", "expects_output", "expects_cotangent"),
    [
        ((TensorMetricFamily.BOUNDARY_OUTPUT,), True, False),
        ((TensorMetricFamily.BOUNDARY_OUTPUT_COTANGENT,), False, True),
        (
            (
                TensorMetricFamily.BOUNDARY_OUTPUT,
                TensorMetricFamily.BOUNDARY_OUTPUT_COTANGENT,
            ),
            True,
            True,
        ),
    ],
)
def test_singleton_output_families_are_independently_selectable(
    fake_world_one: None,
    families: tuple[TensorMetricFamily, ...],
    expects_output: bool,
    expects_cotangent: bool,
) -> None:
    model = _build_model()
    batch = _build_batch(model, families)
    value = torch.tensor([[[1.0, -2.0], [0.0, 4.0]]], requires_grad=True)

    batch.begin_step(should_log=True)
    attention_output = model.layer.attention(value, None, None)
    feed_forward_output = model.layer.feed_forward(value)
    assert len(batch._gradient_hook_handles) == (2 if expects_cotangent else 0)
    (attention_output.sum() + feed_forward_output.sum()).backward()
    snapshot = batch.collect()
    metrics = batch.derive_metrics(snapshot, window_steps=3)

    attention_x = "tensor_metrics/layers.0.attention.output.value"
    attention_dx = "tensor_metrics/layers.0.attention.output.cotangent"
    feed_forward_x = "tensor_metrics/layers.0.feed_forward.output.value"
    feed_forward_dx = "tensor_metrics/layers.0.feed_forward.output.cotangent"
    assert any(key.startswith(attention_x) for key in metrics) is expects_output
    assert any(key.startswith(feed_forward_x) for key in metrics) is expects_output
    assert any(key.startswith(attention_dx) for key in metrics) is expects_cotangent
    assert any(key.startswith(feed_forward_dx) for key in metrics) is expects_cotangent
    if expects_output:
        assert metrics[f"{attention_x}.abs_mean"] == 3.5
        assert metrics[f"{feed_forward_x}.abs_mean"] == 5.25
        assert metrics[f"{attention_x}.observation_count"] == 1
        assert metrics[f"{attention_x}.window_steps"] == 3
    if expects_cotangent:
        assert metrics[f"{attention_dx}.rms"] == 1.0
        assert metrics[f"{feed_forward_dx}.rms"] == 1.0
    assert not batch._gradient_hook_handles
    batch.close()


def test_skipped_step_registers_no_tensor_hooks(fake_world_one: None) -> None:
    model = _build_model()
    batch = _build_batch(
        model,
        (TensorMetricFamily.BOUNDARY_OUTPUT_COTANGENT,),
    )
    value = torch.ones(1, 2, 2, requires_grad=True)

    batch.begin_step(should_log=False)
    output = model.layer.attention(value, None, None)
    output.sum().backward()

    assert not batch._gradient_hook_handles
    assert torch.count_nonzero(batch._counts) == 0
    batch.close()


def test_hook_failure_is_latched_without_changing_backward(
    fake_world_one: None,
) -> None:
    model = _build_model()
    cast(
        Any, model.layer.attention
    ).forward = lambda value, masks, positions=None: value.double()
    cast(Any, model.layer.feed_forward).forward = lambda value: value.double()
    batch = _build_batch(
        model,
        (
            TensorMetricFamily.BOUNDARY_OUTPUT,
            TensorMetricFamily.BOUNDARY_OUTPUT_COTANGENT,
        ),
    )
    value = torch.ones(1, 2, 2, requires_grad=True)

    batch.begin_step(should_log=True)
    output = model.layer.attention(value, None, None)
    output.sum().backward()
    snapshot = batch.collect()

    assert value.grad is not None
    assert torch.equal(value.grad, torch.ones_like(value))
    assert snapshot.local_error is not None
    assert "unsupported dtype" in str(snapshot.local_error)
    batch.close()


def test_close_removes_module_hooks(fake_world_one: None) -> None:
    model = _build_model()
    batch = _build_batch(model, (TensorMetricFamily.BOUNDARY_OUTPUT,))

    assert model.layer.attention._forward_hooks
    assert model.layer.feed_forward._forward_hooks
    batch.close()
    assert not model.layer.attention._forward_hooks
    assert not model.layer.feed_forward._forward_hooks


def test_noncontiguous_output_is_accumulated_without_flatten_copy(
    fake_world_one: None,
) -> None:
    model = _build_model()
    cast(
        Any, model.layer.attention
    ).forward = lambda value, masks, positions=None: value.transpose(1, 2)
    cast(Any, model.layer.feed_forward).forward = lambda value: value.transpose(1, 2)
    batch = _build_batch(model, (TensorMetricFamily.BOUNDARY_OUTPUT,))
    value = torch.ones(1, 2, 3)

    batch.begin_step(should_log=True)
    model.layer.attention(value, None, None)
    model.layer.feed_forward(value)
    snapshot = batch.collect()

    assert snapshot.local_error is None
    metrics = batch.derive_metrics(snapshot, window_steps=1)
    prefix = "tensor_metrics/layers.0.attention.output.value"
    assert metrics[f"{prefix}.numel"] == 6
    assert metrics[f"{prefix}.abs_mean"] == 1.0
    batch.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestOutputStatisticsFourRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_mixed_dp_tp_replica_and_shard_populations(self) -> None:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        tp_mesh = parallel_dims.get_mesh("tp")
        tp_rank = tp_mesh.get_local_rank()
        dp_rank = parallel_dims.get_mesh("loss").get_local_rank()
        device = torch.device(self.device_type, self.rank)

        for enable_sp in (False, True):
            model = _build_model(enable_sp=enable_sp)
            batch = OutputStatisticsBatch(
                model=model,
                parallel_dims=parallel_dims,
                layer_ids=(0,),
                families=(
                    TensorMetricFamily.BOUNDARY_OUTPUT,
                    TensorMetricFamily.BOUNDARY_OUTPUT_COTANGENT,
                ),
                device=device,
            )
            if not enable_sp and tp_rank == 0:
                assert batch._reduction_meshes == (parallel_dims.get_mesh("loss"),)
                assert batch._reduction_meshes != (tp_mesh,)

            local_sequence = 2 if enable_sp else 4
            placements = (Shard(1),) if enable_sp else (Replicate(),)
            global_shape = torch.Size((1, 4, 2))
            global_stride = (8, 2, 1)
            attention_fill = 10 * dp_rank + tp_rank + 1 if enable_sp else dp_rank + 1
            feed_forward_fill = attention_fill + 20

            def distributed_output(
                fill_value: int,
                local_sequence: int = local_sequence,
                placements: tuple[Replicate | Shard, ...] = placements,
                global_shape: torch.Size = global_shape,
                global_stride: tuple[int, ...] = global_stride,
            ) -> DTensor:
                local = torch.full(
                    (1, local_sequence, 2),
                    fill_value,
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True,
                )
                return DTensor.from_local(
                    local,
                    tp_mesh,
                    placements,
                    shape=global_shape,
                    stride=global_stride,
                    run_check=False,
                )

            attention_value = distributed_output(attention_fill)
            feed_forward_value = distributed_output(feed_forward_fill)
            cast(
                Any, model.layer.attention
            ).forward = (
                lambda value, masks, positions=None, output=attention_value: output
            )
            cast(
                Any, model.layer.feed_forward
            ).forward = lambda value, output=feed_forward_value: output

            batch.begin_step(should_log=True)
            unused_input = torch.ones(1, 1, 1, device=device)
            attention_output = model.layer.attention(unused_input, None, None)
            feed_forward_output = model.layer.feed_forward(unused_input)
            (
                attention_output.to_local().sum() + feed_forward_output.to_local().sum()
            ).backward()
            snapshot = batch.collect()

            if self.rank == 0:
                metrics = batch.derive_metrics(snapshot, window_steps=1)
                attention_x = "tensor_metrics/layers.0.attention.output.value"
                attention_dx = "tensor_metrics/layers.0.attention.output.cotangent"
                assert metrics[f"{attention_x}.numel"] == 16
                assert metrics[f"{attention_x}.observation_count"] == 2
                assert metrics[f"{attention_dx}.numel"] == 16
                assert metrics[f"{attention_dx}.observation_count"] == 2
                assert metrics[f"{attention_dx}.rms"] == 1.0
                assert metrics[f"{attention_x}.abs_mean"] == (6.5 if enable_sp else 1.5)

            batch.close()
            dist.barrier()
