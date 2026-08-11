# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from typing import cast
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

from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging import optimizer_statistics
from torchtitan.observability.tensor_logging.families import TensorMetricFamily
from torchtitan.observability.tensor_logging.optimizer_statistics import (
    AdamWStatisticsRecorder,
)


class _Layer(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        bias: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.weight = weight
        if bias is not None:
            self.bias = bias


class _Model(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        bias: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(weight, bias)])


@pytest.fixture
def fake_world_one() -> Iterator[None]:
    dist.init_process_group("fake", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _parallel_dims() -> ParallelDims:
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
    return parallel_dims


def _model(parallel_dims: ParallelDims) -> _Model:
    mesh = parallel_dims.get_mesh("fsdp")
    weight = nn.Parameter(
        DTensor.from_local(
            torch.tensor([1.0, -2.0]),
            mesh,
            (Shard(0),),
            shape=torch.Size((2,)),
            stride=(1,),
            run_check=False,
        )
    )
    return _Model(weight)


def _optimizer(model: _Model, *, optimizer_name: str = "AdamW") -> OptimizersContainer:
    return OptimizersContainer(
        OptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name=optimizer_name,
                    optimizer_kwargs={
                        "lr": 0.1,
                        "betas": (0.9, 0.95),
                        "eps": 1e-8,
                        "weight_decay": 0.2,
                    },
                )
            ],
        ),
        model_parts=[model],
    )


def test_public_adamw_equations_and_cosine(fake_world_one: None) -> None:
    parallel_dims = _parallel_dims()
    model = _model(parallel_dims)
    optimizer = _optimizer(model)
    recorder = AdamWStatisticsRecorder(
        model=model,
        parallel_dims=parallel_dims,
        layer_ids=(0,),
        families=(
            TensorMetricFamily.OPTIMIZER_DISTRIBUTION,
            TensorMetricFamily.MOMENTUM_GRADIENT_COSINE,
        ),
    )
    recorder.bind_optimizer(optimizer)
    layer = cast(_Layer, model.layers[0])
    weight = layer.weight
    weight_dtensor = cast(DTensor, weight)
    weight.grad = DTensor.from_local(
        torch.tensor([0.5, -0.25]),
        weight_dtensor.device_mesh,
        weight_dtensor.placements,
        shape=weight.shape,
        stride=weight.stride(),
        run_check=False,
    )
    old_weight = weight_dtensor.to_local().clone()
    recorder.begin_step(should_log=True)

    with patch.object(optimizer_statistics, "_MAX_CHUNK_ELEMENTS", 1):
        optimizer.step()
    snapshot = recorder.collect()
    assert snapshot.counts[0].shape == (5, 4)
    assert snapshot.sums[0].shape == (5, 3)
    assert snapshot.maxima[0].shape == (4, 1)
    metrics = recorder.derive_metrics(snapshot, window_steps=1)

    prefix = "tensor_metrics/layers.0.weight.optimizer"
    assert metrics[f"{prefix}.numerator.abs_mean"] == pytest.approx(0.0375)
    assert metrics[f"{prefix}.numerator.square_mean"] == pytest.approx(0.0015625)
    assert metrics[f"{prefix}.denominator.abs_mean"] == pytest.approx(0.375)
    assert metrics[f"{prefix}.preconditioned_gradient.abs_mean"] == pytest.approx(1)
    assert metrics[f"{prefix}.update_pre_apply.abs_mean"] == pytest.approx(0.1)
    assert metrics[f"{prefix}.momentum_gradient_cosine"] == pytest.approx(1)
    assert metrics[f"{prefix}.cosine.observation_count"] == 1
    assert metrics[f"{prefix}.cosine.window_steps"] == 1
    assert metrics[f"{prefix}.numerator.numel"] == 2
    expected_weight = old_weight * (1 - 0.1 * 0.2) + torch.tensor([-0.1, 0.1])
    torch.testing.assert_close(weight_dtensor.to_local(), expected_weight)
    recorder.close()


def test_zero_gradient_omits_undefined_cosine(fake_world_one: None) -> None:
    parallel_dims = _parallel_dims()
    model = _model(parallel_dims)
    optimizer = _optimizer(model)
    recorder = AdamWStatisticsRecorder(
        model=model,
        parallel_dims=parallel_dims,
        layer_ids=(0,),
        families=(TensorMetricFamily.MOMENTUM_GRADIENT_COSINE,),
    )
    recorder.bind_optimizer(optimizer)
    layer = cast(_Layer, model.layers[0])
    weight = layer.weight
    weight_dtensor = cast(DTensor, weight)
    weight.grad = DTensor.from_local(
        torch.zeros(2),
        weight_dtensor.device_mesh,
        weight_dtensor.placements,
        shape=weight.shape,
        stride=weight.stride(),
        run_check=False,
    )
    recorder.begin_step(should_log=True)

    with patch.object(optimizer_statistics, "_MAX_CHUNK_ELEMENTS", 1):
        optimizer.step()
    metrics = recorder.derive_metrics(recorder.collect(), window_steps=3)

    prefix = "tensor_metrics/layers.0.weight.optimizer"
    assert f"{prefix}.momentum_gradient_cosine" not in metrics
    assert metrics[f"{prefix}.cosine.observation_count"] == 1
    assert metrics[f"{prefix}.cosine.window_steps"] == 3
    recorder.close()


def test_post_load_hook_reads_live_adamw_group(fake_world_one: None) -> None:
    parallel_dims = _parallel_dims()
    model = _model(parallel_dims)
    optimizers = _optimizer(model)
    recorder = AdamWStatisticsRecorder(
        model=model,
        parallel_dims=parallel_dims,
        layer_ids=(0,),
        families=(TensorMetricFamily.OPTIMIZER_DISTRIBUTION,),
    )
    recorder.bind_optimizer(optimizers)
    optimizer = optimizers.optimizers[0]
    loaded_state = optimizer.state_dict()
    loaded_state["param_groups"][0]["lr"] = 0.03
    optimizer.load_state_dict(loaded_state)

    layer = cast(_Layer, model.layers[0])
    weight = cast(DTensor, layer.weight)
    layer.weight.grad = DTensor.from_local(
        torch.tensor([0.5, -0.25]),
        weight.device_mesh,
        weight.placements,
        shape=weight.shape,
        stride=weight.stride(),
        run_check=False,
    )
    old_weight = weight.to_local().clone()
    recorder.begin_step(should_log=True)

    optimizers.step()
    metrics = recorder.derive_metrics(recorder.collect(), window_steps=1)

    prefix = "tensor_metrics/layers.0.weight.optimizer"
    assert metrics[f"{prefix}.update_pre_apply.abs_mean"] == pytest.approx(0.03)
    expected_weight = old_weight * (1 - 0.03 * 0.2) + torch.tensor([-0.03, 0.03])
    torch.testing.assert_close(weight.to_local(), expected_weight)
    recorder.close()


def test_named_adapter_rejects_adam(fake_world_one: None) -> None:
    parallel_dims = _parallel_dims()
    model = _model(parallel_dims)
    recorder = AdamWStatisticsRecorder(
        model=model,
        parallel_dims=parallel_dims,
        layer_ids=(0,),
        families=(TensorMetricFamily.OPTIMIZER_DISTRIBUTION,),
    )

    with pytest.raises(ValueError, match="public AdamW"):
        recorder.bind_optimizer(_optimizer(model, optimizer_name="Adam"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestAdamWStatisticsFourRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_fused_adamw_reconstructs_tp_shards_and_excludes_replicas(self) -> None:
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
        mesh = parallel_dims.get_mesh(["fsdp", "tp"])
        fsdp_rank = parallel_dims.get_mesh("fsdp").get_local_rank()
        sharded_value = float(self.rank + 1)
        replicated_value = float(fsdp_rank + 1)
        weight = nn.Parameter(
            DTensor.from_local(
                torch.full((2, 2), 10.0, device=self.device_type),
                mesh,
                (Shard(0), Shard(1)),
                shape=torch.Size((4, 4)),
                stride=(4, 1),
                run_check=False,
            )
        )
        bias = nn.Parameter(
            DTensor.from_local(
                torch.full((2,), -2.0, device=self.device_type),
                mesh,
                (Shard(0), Replicate()),
                shape=torch.Size((4,)),
                stride=(1,),
                run_check=False,
            )
        )
        model = _Model(weight, bias)
        optimizer = OptimizersContainer(
            OptimizersContainer.Config(
                implementation="fused",
                param_groups=[
                    ParamGroupConfig(
                        pattern=r".*",
                        optimizer_name="AdamW",
                        optimizer_kwargs={
                            "lr": 0.1,
                            "betas": (0.9, 0.95),
                            "eps": 1e-8,
                            "weight_decay": 0.2,
                        },
                    )
                ],
            ),
            model_parts=[model],
        )
        weight.grad = DTensor.from_local(
            torch.full((2, 2), sharded_value, device=self.device_type),
            mesh,
            (Shard(0), Shard(1)),
            shape=weight.shape,
            stride=weight.stride(),
            run_check=False,
        )
        bias.grad = DTensor.from_local(
            torch.full((2,), replicated_value, device=self.device_type),
            mesh,
            (Shard(0), Replicate()),
            shape=bias.shape,
            stride=bias.stride(),
            run_check=False,
        )
        recorder = AdamWStatisticsRecorder(
            model=model,
            parallel_dims=parallel_dims,
            layer_ids=(0,),
            families=(
                TensorMetricFamily.OPTIMIZER_DISTRIBUTION,
                TensorMetricFamily.MOMENTUM_GRADIENT_COSINE,
            ),
        )
        recorder.bind_optimizer(optimizer)
        recorder.begin_step(should_log=True)
        old_weight = cast(DTensor, weight).to_local().clone()
        old_bias = cast(DTensor, bias).to_local().clone()

        optimizer.step()
        with (
            patch.object(
                optimizer_statistics,
                "reduce_sum",
                wraps=optimizer_statistics.reduce_sum,
            ) as reduce_sum,
            patch.object(
                optimizer_statistics,
                "reduce_max",
                wraps=optimizer_statistics.reduce_max,
            ) as reduce_max,
        ):
            snapshot = recorder.collect()
        # One int64 SUM, one FP32 SUM, and one FP32 MAX per owner cohort.
        self.assertEqual(reduce_sum.call_count, 4)
        self.assertEqual(reduce_max.call_count, 2)
        metrics = recorder.derive_metrics(snapshot, window_steps=1)

        weight_prefix = "tensor_metrics/layers.0.weight.optimizer"
        self.assertEqual(metrics[f"{weight_prefix}.numerator.numel"], 16)
        self.assertAlmostEqual(
            metrics[f"{weight_prefix}.numerator.abs_mean"], 0.25, places=6
        )
        self.assertAlmostEqual(
            metrics[f"{weight_prefix}.denominator.abs_mean"], 2.5, places=6
        )
        self.assertAlmostEqual(
            metrics[f"{weight_prefix}.preconditioned_gradient.abs_mean"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            metrics[f"{weight_prefix}.update_pre_apply.abs_mean"], 0.1, places=6
        )
        self.assertAlmostEqual(
            metrics[f"{weight_prefix}.momentum_gradient_cosine"], 1.0, places=6
        )
        bias_prefix = "tensor_metrics/layers.0.bias.optimizer"
        self.assertEqual(metrics[f"{bias_prefix}.numerator.numel"], 4)
        self.assertAlmostEqual(
            metrics[f"{bias_prefix}.numerator.abs_mean"], 0.15, places=6
        )
        self.assertAlmostEqual(
            metrics[f"{bias_prefix}.denominator.abs_mean"], 1.5, places=6
        )
        torch.testing.assert_close(
            cast(DTensor, weight).to_local(),
            old_weight * 0.98 - 0.1,
        )
        torch.testing.assert_close(
            cast(DTensor, bias).to_local(),
            old_bias * 0.98 - 0.1,
        )
        recorder.close()
