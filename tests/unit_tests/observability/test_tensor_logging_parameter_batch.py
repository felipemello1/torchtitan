# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.testing._internal.distributed.fake_pg  # noqa: F401
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.parameter_batch import (
    ParameterStatisticsBatch,
)
from torchtitan.observability.tensor_logging.sites import TensorMetricSite


class _Projection(nn.Module):
    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__()
        self.weight = weight


class _Attention(nn.Module):
    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__()
        self.wo = _Projection(weight)


class _Layer(nn.Module):
    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__()
        self.attention = _Attention(weight)


class _Model(nn.Module):
    def __init__(self, *weights: nn.Parameter) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(weight) for weight in weights])


class _DistributedAttention(nn.Module):
    def __init__(self, device_type: str) -> None:
        super().__init__()
        self.wo = nn.Linear(8, 8, bias=False, device=device_type)


class _DistributedLayer(nn.Module):
    def __init__(self, device_type: str) -> None:
        super().__init__()
        self.attention = _DistributedAttention(device_type)


class _DistributedModel(nn.Module):
    def __init__(self, device_type: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_DistributedLayer(device_type)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers[0].attention.wo(value)


@pytest.fixture
def fake_world_one() -> Iterator[None]:
    dist.init_process_group("fake", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _build_batch() -> tuple[ParameterStatisticsBatch, nn.Parameter]:
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
    local = torch.tensor([[0.0, -2.0], [float("nan"), float("inf")]])
    weight = nn.Parameter(
        DTensor.from_local(
            local,
            mesh,
            (Shard(0),),
            shape=local.shape,
            stride=local.stride(),
            run_check=False,
        )
    )
    batch = ParameterStatisticsBatch(
        model=_Model(weight),
        parallel_dims=parallel_dims,
        layer_ids=(0,),
        sites=(
            TensorMetricSite.ATTENTION_OUTPUT_WEIGHT,
            TensorMetricSite.ATTENTION_OUTPUT_WEIGHT_GRAD,
        ),
    )
    return batch, weight


def test_singleton_batch_derives_weight_and_completed_gradient(
    fake_world_one: None,
) -> None:
    batch, weight = _build_batch()
    local_grad = torch.ones(2, 2)
    weight.grad = DTensor.from_local(
        local_grad,
        weight.device_mesh,
        weight.placements,
        shape=weight.shape,
        stride=weight.stride(),
        run_check=False,
    )

    snapshot = batch.collect(step=3)
    metrics = batch.derive_metrics(
        snapshot,
        expected_contributors=1,
        window_steps=2,
    )

    weight_prefix = "tensor_metrics/layers.0.attention.wo.weight.w"
    grad_prefix = "tensor_metrics/layers.0.attention.wo.weight.dw_preclip"
    assert metrics[f"{weight_prefix}.numel"] == 4
    assert metrics[f"{weight_prefix}.nonfinite_count"] == 2
    assert metrics[f"{weight_prefix}.abs_mean"] == 1.0
    assert metrics[f"{weight_prefix}.observation_count"] == 1
    assert metrics[f"{weight_prefix}.window_steps"] == 2
    assert metrics[f"{grad_prefix}.numel"] == 4
    assert metrics[f"{grad_prefix}.rms"] == 1.0


def test_all_absent_optional_gradient_is_omitted(fake_world_one: None) -> None:
    batch, _ = _build_batch()

    snapshot = batch.collect(step=1)
    assert snapshot.local_error is None
    metrics = batch.derive_metrics(
        snapshot,
        expected_contributors=1,
        window_steps=1,
    )

    assert any(".w." in key for key in metrics)
    assert not any(".dw_preclip." in key for key in metrics)


def test_parameter_rows_share_one_packed_reduction(fake_world_one: None) -> None:
    batch, _ = _build_batch()
    assert batch._reduction_meshes == ()

    with patch(
        "torchtitan.observability.tensor_logging.parameter_batch.reduce_finite_statistics",
        side_effect=lambda statistics, owner_meshes: statistics,
    ) as reduce_batch:
        snapshot = batch.collect(step=1)

    assert snapshot.statistics.counts.shape == (2, 4)
    assert snapshot.statistics.sums.shape == (2, 2)
    assert snapshot.statistics.abs_max.shape == (2, 1)
    reduce_batch.assert_called_once()


def test_two_layers_share_one_packed_reduction(fake_world_one: None) -> None:
    batch, weight = _build_batch()
    second_weight = nn.Parameter(
        DTensor.from_local(
            torch.ones_like(weight.to_local()),
            weight.device_mesh,
            weight.placements,
            shape=weight.shape,
            stride=weight.stride(),
            run_check=False,
        )
    )
    batch = ParameterStatisticsBatch(
        model=_Model(weight, second_weight),
        parallel_dims=batch._parallel_dims,
        layer_ids=(0, 1),
        sites=(
            TensorMetricSite.ATTENTION_OUTPUT_WEIGHT,
            TensorMetricSite.ATTENTION_OUTPUT_WEIGHT_GRAD,
        ),
    )

    with patch(
        "torchtitan.observability.tensor_logging.parameter_batch.reduce_finite_statistics",
        side_effect=lambda statistics, owner_meshes: statistics,
    ) as reduce_batch:
        snapshot = batch.collect(step=1)

    assert snapshot.statistics.counts.shape == (4, 4)
    assert snapshot.statistics.sums.shape == (4, 2)
    assert snapshot.statistics.abs_max.shape == (4, 1)
    reduce_batch.assert_called_once()


def test_parameter_dimensions_must_divide_owner_degrees() -> None:
    parallel_dims = Mock(dp_shard=2, tp=1)
    weight = nn.Parameter(torch.ones(3, 2))

    with pytest.raises(ValueError, match="divisible by dp_shard"):
        ParameterStatisticsBatch(
            model=_Model(weight),
            parallel_dims=parallel_dims,
            layer_ids=(0,),
            sites=(TensorMetricSite.ATTENTION_OUTPUT_WEIGHT,),
        )


def test_writer_derivation_uses_two_packed_host_copies(fake_world_one: None) -> None:
    batch, _ = _build_batch()
    snapshot = batch.collect(step=1)
    host_counts = snapshot.statistics.counts
    host_floats = torch.cat(
        (snapshot.statistics.sums, snapshot.statistics.abs_max), dim=1
    )

    with patch.object(
        torch.Tensor,
        "cpu",
        side_effect=(host_counts, host_floats),
    ) as copy_to_host:
        batch.derive_metrics(
            snapshot,
            expected_contributors=1,
            window_steps=1,
        )

    assert copy_to_host.call_count == 2


def _build_distributed_batch(
    *,
    world_size: int,
    dp_shard: int,
    tp: int,
    device_type: str,
) -> tuple[ParameterStatisticsBatch, _DistributedModel]:
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=dp_shard,
        cp=1,
        tp=tp,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()
    model = _DistributedModel(device_type)
    if tp > 1:
        parallelize_module(
            model.layers[0].attention.wo,
            parallel_dims.get_mesh("tp"),
            RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
        )
    fully_shard(model, mesh=parallel_dims.get_mesh("fsdp"))
    batch = ParameterStatisticsBatch(
        model=model,
        parallel_dims=parallel_dims,
        layer_ids=(0,),
        sites=(
            TensorMetricSite.ATTENTION_OUTPUT_WEIGHT,
            TensorMetricSite.ATTENTION_OUTPUT_WEIGHT_GRAD,
        ),
    )
    return batch, model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestParameterBatchTwoRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _run_parameter_and_gradient_case(
        self,
        dp_shard: int,
        tp: int,
    ) -> None:
        batch, model = _build_distributed_batch(
            world_size=self.world_size,
            dp_shard=dp_shard,
            tp=tp,
            device_type=self.device_type,
        )
        expected_mesh_name = "fsdp" if dp_shard == 2 else "tp"
        self.assertEqual(len(batch._reduction_meshes), 1)
        self.assertIs(
            batch._reduction_meshes[0],
            batch._parallel_dims.get_mesh(expected_mesh_name),
        )
        parameter = model.layers[0].attention.wo.weight
        assert isinstance(parameter, DTensor)
        with torch.no_grad():
            parameter.to_local().fill_(dist.get_rank() + 1)
        model(torch.ones(2, 8, device=self.device_type)).sum().backward()
        assert isinstance(parameter.grad, DTensor)
        parameter.grad.to_local().fill_(dist.get_rank() + 1)

        metrics = batch.derive_metrics(
            batch.collect(step=1),
            expected_contributors=self.world_size,
            window_steps=1,
        )

        for suffix in ("w", "dw_preclip"):
            prefix = f"tensor_metrics/layers.0.attention.wo.weight.{suffix}"
            self.assertEqual(metrics[f"{prefix}.numel"], 64)
            self.assertEqual(metrics[f"{prefix}.abs_mean"], 1.5)
            self.assertEqual(metrics[f"{prefix}.square_mean"], 2.5)
            self.assertEqual(metrics[f"{prefix}.abs_max"], 2.0)

        if dp_shard == 2:
            if dist.get_rank() == 1:
                parameter.grad = None
            subset_snapshot = batch.collect(step=2)
            if dist.get_rank() == 0:
                with self.assertRaisesRegex(RuntimeError, "present on 1 of 2"):
                    batch.derive_metrics(
                        subset_snapshot,
                        expected_contributors=self.world_size,
                        window_steps=1,
                    )

    @with_comms
    def test_fully_sharded_parameter_and_gradient(self) -> None:
        self._run_parameter_and_gradient_case(dp_shard=2, tp=1)

    @with_comms
    def test_tensor_parallel_parameter_and_gradient(self) -> None:
        self._run_parameter_and_gradient_case(dp_shard=1, tp=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestParameterBatchFourRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_fsdp_shard_by_tp_batch(self) -> None:
        batch, model = _build_distributed_batch(
            world_size=self.world_size,
            dp_shard=2,
            tp=2,
            device_type=self.device_type,
        )
        self.assertEqual(len(batch._reduction_meshes), 1)
        self.assertIs(
            batch._reduction_meshes[0],
            batch._parallel_dims.world_mesh,
        )
        parameter = model.layers[0].attention.wo.weight
        assert isinstance(parameter, DTensor)
        with torch.no_grad():
            parameter.to_local().fill_(dist.get_rank() + 1)
        model(torch.ones(2, 8, device=self.device_type)).sum().backward()
        assert isinstance(parameter.grad, DTensor)
        parameter.grad.to_local().fill_(dist.get_rank() + 1)

        metrics = batch.derive_metrics(
            batch.collect(step=1),
            expected_contributors=self.world_size,
            window_steps=1,
        )

        for suffix in ("w", "dw_preclip"):
            prefix = f"tensor_metrics/layers.0.attention.wo.weight.{suffix}"
            self.assertEqual(metrics[f"{prefix}.numel"], 64)
            self.assertEqual(metrics[f"{prefix}.abs_mean"], 2.5)
            self.assertEqual(metrics[f"{prefix}.square_mean"], 7.5)
            self.assertEqual(metrics[f"{prefix}.abs_max"], 4.0)
