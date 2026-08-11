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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging import statistics
from torchtitan.observability.tensor_logging.parameter_ownership import (
    resolve_parameter_owner_meshes,
)
from torchtitan.observability.tensor_logging.statistics import (
    derive_finite_statistics,
    finite_statistics,
    FiniteStatistics,
    reduce_finite_statistics,
)


def test_empty_owner_reduction_returns_an_owned_snapshot() -> None:
    statistics = finite_statistics(torch.tensor([0.0, -2.0]))

    reduced = reduce_finite_statistics(statistics, ())

    assert reduced.counts.data_ptr() != statistics.counts.data_ptr()
    assert reduced.sums.data_ptr() != statistics.sums.data_ptr()
    assert reduced.abs_max.data_ptr() != statistics.abs_max.data_ptr()
    statistics.counts.zero_()
    statistics.sums.zero_()
    statistics.abs_max.zero_()
    assert reduced.counts.tolist() == [2, 0, 1]
    assert reduced.sums.tolist() == [2.0, 4.0]
    assert reduced.abs_max.tolist() == [2.0]


def test_chunked_statistics_match_single_chunk() -> None:
    value = torch.tensor([0.0, -2.0, float("nan"), float("inf"), 3.0])

    single_chunk = finite_statistics(value)
    chunked = finite_statistics(value, max_chunk_elements=2)

    assert torch.equal(chunked.counts, single_chunk.counts)
    assert torch.equal(chunked.sums, single_chunk.sums)
    assert torch.equal(chunked.abs_max, single_chunk.abs_max)


def test_chunked_statistics_match_noncontiguous_storage() -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)

    single_chunk = finite_statistics(value)
    chunked = finite_statistics(value, max_chunk_elements=5)

    assert torch.equal(chunked.counts, single_chunk.counts)
    assert torch.equal(chunked.sums, single_chunk.sums)
    assert torch.equal(chunked.abs_max, single_chunk.abs_max)


@pytest.mark.parametrize(
    ("sums", "missing"),
    [
        ((float("inf"), 4.0), {"abs_mean"}),
        ((2.0, float("inf")), {"square_mean", "rms"}),
    ],
)
def test_derived_statistics_omit_overflowed_accumulators(
    sums: tuple[float, float],
    missing: set[str],
) -> None:
    statistics = FiniteStatistics(
        counts=torch.tensor([2, 0, 0], dtype=torch.int64),
        sums=torch.tensor(sums, dtype=torch.float32),
        abs_max=torch.tensor([3.0], dtype=torch.float32),
    )

    derived = derive_finite_statistics(statistics)

    assert missing.isdisjoint(derived)
    assert derived["numel"] == 2
    assert derived["abs_max"] == 3.0


def test_finite_fp32_square_overflow_omits_square_metrics() -> None:
    statistics = finite_statistics(torch.tensor([3e19], dtype=torch.float32))

    derived = derive_finite_statistics(statistics)

    assert torch.isinf(statistics.sums[1])
    assert "square_mean" not in derived
    assert "rms" not in derived
    assert derived["abs_mean"] == pytest.approx(3e19)


def test_finite_stat_reduction_uses_axis_then_lane_order(monkeypatch) -> None:
    calls = []
    first_mesh = cast(DeviceMesh, object())
    second_mesh = cast(DeviceMesh, object())

    def record_sum(value, mesh):
        calls.append(("sum", mesh))
        return value

    def record_max(value, mesh):
        calls.append(("max", mesh))
        return value

    monkeypatch.setattr(statistics, "reduce_sum", record_sum)
    monkeypatch.setattr(statistics, "reduce_max", record_max)
    finite_stats = finite_statistics(torch.tensor([1.0]))

    reduce_finite_statistics(
        finite_stats,
        (first_mesh, second_mesh),
    )

    assert calls == [
        ("sum", first_mesh),
        ("sum", first_mesh),
        ("max", first_mesh),
        ("sum", second_mesh),
        ("sum", second_mesh),
        ("max", second_mesh),
    ]


def _build_rowwise_linear(
    parallel_dims: ParallelDims,
    device_type: str,
) -> nn.Linear:
    parallel_dims.build_mesh()
    linear = nn.Linear(8, 8, bias=False, device=device_type)
    if parallel_dims.tp > 1:
        parallelize_module(
            linear,
            parallel_dims.get_mesh("tp"),
            RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
        )
    data_parallel_mesh_axis_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate > 1 else ["fsdp"]
    )
    fully_shard(
        linear,
        mesh=parallel_dims.get_mesh(data_parallel_mesh_axis_names),
    )
    return linear


def _fill_local(value: DTensor, fill_value: float) -> None:
    with torch.no_grad():
        value.to_local().fill_(fill_value)


def _to_cpu(statistics: FiniteStatistics) -> FiniteStatistics:
    return FiniteStatistics(
        counts=statistics.counts.cpu(),
        sums=statistics.sums.cpu(),
        abs_max=statistics.abs_max.cpu(),
    )


def _assert_two_shard_statistics(statistics: FiniteStatistics) -> None:
    assert statistics.counts.tolist() == [64, 0, 0]
    assert statistics.sums.tolist() == [128.0, 320.0]
    assert statistics.abs_max.tolist() == [3.0]


@pytest.fixture
def fake_world_four() -> Iterator[None]:
    dist.init_process_group("fake", rank=0, world_size=4)
    yield
    dist.destroy_process_group()


def test_fake_pg_resolves_and_rejects_parameter_signatures(
    fake_world_four: None,
) -> None:
    with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=4,
        )
        parallel_dims.build_mesh()
    expected_mesh = parallel_dims.get_mesh(["fsdp", "tp"])
    parameter = DTensor.from_local(
        torch.ones(4, 4),
        expected_mesh,
        (Shard(0), Shard(1)),
        shape=torch.Size((8, 8)),
        stride=(8, 1),
        run_check=False,
    )

    owner_meshes = resolve_parameter_owner_meshes(
        parameter,
        parallel_dims=parallel_dims,
    )
    assert owner_meshes == (
        parallel_dims.get_mesh("fsdp"),
        parallel_dims.get_mesh("tp"),
    )

    replicated_parameter = DTensor.from_local(
        torch.ones(4, 8),
        expected_mesh,
        (Shard(0), Replicate()),
        shape=torch.Size((8, 8)),
        stride=(8, 1),
        run_check=False,
    )
    assert resolve_parameter_owner_meshes(
        replicated_parameter,
        parallel_dims=parallel_dims,
    ) == (parallel_dims.get_mesh("fsdp"),)

    colwise_parameter = DTensor.from_local(
        torch.ones(2, 8),
        expected_mesh,
        (Shard(0), Shard(0)),
        shape=torch.Size((8, 8)),
        stride=(8, 1),
        run_check=False,
    )
    assert resolve_parameter_owner_meshes(
        colwise_parameter,
        parallel_dims=parallel_dims,
    ) == (
        parallel_dims.get_mesh("fsdp"),
        parallel_dims.get_mesh("tp"),
    )

    strided_parameter = DTensor.from_local(
        torch.ones(2, 8),
        expected_mesh,
        (_StridedShard(0, split_factor=2), Shard(0)),
        shape=torch.Size((8, 8)),
        stride=(8, 1),
        run_check=False,
    )
    assert resolve_parameter_owner_meshes(
        strided_parameter,
        parallel_dims=parallel_dims,
    ) == (
        parallel_dims.get_mesh("fsdp"),
        parallel_dims.get_mesh("tp"),
    )

    with pytest.raises(ValueError, match="require a DTensor"):
        resolve_parameter_owner_meshes(
            parameter.to_local(),
            parallel_dims=parallel_dims,
        )

    partial = DTensor.from_local(
        torch.ones(8, 4),
        expected_mesh,
        (Partial(), Shard(1)),
        shape=torch.Size((8, 8)),
        stride=(8, 1),
        run_check=False,
    )
    with pytest.raises(ValueError, match="do not accept Partial"):
        resolve_parameter_owner_meshes(
            partial,
            parallel_dims=parallel_dims,
        )

    alternate_fsdp_shard = DTensor.from_local(
        torch.ones(8, 2),
        expected_mesh,
        (Shard(1), Shard(1)),
        shape=torch.Size((8, 8)),
        stride=(8, 1),
        run_check=False,
    )
    assert resolve_parameter_owner_meshes(
        alternate_fsdp_shard,
        parallel_dims=parallel_dims,
    ) == (
        parallel_dims.get_mesh("fsdp"),
        parallel_dims.get_mesh("tp"),
    )

    unsharded_storage = DTensor.from_local(
        torch.ones(8, 4),
        expected_mesh,
        (Replicate(), Shard(1)),
        shape=torch.Size((8, 8)),
        stride=(8, 1),
        run_check=False,
    )
    with pytest.raises(ValueError, match="a shard on the fsdp axis"):
        resolve_parameter_owner_meshes(
            unsharded_storage,
            parallel_dims=parallel_dims,
        )

    unnamed_mesh = DeviceMesh("cpu", expected_mesh.mesh)
    unnamed_parameter = DTensor.from_local(
        torch.ones(4, 4),
        unnamed_mesh,
        (Shard(0), Shard(1)),
        shape=torch.Size((8, 8)),
        stride=(8, 1),
        run_check=False,
    )
    with pytest.raises(ValueError, match="expected dense mesh axes"):
        resolve_parameter_owner_meshes(
            unnamed_parameter,
            parallel_dims=parallel_dims,
        )

    for backend in ("full_dtensor", "spmd_types"):
        unsupported_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=4,
            spmd_backend=backend,
        )
        with pytest.raises(ValueError, match="spmd_backend='default'"):
            resolve_parameter_owner_meshes(
                parameter,
                parallel_dims=unsupported_dims,
            )

    permuted_mesh = DeviceMesh(
        "cpu",
        expected_mesh.mesh.transpose(0, 1).contiguous(),
        mesh_dim_names=("fsdp", "tp"),
    )
    permuted_parameter = DTensor.from_local(
        torch.ones(4, 4),
        permuted_mesh,
        (Shard(0), Shard(1)),
        shape=torch.Size((8, 8)),
        stride=(8, 1),
        run_check=False,
    )
    with pytest.raises(ValueError, match="ParallelDims rank grid"):
        resolve_parameter_owner_meshes(
            permuted_parameter,
            parallel_dims=parallel_dims,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestRowwiseParameterSingleton(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 1

    @with_comms
    def test_singleton_has_no_owner_collective(self) -> None:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        linear = _build_rowwise_linear(parallel_dims, self.device_type)
        parameter = linear.weight
        assert isinstance(parameter, DTensor)
        _fill_local(parameter, 2.0)

        owner_meshes = resolve_parameter_owner_meshes(
            parameter,
            parallel_dims=parallel_dims,
        )
        statistics = reduce_finite_statistics(
            finite_statistics(parameter.to_local()),
            owner_meshes,
        )
        host_statistics = _to_cpu(statistics)

        self.assertEqual(owner_meshes, ())
        self.assertEqual(host_statistics.counts.tolist(), [64, 0, 0])
        self.assertEqual(host_statistics.sums.tolist(), [128.0, 256.0])
        self.assertEqual(host_statistics.abs_max.tolist(), [2.0])
        self.assertEqual(int(host_statistics.counts[0]), parameter.numel())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestRowwiseParameterTwoRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _parallel_dims(self, *, dp_shard: int, cp: int, tp: int) -> ParallelDims:
        return ParallelDims(
            dp_replicate=1,
            dp_shard=dp_shard,
            cp=cp,
            tp=tp,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )

    def _assert_one_owner_axis(
        self,
        parallel_dims: ParallelDims,
        expected_mesh_axis_name: str,
    ) -> None:
        linear = _build_rowwise_linear(parallel_dims, self.device_type)
        parameter = linear.weight
        assert isinstance(parameter, DTensor)
        expected_mesh = parallel_dims.get_mesh(expected_mesh_axis_name)
        _fill_local(parameter, 1.0 if expected_mesh.get_local_rank() == 0 else 3.0)

        owner_meshes = resolve_parameter_owner_meshes(
            parameter,
            parallel_dims=parallel_dims,
        )
        statistics = reduce_finite_statistics(
            finite_statistics(parameter.to_local()),
            owner_meshes,
        )
        host_statistics = _to_cpu(statistics)

        self.assertEqual(len(owner_meshes), 1)
        self.assertIs(owner_meshes[0], expected_mesh)
        _assert_two_shard_statistics(host_statistics)
        self.assertEqual(int(host_statistics.counts[0]), parameter.numel())

    @with_comms
    def test_tp_owns_rowwise_parameter_storage(self) -> None:
        self._assert_one_owner_axis(
            self._parallel_dims(dp_shard=1, cp=1, tp=2),
            "tp",
        )

    @with_comms
    def test_fsdp_owns_parameter_storage(self) -> None:
        self._assert_one_owner_axis(
            self._parallel_dims(dp_shard=2, cp=1, tp=1),
            "fsdp",
        )

    @with_comms
    def test_cp_is_folded_once_into_fsdp(self) -> None:
        self._assert_one_owner_axis(
            self._parallel_dims(dp_shard=1, cp=2, tp=1),
            "fsdp",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestRowwiseParameterFourRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_compound_reduction_uses_fsdp_then_tp(self) -> None:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        linear = _build_rowwise_linear(parallel_dims, self.device_type)
        parameter = linear.weight
        assert isinstance(parameter, DTensor)
        fsdp_mesh = parallel_dims.get_mesh("fsdp")
        tp_mesh = parallel_dims.get_mesh("tp")
        fill_value = (
            100 * fsdp_mesh.get_local_rank() + 10 * tp_mesh.get_local_rank() + 1
        )
        _fill_local(parameter, fill_value)

        owner_meshes = resolve_parameter_owner_meshes(
            parameter,
            parallel_dims=parallel_dims,
        )
        statistics = reduce_finite_statistics(
            finite_statistics(parameter.to_local()),
            owner_meshes,
        )
        host_statistics = _to_cpu(statistics)

        self.assertIs(owner_meshes[0], fsdp_mesh)
        self.assertIs(owner_meshes[1], tp_mesh)
        self.assertEqual(host_statistics.counts.tolist(), [64, 0, 0])
        self.assertEqual(host_statistics.sums.tolist(), [3584.0, 362304.0])
        self.assertEqual(host_statistics.abs_max.tolist(), [111.0])
        self.assertEqual(int(host_statistics.counts[0]), parameter.numel())

        wrong_statistics = _to_cpu(
            reduce_finite_statistics(
                finite_statistics(parameter.to_local()),
                (tp_mesh, tp_mesh),
            )
        )
        self.assertEqual(wrong_statistics.counts[0], parameter.numel())
        self.assertNotEqual(wrong_statistics.sums, host_statistics.sums)

        linear(torch.ones(2, 8, device=self.device_type)).sum().backward()
        gradient = parameter.grad
        assert isinstance(gradient, DTensor)
        gradient_owner_meshes = resolve_parameter_owner_meshes(
            gradient,
            parallel_dims=parallel_dims,
        )
        gradient_statistics = _to_cpu(
            reduce_finite_statistics(
                finite_statistics(gradient.to_local()),
                gradient_owner_meshes,
            )
        )
        self.assertIs(gradient_owner_meshes[0], fsdp_mesh)
        self.assertIs(gradient_owner_meshes[1], tp_mesh)
        self.assertEqual(gradient_statistics.counts.tolist(), [64, 0, 0])
        self.assertEqual(gradient_statistics.sums.tolist(), [128.0, 256.0])
        self.assertEqual(gradient_statistics.abs_max.tolist(), [2.0])
        self.assertEqual(int(gradient_statistics.counts[0]), parameter.numel())
