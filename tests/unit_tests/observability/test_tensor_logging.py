# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.recorders import (
    derive_finite_statistics,
    finite_statistics,
    FiniteStatistics,
)
from torchtitan.observability.tensor_logging.reduction import reduce_max, reduce_sum


def test_finite_statistics_uses_the_finite_denominator() -> None:
    statistics = finite_statistics(
        torch.tensor([0.0, -2.0, float("nan"), float("inf")])
    )
    result = derive_finite_statistics(statistics)

    assert result == {
        "numel": 4,
        "nonfinite_count": 2,
        "zero_count": 1,
        "zero_fraction": 0.5,
        "abs_mean": 1.0,
        "square_mean": 2.0,
        "rms": 2.0**0.5,
        "abs_max": 2.0,
    }


def test_all_nonfinite_omits_undefined_statistics() -> None:
    statistics = finite_statistics(torch.tensor([float("nan"), float("-inf")]))

    assert derive_finite_statistics(statistics) == {
        "numel": 2,
        "nonfinite_count": 2,
        "zero_count": 0,
    }


def test_finite_statistics_preserves_lane_contract() -> None:
    value = torch.tensor(
        [0.0, -2.0, float("nan")], dtype=torch.bfloat16, requires_grad=True
    )

    statistics = finite_statistics(value)

    assert statistics.counts.shape == (3,)
    assert statistics.counts.dtype == torch.int64
    assert statistics.sums.shape == (2,)
    assert statistics.sums.dtype == torch.float32
    assert statistics.abs_max.shape == (1,)
    assert statistics.abs_max.dtype == torch.float32
    assert all(
        lane.device == value.device
        for lane in (statistics.counts, statistics.sums, statistics.abs_max)
    )
    assert all(
        not lane.requires_grad
        for lane in (statistics.counts, statistics.sums, statistics.abs_max)
    )


def test_empty_finite_statistics_uses_max_identity() -> None:
    statistics = finite_statistics(torch.empty(0))

    assert statistics.counts.tolist() == [0, 0, 0]
    assert statistics.sums.tolist() == [0.0, 0.0]
    assert statistics.abs_max.tolist() == [0.0]
    assert derive_finite_statistics(statistics) == {
        "numel": 0,
        "nonfinite_count": 0,
        "zero_count": 0,
    }


class TestTensorLoggingReductionTwoRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_dp_sum(self) -> None:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        mesh = parallel_dims.get_mesh("batch")

        one = torch.ones((), device=self.device_type)
        rank_value = torch.tensor(self.rank + 1, device=self.device_type)

        self.assertEqual(reduce_sum(one, mesh).item(), 2)
        self.assertEqual(reduce_sum(rank_value, mesh).item(), 3)

    @with_comms
    def test_tp_sum(self) -> None:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        mesh = parallel_dims.get_mesh("tp")

        one = torch.ones((), device=self.device_type)
        rank_value = torch.tensor(self.rank + 1, device=self.device_type)

        self.assertEqual(reduce_sum(one, mesh).item(), 2)
        self.assertEqual(reduce_sum(rank_value, mesh).item(), 3)

    @with_comms
    def test_unequal_local_counts_use_a_global_denominator(self) -> None:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        mesh = parallel_dims.get_mesh("batch")
        local_value = (
            torch.tensor([1.0], device=self.device_type)
            if self.rank == 0
            else torch.tensor([1.0, 3.0, 5.0], device=self.device_type)
        )
        local_statistics = finite_statistics(local_value)
        global_statistics = FiniteStatistics(
            counts=reduce_sum(local_statistics.counts, mesh).cpu(),
            sums=reduce_sum(local_statistics.sums, mesh).cpu(),
            abs_max=reduce_max(local_statistics.abs_max, mesh).cpu(),
        )

        result = derive_finite_statistics(global_statistics)

        self.assertEqual(result["numel"], 4)
        self.assertEqual(result["abs_mean"], 2.5)
        self.assertEqual(result["square_mean"], 9.0)
        self.assertEqual(result["rms"], 3.0)
        self.assertEqual(result["abs_max"], 5.0)

    @with_comms
    def test_mixed_nonfinite_values_reduce_exact_counts(self) -> None:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        mesh = parallel_dims.get_mesh("batch")
        local_value = (
            torch.tensor([0.0, float("nan")], device=self.device_type)
            if self.rank == 0
            else torch.tensor([-2.0, float("inf"), 4.0], device=self.device_type)
        )
        local_statistics = finite_statistics(local_value)
        reduced_counts = reduce_sum(local_statistics.counts, mesh)
        reduced_sums = reduce_sum(local_statistics.sums, mesh)
        reduced_abs_max = reduce_max(local_statistics.abs_max, mesh)

        self.assertEqual(type(reduced_counts), torch.Tensor)
        self.assertEqual(type(reduced_sums), torch.Tensor)
        self.assertEqual(type(reduced_abs_max), torch.Tensor)
        global_statistics = FiniteStatistics(
            counts=reduced_counts.cpu(),
            sums=reduced_sums.cpu(),
            abs_max=reduced_abs_max.cpu(),
        )

        self.assertEqual(
            derive_finite_statistics(global_statistics),
            {
                "numel": 5,
                "nonfinite_count": 2,
                "zero_count": 1,
                "zero_fraction": 1 / 3,
                "abs_mean": 2.0,
                "square_mean": 20 / 3,
                "rms": (20 / 3) ** 0.5,
                "abs_max": 4.0,
            },
        )


class TestTensorLoggingReductionFourRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_mixed_dp_tp_uses_the_requested_axis(self) -> None:
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
        dp_mesh = parallel_dims.get_mesh("batch")
        tp_mesh = parallel_dims.get_mesh("tp")
        dp_coordinate = dp_mesh.get_local_rank()
        tp_coordinate = tp_mesh.get_local_rank()
        value = torch.tensor(
            100 * dp_coordinate + 10 * tp_coordinate + 1,
            device=self.device_type,
        )

        dp_result = reduce_sum(value, dp_mesh)
        tp_result = reduce_sum(value, tp_mesh)

        self.assertEqual(dp_result.item(), 102 + 20 * tp_coordinate)
        self.assertEqual(tp_result.item(), 12 + 200 * dp_coordinate)
