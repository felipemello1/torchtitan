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
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.data_statistics import (
    DataStatisticsRecorder,
)
from torchtitan.observability.tensor_logging.families import TensorMetricFamily


_ALL_DATA_FAMILIES = (
    TensorMetricFamily.DATASET_LOSS,
    TensorMetricFamily.DOCUMENT_SEGMENTS,
    TensorMetricFamily.BLOCK_CAUSAL_MOMENTS,
)


@pytest.fixture
def fake_world_one() -> Iterator[None]:
    dist.init_process_group("fake", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _parallel_dims(
    *,
    dp_shard: int = 1,
    cp: int = 1,
    tp: int = 1,
    world_size: int = 1,
) -> ParallelDims:
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=dp_shard,
        cp=cp,
        tp=tp,
        pp=1,
        ep=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()
    return parallel_dims


def test_exact_loss_document_and_block_moments(fake_world_one: None) -> None:
    with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
        parallel_dims = _parallel_dims()
    recorder = DataStatisticsRecorder(
        parallel_dims=parallel_dims,
        families=_ALL_DATA_FAMILIES,
        dataset_id="c4_test",
        device=torch.device("cpu"),
    )
    positions = torch.tensor(
        [[0, 1, 2, 0, 1], [0, 1, 0, 1, 2]],
        dtype=torch.int64,
    )
    labels = torch.tensor(
        [[1, 2, IGNORE_INDEX, 3, 4], [5, IGNORE_INDEX, 6, 7, 8]],
        dtype=torch.int64,
    )

    recorder.record_batch(labels=labels, positions=positions)
    recorder.record_loss(
        normalized_loss=torch.tensor(0.25),
        global_valid_tokens=8,
    )
    metrics = recorder.derive_metrics(recorder.collect(), window_steps=3)

    dataset = "tensor_metrics/data/datasets.c4_test"
    assert metrics[f"{dataset}.loss_mean"] == pytest.approx(0.25)
    assert metrics[f"{dataset}.valid_token_count"] == 8
    assert metrics[f"{dataset}.all_token_count"] == 10
    assert metrics[f"{dataset}.masked_fraction"] == pytest.approx(0.2)
    assert metrics[f"{dataset}.observation_count"] == 1
    assert metrics[f"{dataset}.window_steps"] == 3

    documents = "tensor_metrics/data/documents"
    assert metrics[f"{documents}.segment_length_mean"] == pytest.approx(2.5)
    assert metrics[f"{documents}.segment_count"] == 4
    assert metrics[f"{documents}.observation_count"] == 1

    block_causal = "tensor_metrics/data/block_causal"
    assert metrics[f"{block_causal}.sum_length_squared"] == 26
    assert metrics[f"{block_causal}.observation_count"] == 1
    assert metrics[f"{block_causal}.window_steps"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestDataStatisticsFourRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_dp_batches_exclude_tp_replicas(self) -> None:
        device = torch.device(self.device_type)
        parallel_dims = _parallel_dims(dp_shard=2, tp=2, world_size=self.world_size)
        dp_rank = parallel_dims.get_mesh("batch").get_local_rank()
        recorder = DataStatisticsRecorder(
            parallel_dims=parallel_dims,
            families=_ALL_DATA_FAMILIES,
            dataset_id="c4_test",
            device=device,
        )
        positions = torch.tensor(
            [[0, 1, 0, 1] if dp_rank == 0 else [0, 1, 2, 3]],
            dtype=torch.int64,
            device=device,
        )
        labels = torch.tensor(
            [[1, 2, 3, 4] if dp_rank == 0 else [5, 6, 7, IGNORE_INDEX]],
            dtype=torch.int64,
            device=device,
        )
        local_loss_sum = 2.0 if dp_rank == 0 else 5.0

        recorder.record_batch(labels=labels, positions=positions)
        recorder.record_loss(
            normalized_loss=torch.tensor(local_loss_sum / 7, device=device),
            global_valid_tokens=7,
        )
        metrics = recorder.derive_metrics(recorder.collect(), window_steps=1)

        dataset = "tensor_metrics/data/datasets.c4_test"
        self.assertEqual(metrics[f"{dataset}.all_token_count"], 8)
        self.assertEqual(metrics[f"{dataset}.valid_token_count"], 7)
        self.assertAlmostEqual(metrics[f"{dataset}.loss_mean"], 1.0)
        self.assertEqual(metrics[f"{dataset}.observation_count"], 2)
        self.assertEqual(
            metrics["tensor_metrics/data/documents.segment_count"],
            3,
        )
        self.assertEqual(
            metrics["tensor_metrics/data/block_causal.sum_length_squared"],
            24,
        )

    @with_comms
    def test_cp_loss_parts_and_full_data_use_different_representatives(self) -> None:
        device = torch.device(self.device_type)
        parallel_dims = _parallel_dims(cp=2, tp=2, world_size=self.world_size)
        cp_rank = parallel_dims.get_mesh("cp").get_local_rank()
        recorder = DataStatisticsRecorder(
            parallel_dims=parallel_dims,
            families=_ALL_DATA_FAMILIES,
            dataset_id="c4_test",
            device=device,
        )
        positions = torch.tensor([[0, 1, 0, 1]], device=device)
        labels = torch.tensor([[1, 2, 3, IGNORE_INDEX]], device=device)
        local_loss_sum = 1.0 if cp_rank == 0 else 3.0

        recorder.record_batch(labels=labels, positions=positions)
        recorder.record_loss(
            normalized_loss=torch.tensor(local_loss_sum / 3, device=device),
            global_valid_tokens=3,
        )
        metrics = recorder.derive_metrics(recorder.collect(), window_steps=1)

        dataset = "tensor_metrics/data/datasets.c4_test"
        self.assertEqual(metrics[f"{dataset}.all_token_count"], 4)
        self.assertEqual(metrics[f"{dataset}.valid_token_count"], 3)
        self.assertAlmostEqual(metrics[f"{dataset}.loss_mean"], 4 / 3)
        self.assertEqual(metrics[f"{dataset}.observation_count"], 1)
        self.assertEqual(
            metrics["tensor_metrics/data/documents.segment_count"],
            2,
        )
        self.assertEqual(
            metrics["tensor_metrics/data/block_causal.sum_length_squared"],
            8,
        )
