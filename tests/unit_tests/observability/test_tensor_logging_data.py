# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.observability.tensor_logging.data import DataWindowStatistics


def test_weighted_loss_and_document_statistics() -> None:
    statistics = DataWindowStatistics(
        dataset_id="c4_test",
        data_contributor=True,
        loss_contributor=True,
        step_contributor=True,
        ignore_index=IGNORE_INDEX,
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

    statistics.record_batch(labels, positions)
    statistics.record_step_loss(torch.tensor(0.25), global_valid_tokens=8)
    statistics.record_batch(labels, positions)
    statistics.record_step_loss(torch.tensor(0.75), global_valid_tokens=8)
    metrics = statistics.collect()

    dataset = "data/datasets.c4_test"
    assert metrics[f"{dataset}.loss_mean"] == pytest.approx(0.5)
    assert metrics[f"{dataset}.valid_token_count"] == 16
    assert metrics[f"{dataset}.all_token_count"] == 20
    assert metrics[f"{dataset}.masked_fraction"] == pytest.approx(0.2)
    assert metrics[f"{dataset}.batch_count"] == 2
    assert metrics[f"{dataset}.window_steps"] == 2
    assert metrics["data/documents.segment_length_mean"] == pytest.approx(2.5)
    assert metrics["data/documents.segment_count"] == 8
    assert metrics["data/block_causal.sum_length_squared"] == 52

    cleared = statistics.collect()
    assert cleared[f"{dataset}.valid_token_count"] == 0
    assert cleared[f"{dataset}.window_steps"] == 0


def test_positionless_batch_keeps_token_metrics() -> None:
    statistics = DataWindowStatistics(
        dataset_id="positionless",
        data_contributor=True,
        loss_contributor=True,
        step_contributor=True,
        ignore_index=IGNORE_INDEX,
        device=torch.device("cpu"),
    )
    labels = torch.tensor([[1, IGNORE_INDEX, 2]], dtype=torch.int64)

    statistics.record_batch(labels, positions=None)
    statistics.record_step_loss(torch.tensor(0.5), global_valid_tokens=2)
    metrics = statistics.collect()

    dataset = "data/datasets.positionless"
    assert metrics[f"{dataset}.valid_token_count"] == 2
    assert metrics[f"{dataset}.all_token_count"] == 3
    assert metrics[f"{dataset}.masked_fraction"] == pytest.approx(1 / 3)
    assert metrics[f"{dataset}.loss_mean"] == pytest.approx(0.5)
    assert metrics["data/documents.segment_count"] == 0
    assert metrics["data/block_causal.sum_length_squared"] == 0
