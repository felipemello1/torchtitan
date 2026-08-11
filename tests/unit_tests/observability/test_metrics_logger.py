# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torchtitan.components.metrics import TensorBoardLogger


def test_tensorboard_logger_writes_one_event_with_all_scalars(tmp_path: Path) -> None:
    logger = TensorBoardLogger(str(tmp_path), tag="train")
    logger.log({"loss": 1.25, "tokens": 8}, step=3)
    logger.close()

    event_file = next(tmp_path.glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()

    assert accumulator.Tags()["scalars"] == ["train/loss", "train/tokens"]
    assert [
        (event.step, event.value) for event in accumulator.Scalars("train/loss")
    ] == [(3, 1.25)]
    assert [
        (event.step, event.value) for event in accumulator.Scalars("train/tokens")
    ] == [(3, 8.0)]
