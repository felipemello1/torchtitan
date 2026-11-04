# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time

from torchtitan.observability import metrics
from torchtitan.observability.metrics import TensorBoardLogger


def test_close_gives_up_on_a_stuck_writer(tmp_path, monkeypatch):
    """A writer whose close() never returns (its thread stuck in a write() to a stalled
    filesystem) must not block TensorBoardLogger.close()."""
    tensorboard = TensorBoardLogger(TensorBoardLogger.Config(), log_dir=str(tmp_path))
    real_close = tensorboard.writer.close
    release = threading.Event()
    # Blocks for 5 s, standing in for a close() that never returns.
    tensorboard.writer.close = lambda: release.wait(5)
    monkeypatch.setattr(metrics, "_TENSORBOARD_CLOSE_TIMEOUT_S", 0.1)

    started = time.monotonic()
    tensorboard.close()
    release.set()
    real_close()

    assert time.monotonic() - started < 2
