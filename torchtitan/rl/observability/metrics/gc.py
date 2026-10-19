# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import time

from .types import Max, Metric, Sum


class GCTimer:
    """Time this process spends in Python's garbage collector; flush() drains it.

    Example:
        gc_timer = GCTimer("batcher")  # registers a gc callback until close()
        ...                            # collections run until the next flush
        gc_timer.flush()
        # -> [Metric("perf/batcher/gc_seconds", Sum(1.9)),  # every collection since the last flush
        #     Metric("perf/batcher/gc_seconds", Max(1.2))]  # the longest pause
        gc_timer.close()
    """

    def __init__(self, role: str) -> None:
        self._key = f"perf/{role}/gc_seconds"
        self._start = time.perf_counter()
        self._total_s = self._max_s = 0.0
        gc.callbacks.append(self._on_gc)

    def _on_gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._start = time.perf_counter()
            return
        # Wall time: if another thread takes the GIL during these callbacks, its run time counts too.
        seconds = time.perf_counter() - self._start
        self._total_s += seconds
        self._max_s = max(self._max_s, seconds)

    def flush(self) -> list[Metric]:
        """Return the GC time since the last flush, then reset."""
        metrics = [
            Metric(self._key, Sum(self._total_s)),
            Metric(self._key, Max(self._max_s)),
        ]
        self._total_s = self._max_s = 0.0
        return metrics

    def close(self) -> None:
        gc.callbacks.remove(self._on_gc)
