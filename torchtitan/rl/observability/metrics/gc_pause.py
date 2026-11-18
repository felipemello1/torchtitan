# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import time

from .types import Max, Metric


class GcPause:
    """Measure the longest garbage collection in this process, to tell it from other event loop lag.

    While Python's garbage collector runs, no other Python code in the process runs, including the
    event loop. A full collection takes longer as the process holds more objects.

    Example:

        gc_pause = GcPause("rollout_worker")
        gc.collect()  # e.g. takes 0.4 s
        gc_pause.flush()
        # -> [Metric("rollout_worker/gc_pause_ms", Max(400.0))]
        gc_pause.flush()
        # -> Max(0.0), unless a collection ran since: each flush starts over
        gc_pause.close()

    Args:
        role: The kind of process, used as the key's first part: "rollout_worker" ->
            `rollout_worker/gc_pause_ms`.
    """

    def __init__(self, role: str) -> None:
        self._key = f"{role}/gc_pause_ms"
        self._max_pause_s = 0.0
        self._start_time = time.monotonic()
        gc.callbacks.append(self._on_gc)

    def _on_gc(self, phase: str, info: dict[str, int]) -> None:
        # Runs on the thread whose allocation started the collection. No lock: a lock here would
        # deadlock when a collection starts on a thread that already holds it. CPython runs one
        # collection at a time, so one start time is enough. A pause that ends between `flush`'s
        # read and its reset is lost.
        if phase == "start":
            self._start_time = time.monotonic()
        else:
            pause_s = time.monotonic() - self._start_time
            self._max_pause_s = max(self._max_pause_s, pause_s)

    def flush(self) -> list[Metric]:
        """Return the longest collection since the last flush, then reset."""
        max_pause_s, self._max_pause_s = self._max_pause_s, 0.0
        return [Metric(self._key, Max(1000 * max_pause_s))]

    def close(self) -> None:
        gc.callbacks.remove(self._on_gc)
