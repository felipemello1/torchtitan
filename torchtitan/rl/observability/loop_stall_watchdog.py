# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Log the event-loop thread's Python stack while its loop is blocked."""

import asyncio
import json
import logging
import sys
import threading
import time
import traceback

from torchtitan.observability import structured_logger as sl

logger = logging.getLogger(__name__)


class LoopStallWatchdog:
    """Sample the loop thread's stack whenever its event loop hasn't ticked for `threshold_s`.

    A heartbeat task on the loop stamps the time every `tick_s`. A daemon thread checks the
    stamp; once the loop has been blocked for `threshold_s` it records the loop thread's stack
    (`sys._current_frames`), and again every `resample_s` while the block lasts. When the loop
    ticks again, one record with the block's duration and its sampled stacks is appended to
    `path` (JSONL) and a `loop_stall_s` scalar is logged.

    Python threads share the GIL, so the watchdog runs only when the blocked thread releases it:
    every switch interval (5 ms) for pure-Python work, not during one long C call such as a GC
    collection (its stack then shows the frame that triggered it, once the collection ends).

    Example:
        watchdog = LoopStallWatchdog(path="outputs/rl/loop_stalls.jsonl", threshold_s=1.0)
        task = asyncio.create_task(watchdog.heartbeat())  # on the loop to watch
        # a handler runs time.sleep(3) on the loop ->
        # {"start": 1792..., "duration_s": 3.0, "stacks": [{"at_s": 1.0, "stack": "..."}, ...]}
    """

    def __init__(
        self,
        *,
        path: str,
        threshold_s: float = 1.0,
        resample_s: float = 2.0,
        tick_s: float = 0.05,
        max_stacks: int = 10,
    ) -> None:
        self._path = path
        self._threshold_s = threshold_s
        self._resample_s = resample_s
        self._tick_s = tick_s
        self._max_stacks = max_stacks
        self._beat = time.monotonic()
        self._loop_thread_id: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._watch, name="loop_stall_watchdog", daemon=True
        )

    async def heartbeat(self) -> None:
        """Run on the watched loop until cancelled."""
        self._loop_thread_id = threading.get_ident()
        self._beat = time.monotonic()
        self._thread.start()
        try:
            while True:
                self._beat = time.monotonic()
                await asyncio.sleep(self._tick_s)
        finally:
            self._stop.set()

    def _watch(self) -> None:
        stall_beat: float | None = None  # the beat the current block started after
        stacks: list[dict] = []
        next_sample = 0.0
        while not self._stop.wait(self._tick_s):
            beat = self._beat
            blocked_s = time.monotonic() - beat
            if stall_beat is not None and beat != stall_beat:
                # The loop ticked: the block ended at about the new beat.
                self._record(stall_beat, beat - stall_beat, stacks)
                stall_beat, stacks = None, []
            if blocked_s < self._threshold_s:
                continue
            if stall_beat is None:
                stall_beat, next_sample = beat, 0.0
            if blocked_s >= next_sample and len(stacks) < self._max_stacks:
                frame = sys._current_frames().get(self._loop_thread_id)
                if frame is not None:
                    stacks.append(
                        {
                            "at_s": round(blocked_s, 3),
                            "stack": "".join(traceback.format_stack(frame, limit=40)),
                        }
                    )
                next_sample = blocked_s + self._resample_s

    def _record(self, start: float, duration_s: float, stacks: list[dict]) -> None:
        record = {
            "start_unix": round(time.time() - (time.monotonic() - start), 3),
            "duration_s": round(duration_s, 3),
            "stacks": stacks,
        }
        with open(self._path, "a") as file:
            file.write(json.dumps(record) + "\n")
        sl.log_trace_scalar({"loop_stall_s": duration_s})
        top = stacks[0]["stack"].rstrip().rsplit("\n", 2)[-2:] if stacks else []
        logger.warning(
            f"Event loop blocked {duration_s:.2f} s; first sampled frame: {' | '.join(s.strip() for s in top)}"
        )
