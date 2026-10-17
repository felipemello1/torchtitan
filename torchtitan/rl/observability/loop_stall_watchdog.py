# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dump every thread's Python stack when the controller's event loop stays blocked."""

import asyncio
import faulthandler
import time

from torchtitan.observability import structured_logger as sl

_TICK_S = 0.05


async def watch_loop_stalls(*, path: str, threshold_s: float) -> None:
    """Run on the event loop to watch; dump all threads' stacks to `path` when it blocks.

    Every 50 ms this task re-arms `faulthandler.dump_traceback_later(threshold_s)`. If the loop
    doesn't run the task again within `threshold_s`, faulthandler's own C thread writes every
    thread's stack to `path`, even mid-way through a C call that holds the GIL (a GC pass, a large
    `json.dumps`). When the loop resumes, the task logs a `loop_stall_s` scalar and appends a
    line with the stall's end time and duration. Runs until cancelled.

    faulthandler holds one timer per process, so nothing else in the process may use
    `dump_traceback_later` while this runs.

    Example:
        task = asyncio.create_task(watch_loop_stalls(path="loop_stalls.txt", threshold_s=1.0))
        # a handler blocks the loop for 3 s ->
        # Timeout (0:00:01)!
        # Thread 0x... (most recent call first):
        #   File ".../generation_server.py", line 231 in _handle_generate_request
        #   ...
        # # loop stall ended at 1792112345.6, 3.02 s
    """
    with open(path, "a") as file:
        last_tick = time.monotonic()
        try:
            while True:
                faulthandler.dump_traceback_later(threshold_s, file=file)
                await asyncio.sleep(_TICK_S)
                now = time.monotonic()
                stall_s = now - last_tick - _TICK_S
                if stall_s >= threshold_s:
                    file.write(f"# loop stall ended at {time.time():.1f}, {stall_s:.2f} s\n")
                    file.flush()
                    sl.log_trace_scalar({"loop_stall_s": stall_s})
                last_tick = now
        finally:
            faulthandler.cancel_dump_traceback_later()
