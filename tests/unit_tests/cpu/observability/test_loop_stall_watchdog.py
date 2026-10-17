# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import time

from torchtitan.rl.observability.loop_stall_watchdog import LoopStallWatchdog


def _busy_python(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        sum(range(1000))


def test_watchdog_records_blocking_frame_and_duration(tmp_path) -> None:
    path = tmp_path / "stalls.jsonl"
    watchdog = LoopStallWatchdog(path=str(path), threshold_s=0.3, resample_s=0.4)

    async def run() -> None:
        heartbeat = asyncio.create_task(watchdog.heartbeat())
        await asyncio.sleep(0.2)  # healthy: nothing recorded
        _busy_python(1.0)  # blocks the loop for 1 s in pure Python
        await asyncio.sleep(0.3)  # the loop ticks; the watchdog writes the record
        heartbeat.cancel()

    asyncio.run(run())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert 0.9 <= records[0]["duration_s"] <= 1.3
    # Sampled at ~0.3 s and ~0.7 s into the block, inside the blocking function.
    assert 2 <= len(records[0]["stacks"]) <= 3
    assert all("_busy_python" in sample["stack"] for sample in records[0]["stacks"])


def test_watchdog_records_nothing_on_a_healthy_loop(tmp_path) -> None:
    path = tmp_path / "stalls.jsonl"
    watchdog = LoopStallWatchdog(path=str(path), threshold_s=0.3)

    async def run() -> None:
        heartbeat = asyncio.create_task(watchdog.heartbeat())
        for _ in range(10):
            _busy_python(0.05)  # short blocks under the threshold
            await asyncio.sleep(0.05)
        heartbeat.cancel()

    asyncio.run(run())
    assert not path.exists()
