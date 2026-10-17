# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from torchtitan.rl.controller.loop_stall_watchdog import watch_loop_stalls


def _blocking_c_call() -> int:
    return sum(range(60_000_000))  # one C call that holds the GIL for ~1 s


def test_dumps_the_blocking_frame_and_the_duration(tmp_path) -> None:
    path = tmp_path / "stalls.txt"

    async def run() -> None:
        watchdog = asyncio.create_task(watch_loop_stalls(path=str(path), threshold_s=0.3))
        await asyncio.sleep(0.2)
        _blocking_c_call()
        await asyncio.sleep(0.2)
        watchdog.cancel()

    asyncio.run(run())
    dump = path.read_text()
    assert dump.startswith("Timeout")
    assert "in _blocking_c_call" in dump
    assert dump.count("# loop stall ended at") == 1


def test_healthy_loop_writes_nothing(tmp_path) -> None:
    path = tmp_path / "stalls.txt"

    async def run() -> None:
        watchdog = asyncio.create_task(watch_loop_stalls(path=str(path), threshold_s=0.3))
        for _ in range(10):
            sum(range(1_000_000))  # well under the threshold
            await asyncio.sleep(0.05)
        watchdog.cancel()

    asyncio.run(run())
    assert path.read_text() == ""
