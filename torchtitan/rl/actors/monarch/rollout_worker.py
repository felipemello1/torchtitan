# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
from typing import Any

from monarch.actor import Actor, concurrent_endpoint, current_rank

# TODO: load the `grain` and `datasets` packages only where they are used, to save ~450 MiB per rollout worker.
#   This import runs `torchtitan/components/data/__init__.py`, which loads both. The renderer and
#   `rl/rollout/data.py` import them too. First check that a worker doesn't need them.
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker
from torchtitan.rl.rollout.types import RolloutGroup
from torchtitan.tools.garbage_collector import GarbageCollector


class RolloutWorkerActor(Actor):
    """Monarch adapter that owns a configurable standalone rollout worker."""

    def __init__(
        self,
        *,
        worker_config: BaseRolloutWorker.Config,
        worker_kwargs: dict[str, Any],
        num_threads: int,
        structured_logger_init_fn: functools.partial[None] | None,
    ) -> None:
        if structured_logger_init_fn is not None:
            structured_logger_init_fn(
                source="rl_rollout_worker", rank=current_rank().rank
            )
        # The pool behind every `asyncio.to_thread` call in this process. It also measures how
        # long each call waits for a free thread.
        self._executor = m.QueueWaitThreadPoolExecutor(
            "rollout_worker", max_workers=num_threads
        )
        asyncio.get_running_loop().set_default_executor(self._executor)
        self._worker = worker_config.build(**worker_kwargs)
        # group id -> the task running its `run_group` here, for `cancel_group`.
        self._group_tasks: dict[int, asyncio.Task] = {}
        # Created at the end of `setup_async`, so setup isn't counted as lag or as a GC pause.
        self._event_loop_lag: m.EventLoopLag
        self._gc_pause: m.GcPause

    @concurrent_endpoint
    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        await self._worker.setup_async(
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
            vision_processor_config=vision_processor_config,
        )
        # Setup objects (imported modules, the tokenizer: ~0.7-1M objects) live until exit. Freeze
        # them, so a full collection skips them and stalls the event loop for less time. Trade-off:
        # CPython 3.12 starts a full collection when its long-lived objects grow by 25%, and frozen
        # objects no longer count, so full collections come more often. Near the event loop's limit,
        # the total pause grows.
        GarbageCollector.freeze(
            "Full GC before freezing the rollout worker's setup objects"
        )
        # Start measuring after setup: setup may hold the loop (e.g. building a tokenizer), and that
        # isn't rollout work.
        self._event_loop_lag = m.EventLoopLag("rollout_worker")
        self._gc_pause = m.GcPause("rollout_worker")

    @concurrent_endpoint
    async def run_group(
        self,
        *,
        generate_fn: Any,
        release_session_fn: Any,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: Any,
    ) -> RolloutGroup:
        self._group_tasks[group_id] = asyncio.current_task()
        try:
            group = await self._worker.run_group(
                generate_fn=generate_fn,
                release_session_fn=release_session_fn,
                sample=sample,
                group_id=group_id,
                group_size=group_size,
                sampling=sampling,
            )
        finally:
            self._group_tasks.pop(group_id, None)
        # Add this worker's metrics to each training group. What each shows and when to act:
        # "Rollout worker metrics" in `torchtitan/rl/observability/metrics/README.md`.
        #   Since its previous training group: event_loop_lag_ms, event_loop_lag_ms_per_s,
        #   executor_queue_wait_ms, num_executor_jobs_in_flight, gc_pause_ms.
        #   Read now: rss_gib, pss_gib.
        # The controller ignores the metrics of validation groups (negative ids), so what is measured
        # during validation goes with this worker's next training group.
        if group_id >= 0:
            group.metrics += [
                *self._event_loop_lag.flush(),
                *self._executor.flush(),
                *self._gc_pause.flush(),
                m.Metric(
                    "rollout_worker/rss_gib",
                    m.Max(_read_proc_kib("/proc/self/status", "VmRSS") / 2**20),
                ),
                # Reading Pss walks the page tables on the event loop: ~0.6 ms per training group with
                # 64 KiB pages, more with 4 KiB pages.
                m.Metric(
                    "rollout_worker/pss_gib",
                    m.Max(_read_proc_kib("/proc/self/smaps_rollup", "Pss") / 2**20),
                ),
            ]
        return group

    @concurrent_endpoint
    async def cancel_group(self, group_id: int) -> None:
        """Cancel this group's `run_group` if this worker runs it; return once it ended.

            group runs on this worker      cancel it, return when it ended
            group not on this worker       return at once

        Monarch starts one sender's concurrent calls in order, so a `run_group` sent before this call is found.
        The cancelled `run_group` call fails with an `ActorError` at its caller.
        """
        task = self._group_tasks.get(group_id)
        if task is not None:
            task.cancel()
            await asyncio.wait([task])

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)


def _read_proc_kib(path: str, field: str) -> int:
    """Return the kB value of one field of a Linux /proc file.

    Example: the line "VmRSS:  1745628 kB" in /proc/self/status -> 1745628.
    """
    with open(path) as proc_file:
        for line in proc_file:
            if line.startswith(f"{field}:"):
                return int(line.split()[1])
    raise RuntimeError(f"{path} has no {field} line")
