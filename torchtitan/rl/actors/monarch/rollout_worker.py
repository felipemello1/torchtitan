# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
import os
import socket
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
        num_threads: int,
        structured_logger_init_fn: functools.partial[None] | None,
        first_worker_index: int,
    ) -> None:
        if structured_logger_init_fn is not None:
            # Each host's workers form their own proc mesh, so `current_rank()` restarts at 0 on
            # every host. The offset keeps each worker's log rank unique.
            structured_logger_init_fn(
                source="rl_rollout_worker",
                rank=first_worker_index + current_rank().rank,
            )
        # The pool behind every `asyncio.to_thread` call in this process. It also measures how
        # long each call waits for a free thread.
        self._executor = m.QueueWaitThreadPoolExecutor(
            "rollout_worker", max_workers=num_threads
        )
        asyncio.get_running_loop().set_default_executor(self._executor)
        self._worker_config = worker_config
        # Built in `setup_async`: a build error there, e.g. a Verifiers env that fails to load,
        # reaches the caller as an endpoint error. Raised here, it would kill the actor.
        self._worker: BaseRolloutWorker
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
        paths_on_every_host: list[str],
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        # The training dataset's paths, e.g. its image dir. A host without one fails the job here, at
        # start, instead of failing each rollout that reads it.
        for path in paths_on_every_host:
            if not os.path.exists(path):
                raise RuntimeError(
                    f"{os.path.abspath(path)} is missing on host {socket.gethostname()}. The "
                    "training dataset's samples point there (rollouter.training_dataloader.dataset). "
                    "Please make it available at this path on every host that runs rollout workers, "
                    "best in a shared folder, or keep every rollout worker on the controller's host: "
                    "set rollouter.worker_roles.generator=None and "
                    "rollouter.worker_roles.controller=True."
                )
        self._worker = self._worker_config.build()
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
        group = await self._worker.run_group(
            generate_fn=generate_fn,
            release_session_fn=release_session_fn,
            sample=sample,
            group_id=group_id,
            group_size=group_size,
            sampling=sampling,
        )
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
    async def close(self) -> None:
        # The controller closes every host's workers, also when `setup_async` failed before it
        # built this one, e.g. on a missing path. Then there is nothing to close.
        if hasattr(self, "_worker"):
            await self._worker.close()

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
