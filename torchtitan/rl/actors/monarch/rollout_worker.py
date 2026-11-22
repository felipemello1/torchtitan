# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from monarch.actor import concurrent_endpoint, current_rank

# TODO: load the `grain` and `datasets` packages only where they are used, to save ~450 MiB per rollout worker.
#   This import runs `torchtitan/components/data/__init__.py`, which loads both. The renderer and
#   `rl/rollout/data.py` import them too. First check that a worker doesn't need them.
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch.actor import Actor
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker
from torchtitan.rl.rollout.types import RolloutGroup


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
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=num_threads)
        )
        self._worker = worker_config.build(**worker_kwargs)
        # group id -> the task running its `run_group` here, for `cancel_group`.
        self._group_tasks: dict[int, asyncio.Task] = {}

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
            return await self._worker.run_group(
                generate_fn=generate_fn,
                release_session_fn=release_session_fn,
                sample=sample,
                group_id=group_id,
                group_size=group_size,
                sampling=sampling,
            )
        finally:
            self._group_tasks.pop(group_id, None)

    @concurrent_endpoint
    async def cancel_group(self, group_id: int) -> None:
        """Cancel this group's `run_group` if this worker runs it; return once it ended.

            group runs on this worker      cancel it, return when it ended
            group not on this worker       return at once

        Monarch starts one sender's concurrent calls in order, so a `run_group` sent before this call is found.
        The cancelled `run_group` call fails with an `ActorError` at its caller.
        """
        # TODO: a cancel that reaches this worker before the group's `run_group` does nothing, and the group runs
        # to its end. The buffer cancels only groups past `max_offpolicy_steps` (a fresh claim is ~3 versions
        # behind at most), so it needs max < 3. Fix: remember cancelled ids and have `run_group` return at once
        # for one, or require `max_offpolicy_steps >= 3`.
        task = self._group_tasks.get(group_id)
        if task is not None:
            task.cancel()
            await asyncio.wait([task])

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)
