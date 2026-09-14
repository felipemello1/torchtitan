# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from monarch.actor import Actor, concurrent_endpoint

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.rollouter import RolloutWorker
from torchtitan.experiments.rl.rollout.types import RolloutGroup
from torchtitan.observability import structured_logger as sl


class RolloutWorkerActor(Actor):
    """Hosts a rollout worker in one CPU process."""

    def __init__(
        self,
        *,
        worker_config: RolloutWorker.Config,
        num_threads: int,
    ) -> None:
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=num_threads)
        )
        self._worker: RolloutWorker = worker_config.build()
        self._active_run_groups = 0
        self._closing = False
        self._drained = asyncio.Condition()

    @concurrent_endpoint
    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
    ) -> None:
        await self._worker.setup_async(
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
        )

    @concurrent_endpoint
    async def run_group(
        self,
        *,
        generate_fn: Any,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: Any,
    ) -> RolloutGroup:
        async with self._drained:
            if self._closing:
                raise RuntimeError("rollout worker is closing")
            self._active_run_groups += 1
        try:
            return await self._worker.run_group(
                generate_fn=generate_fn,
                sample=sample,
                group_id=group_id,
                group_size=group_size,
                sampling=sampling,
            )
        finally:
            async with self._drained:
                self._active_run_groups -= 1
                if self._active_run_groups == 0:
                    self._drained.notify_all()

    @concurrent_endpoint
    async def close(self) -> None:
        """Reject new groups and wait for all active group calls to unwind."""
        async with self._drained:
            self._closing = True
            await self._drained.wait_for(lambda: self._active_run_groups == 0)

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)
