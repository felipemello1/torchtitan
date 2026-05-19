# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Controller-side batching and weight-sync admission for generation."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.types import Completion
from torchtitan.observability import structured_logger as sl


GenerateBatchFn = Callable[
    [list[list[int]], SamplingConfig],
    Awaitable[tuple[list[Completion], list[m.Metric]]],
]


@dataclass(slots=True)
class _PendingGeneration:
    """One controller-side generation request awaiting a batched flush."""

    prompt_token_ids: list[int]
    sampling: SamplingConfig
    request_id: str
    future: asyncio.Future[Completion]


def _sampling_key(
    sampling: SamplingConfig,
) -> tuple[float, float, int, tuple[int, ...]]:
    return (
        sampling.temperature,
        sampling.top_p,
        sampling.max_tokens,
        tuple(sampling.stop_token_ids),
    )


class GenerationScheduler:
    """Ordered controller-side admission and draining for token generation.

    The controller owns batching so every tensor-parallel generator rank
    receives one ordered ``generate`` call per flush. Weight sync pauses
    admission at request boundaries: active generator calls drain, new rollout
    turns queue on the controller, and generation resumes after fresh weights
    are loaded.
    """

    def __init__(self, generate_batch: GenerateBatchFn):
        self._generate_batch = generate_batch
        self._pending: list[_PendingGeneration] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._metrics: list[m.Metric] = []
        self._condition = asyncio.Condition()
        self._loading_weights = False
        self._active_requests = 0
        self._closed = False

    def pop_metrics(self) -> list[m.Metric]:
        metrics = self._metrics
        self._metrics = []
        return metrics

    @staticmethod
    def _closed_error() -> RuntimeError:
        return RuntimeError("generation scheduler is closed")

    @staticmethod
    def _fail_pending(pending: list[_PendingGeneration]) -> None:
        for request in pending:
            if not request.future.done():
                request.future.set_exception(GenerationScheduler._closed_error())

    async def submit(
        self,
        *,
        prompt_token_ids: list[int],
        sampling: SamplingConfig,
        request_id: str,
    ) -> Completion:
        if sampling.n != 1:
            raise ValueError(f"GenerationScheduler requires n=1, got {sampling.n}")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Completion] = loop.create_future()
        async with self._condition:
            if self._closed:
                raise self._closed_error()
            self._pending.append(
                _PendingGeneration(
                    prompt_token_ids=list(prompt_token_ids),
                    sampling=sampling,
                    request_id=request_id,
                    future=future,
                )
            )
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush_loop())
        return await future

    async def _flush_loop(self) -> None:
        await asyncio.sleep(0)
        while True:
            async with self._condition:
                if self._closed or not self._pending:
                    self._fail_pending(self._pending)
                    self._pending = []
                    self._condition.notify_all()
                    return
                batch = self._pending
                self._pending = []

            pending_by_sampling: dict[
                tuple[float, float, int, tuple[int, ...]],
                list[_PendingGeneration],
            ] = defaultdict(list)
            for pending in batch:
                pending_by_sampling[_sampling_key(pending.sampling)].append(pending)

            for pending_group in pending_by_sampling.values():
                await self._flush_group(pending_group)
            await asyncio.sleep(0)

    async def _flush_group(self, pending_group: list[_PendingGeneration]) -> None:
        pending_group = [
            pending for pending in pending_group if not pending.future.done()
        ]
        if not pending_group:
            return

        sampling = pending_group[0].sampling
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or not self._loading_weights
            )
            if self._closed:
                self._fail_pending(pending_group)
                return
            pending_group = [
                pending for pending in pending_group if not pending.future.done()
            ]
            if not pending_group:
                return
            self._active_requests += len(pending_group)

        try:
            with sl.log_trace_span("generation_scheduler_flush"):
                completions, metrics = await self._generate_batch(
                    [pending.prompt_token_ids for pending in pending_group],
                    sampling,
                )
            self._metrics.extend(metrics)
            if len(completions) != len(pending_group):
                raise RuntimeError(
                    "generator returned "
                    f"{len(completions)} completions for "
                    f"{len(pending_group)} requests "
                    f"({[pending.request_id for pending in pending_group]})"
                )
            for pending, completion in zip(
                pending_group,
                completions,
                strict=True,
            ):
                if not pending.future.done():
                    pending.future.set_result(completion)
        except Exception as exc:
            for pending in pending_group:
                if not pending.future.done():
                    pending.future.set_exception(exc)
        finally:
            async with self._condition:
                self._active_requests -= len(pending_group)
                self._condition.notify_all()

    async def pause_for_weight_sync(self) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or not self._loading_weights
            )
            if self._closed:
                raise self._closed_error()
            self._loading_weights = True
            await self._condition.wait_for(lambda: self._active_requests == 0)

    async def resume_after_weight_sync(self) -> None:
        async with self._condition:
            self._loading_weights = False
            self._condition.notify_all()

    async def close(self) -> None:
        task: asyncio.Task[None] | None
        async with self._condition:
            self._closed = True
            self._loading_weights = False
            self._fail_pending(self._pending)
            self._pending = []
            self._condition.notify_all()
            task = self._flush_task

        if task is not None and task is not asyncio.current_task():
            await task

        async with self._condition:
            await self._condition.wait_for(lambda: self._active_requests == 0)
