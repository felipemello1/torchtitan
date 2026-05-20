# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Controller-side batching and weight-sync admission for generation."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.types import Completion
from torchtitan.observability import structured_logger as sl


GenerateBatchFn = Callable[
    [list[list[int]], list[str], SamplingConfig],
    Awaitable[tuple[list[Completion], list[m.Metric]]],
]


@dataclass(slots=True)
class _PendingGeneration:
    """One controller-side generation request awaiting a batched flush."""

    prompt_token_ids: list[int]
    sampling: SamplingConfig
    request_id: str
    future: asyncio.Future[Completion]
    submitted_at_s: float


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

    The controller owns batching so every tensor-parallel generator rank receives
    deterministic ``generate`` calls from the same controller tick. Mixed
    sampling configs are partitioned into separate sub-batches, and result
    futures preserve each request's prompt/completion association. Weight sync
    pauses admission at request boundaries: active generator calls drain, new
    rollout turns queue on the controller, and generation resumes after fresh
    weights are loaded. Flush tasks may overlap, so later rollout turns can be
    admitted to the generator actor while earlier turns are still decoding.
    ``max_admitted_prompts`` bounds controller-admitted prompts whose generator
    calls have started and not returned yet; extra flushed chunks remain queued.
    """

    def __init__(
        self,
        generate_batch: GenerateBatchFn,
        *,
        max_admitted_prompts: int | None = None,
        flush_window_s: float = 0.0,
    ):
        if max_admitted_prompts is not None and max_admitted_prompts <= 0:
            raise ValueError(
                "max_admitted_prompts must be positive or None, "
                f"got {max_admitted_prompts}"
            )
        if flush_window_s < 0:
            raise ValueError(
                f"flush_window_s must be non-negative, got {flush_window_s}"
            )
        self._generate_batch = generate_batch
        self._max_admitted_prompts = max_admitted_prompts
        self._flush_window_s = flush_window_s
        self._pending: list[_PendingGeneration] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._active_flush_tasks: set[asyncio.Task[None]] = set()
        self._metrics: list[m.Metric] = []
        self._condition = asyncio.Condition()
        self._loading_weights = False
        self._admitted_prompts = 0
        self._queued_flush_prompts = 0
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
                    submitted_at_s=time.perf_counter(),
                )
            )
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush_loop())
        return await future

    async def _flush_loop(self) -> None:
        # Coalesce sibling rollout submits arriving within `flush_window_s`
        # so each batch admitted to the actor exercises vLLM continuous
        # batching instead of shipping one prompt at a time.
        await asyncio.sleep(self._flush_window_s)
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

            chunks = [
                chunk
                for pending_group in pending_by_sampling.values()
                for chunk in self._flush_chunks(pending_group)
            ]
            async with self._condition:
                self._queued_flush_prompts += sum(len(chunk) for chunk in chunks)
                self._condition.notify_all()

            for chunk in chunks:
                task = asyncio.create_task(self._flush_group(chunk))
                self._active_flush_tasks.add(task)
                task.add_done_callback(self._active_flush_tasks.discard)
            await asyncio.sleep(0)

    def _flush_chunks(
        self, pending_group: list[_PendingGeneration]
    ) -> list[list[_PendingGeneration]]:
        max_admitted_prompts = self._max_admitted_prompts
        if max_admitted_prompts is None or len(pending_group) <= max_admitted_prompts:
            return [pending_group]
        return [
            pending_group[start : start + max_admitted_prompts]
            for start in range(0, len(pending_group), max_admitted_prompts)
        ]

    def _can_admit(self, batch_size: int) -> bool:
        return (
            self._max_admitted_prompts is None
            or self._admitted_prompts + batch_size <= self._max_admitted_prompts
        )

    async def _flush_group(self, pending_group: list[_PendingGeneration]) -> None:
        queued_count = len(pending_group)
        admitted_count = 0
        pending_group = [
            pending for pending in pending_group if not pending.future.done()
        ]
        if not pending_group:
            async with self._condition:
                self._queued_flush_prompts -= queued_count
                self._condition.notify_all()
            return

        sampling = pending_group[0].sampling
        async with self._condition:
            try:
                await self._condition.wait_for(
                    lambda: self._closed
                    or (
                        not self._loading_weights
                        and self._can_admit(len(pending_group))
                    )
                )
            finally:
                self._queued_flush_prompts -= queued_count
                queued_count = 0
            if self._closed:
                self._fail_pending(pending_group)
                return
            pending_group = [
                pending for pending in pending_group if not pending.future.done()
            ]
            if not pending_group:
                return
            self._admitted_prompts += len(pending_group)
            admitted_count = len(pending_group)
            batch_size = len(pending_group)
            queued_prompts = len(self._pending) + self._queued_flush_prompts
            admitted_prompts = self._admitted_prompts

        try:
            queue_wait_s = [
                time.perf_counter() - pending.submitted_at_s
                for pending in pending_group
            ]
            self._metrics.extend(
                [
                    m.Metric("generation_scheduler/batch_size", m.Mean(batch_size)),
                    m.Metric("generation_scheduler/batch_size", m.Max(batch_size)),
                    m.Metric(
                        "generation_scheduler/queued_prompts",
                        m.Mean(queued_prompts),
                    ),
                    m.Metric(
                        "generation_scheduler/queued_prompts",
                        m.Max(queued_prompts),
                    ),
                    m.Metric(
                        "generation_scheduler/admitted_prompts",
                        m.Mean(admitted_prompts),
                    ),
                    m.Metric(
                        "generation_scheduler/admitted_prompts",
                        m.Max(admitted_prompts),
                    ),
                    m.Metric(
                        "generation_scheduler/queue_wait_seconds",
                        m.Mean.from_list(queue_wait_s),
                    ),
                    m.Metric(
                        "generation_scheduler/queue_wait_seconds",
                        m.Max.from_list(queue_wait_s),
                    ),
                ]
            )
            sl.log_trace_scalar(
                {
                    "generation_scheduler.batch_size": batch_size,
                    "generation_scheduler.queued_prompts": queued_prompts,
                    "generation_scheduler.admitted_prompts": admitted_prompts,
                    "generation_scheduler.queue_wait_ms.max": max(queue_wait_s) * 1000,
                }
            )
            with sl.log_trace_span("generation_scheduler_flush"):
                completions, metrics = await self._generate_batch(
                    [pending.prompt_token_ids for pending in pending_group],
                    [pending.request_id for pending in pending_group],
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
        except asyncio.CancelledError:
            for pending in pending_group:
                if not pending.future.done():
                    pending.future.cancel()
            raise
        except Exception as exc:
            for pending in pending_group:
                if not pending.future.done():
                    pending.future.set_exception(exc)
        finally:
            async with self._condition:
                if queued_count:
                    self._queued_flush_prompts -= queued_count
                self._admitted_prompts -= admitted_count
                self._condition.notify_all()

    async def pause_for_weight_sync(self) -> None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or not self._loading_weights
            )
            if self._closed:
                raise self._closed_error()
            self._loading_weights = True
            await self._condition.wait_for(lambda: self._admitted_prompts == 0)

    async def resume_after_weight_sync(self) -> None:
        async with self._condition:
            self._loading_weights = False
            self._condition.notify_all()

    async def close(self) -> None:
        task: asyncio.Task[None] | None
        active_flush_tasks: list[asyncio.Task[None]]
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
            active_flush_tasks = list(self._active_flush_tasks)
        if active_flush_tasks:
            await asyncio.gather(*active_flush_tasks, return_exceptions=True)

        async with self._condition:
            await self._condition.wait_for(lambda: self._admitted_prompts == 0)
