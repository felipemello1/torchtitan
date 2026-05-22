# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest

from torchtitan.experiments.rl.generation_scheduler import GenerationScheduler
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.types import Completion


def _completion(token_id: int) -> Completion:
    return Completion(
        policy_version=1,
        token_ids=[token_id],
        token_logprobs=[-0.1],
        finish_reason="stop",
    )


def test_batches_same_sampling_key_requests() -> None:
    async def scenario():
        calls = []
        sampling = SamplingConfig(temperature=0.7, top_p=1.0, max_tokens=8)

        async def generate_batch(prompts, request_ids, sampling):
            calls.append((prompts, request_ids, sampling))
            return [_completion(i) for i, _ in enumerate(request_ids)], []

        scheduler = GenerationScheduler(generate_batch, flush_window_s=0.01)
        try:
            results = await asyncio.gather(
                scheduler.submit(
                    prompt_token_ids=[1],
                    sampling=sampling,
                    request_id="a",
                ),
                scheduler.submit(
                    prompt_token_ids=[2],
                    sampling=sampling,
                    request_id="b",
                ),
            )
        finally:
            await scheduler.close()

        return calls, results

    calls, results = asyncio.run(scenario())

    assert len(calls) == 1
    prompts, request_ids, _ = calls[0]
    assert prompts == [[1], [2]]
    assert request_ids == ["a", "b"]
    assert [result.token_ids for result in results] == [[0], [1]]


def test_splits_different_sampling_keys() -> None:
    async def scenario():
        calls = []

        async def generate_batch(prompts, request_ids, sampling):
            calls.append((request_ids, sampling.temperature))
            return [_completion(0) for _ in request_ids], []

        scheduler = GenerationScheduler(generate_batch, flush_window_s=0.01)
        try:
            await asyncio.gather(
                scheduler.submit(
                    prompt_token_ids=[1],
                    sampling=SamplingConfig(temperature=0.7, top_p=1.0, max_tokens=8),
                    request_id="a",
                ),
                scheduler.submit(
                    prompt_token_ids=[2],
                    sampling=SamplingConfig(temperature=0.8, top_p=1.0, max_tokens=8),
                    request_id="b",
                ),
            )
        finally:
            await scheduler.close()

        return calls

    calls = asyncio.run(scenario())

    assert sorted((tuple(ids), temp) for ids, temp in calls) == [
        (("a",), 0.7),
        (("b",), 0.8),
    ]


def test_max_admitted_prompts_chunks_batches() -> None:
    async def scenario():
        call_sizes = []
        release_first = asyncio.Event()

        async def generate_batch(prompts, request_ids, sampling):
            call_sizes.append(len(request_ids))
            if len(call_sizes) == 1:
                await release_first.wait()
            return [_completion(i) for i, _ in enumerate(request_ids)], []

        scheduler = GenerationScheduler(
            generate_batch,
            max_admitted_prompts=2,
            flush_window_s=0.01,
        )
        try:
            tasks = [
                asyncio.create_task(
                    scheduler.submit(
                        prompt_token_ids=[idx],
                        sampling=SamplingConfig(max_tokens=8),
                        request_id=str(idx),
                    )
                )
                for idx in range(3)
            ]
            await asyncio.sleep(0.05)
            assert call_sizes == [2]
            release_first.set()
            await asyncio.gather(*tasks)
        finally:
            await scheduler.close()

        return call_sizes

    assert asyncio.run(scenario()) == [2, 1]


def test_pause_drains_active_work_and_blocks_new_admission() -> None:
    async def scenario():
        calls = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def generate_batch(prompts, request_ids, sampling):
            calls.append(request_ids)
            started.set()
            if request_ids == ["active"]:
                await release.wait()
            return [_completion(0) for _ in request_ids], []

        scheduler = GenerationScheduler(generate_batch)
        try:
            active = asyncio.create_task(
                scheduler.submit(
                    prompt_token_ids=[1],
                    sampling=SamplingConfig(max_tokens=8),
                    request_id="active",
                )
            )
            await started.wait()

            pause = asyncio.create_task(scheduler.pause_for_weight_sync())
            await asyncio.sleep(0)
            assert not pause.done()
            release.set()
            await pause
            await active

            blocked = asyncio.create_task(
                scheduler.submit(
                    prompt_token_ids=[2],
                    sampling=SamplingConfig(max_tokens=8),
                    request_id="blocked",
                )
            )
            await asyncio.sleep(0.05)
            assert calls == [["active"]]
            await scheduler.resume_after_weight_sync()
            await blocked
        finally:
            await scheduler.close()

        return calls

    assert asyncio.run(scenario()) == [["active"], ["blocked"]]


def test_cancelled_pause_reopens_admission() -> None:
    async def scenario():
        calls = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def generate_batch(prompts, request_ids, sampling):
            calls.append(request_ids)
            started.set()
            if request_ids == ["active"]:
                await release.wait()
            return [_completion(0) for _ in request_ids], []

        scheduler = GenerationScheduler(generate_batch)
        try:
            active = asyncio.create_task(
                scheduler.submit(
                    prompt_token_ids=[1],
                    sampling=SamplingConfig(max_tokens=8),
                    request_id="active",
                )
            )
            await started.wait()

            pause = asyncio.create_task(scheduler.pause_for_weight_sync())
            await asyncio.sleep(0)
            pause.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pause

            release.set()
            await active
            reopened = await scheduler.submit(
                prompt_token_ids=[2],
                sampling=SamplingConfig(max_tokens=8),
                request_id="reopened",
            )
        finally:
            await scheduler.close()

        return calls, reopened

    calls, reopened = asyncio.run(scenario())

    assert calls == [["active"], ["reopened"]]
    assert reopened.token_ids == [0]


def test_generation_failure_surfaces_and_scheduler_recovers() -> None:
    async def scenario():
        calls = 0

        async def generate_batch(prompts, request_ids, sampling):
            nonlocal calls
            calls += 1
            if calls == 1:
                return [], []
            return [_completion(7) for _ in request_ids], []

        scheduler = GenerationScheduler(generate_batch)
        try:
            with pytest.raises(RuntimeError, match="generator returned 0 completions"):
                await scheduler.submit(
                    prompt_token_ids=[1],
                    sampling=SamplingConfig(max_tokens=8),
                    request_id="bad",
                )
            result = await scheduler.submit(
                prompt_token_ids=[2],
                sampling=SamplingConfig(max_tokens=8),
                request_id="good",
            )
        finally:
            await scheduler.close()

        return result

    assert asyncio.run(scenario()).token_ids == [7]


def test_close_fails_queued_requests() -> None:
    async def scenario():
        async def generate_batch(prompts, request_ids, sampling):
            return [_completion(0) for _ in request_ids], []

        scheduler = GenerationScheduler(generate_batch, flush_window_s=0.1)
        task = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=SamplingConfig(max_tokens=8),
                request_id="queued",
            )
        )
        await asyncio.sleep(0)
        await scheduler.close()
        with pytest.raises(RuntimeError, match="generation scheduler is closed"):
            await task

    asyncio.run(scenario())
