# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for pacing sandbox creation in the Verifiers rollouter."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers")

from torchtitan.rl.examples.dapo_math import DapoMathSample
from torchtitan.rl.examples.verifiers import SandboxCreationLimiter
from torchtitan.rl.examples.verifiers.dapo_math import data
from torchtitan.rl.examples.verifiers.generation_server import (
    VerifiersGenerationMetadata,
)
from torchtitan.rl.generator import SamplingConfig
from torchtitan.rl.rollout.types import RolloutStatus
from torchtitan_recipes.rl.verifiers_dapo_math import _verifiers_math_rollouter_config
from verifiers.v1.trace import Error
from verifiers.v1.types import AssistantMessage


class FakeSandboxProvider:
    """Stands in for `VerifiersEnvClient`: each `run` creates one sandbox, and a create
    within `min_gap_sec` of the previous one is rate-limited (HTTP 429), so that
    rollout comes back failed."""

    def __init__(self, min_gap_sec: float) -> None:
        self.min_gap_sec = min_gap_sec
        self.create_times: list[float] = []

    async def run(self, **kwargs) -> SimpleNamespace:
        now = asyncio.get_running_loop().time()
        rate_limited = bool(self.create_times) and (
            now - self.create_times[-1] < self.min_gap_sec
        )
        self.create_times.append(now)
        return _episode(trace_id=f"trace-{len(self.create_times)}", ok=not rate_limited)


def _episode(*, trace_id: str, ok: bool) -> SimpleNamespace:
    node = SimpleNamespace(
        token_ids=[1, 2],
        mask=[False, True],
        sampled=True,
        message=AssistantMessage(content="42"),
    )
    span = SimpleNamespace(duration=0.0)
    trace = SimpleNamespace(
        id=trace_id,
        agent=SimpleNamespace(trainable=True),
        ok=ok,
        is_truncated=False,
        stop_condition="completed" if ok else "error",
        reward=1.0,
        nodes=[node] if ok else [],
        branches=(
            [SimpleNamespace(nodes=[node], token_ids=[1, 2], logprobs=[0.0, -0.1])]
            if ok
            else []
        ),
        task=SimpleNamespace(key="task-0"),
        errors=[] if ok else [Error(type="SandboxError", message="HTTP 429")],
        timing=SimpleNamespace(
            setup=span,
            agent=SimpleNamespace(duration=0.0, model=span, harness=span),
            scoring=span,
        ),
        calls=[],
    )
    return SimpleNamespace(ok=ok, errors=[], traces=[trace])


def _rollouter_with_provider(monkeypatch, provider, limiter_config):
    """Build a Verifiers rollouter whose env client is `provider`, plus one sample."""
    monkeypatch.setattr(
        data,
        "_load_math_dataset",
        lambda name: (iter([DapoMathSample(prompt="1+1?", ground_truth="2")]), 1),
    )
    config = replace(
        _verifiers_math_rollouter_config(max_rollout_tokens=64),
        sandbox_creation_limiter=limiter_config,
    )
    rollouter = config.build()
    rollouter._generation_server = SimpleNamespace(
        model_id="torchtitan",
        set_generate_fn=lambda generate_fn: None,
        pop_generation_metadata=lambda trace_id: VerifiersGenerationMetadata(
            min_policy_version=0, max_policy_version=0, metrics=[]
        ),
    )
    rollouter._verifiers_env_client = provider
    rollouter._verifiers_train_client_config = object()
    _, sample = rollouter.get_training_sample()
    return rollouter, sample


def _run_group(rollouter, sample, group_size: int):
    return asyncio.run(
        rollouter.run_group_rollouts(
            generate_fn=None,
            sample=sample,
            group_id=0,
            group_size=group_size,
            sampling=SamplingConfig(),
        )
    )


@pytest.mark.parametrize(
    "limiter_config, expected_rate_limited",
    [
        # Six sandboxes requested at once: the provider rate-limits all but the first.
        (None, 5),
        # Paced to one create per 0.1 s, twice the provider's minimum gap.
        (SandboxCreationLimiter.Config(creates_per_minute=600), 0),
    ],
)
def test_limiter_keeps_sandbox_creation_under_the_provider_rate(
    monkeypatch, limiter_config, expected_rate_limited
) -> None:
    provider = FakeSandboxProvider(min_gap_sec=0.05)
    rollouter, sample = _rollouter_with_provider(monkeypatch, provider, limiter_config)

    group = _run_group(rollouter, sample, group_size=6)

    statuses = [rollout.status for rollout in group.rollouts]
    assert statuses.count(RolloutStatus.ERROR) == expected_rate_limited


def test_rollout_that_waits_too_long_errors_without_a_sandbox(monkeypatch) -> None:
    provider = FakeSandboxProvider(min_gap_sec=0.0)
    rollouter, sample = _rollouter_with_provider(
        monkeypatch,
        provider,
        # One create per second: the second rollout would wait 1 s, past its 0.2 s budget.
        SandboxCreationLimiter.Config(creates_per_minute=60, max_wait_sec=0.2),
    )

    group = _run_group(rollouter, sample, group_size=2)

    first, second = group.rollouts
    assert first.status == RolloutStatus.COMPLETED
    assert second.status == RolloutStatus.ERROR
    assert second.turns == []
    assert "sandbox creation slot" in second.logs["errors"][0]["message"]
    assert len(provider.create_times) == 1
    wait_metrics = [
        metric for metric in first.turns[0].metrics if metric.key == "sandbox/wait_sec"
    ]
    assert len(wait_metrics) == 2


def test_timed_out_rollout_hands_its_slot_to_the_next() -> None:
    async def run_test() -> dict[str, tuple[str, float]]:
        # Slots are 0.2 s apart and a rollout may wait at most 0.1 s.
        limiter = SandboxCreationLimiter.Config(
            creates_per_minute=300, max_wait_sec=0.1
        ).build()
        loop = asyncio.get_running_loop()
        start = loop.time()
        outcomes = {}

        async def rollout(name: str, arrive_sec: float) -> None:
            await asyncio.sleep(arrive_sec)
            try:
                await limiter.wait_for_slot()
                outcomes[name] = ("started", loop.time() - start)
            except TimeoutError:
                outcomes[name] = ("timed out", loop.time() - start)

        await asyncio.gather(rollout("A", 0.0), rollout("B", 0.0), rollout("C", 0.15))
        return outcomes

    outcomes = asyncio.run(run_test())
    assert outcomes["A"][0] == "started" and outcomes["A"][1] < 0.1
    # B waits for the 0.2 s slot and gives up at 0.1 s.
    assert outcomes["B"][0] == "timed out" and 0.09 <= outcomes["B"][1] < 0.2
    # C gets that slot instead of the next one at 0.4 s, which is past its own budget.
    assert outcomes["C"][0] == "started" and 0.19 <= outcomes["C"][1] < 0.35


def test_cancelled_rollouts_leave_no_backlog() -> None:
    """Shutdown cancels in-flight groups, then the final validation pass starts rollouts."""

    async def run_test() -> float:
        # Slots are 0.2 s apart: four queued rollouts hold 0.8 s of slots until cancelled.
        limiter = SandboxCreationLimiter.Config(
            creates_per_minute=300, max_wait_sec=0.5
        ).build()
        loop = asyncio.get_running_loop()
        start = loop.time()
        await limiter.wait_for_slot()
        queued = [asyncio.create_task(limiter.wait_for_slot()) for _ in range(4)]
        await asyncio.sleep(0.05)
        for task in queued:
            task.cancel()
        await asyncio.gather(*queued, return_exceptions=True)
        await limiter.wait_for_slot()
        return loop.time() - start

    # The next rollout takes the 0.2 s slot, not one after the cancelled rollouts' slots.
    assert 0.19 <= asyncio.run(run_test()) < 0.35
