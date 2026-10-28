# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the repeated-tool-call stops and the rollout status they map to."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf

from torchtitan.rl.examples.verifiers.dapo_math import data
from torchtitan.rl.examples.verifiers.rollouter import VerifiersRolloutWorker
from torchtitan.rl.examples.verifiers.stops import (
    repeated_call_in_turn,
    REPEATED_CALL_IN_TURN_STOP,
    repeated_tool_calls,
    REPEATED_TOOL_CALLS_STOP,
    REPETITION_STOPS,
)
from torchtitan.rl.rollout.types import RolloutStatus
from verifiers.v1.configs.task import DecoratedFunctionConfig
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import AgentInfo, MessageNode, Trace, TraceTask
from verifiers.v1.types import AssistantMessage, ToolCall


def _trace(*responses: list[str] | None) -> Trace:
    """A trace with one sampled response per entry: a list of bash commands, one tool call
    each, or None for a response without tool calls."""
    trace = Trace(
        task=TraceTask(
            type="VerifiersMathTask",
            data=data.VerifiersMathData(idx=0, prompt="p", ground_truth="1"),
        ),
        agent=AgentInfo(config=vf.AgentConfig()),
    )
    for turn, commands in enumerate(responses):
        tool_calls = (
            None
            if commands is None
            else [
                ToolCall(
                    id=f"call_{turn}_{i}",
                    name="bash",
                    arguments=f'{{"command": "{command}"}}',
                )
                for i, command in enumerate(commands)
            ]
        )
        trace.nodes.append(
            MessageNode(
                message=AssistantMessage(reasoning_content="r", tool_calls=tool_calls),
                sampled=True,
            )
        )
    return trace


def test_repeated_tool_calls_fires_at_five_identical_responses() -> None:
    assert not repeated_tool_calls(_trace(["ls"], *[["pytest -x"]] * 4))
    assert not repeated_tool_calls(_trace(*[["pytest -x"]] * 4))
    assert repeated_tool_calls(_trace(["ls"], *[["pytest -x"]] * 5))
    # The whole response's calls must match, not just one of them.
    assert not repeated_tool_calls(_trace(*[["pytest -x"]] * 4, ["pytest -x", "ls"]))
    assert repeated_tool_calls(_trace(*[["pytest -x", "ls"]] * 5))


def test_repeated_tool_calls_ignores_whitespace_and_ids_but_not_order_or_gaps() -> None:
    assert repeated_tool_calls(
        _trace(
            ["pytest  -x"], ["pytest -x"], ["pytest -x"], ["pytest -x"], ["pytest -x"]
        )
    )
    assert not repeated_tool_calls(
        _trace(["pytest"], ["pytest"], ["ls"], ["pytest"], ["pytest"], ["pytest"])
    )
    # Responses without tool calls (e.g. a final answer) never trigger it.
    assert not repeated_tool_calls(_trace(*[None] * 5))


def test_repeated_call_in_turn_fires_at_ten_copies_in_the_last_response() -> None:
    assert repeated_call_in_turn(_trace(["ls"], ["cat a.py"] * 10))
    assert not repeated_call_in_turn(_trace(["ls"], ["cat a.py"] * 9))
    assert not repeated_call_in_turn(_trace(["ls"], [f"cat {i}.py" for i in range(10)]))
    # Only the last response counts.
    assert not repeated_call_in_turn(_trace(["cat a.py"] * 10, ["ls"]))
    assert not repeated_call_in_turn(_trace(None))


@pytest.mark.parametrize(
    "responses, stop",
    [
        ([["ls"], *[["pytest"]] * 5], REPEATED_TOOL_CALLS_STOP),
        ([["ls"], ["cat a.py"] * 10], REPEATED_CALL_IN_TURN_STOP),
        ([["ls"], ["pytest"], ["sed -i s/a/b/ f.py"], ["pytest"]], None),
    ],
)
def test_plugged_stops_end_the_session(responses, stop) -> None:
    """Config entries -> task hooks -> RolloutSession.refused(), as the interception
    server runs it before each model call."""
    config = data.VerifiersMathTask.config_type()(
        stops={
            name: DecoratedFunctionConfig(
                fn=f"torchtitan.rl.examples.verifiers.stops:{name}"
            )
            for name in REPETITION_STOPS
        }
    )
    task = data.VerifiersMathTask(
        data.VerifiersMathData(idx=0, prompt="p", ground_truth="1"), config
    )
    hooks = [fn for fn in task.hooks("stop") if fn.__name__ in REPETITION_STOPS]
    assert len(hooks) == 2

    trace = _trace(*responses)
    session = RolloutSession(ctx=None, trace=trace, trace_stops=hooks)
    assert asyncio.run(session.refused()) == stop
    assert trace.stop_condition == stop


@pytest.mark.parametrize("stop", REPETITION_STOPS)
def test_rollout_status_maps_repetition_stops_to_truncated(stop: str) -> None:
    trace = _trace(*[["pytest"]] * 5)
    trace.ok = True  # graded like a max_turns rollout
    trace.stop(stop)
    assert not trace.is_truncated  # Verifiers' own flag leaves custom stops out
    status = VerifiersRolloutWorker.rollout_status(
        verifiers_episode=SimpleNamespace(ok=True, is_timeout=False), trace=trace
    )
    assert status is RolloutStatus.TRUNCATED_REPETITION
    assert status.is_truncated() and not status.is_error()


def test_rollout_status_keeps_max_turns() -> None:
    trace = _trace(["ls"])
    trace.ok = True
    trace.stop("max_turns")
    status = VerifiersRolloutWorker.rollout_status(
        verifiers_episode=SimpleNamespace(ok=True, is_timeout=False), trace=trace
    )
    assert status is RolloutStatus.TRUNCATED_MAX_TURNS
