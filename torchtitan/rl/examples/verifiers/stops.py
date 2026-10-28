# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verifiers ``@stop`` hooks that end a rollout which keeps repeating a tool call.

Plug them into a verifiers task through its config, by import path; the entry's name
becomes the trace's stop condition:

    task_config.stops = {
        name: DecoratedFunctionConfig(fn=f"torchtitan.rl.examples.verifiers.stops:{name}")
        for name in REPETITION_STOPS
    }

``VerifiersRolloutWorker.rollout_status`` maps both stop conditions to
``RolloutStatus.TRUNCATED_REPETITION``, which counts as truncated. Both hooks run before
each model call, so the repeated calls themselves still execute.
"""

from __future__ import annotations

from collections import Counter

from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage

REPEATED_TOOL_CALLS_STOP = "repeated_tool_calls"
"""Stop condition of `repeated_tool_calls`."""

REPEATED_CALL_IN_TURN_STOP = "repeated_call_in_turn"
"""Stop condition of `repeated_call_in_turn`."""

REPETITION_STOPS = (REPEATED_TOOL_CALLS_STOP, REPEATED_CALL_IN_TURN_STOP)

REPEATED_TURNS_LIMIT = 5
"""Consecutive model responses with identical tool calls that end the rollout."""

REPEATED_CALLS_PER_TURN_LIMIT = 10
"""Copies of one identical tool call within one model response that end the rollout."""


def _tool_calls(message: AssistantMessage) -> tuple[tuple[str, str], ...]:
    """The response's tool calls as (name, whitespace-normalized arguments); call ids
    differ on every turn, so they are left out.

    Example:
        bash {"command":  "ls"} -> (("bash", '{"command": "ls"}'),)
    """
    return tuple(
        (call.name, " ".join(call.arguments.split()))
        for call in message.tool_calls or ()
    )


def repeated_tool_calls(trace: Trace) -> bool:
    """True once the last 5 model responses all made the same tool calls.

    A response without tool calls never matches.

    Example:
        read a.py, read a.py, read a.py, read a.py, read a.py -> True
        read a.py x4, then read b.py                          -> False
    """
    responses = trace.assistant_messages[-REPEATED_TURNS_LIMIT:]
    if len(responses) < REPEATED_TURNS_LIMIT:
        return False
    first = _tool_calls(responses[0])
    return bool(first) and all(_tool_calls(m) == first for m in responses[1:])


def repeated_call_in_turn(trace: Trace) -> bool:
    """True once the last model response repeated one tool call 10 or more times.

    Example:
        one response with 10 x read {"filePath": "a.py"} -> True
        one response with 10 distinct reads             -> False
    """
    responses = trace.assistant_messages
    if not responses:
        return False
    counts = Counter(_tool_calls(responses[-1]))
    return bool(counts) and max(counts.values()) >= REPEATED_CALLS_PER_TURN_LIMIT
