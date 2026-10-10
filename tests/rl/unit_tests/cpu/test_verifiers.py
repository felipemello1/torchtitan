# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout integration."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

pytest.importorskip("verifiers")

from aiohttp import ClientSession

from torchtitan.rl.examples.verifiers.generation_server import (
    _parse_sampling_config,
    GenerationServer,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.examples.verifiers.rollouter import (
    _trainable_token_spans,
    log_failed_rollout,
    verifiers_rollout_logs,
    VerifiersRollouter,
)
from torchtitan.rl.generator import SamplingConfig
from torchtitan.rl.rollout.types import Rollout, RolloutStatus
from torchtitan.rl.types import Completion


def test_trainable_token_spans() -> None:
    assert _trainable_token_spans([False, True, True, False, True]) == [
        (1, 3),
        (4, 5),
    ]


def test_verifiers_trace_preserves_generation_metadata() -> None:
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    node = SimpleNamespace(
        token_ids=[10, 11, 12, 13],
        mask=[False, False, True, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="Answer: $42$"),
    )
    trace = SimpleNamespace(
        nodes=[node],
        branches=[
            SimpleNamespace(
                nodes=[node],
                token_ids=[10, 11, 12, 13],
                logprobs=[0.0, 0.0, -0.2, -0.3],
            )
        ],
    )
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=trace,
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=3,
            max_policy_version=4,
            metrics=[],
        ),
        group_id=5,
        rollout_id=2,
    )

    assert len(turns) == 1
    assert turns[0].prompt_prefix_len == 0
    assert turns[0].prompt_delta_token_ids == [10, 11]
    assert turns[0].completion_token_ids == [12, 13]
    assert turns[0].completion_logprobs == [-0.2, -0.3]
    assert turns[0].completion_message == {
        "role": "assistant",
        "content": "Answer: $42$",
    }
    assert turns[0].min_policy_version == 3
    assert turns[0].max_policy_version == 4


def test_verifiers_multiturn_trace_matches_titanrl_rollout_structure() -> None:
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    first_node = SimpleNamespace(
        token_ids=[10, 11],
        mask=[False, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="first"),
    )
    second_node = SimpleNamespace(
        token_ids=[12, 13],
        mask=[False, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="second"),
    )
    trace = SimpleNamespace(
        nodes=[first_node, second_node],
        branches=[
            SimpleNamespace(
                nodes=[first_node, second_node],
                token_ids=[10, 11, 12, 13],
                logprobs=[0.0, -0.1, 0.0, -0.2],
            )
        ],
    )
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=trace,
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=3,
            max_policy_version=8,
            metrics=[],
        ),
        group_id=5,
        rollout_id=2,
    )

    assert [turn.min_policy_version for turn in turns] == [3, 3]
    assert [turn.max_policy_version for turn in turns] == [8, 8]
    # turn 1's prompt [10, 11, 12] continues turn 0's [10] + [11], so it stores only [12]
    stored = [(turn.prompt_prefix_len, turn.prompt_delta_token_ids) for turn in turns]
    assert stored == [(0, [10]), (2, [12])]
    assert [turn.completion_token_ids for turn in turns] == [[11], [13]]
    assert [turn.completion_logprobs for turn in turns] == [[-0.1], [-0.2]]


def test_verifiers_second_branch_stores_its_prompt_as_a_delta_on_the_first() -> None:
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    def node(token_ids: list[int], content: str) -> SimpleNamespace:
        message = VerifiersAssistantMessage(content=content)
        return SimpleNamespace(
            token_ids=token_ids, mask=[False, True], sampled=True, message=message
        )

    shared = node([10, 11], "a")
    first_leaf = node([12, 13], "b")
    second_leaf = node([14, 15], "c")
    trace = SimpleNamespace(
        nodes=[shared, first_leaf, second_leaf],
        branches=[
            SimpleNamespace(
                nodes=[shared, first_leaf],
                token_ids=[10, 11, 12, 13],
                logprobs=[0.0, -0.1, 0.0, -0.2],
            ),
            SimpleNamespace(
                nodes=[shared, second_leaf],
                token_ids=[10, 11, 14, 15],
                logprobs=[0.0, -0.1, 0.0, -0.3],
            ),
        ],
    )
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=trace,
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=1, max_policy_version=1, metrics=[]
        ),
        group_id=5,
        rollout_id=2,
    )

    # The shared node trains once; the second branch's prompt [10, 11, 14] keeps [10, 11] of the
    # previous turn's [10, 11, 12, 13], so it stores (2, [14]) and the builder opens a new sample.
    stored = [(turn.prompt_prefix_len, turn.prompt_delta_token_ids) for turn in turns]
    assert stored == [(0, [10]), (2, [12]), (2, [14])]
    assert [turn.completion_token_ids for turn in turns] == [[11], [13], [15]]


def _two_turn_trace():
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    first_node = SimpleNamespace(
        token_ids=[10, 11],
        mask=[False, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="first"),
    )
    second_node = SimpleNamespace(
        token_ids=[12, 13, 14],
        mask=[False, True, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="second"),
    )
    return SimpleNamespace(
        nodes=[first_node, second_node],
        branches=[
            SimpleNamespace(
                nodes=[first_node, second_node],
                token_ids=[10, 11, 12, 13, 14],
                logprobs=[0.0, -0.1, 0.0, -0.2, -0.3],
            )
        ],
    )


def test_verifiers_trace_attaches_each_generations_topk_rows() -> None:
    # Generations are keyed by (prompt length, completion tokens): prompts [10] and
    # [10, 11, 12] sampled completions [11] and [13, 14].
    topk_by_generation = {
        (1, (11,)): (torch.tensor([[11, 7]]), torch.tensor([[-0.1, -2.0]])),
        (3, (13, 14)): (
            torch.tensor([[13, 8], [14, 9]]),
            torch.tensor([[-0.2, -1.5], [-0.3, -1.2]]),
        ),
    }
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=_two_turn_trace(),
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=3,
            max_policy_version=4,
            metrics=[],
            topk_by_generation=topk_by_generation,
        ),
        group_id=5,
        rollout_id=2,
    )

    assert [turn.completion_token_ids for turn in turns] == [[11], [13, 14]]
    assert [turn.completion_topk_token_ids.tolist() for turn in turns] == [
        [[11, 7]],
        [[13, 8], [14, 9]],
    ]
    torch.testing.assert_close(
        turns[1].completion_topk_logprobs, torch.tensor([[-0.2, -1.5], [-0.3, -1.2]])
    )


def test_verifiers_trace_rejects_a_node_without_its_topk_rows() -> None:
    # Top-k was requested, but no generation produced the second node's tokens.
    topk_by_generation = {
        (1, (11,)): (torch.tensor([[11, 7]]), torch.tensor([[-0.1, -2.0]])),
    }
    with pytest.raises(ValueError, match="top-k logprobs are unknown"):
        VerifiersRollouter.trace_to_rollout_turns(
            trace=_two_turn_trace(),
            generation_metadata=VerifiersGenerationMetadata(
                min_policy_version=3,
                max_policy_version=4,
                metrics=[],
                topk_by_generation=topk_by_generation,
            ),
            group_id=5,
            rollout_id=2,
        )


def test_verifiers_trace_attaches_env_replies_to_the_preceding_turn() -> None:
    from verifiers.v1.types import (
        AssistantMessage as VerifiersAssistantMessage,
        UserMessage as VerifiersUserMessage,
    )

    def node(content: str, sampled: bool) -> SimpleNamespace:
        message = (
            VerifiersAssistantMessage(content=content)
            if sampled
            else VerifiersUserMessage(content=content)
        )
        return SimpleNamespace(
            token_ids=[0], mask=[sampled], sampled=sampled, message=message
        )

    # Two branches share the prompt and the first command, then diverge.
    task, first, out, second, other_out, other = (
        node("task", False),
        node("ls", True),
        node("a.txt", False),
        node("cat a.txt", True),
        node("No such file", False),
        node("pwd", True),
    )
    trace = SimpleNamespace(
        nodes=[task, first, out, second, other_out, other],
        branches=[
            SimpleNamespace(
                nodes=branch_nodes,
                token_ids=[0] * len(branch_nodes),
                logprobs=[0.0] * len(branch_nodes),
            )
            for branch_nodes in (
                [task, first, out, second],
                [task, first, other_out, other],
            )
        ],
    )
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=trace,
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=0, max_policy_version=0, metrics=[]
        ),
        group_id=0,
        rollout_id=0,
    )

    assert [turn.completion_message["content"] for turn in turns] == [
        "ls",
        "cat a.txt",
        "pwd",
    ]
    assert [turn.env_messages for turn in turns] == [
        [
            {"role": "user", "content": "a.txt"},
            {"role": "user", "content": "No such file"},
        ],
        [],
        [],
    ]


def test_verifiers_rollout_logs_keep_the_failure_reason(caplog) -> None:
    from verifiers.v1.trace import Error

    def span(seconds: float) -> SimpleNamespace:
        return SimpleNamespace(duration=seconds)

    trace = SimpleNamespace(
        id="trace-1",
        task=SimpleNamespace(key="task-1"),
        stop_condition="error",
        errors=[
            Error(
                type="HarnessError",
                message="agent timeout",
                traceback="Traceback\n  ...\nTimeoutError",
            )
        ],
        timing=SimpleNamespace(
            setup=span(5.0),
            agent=SimpleNamespace(
                duration=7200.0, model=span(6900.0), harness=span(300.0)
            ),
            scoring=span(0.0),
        ),
        calls=[
            SimpleNamespace(time=span(1801.0), error=None),
            SimpleNamespace(
                time=span(2.0), error=Error(type="APIError", message="502")
            ),
        ],
    )
    logs = verifiers_rollout_logs(SimpleNamespace(errors=[]), trace)

    assert logs["errors"] == [
        {
            "type": "HarnessError",
            "message": "agent timeout",
            "status_code": None,
            "traceback": "Traceback\n  ...\nTimeoutError",
        }
    ]
    assert (logs["agent_sec"], logs["model_sec"], logs["harness_sec"]) == (
        7200.0,
        6900.0,
        300.0,
    )
    assert logs["model_calls"] == 2
    assert logs["failed_model_calls"] == 1
    assert logs["slowest_model_call_sec"] == 1801.0

    with caplog.at_level(logging.WARNING):
        log_failed_rollout(logs, group_id=3, rollout_id=1)
        log_failed_rollout({**logs, "errors": []}, group_id=3, rollout_id=2)
    assert "error=HarnessError: agent timeout" in caplog.text
    assert "error=None: None" in caplog.text


def test_generation_server_forwards_token_request() -> None:
    async def run_test() -> None:
        received = []

        async def generate_fn(
            prompt_token_ids,
            *,
            request_id,
            group_id,
            routing_session_id=None,
            sampling_config=None,
        ):
            received.append(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "request_id": request_id,
                    "group_id": group_id,
                    "routing_session_id": routing_session_id,
                    "sampling_config": sampling_config,
                }
            )
            request_index = int(request_id.rsplit("=", 1)[1])
            return Completion(
                min_policy_version=7 - request_index,
                max_policy_version=8 + request_index,
                request_id=request_id,
                token_ids=[31, 32],
                token_logprobs=[-0.1, -0.2],
                finish_reason="stop",
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.generate_fns[1] = generate_fn
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.get(f"{server.base_url}/models")
                assert response.status == 200
                assert await response.json() == {
                    "object": "list",
                    "data": [
                        {
                            "id": "torchtitan",
                            "object": "model",
                            "created": 0,
                            "owned_by": "torchtitan",
                            "max_model_len": 40960,
                        }
                    ],
                }
                for _ in range(2):
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": "group=1/rollout=2"},
                        json={
                            "token_ids": [10, 11],
                            "sampling_params": {
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "max_tokens": 2,
                                "seed": 4,
                                "logprobs": 1,
                                "torchtitan_group_id": 1,
                                "stop_token_ids": [99],
                            },
                        },
                    )
                    assert response.status == 200
                    payload = await response.json()
            generation_metadata = server.pop_generation_metadata("group=1/rollout=2")
        finally:
            await server.close()

        assert [request["request_id"] for request in received] == [
            "group=1/rollout=2/request=0",
            "group=1/rollout=2/request=1",
        ]
        assert all(request["prompt_token_ids"] == [10, 11] for request in received)
        assert all(request["group_id"] == 1 for request in received)
        assert all(
            request["routing_session_id"] == "group=1/rollout=2" for request in received
        )
        assert all(request["sampling_config"].seed == 4 for request in received)
        assert payload["choices"][0]["token_ids"] == [31, 32]
        assert generation_metadata is not None
        assert generation_metadata.min_policy_version == 6
        assert generation_metadata.max_policy_version == 9

    asyncio.run(run_test())


def test_generation_server_records_topk_logprobs_per_generation() -> None:
    async def run_test() -> None:
        received = []

        async def generate_fn(prompt_token_ids, *, request_id, group_id, **kwargs):
            received.append(kwargs["sampling_config"])
            return Completion(
                min_policy_version=7,
                max_policy_version=7,
                request_id=request_id,
                token_ids=[31, 32],
                token_logprobs=[-0.1, -0.2],
                topk_token_ids=torch.tensor([[31, 5], [32, 6]], dtype=torch.int32),
                topk_logprobs=torch.tensor([[-0.1, -2.4], [-0.2, -1.9]]),
                finish_reason="stop",
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.generate_fns[1] = generate_fn
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11],
                        "sampling_params": {
                            "max_tokens": 2,
                            "torchtitan_group_id": 1,
                            "stop_token_ids": [99],
                            "num_topk_logprobs": 2,
                        },
                    },
                )
                assert response.status == 200
            generation_metadata = server.pop_generation_metadata("group=1/rollout=2")
        finally:
            await server.close()

        assert received[0].num_topk_logprobs == 2
        [
            (key, (topk_token_ids, topk_logprobs))
        ] = generation_metadata.topk_by_generation.items()
        assert key == (2, (31, 32))
        assert topk_token_ids.tolist() == [[31, 5], [32, 6]]
        torch.testing.assert_close(
            topk_logprobs, torch.tensor([[-0.1, -2.4], [-0.2, -1.9]])
        )

    asyncio.run(run_test())


def test_generation_server_rejects_aborted_generation() -> None:
    async def run_test() -> None:
        async def generate_fn(
            prompt_token_ids,
            *,
            request_id,
            group_id,
            routing_session_id=None,
            sampling_config=None,
        ):
            return Completion(
                min_policy_version=7,
                max_policy_version=7,
                request_id=request_id,
                token_ids=[],
                token_logprobs=[],
                finish_reason="abort",
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.generate_fns[1] = generate_fn
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11],
                        "sampling_params": {
                            "torchtitan_group_id": 1,
                            "stop_token_ids": [99],
                        },
                    },
                )
                assert response.status == 502
                payload = await response.json()
            generation_metadata = server.pop_generation_metadata("group=1/rollout=2")
        finally:
            await server.close()

        assert payload == {
            "error": "generation finished without a usable completion: abort"
        }
        assert generation_metadata is None

    asyncio.run(run_test())


def test_generation_server_requires_group_id() -> None:
    async def run_test() -> None:
        async def generate_fn(*args, **kwargs):
            raise AssertionError("generate_fn must not run without a group id")

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.generate_fns[1] = generate_fn
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={"token_ids": [10, 11], "sampling_params": {}},
                )
                assert response.status == 400
                payload = await response.json()
        finally:
            await server.close()

        assert "torchtitan_group_id" in payload["error"]

    asyncio.run(run_test())


def test_generation_server_uses_each_groups_generate_fn() -> None:
    """Each request goes to its own group's `GenerateFn`, even after another group registers
    one: a validation group (-1) and a training group (7) run at once."""

    async def run_test() -> None:
        received: list[tuple[str, int]] = []

        def make_generate_fn(name: str):
            async def generate_fn(prompt_token_ids, *, request_id, group_id, **kwargs):
                received.append((name, group_id))
                return Completion(
                    min_policy_version=0,
                    max_policy_version=0,
                    request_id=request_id,
                    token_ids=[31],
                    token_logprobs=[-0.1],
                    finish_reason="stop",
                )

            return generate_fn

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        await server.start()
        try:
            async with ClientSession() as session:

                async def generate(group_id: int) -> int:
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": f"group={group_id}/rollout=0"},
                        json={
                            "token_ids": [10, 11],
                            "sampling_params": {
                                "torchtitan_group_id": group_id,
                                "stop_token_ids": [99],
                            },
                        },
                    )
                    return response.status

                server.generate_fns[-1] = make_generate_fn("validation")
                assert await generate(-1) == 200
                server.generate_fns[7] = make_generate_fn("training")
                assert await generate(-1) == 200
                assert await generate(7) == 200
                # A group with no registered function gets an error, not another group's.
                assert await generate(3) == 503
        finally:
            await server.close()

        assert received == [("validation", -1), ("validation", -1), ("training", 7)]

    asyncio.run(run_test())


def test_rollouter_registers_the_groups_generate_fn_while_it_runs() -> None:
    """`run_group_rollouts` keeps its group's `GenerateFn` registered for the group's rollouts,
    while another group starts, and removes it after."""

    async def run_test() -> None:
        rollouter = object.__new__(VerifiersRollouter)
        rollouter._thinking_budget = None
        rollouter._generation_server = GenerationServer.Config(
            max_rollout_tokens=40960
        ).build()
        rollouter._rubric = SimpleNamespace(
            score_group=AsyncMock(
                return_value=[SimpleNamespace(reward=1.0, reward_breakdown={})]
            )
        )
        rollouter._advantage_estimator = lambda group: [0.0]
        validation_rollout_can_end = asyncio.Event()
        seen: list[tuple[int, str]] = []

        async def run_single_rollout(*, sample, sampling, group_id, rollout_id):
            seen.append((group_id, rollouter._generation_server.generate_fns[group_id]))
            if group_id == -1:
                await validation_rollout_can_end.wait()
                seen.append(
                    (group_id, rollouter._generation_server.generate_fns[group_id])
                )
            return Rollout(
                group_id=group_id, rollout_id=rollout_id, status=RolloutStatus.COMPLETED
            )

        rollouter._run_single_rollout = run_single_rollout

        def run_group(group_id: int, generate_fn: str):
            return rollouter.run_group_rollouts(
                generate_fn=generate_fn,
                sample=None,
                group_id=group_id,
                group_size=1,
                sampling=SamplingConfig(),
            )

        validation_group = asyncio.create_task(run_group(-1, "validation_fn"))
        await asyncio.sleep(0)
        await run_group(7, "training_fn")
        validation_rollout_can_end.set()
        await validation_group

        assert seen == [
            (-1, "validation_fn"),
            (7, "training_fn"),
            (-1, "validation_fn"),
        ]
        assert rollouter._generation_server.generate_fns == {}

    asyncio.run(run_test())


def test_rollouter_removes_the_groups_generate_fn_when_a_rollout_raises() -> None:
    rollouter = object.__new__(VerifiersRollouter)
    rollouter._thinking_budget = None
    rollouter._generation_server = GenerationServer.Config(
        max_rollout_tokens=40960
    ).build()

    async def run_single_rollout(*, sample, sampling, group_id, rollout_id):
        raise RuntimeError("sandbox lost")

    rollouter._run_single_rollout = run_single_rollout

    with pytest.raises(RuntimeError, match="sandbox lost"):
        asyncio.run(
            rollouter.run_group_rollouts(
                generate_fn="training_fn",
                sample=None,
                group_id=7,
                group_size=2,
                sampling=SamplingConfig(),
            )
        )
    assert rollouter._generation_server.generate_fns == {}


def test_parse_sampling_config_requires_stop_token_ids() -> None:
    with pytest.raises(ValueError, match="stop_token_ids"):
        _parse_sampling_config({"temperature": 1.0})


def test_forced_turns_reach_verifiers_client_and_keep_their_loss_mask() -> None:
    """Completions with appended tokens go through Verifiers' own client and trace graph, then
    back to turns whose appended tokens are out of the loss, including two identical turns."""
    from openai import AsyncOpenAI
    from renderers.client import generate

    from verifiers.v1.clients.train import response_from_generate
    from verifiers.v1.graph import prepare_turn
    from verifiers.v1.trace import AgentInfo, Trace, TraceTask
    from verifiers.v1.types import UserMessage

    completion_ids = [31, 32, 33, 34]
    loss_mask = [True, False, False, True]  # 32 and 33 were appended

    class _Renderer:
        def get_stop_token_ids(self) -> list[int]:
            return [99]

        def parse_response(self, token_ids, tools=None):
            return SimpleNamespace(content="", reasoning_content=None, tool_calls=[])

    async def generate_fn(prompt_token_ids, **kwargs):
        return Completion(
            min_policy_version=3,
            max_policy_version=3,
            request_id=kwargs["request_id"],
            token_ids=completion_ids,
            token_logprobs=[-0.1, math.nan, math.nan, -0.4],
            loss_mask=loss_mask,
            finish_reason="stop",
        )

    trace = Trace(task=TraceTask(type="task", data={}), agent=AgentInfo(config={}))

    async def run_test():
        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.generate_fns[1] = generate_fn
        await server.start()
        try:
            prompt, prompt_ids = [UserMessage(content="q")], [10, 11]
            for _ in range(2):  # two forced turns with identical completions
                turn = prepare_turn(trace, prompt)
                reply = await generate(
                    client=AsyncOpenAI(base_url=server.base_url, api_key="EMPTY"),
                    renderer=_Renderer(),
                    messages=[],
                    model=server.model_id,
                    prompt_ids=prompt_ids,
                    sampling_params={"max_tokens": 8, "torchtitan_group_id": 1},
                    extra_headers={"X-Session-ID": trace.id},
                )
                # Verifiers' client rejects NaN, so the reply carries 0.0 on appended tokens.
                assert reply["completion_logprobs"] == [-0.1, 0.0, 0.0, -0.4]
                response = response_from_generate(reply, model=server.model_id)
                turn.commit(response)
                prompt = [*prompt, response.message, UserMessage(content="tool")]
                prompt_ids = [*prompt_ids, *completion_ids, 12]
            return server.pop_generation_metadata(trace.id)
        finally:
            await server.close()

    generation_metadata = asyncio.run(run_test())
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=trace,
        generation_metadata=generation_metadata,
        group_id=1,
        rollout_id=0,
    )
    assert len(turns) == 2
    for turn in turns:
        assert turn.completion_token_ids == completion_ids
        assert turn.completion_loss_mask == loss_mask
        assert [math.isnan(logprob) for logprob in turn.completion_logprobs] == [
            False,
            True,
            True,
            False,
        ]


def test_terminus_max_tokens_turn_stays_on_its_branch() -> None:
    """Terminus-2 re-sends a max_tokens turn without its reasoning; the plugin restores it so
    the next prompt still bridges from the sampled tokens instead of forking a new branch."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins.terminal_bench_sandoq import (
        restore_sampled_reasoning,
    )
    from verifiers.v1 import graph
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.dialects.chat import parse_message
    from verifiers.v1.trace import AgentInfo, Trace, TraceTask
    from verifiers.v1.types import AssistantMessage, Response, TurnTokens

    trace = Trace(
        task=TraceTask(type="Task", data={}), agent=AgentInfo(config=AgentConfig())
    )
    task = {"role": "user", "content": "Write a.py"}
    # Hit max_tokens after </think>: reasoning is set, content is partial JSON.
    truncated = AssistantMessage(
        content='{"analysis": "Wri', reasoning_content="I will write it."
    )
    graph.prepare_turn(trace, [parse_message(task)]).commit(
        Response(
            id="r0",
            created=0,
            model="torchtitan",
            message=truncated,
            finish_reason="length",
            tokens=TurnTokens(
                prompt_ids=[1, 2], completion_ids=[3, 4], completion_logprobs=[0.0] * 2
            ),
        )
    )
    # Terminus-2's history after its max_tokens re-prompt (terminus_2.py:1142-1143).
    history = [
        task,
        {"role": "assistant", "content": '{"analysis": "Wri'},
        {
            "role": "user",
            "content": "ERROR!! NONE of the actions you just requested ...",
        },
    ]

    def bridges(messages: list[dict]) -> bool:
        prompt = [parse_message(message) for message in messages]
        return graph.prepare_turn(trace, prompt).previous_token_ids() is not None

    assert not bridges(history)
    assert bridges(restore_sampled_reasoning(history, trace))


def test_terminus_turns_with_the_same_content_stay_on_one_branch() -> None:
    """Two sampled turns with content "" but different reasoning (a stray </think> with thinking
    off) each get their own reasoning back, so the next prompt bridges instead of forking."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins.terminal_bench_sandoq import (
        restore_sampled_reasoning,
    )
    from verifiers.v1 import graph
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.dialects.chat import parse_message
    from verifiers.v1.trace import AgentInfo, Trace, TraceTask
    from verifiers.v1.types import AssistantMessage, Response, TurnTokens

    trace = Trace(
        task=TraceTask(type="Task", data={}), agent=AgentInfo(config=AgentConfig())
    )
    task = {"role": "user", "content": "Write a.py"}
    reply = {"role": "user", "content": "New Terminal Output: $"}
    first = {"role": "assistant", "content": "", "reasoning_content": "junk A"}
    second = {"role": "assistant", "content": "", "reasoning_content": "junk B"}

    def commit(messages: list[dict], sampled: dict, prompt_ids: list[int]) -> None:
        prompt = [parse_message(message) for message in messages]
        graph.prepare_turn(trace, prompt).commit(
            Response(
                id=str(len(prompt_ids)),
                created=0,
                model="torchtitan",
                message=AssistantMessage(
                    content="", reasoning_content=sampled["reasoning_content"]
                ),
                finish_reason="stop",
                tokens=TurnTokens(
                    prompt_ids=prompt_ids, completion_ids=[9], completion_logprobs=[0.0]
                ),
            )
        )

    commit([task], first, [1, 2])
    commit([task, first, reply], second, [1, 2, 9, 3])
    # Terminus-2 with interleaved thinking off re-sends both turns without their reasoning.
    resent = {"role": "assistant", "content": ""}
    history = [task, resent, reply, resent, reply]

    prompt = [
        parse_message(message) for message in restore_sampled_reasoning(history, trace)
    ]
    assert graph.prepare_turn(trace, prompt).previous_token_ids() is not None


def test_sandoq_rollout_log_explains_a_tmux_failure(tmp_path, caplog) -> None:
    """A rollout whose tmux never starts leaves a log with Terminus-2's traceback and log lines."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.errors import HarnessError
    from verifiers.v1.runtimes import ProgramResult

    class NoTmuxRuntime:
        async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
            return ProgramResult(exit_code=127, stdout="", stderr="tmux: not found")

    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    runtime = NoTmuxRuntime()

    async def rollout() -> None:
        with pytest.raises(HarnessError):
            await _launch(harness, trace, runtime)
        sandoq.harbor_logger.warning("logged outside any launch")
        await harness.cleanup(trace, runtime)
        # Verifiers' abort() after a cancelled close() calls cleanup again.
        await harness.cleanup(trace, runtime)

    asyncio.run(rollout())
    log = _read_rollout_log(tmp_path, trace)

    assert log["verifiers_trace_id"] == trace.id
    assert "Failed to start tmux session" in log["harness_stderr"]
    assert "ERROR harbor.utils.logger: Failed to install tmux" in log["terminus_log"]
    assert "outside any launch" not in log["terminus_log"]
    # The filter copies Harbor's records; they still reach the job log.
    assert "Failed to install tmux" in caplog.text
    assert log["trajectory"] is None
    assert log["tests"] is None
    assert log["pane"] is None


def test_sandoq_rollout_log_keeps_trajectory_and_test_output(
    tmp_path, monkeypatch
) -> None:
    """A finished rollout's log holds Terminus-2's trajectory, without the API key, and test.sh's
    capped output; the reward is the one the stock HarborTask reads."""
    pytest.importorskip("harbor")
    from harbor.models.trajectories import Step
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.runtimes import ProgramResult
    from verifiers.v1.tasksets.harbor import HarborTask

    sent_api_keys = []
    test_stdout = "apt-get update\n" + "x" * 100_000 + "\n1 failed"

    class Terminus2WithoutModel(_terminus2_without_model(sandoq)):
        async def run(self, instruction, environment, context) -> None:
            sent_api_keys.append(self._llm._build_base_kwargs()["api_key"])
            self._context = context
            self._trajectory_steps = [
                Step(
                    step_id=1,
                    timestamp="2026-10-07T12:00:00Z",
                    source="user",
                    message=instruction,
                )
            ]
            self._dump_trajectory()

    class VMRuntime:
        async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
            if argv == ["bash", "/tests/test.sh"]:
                return ProgramResult(exit_code=1, stdout=test_stdout, stderr="")
            return ProgramResult(exit_code=0, stdout="", stderr="")

        async def read(self, path: str, max_bytes: int) -> bytes:
            if path.endswith("reward.json"):
                raise OSError(path)
            return b"0"

    monkeypatch.setattr(sandoq, "Terminus2", Terminus2WithoutModel)
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    runtime = VMRuntime()
    data = trace.task.data

    async def rollout() -> tuple[ProgramResult, float, float]:
        result = await _launch(harness, trace, runtime)
        reward = await sandoq.SandoqHarborTask(data)._graded(runtime, trace)
        stock_reward = await HarborTask(data)._graded(runtime, trace)
        trace.record_reward("solved", reward)
        await harness.cleanup(trace, runtime)
        return result, reward, stock_reward

    result, reward, stock_reward = asyncio.run(rollout())
    log = _read_rollout_log(tmp_path, trace)

    # The env server builds tasks as the taskset's task type.
    assert sandoq.SandoqTerminalTaskset.task_type() is sandoq.SandoqHarborTask
    assert reward == stock_reward == 0.0
    assert result.exit_code == 0
    assert log["task_name"] == "allenai-tmax/task_000000_c19dda5b"
    assert log["harness_stderr"] == ""
    assert log["trajectory"]["steps"][0]["message"] == "Write a.py"
    assert log["trajectory"]["agent"]["extra"]["llm_kwargs"] == {
        "custom_llm_provider": "openai"
    }
    assert sent_api_keys == ["secret"]
    assert log["tests"]["exit_code"] == 1
    assert log["tests"]["stdout"].startswith("apt-get update\n")
    assert log["tests"]["stdout"].endswith("\n1 failed")
    assert "characters omitted" in log["tests"]["stdout"]


def test_sandoq_tests_get_requests_from_a_bundle(tmp_path, monkeypatch) -> None:
    """A test.sh that imports `requests` and installs nothing gets the packages the image lacks
    from a pure-Python bundle on its PYTHONPATH; one that pip-installs is left alone."""
    pytest.importorskip("harbor")
    import idna
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.runtimes import ProgramResult
    from verifiers.v1.tasksets.harbor.taskset import HarborData

    def task(name: str, test_sh: str) -> str:
        (tmp_path / name / "tests").mkdir(parents=True)
        (tmp_path / name / "tests" / "test.sh").write_text(test_sh)
        return str(tmp_path / name)

    heredoc = (
        "cat << 'EOF' > /tmp/t.py\nimport requests\nEOF\npython3 -m pytest /tmp/t.py\n"
    )
    tmax = task("tmax", heredoc)
    installs = task("tb21", "pip install requests==2.32.4\n" + heredoc)
    assert sandoq.tests_need_requests(tmax)
    assert not sandoq.tests_need_requests(installs)

    # The image: a python3 without site-packages, whose own PYTHONPATH has idna. Staging must
    # extract the other 4 packages, and test.sh must keep that PYTHONPATH.
    deps = tmp_path / "deps"
    monkeypatch.setattr(sandoq, "_TEST_PYTHON_DEPS", str(deps))
    (tmp_path / "image_site").mkdir()
    (tmp_path / "image_site" / "idna").symlink_to(Path(idna.__file__).parent)
    image_python = tmp_path / "bin" / "python3"
    image_python.parent.mkdir()
    image_python.write_text(f'#!/bin/sh\nexec {sys.executable} -S "$@"\n')
    image_python.chmod(0o755)
    image_env = {
        "PATH": f"{image_python.parent}:/usr/bin:/bin",
        "PYTHONPATH": str(tmp_path / "image_site"),
    }
    # Stands in for the pytest file test.sh writes, which starts with `import requests`.
    pytest_file = ["python3", "-c", "import requests; print(requests.__file__)"]

    class VMRuntime:
        async def write(self, path: str, data: bytes) -> None:
            if path.startswith(str(deps)):
                Path(path).write_bytes(data)

        async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
            if argv[-2:] == ["bash", "/tests/test.sh"]:
                argv = [*argv[:-2], *pytest_file]
            elif str(deps) not in argv[-1]:
                # Harbor's own staging of /tests.
                return ProgramResult(exit_code=0, stdout="", stderr="")
            done = subprocess.run(
                argv, env={**image_env, **env}, capture_output=True, text=True
            )
            return ProgramResult(
                exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr
            )

        async def read(self, path: str, max_bytes: int) -> bytes:
            if path.endswith("reward.json"):
                raise OSError(path)
            return b"1"

    runtime = VMRuntime()
    harbor_task = sandoq.SandoqHarborTask(HarborData(prompt="x", task_dir=tmax))
    trace = _sandoq_harness_and_trace(sandoq, tmp_path)[1]
    assert asyncio.run(harbor_task.solved(runtime, trace)) == 1.0
    extracted = sorted(path.name for path in deps.iterdir())
    assert extracted == ["certifi", "charset_normalizer", "requests", "urllib3"]
    assert not list(deps.rglob("*.so"))
    assert trace.info["tests"]["stdout"].startswith(str(deps)), trace.info["tests"]


def test_sandoq_task_scores_within_harbor_verifier_timeout(
    tmp_path, monkeypatch
) -> None:
    """Through the recipe's Sandoq config, scoring stops at the task's [verifier] timeout_sec,
    else Harbor's 600 s; the run-level rollout timeout is unchanged."""
    pytest.importorskip("harbor")
    import verifiers.v1.tasksets.harbor.taskset as harbor_taskset
    from torchtitan.rl.examples.verifiers.terminal_bench.prepare_tmax import task_toml
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq
    from torchtitan_recipes.rl.verifiers_terminal_bench import (
        _on_sandoq,
        _terminal_bench_rollouter_config,
    )
    from verifiers.v1.utils.loaders import load_taskset

    monkeypatch.setenv("VF_SANDBOX_PROVIDER", "oci-runner")
    monkeypatch.setenv("OCI_RUNNER_TASK_NETWORK", "host")
    monkeypatch.syspath_prepend(str(Path(terminal_bench_sandoq.__file__).parent))
    monkeypatch.setattr(harbor_taskset, "CACHE", tmp_path)
    declared = "[verifier]\ntimeout_sec = 1800.0\n"
    for name, verifier in [("undeclared", ""), ("declared", declared)]:
        task_dir = tmp_path / "train_1" / name
        (task_dir / "tests").mkdir(parents=True)
        (task_dir / "tests" / "test.sh").write_text("true\n")
        (task_dir / "instruction.md").write_text("x")
        (task_dir / "task.toml").write_text(
            task_toml(name, "ubuntu:22.04", 1, 2048, "/app") + verifier
        )

    rollouter = _on_sandoq(
        _terminal_bench_rollouter_config(
            "train@1",
            "validation@1",
            max_context_length=1024,
            max_turns=1,
            max_concurrent_rollouts=1,
            num_env_workers=1,
        ),
        interleaved_thinking=False,
        rollout_log_dir=str(tmp_path / "logs"),
    )
    agent = rollouter.verifiers_env_server.environment.agent
    scoring = {
        task.data.name: agent.timeout.scoring or task.data.timeout.scoring
        for task in load_taskset(
            rollouter.training_dataloader.dataset.verifiers_taskset
        )
    }

    assert scoring == {
        "allenai-tmax/declared": 1800.0,
        "allenai-tmax/undeclared": 600.0,
    }
    assert rollouter.validation_dataset.verifiers_taskset.ignore_timeouts is False
    assert agent.timeout.rollout == 7200


def test_sandoq_tmux_outlives_grading(tmp_path, monkeypatch) -> None:
    """launch leaves tmux running, so test.sh sees the agent's shell jobs as under `harbor run`;
    cleanup writes the log with the pane, then kills tmux once."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.runtimes import ProgramResult

    monkeypatch.setattr(sandoq, "Terminus2", _terminus2_without_model(sandoq))
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    pane = ProgramResult(exit_code=0, stdout="$ python3 server.py &\n", stderr="")
    runtime = _RecordingRuntime({"tail": pane})

    asyncio.run(_launch(harness, trace, runtime))
    assert not any("kill-server" in " ".join(argv) for argv, _ in runtime.calls)
    asyncio.run(harness.cleanup(trace, runtime))
    # Verifiers' abort() after a cancelled close() calls cleanup again.
    asyncio.run(harness.cleanup(trace, runtime))
    commands = [" ".join(argv) for argv, _ in runtime.calls]
    kills = [i for i, command in enumerate(commands) if "kill-server" in command]
    pane_read = next(
        i for i, command in enumerate(commands) if command.startswith("tail")
    )
    assert len(kills) == 1 and pane_read < kills[0]
    assert _read_rollout_log(tmp_path, trace)["pane"] == "$ python3 server.py &\n"


def test_sandoq_exec_keeps_keystrokes_out_of_argv() -> None:
    """Each exec ships its command in an env var, so an agent's `pkill -f` pattern can't match
    the exec that typed it; the command still runs as written."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq

    command = "tmux send-keys -t terminus-2 -- 'pkill -f server.py' Enter"
    runtime = _RecordingRuntime()
    environment = sandoq.RuntimeEnvironment(runtime, {"TMUX_TMPDIR": "/tmp/t"})
    asyncio.run(environment.exec(command, cwd="/app"))

    [(argv, env)] = runtime.calls
    assert "server.py" not in " ".join(argv)
    assert env == {"TMUX_TMPDIR": "/tmp/t", "TERMINUS_EXEC": f"cd /app && {command}"}
    # The wrapper runs a quoted, multi-line command unchanged.
    script = "cat <<'EOF'\nit's \"quoted\" $HOME\nEOF"
    output = subprocess.run(
        argv, env={"TERMINUS_EXEC": script}, capture_output=True, text=True
    ).stdout
    assert output == 'it\'s "quoted" $HOME\n'


def test_sandoq_exec_rejects_a_null_byte_like_harbor() -> None:
    """A command with a NUL byte raises ValueError, as Harbor's subprocess exec does, so the
    rollout fails as HarnessError instead of a retried SandboxError."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq

    runtime = _RecordingRuntime()
    environment = sandoq.RuntimeEnvironment(runtime, {})
    with pytest.raises(ValueError, match="embedded null byte"):
        asyncio.run(environment.exec("printf 'a\0b'"))
    assert runtime.calls == []


def test_sandoq_session_nests_a_shell(tmp_path, monkeypatch) -> None:
    """After tmux setup the agent's shell is a child shell, as under Harbor's recording, so its
    first `exit` returns to the outer shell instead of ending the session."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq

    sessions = []

    class Terminus2KeepingSession(_terminus2_without_model(sandoq)):
        async def run(self, instruction, environment, context) -> None:
            sessions.append(self._session)

    monkeypatch.setattr(sandoq, "Terminus2", Terminus2KeepingSession)
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    result = asyncio.run(_launch(harness, trace, _RecordingRuntime()))

    assert result.exit_code == 0
    assert sessions[0].keys == [["bash", "Enter"], ["clear", "Enter"]]


def test_sandoq_exec_failure_raises_sandbox_error(tmp_path, monkeypatch) -> None:
    """A lost exec channel leaves launch as a SandboxError, which the agent's retries can rerun
    on a fresh VM; the rollout log still keeps the traceback."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.errors import SandboxError

    class Terminus2LosingTheVM(_terminus2_without_model(sandoq)):
        async def run(self, instruction, environment, context) -> None:
            raise SandboxError("prime exec failed: uncertain transport failure")

    monkeypatch.setattr(sandoq, "Terminus2", Terminus2LosingTheVM)
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    runtime = _RecordingRuntime()
    with pytest.raises(SandboxError):
        asyncio.run(_launch(harness, trace, runtime))
    asyncio.run(harness.cleanup(trace, runtime))
    assert (
        "uncertain transport failure"
        in _read_rollout_log(tmp_path, trace)["harness_stderr"]
    )


def test_sandoq_dead_container_is_not_retried(tmp_path, monkeypatch) -> None:
    """A container the agent killed fails as HarnessError, which the SandboxError retry skips."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.errors import HarnessError

    class Terminus2KillingItsContainer(_terminus2_without_model(sandoq)):
        async def run(self, instruction, environment, context) -> None:
            # What Terminus-2 raises on its next keystroke after `pkill -9 -f sleep`.
            raise RuntimeError(
                "failed to send non-blocking keys: return_code=255, stderr='Error: can only "
                "create exec sessions on running containers'"
            )

    monkeypatch.setattr(sandoq, "Terminus2", Terminus2KillingItsContainer)
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    with pytest.raises(HarnessError, match="running containers"):
        asyncio.run(_launch(harness, trace, _RecordingRuntime()))


def test_sandoq_model_call_stops_with_the_rollout(tmp_path, monkeypatch) -> None:
    """Once Verifiers has stopped the rollout, the model call raises at once instead of letting
    Terminus-2 retry a request Verifiers will refuse."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.errors import HarnessError

    class Terminus2AtTheCap(_terminus2_without_model(sandoq)):
        async def run(self, instruction, environment, context) -> None:
            trace.stop_condition = "context_length"
            await self._llm.call(prompt="next turn", message_history=[])

    monkeypatch.setattr(sandoq, "Terminus2", Terminus2AtTheCap)
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    with pytest.raises(
        HarnessError,
        match="ContextLengthExceededError: Verifiers stopped the rollout: context_length",
    ):
        asyncio.run(_launch(harness, trace, _RecordingRuntime()))


class _RecordingRuntime:
    """Runtime that records each `run` and answers by the command's first word."""

    def __init__(self, results: dict | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self._results = results or {}

    async def run(self, argv: list[str], env: dict[str, str]):
        from verifiers.v1.runtimes import ProgramResult

        self.calls.append((argv, env))
        return self._results.get(
            argv[0], ProgramResult(exit_code=0, stdout="", stderr="")
        )


def _terminus2_without_model(sandoq):
    """Terminus-2 that runs no model and no tmux; its session records the keys it is sent."""

    class Session:
        def __init__(self) -> None:
            self.keys: list = []

        async def send_keys(self, keys, **kwargs) -> None:
            self.keys.append(keys)

    class Terminus2WithoutModel(sandoq.Terminus2):
        async def setup(self, environment) -> None:
            self._session = Session()

        async def run(self, instruction, environment, context) -> None:
            pass

    return Terminus2WithoutModel


def _sandoq_harness_and_trace(sandoq, log_dir):
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.tasksets.harbor.taskset import HarborData
    from verifiers.v1.trace import AgentInfo, Trace, TraceTask

    config = sandoq.StockTerminusOutsideConfig(
        id=sandoq.PLUGIN_ID, rollout_log_dir=str(log_dir)
    )
    data = HarborData(name="allenai-tmax/task_000000_c19dda5b", prompt="Write a.py")
    trace = Trace(
        task=TraceTask(type="Task", data=data), agent=AgentInfo(config=AgentConfig())
    )
    return sandoq.StockTerminusOutsideHarness(config), trace


async def _launch(harness, trace, runtime):
    from verifiers.v1.task import TaskData

    return await harness.launch(
        SimpleNamespace(model="policy"),
        trace,
        runtime,
        endpoint="http://127.0.0.1:1",
        secret="secret",
        mcp_urls={},
        data=TaskData(prompt="Write a.py"),
    )


def _read_rollout_log(log_dir, trace) -> dict:
    return json.loads(gzip.decompress((log_dir / f"{trace.id}.json.gz").read_bytes()))


def test_text_only_parse_keeps_tool_call_markup(monkeypatch) -> None:
    """With no tools declared, a reply wrapped in <tool_call> stays in the content, where
    Terminus-2 reads its JSON; declared tools use the stock parser."""
    pytest.importorskip("renderers")
    from renderers.qwen35 import Qwen35Renderer
    from torchtitan_recipes.rl.verifiers_plugins.terminal_bench_sandoq import (
        install_text_only_parse,
    )

    renderer = _fake_qwen35_renderer(monkeypatch)
    stock_parse_response = Qwen35Renderer.parse_response
    wrapped = [4, 7, 6, 7, 5, 0]  # <tool_call>\n{json}\n</tool_call><|im_end|>
    tools = [{"name": "bash", "parameters": {}}]
    assert stock_parse_response(renderer, wrapped, tools=None).content == ""

    install_text_only_parse()

    assert (
        renderer.parse_response(wrapped, tools=None).content
        == '<tool_call>\n{"analysis": "x"}\n</tool_call>'
    )
    assert renderer.parse_response(wrapped, tools=tools) == stock_parse_response(
        renderer, wrapped, tools=tools
    )


def test_text_only_parse_keeps_a_reply_before_a_stray_think_end(monkeypatch) -> None:
    """With thinking off, a reply that ends in a stray </think> reaches Terminus-2 as content
    instead of "", and an empty reply stays "" (not None); a thought before </think> and JSON
    after it still split, and thinking on keeps the stock split."""
    pytest.importorskip("renderers")
    from torchtitan_recipes.rl.verifiers_plugins.terminal_bench_sandoq import (
        install_text_only_parse,
    )

    renderer = _fake_qwen35_renderer(monkeypatch)
    install_text_only_parse()

    json_then_think_end = renderer.parse_response([6, 3, 0])
    assert json_then_think_end.content == '{"analysis": "x"}'
    assert json_then_think_end.reasoning_content is None
    assert renderer.parse_response([7, 7, 0]).content == ""
    thought_then_json = renderer.parse_response([8, 3, 6, 0])
    assert thought_then_json.content == '{"analysis": "x"}'
    assert thought_then_json.reasoning_content == "I will look."
    renderer.config.enable_thinking = True
    assert renderer.parse_response([6, 3, 0]).content == ""


def _fake_qwen35_renderer(monkeypatch):
    """Thinking-off Qwen3.5 renderer over a toy vocabulary; restores the class
    install_text_only_parse patches.

    Token ids: 0 <|im_end|>, 1 <|endoftext|>, 2 <think>, 3 </think>, 4 <tool_call>,
    5 </tool_call>, 6 '{"analysis": "x"}', 7 "\n", 8 "I will look."
    """
    from renderers.qwen35 import Qwen35Renderer

    vocab = ["<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>"]
    vocab += ["</tool_call>", '{"analysis": "x"}', "\n", "I will look."]

    class Tokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return "".join(vocab[i] for i in ids)

    monkeypatch.setattr(Qwen35Renderer, "parse_response", Qwen35Renderer.parse_response)
    monkeypatch.setattr(
        Qwen35Renderer, "_text_only_parse_installed", False, raising=False
    )
    renderer = object.__new__(Qwen35Renderer)
    renderer._tokenizer = Tokenizer()
    renderer._im_end, renderer._endoftext = 0, 1
    renderer._think, renderer._think_end = 2, 3
    renderer._tool_call, renderer._tool_call_end = 4, 5
    renderer.config = SimpleNamespace(enable_thinking=False)
    return renderer
