# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout integration."""

from __future__ import annotations

import asyncio
import base64
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

pytest.importorskip("verifiers")

from aiohttp import ClientSession

from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
from torchtitan.rl.examples.verifiers.generation_server import (
    _parse_sampling_config,
    _validate_token_ids,
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
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Completion, RolloutStatus


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
                routed_experts=None,
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
    assert turns[0].prompt_delta_token_ids.tolist() == [10, 11]
    assert turns[0].completion_token_ids.tolist() == [12, 13]
    assert turns[0].completion_logprobs.tolist() == pytest.approx([-0.2, -0.3])
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
                # Row i holds the experts of position i.
                routed_experts=np.arange(4).reshape(4, 1, 1),
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
    stored = [
        (turn.prompt_prefix_len, turn.prompt_delta_token_ids.tolist()) for turn in turns
    ]
    assert stored == [(0, [10]), (2, [12])]
    assert [turn.completion_token_ids.tolist() for turn in turns] == [[11], [13]]
    # Like the prompt, a delta: turn 1's rows start at 11, which only its prefill ran.
    assert [turn.routed_expert_ids.flatten().tolist() for turn in turns] == [
        [0],
        [1, 2],
    ]
    assert [turn.completion_logprobs.tolist() for turn in turns] == [
        pytest.approx([-0.1]),
        pytest.approx([-0.2]),
    ]


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
                routed_experts=None,
            ),
            SimpleNamespace(
                nodes=[shared, second_leaf],
                token_ids=[10, 11, 14, 15],
                logprobs=[0.0, -0.1, 0.0, -0.3],
                routed_experts=None,
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
    stored = [
        (turn.prompt_prefix_len, turn.prompt_delta_token_ids.tolist()) for turn in turns
    ]
    assert stored == [(0, [10]), (2, [12]), (2, [14])]
    assert [turn.completion_token_ids.tolist() for turn in turns] == [[11], [13], [15]]


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
                routed_experts=None,
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

    assert [turn.completion_token_ids.tolist() for turn in turns] == [[11], [13, 14]]
    assert [turn.completion_topk_token_ids.tolist() for turn in turns] == [
        [[11, 7]],
        [[13, 8], [14, 9]],
    ]
    np.testing.assert_allclose(
        turns[1].completion_topk_logprobs,
        np.array([[-0.2, -1.5], [-0.3, -1.2]], dtype=np.float32),
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
                routed_experts=None,
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


@pytest.mark.parametrize(
    "value", [[1, True], [1, 2.0], [1, "2"], [1, None], (1, 2), None]
)
def test_validate_token_ids_rejects_non_int_tokens(value) -> None:
    with pytest.raises(
        ValueError, match="token_ids must be a list of integer token IDs"
    ):
        _validate_token_ids(value, field_name="token_ids")


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
                metrics=[
                    m.Metric("generator/queue_time_ms", m.Mean(float(request_index))),
                    m.Metric("generator/queue_time_ms", m.Max(float(request_index))),
                ],
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
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
        # Both generations folded into one Mean and one Max
        assert len(generation_metadata.metrics) == 2
        assert m.MetricsProcessor._aggregate_metrics(generation_metadata.metrics) == {
            "generator/queue_time_ms/mean": 0.5,
            "generator/queue_time_ms/max": 1.0,
        }

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
                topk_token_ids=np.array([[31, 5], [32, 6]], dtype=np.int32),
                topk_logprobs=np.array([[-0.1, -2.4], [-0.2, -1.9]], dtype=np.float32),
                finish_reason="stop",
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11],
                        "sampling_params": {
                            "temperature": 1.0,
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
        np.testing.assert_array_equal(
            topk_logprobs, np.array([[-0.1, -2.4], [-0.2, -1.9]], dtype=np.float32)
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
        server.set_generate_fn(generate_fn)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11],
                        "sampling_params": {
                            "temperature": 1.0,
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


def test_generation_server_returns_routed_experts_from_the_prompt_start() -> None:
    # Prompt [10, 11, 12] + completion [31, 32]: rows for positions 0..3.
    routed_expert_ids = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)

    async def run_test() -> dict:
        async def generate_fn(prompt_token_ids, *, request_id, **kwargs):
            return Completion(
                min_policy_version=0,
                max_policy_version=0,
                request_id=request_id,
                token_ids=[31, 32],
                token_logprobs=[-0.1, -0.2],
                finish_reason="stop",
                routed_expert_ids=routed_expert_ids,
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11, 12],
                        "sampling_params": {
                            "temperature": 1.0,
                            "max_tokens": 2,
                            "torchtitan_group_id": 1,
                            "stop_token_ids": [99],
                            # The previous turn already returned positions 0 and 1.
                            "routed_experts_prompt_start": 2,
                        },
                    },
                )
                assert response.status == 200
                return await response.json()
        finally:
            await server.close()

    payload = asyncio.run(run_test())["choices"][0]["routed_experts"]
    assert (payload["shape"], payload["start"], payload["dtype"]) == (
        [2, 2, 2],
        2,
        "uint8",
    )
    rows = np.frombuffer(base64.b64decode(payload["data"]), dtype=np.uint8)
    assert rows.reshape(payload["shape"]).tolist() == routed_expert_ids[2:].tolist()


def test_generation_server_requires_group_id() -> None:
    async def run_test() -> None:
        async def generate_fn(*args, **kwargs):
            raise AssertionError("generate_fn must not run without a group id")

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
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


def test_generation_server_rejects_prompt_without_room_for_output() -> None:
    async def run_test() -> None:
        prompt_lengths = []

        async def generate_fn(prompt_token_ids, **kwargs):
            prompt_lengths.append(len(prompt_token_ids))
            raise RuntimeError("generator reached")

        server = GenerationServer.Config(max_rollout_tokens=8).build()
        server.set_generate_fn(generate_fn)
        await server.start()
        try:
            async with ClientSession() as session:
                statuses, errors = [], []
                for num_tokens in (7, 8, 9):
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": "group=1/rollout=2"},
                        json={
                            "token_ids": [1] * num_tokens,
                            "sampling_params": {
                                "temperature": 1.0,
                                "torchtitan_group_id": 1,
                                "stop_token_ids": [99],
                            },
                        },
                    )
                    statuses.append(response.status)
                    errors.append((await response.json())["error"])
        finally:
            await server.close()

        assert prompt_lengths == [7]
        assert statuses == [500, 400, 400]
        # vLLM's and the renderer's overlong-prompt wording, which Verifiers stops as truncated.
        assert "exceeds maximum context length" in errors[1]

    asyncio.run(run_test())


def test_rollout_whose_first_prompt_does_not_fit_is_flagged(caplog) -> None:
    # Verifiers stops a rollout for context length before its first model call.
    span = SimpleNamespace(duration=0.0)
    trace = SimpleNamespace(
        id="trace",
        task=SimpleNamespace(key="huge-task"),
        agent=SimpleNamespace(trainable=True),
        ok=True,
        is_truncated=True,
        stop_condition="context_length",
        errors=[],
        timing=SimpleNamespace(
            setup=span,
            agent=SimpleNamespace(duration=0.0, model=span, harness=span),
            scoring=span,
        ),
        calls=[],
        nodes=[],
        branches=[],
    )
    rollouter = object.__new__(VerifiersRollouter)
    rollouter._verifiers_config = SimpleNamespace(
        max_rollout_tokens=None,
        validation_max_rollout_tokens=None,
        extra_validation_sets={},
    )
    rollouter._generation_server = SimpleNamespace(
        model_id="model", pop_generation_metadata=lambda trace_id: None
    )
    rollouter._verifiers_env_client = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(ok=True, errors=[], traces=[trace]))
    )
    rollouter._verifiers_train_client_config = object()

    with caplog.at_level(logging.WARNING):
        rollout = asyncio.run(
            rollouter._run_single_rollout(
                release_session_fn=AsyncMock(),
                sample=VerifiersTaskSample(verifiers_task_data={"name": "huge-task"}),
                sampling=SamplingConfig(),
                group_id=1,
                rollout_id=2,
            )
        )

    assert rollout.status is RolloutStatus.ERROR
    assert "first prompt of task huge-task does not fit" in caplog.text


def test_rollout_releases_every_trace_session_even_if_one_release_fails(caplog) -> None:
    """Each trace generated under its own id; another agent's trace has one too."""
    span = SimpleNamespace(duration=0.0)
    traces = [
        SimpleNamespace(
            id=trace_id,
            task=SimpleNamespace(key="task"),
            agent=SimpleNamespace(trainable=trainable),
            ok=True,
            is_truncated=False,
            stop_condition=None,
            errors=[],
            timing=SimpleNamespace(
                setup=span,
                agent=SimpleNamespace(duration=0.0, model=span, harness=span),
                scoring=span,
            ),
            calls=[],
            nodes=[],
            branches=[],
        )
        for trace_id, trainable in [("opponent", False), ("policy", True)]
    ]
    rollouter = object.__new__(VerifiersRollouter)
    rollouter._verifiers_config = SimpleNamespace(
        max_rollout_tokens=None,
        validation_max_rollout_tokens=None,
        extra_validation_sets={},
    )
    rollouter._generation_server = SimpleNamespace(
        model_id="model", pop_generation_metadata=lambda trace_id: None
    )
    rollouter._verifiers_env_client = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(ok=True, errors=[], traces=traces))
    )
    rollouter._verifiers_train_client_config = object()
    release_session_fn = AsyncMock(side_effect=[RuntimeError("router down"), None])

    with caplog.at_level(logging.ERROR):
        rollout = asyncio.run(
            rollouter._run_single_rollout(
                release_session_fn=release_session_fn,
                sample=VerifiersTaskSample(verifiers_task_data={"name": "task"}),
                sampling=SamplingConfig(),
                group_id=1,
                rollout_id=0,
            )
        )

    assert [call.kwargs for call in release_session_fn.call_args_list] == [
        {"routing_session_id": "opponent"},
        {"routing_session_id": "policy"},
    ]
    assert "releasing routing session opponent failed" in caplog.text
    assert rollout.group_id == 1


def test_a_failing_rollout_fails_its_group_only_after_every_sibling_finished() -> None:
    """The caller releases the group once this raises; no sibling may still be generating."""

    async def run_test() -> list[int]:
        finished = []

        async def run_single_rollout(*, rollout_id: int, **kwargs) -> None:
            if rollout_id == 0:
                raise RuntimeError("rollout 0 failed")
            await asyncio.sleep(0.01)
            finished.append(rollout_id)

        rollouter = object.__new__(VerifiersRollouter)
        rollouter._generation_server = SimpleNamespace(set_generate_fn=lambda fn: None)
        rollouter._run_single_rollout = run_single_rollout
        with pytest.raises(RuntimeError, match="rollout 0 failed"):
            await rollouter.run_group_rollouts(
                generate_fn=AsyncMock(),
                release_session_fn=AsyncMock(),
                sample=None,
                group_id=1,
                group_size=2,
                sampling=SamplingConfig(),
            )
        return finished

    assert asyncio.run(run_test()) == [1]


def test_parse_sampling_config_requires_stop_token_ids() -> None:
    with pytest.raises(ValueError, match="stop_token_ids"):
        _parse_sampling_config({"temperature": 1.0})


def test_parse_sampling_config_requires_temperature() -> None:
    # A missing temperature must not silently fall back to SamplingConfig's default.
    with pytest.raises(ValueError, match="must include temperature"):
        _parse_sampling_config({"stop_token_ids": [1]})
    assert _parse_sampling_config(
        {"temperature": 0.0, "stop_token_ids": [1]}
    ).temperature == 0.0


def test_generation_server_enforces_each_request_context_cap() -> None:
    """Server cap 131,072. A training request capped at 64 (the rollouter's per-source cap) with a
    70-token prompt is rejected with vLLM's overlong-prompt wording (Verifiers stops it as
    truncated); the same prompt without a request cap reaches the generator, and a capped request
    has its `max_tokens` clamped to the room left."""

    async def run_test() -> None:
        calls = []

        async def generate_fn(prompt_token_ids, **kwargs):
            calls.append((len(prompt_token_ids), kwargs["sampling_config"].max_tokens))
            raise RuntimeError("generator reached")

        server = GenerationServer.Config(max_rollout_tokens=131072).build()
        server.set_generate_fn(generate_fn)
        await server.start()

        async def post(session, num_tokens, **extra):
            response = await session.post(
                f"http://{server.host}:{server.port}/inference/v1/generate",
                headers={"X-Session-ID": "group=1/rollout=2"},
                json={
                    "token_ids": [1] * num_tokens,
                    "sampling_params": {
                        "temperature": 1.0,
                        "max_tokens": 50,
                        "torchtitan_group_id": 1,
                        "stop_token_ids": [99],
                        **extra,
                    },
                },
            )
            return response.status, (await response.json())["error"]

        try:
            async with ClientSession() as session:
                over = await post(session, 70, torchtitan_max_rollout_tokens=64)
                uncapped = await post(session, 70)
                clamped = await post(session, 40, torchtitan_max_rollout_tokens=64)
        finally:
            await server.close()

        assert over[0] == 400
        assert "Prompt length (70) exceeds maximum context length (64)" in over[1]
        assert uncapped[0] == 500  # reached the generator
        assert clamped[0] == 500
        assert calls == [(70, 50), (40, 24)]

    asyncio.run(run_test())


def test_rollouter_sends_each_sources_context_cap() -> None:
    from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
    from torchtitan.rl.examples.verifiers.generation_server import (
        MAX_ROLLOUT_TOKENS_SAMPLING_PARAM,
    )
    from torchtitan.rl.examples.verifiers.rollouter import VerifiersRollouter

    rollouter = object.__new__(VerifiersRollouter)
    rollouter._verifiers_config = SimpleNamespace(
        max_rollout_tokens=65536,
        validation_max_rollout_tokens=None,
        extra_validation_sets={
            "deepswe": SimpleNamespace(max_rollout_tokens=131072)
        },
    )
    train = VerifiersTaskSample(verifiers_task_data={})
    deepswe = VerifiersTaskSample(verifiers_task_data={}, validation_set="deepswe")
    key = MAX_ROLLOUT_TOKENS_SAMPLING_PARAM
    assert rollouter._max_rollout_tokens_param(train, 7) == {key: 65536}
    assert rollouter._max_rollout_tokens_param(train, -3) == {key: 65536}
    assert rollouter._max_rollout_tokens_param(deepswe, -1_000_001) == {key: 131072}
    rollouter._verifiers_config.max_rollout_tokens = None
    assert rollouter._max_rollout_tokens_param(train, 7) == {}
