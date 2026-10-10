# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout integration."""

from __future__ import annotations

import asyncio
import binascii
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
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


def _true_routing(position: int, *, branch: int = 0) -> list[list[int]]:
    """One routed-expert row [1 layer, 2 experts]: position p routes to (p, 100 + p), plus
    50 on a second branch's own tokens."""
    return [[position + 50 * branch, 100 + position + 50 * branch]]


def _routed_node(role: str, token_ids, first_position: int, *, branch: int = 0):
    """A real Verifiers node holding the routed-expert rows Verifiers attributes to it.

    An assistant node's first token is the generation scaffold and the rest is sampled;
    its last token never ran forward in its turn, so Verifiers repeats the row before it.
    """
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage

    sampled = role == "assistant"
    rows = [
        _true_routing(first_position + offset, branch=branch)
        for offset in range(len(token_ids))
    ]
    if sampled:
        rows[-1] = rows[-2]
    message = {
        "user": lambda: UserMessage(content="task"),
        "tool": lambda: ToolMessage(content="result", tool_call_id="call"),
        "assistant": lambda: AssistantMessage(content="reply"),
    }[role]()
    return MessageNode(
        message=message,
        sampled=sampled,
        token_ids=list(token_ids),
        mask=[False] + [True] * (len(token_ids) - 1)
        if sampled
        else [False] * len(token_ids),
        logprobs=[-0.1] * (len(token_ids) - 1) if sampled else [],
        routed_experts=np.array(rows, dtype=np.uint8),
    )


def _routed_three_turn_trace():
    """user [10, 11]; turn 0 [12 | 13, 14]; tool [15, 16]; turn 1 [17 | 18, 19];
    tool [20]; turn 2 [21 | 22, 23]. Positions 0..13."""
    from verifiers.v1.trace import Branch

    nodes = [
        _routed_node("user", [10, 11], 0),
        _routed_node("assistant", [12, 13, 14], 2),
        _routed_node("tool", [15, 16], 5),
        _routed_node("assistant", [17, 18, 19], 7),
        _routed_node("tool", [20], 10),
        _routed_node("assistant", [21, 22, 23], 11),
    ]
    return SimpleNamespace(nodes=nodes, branches=[Branch(index=0, nodes=nodes)])


def _routed_metadata(**boundary_rows) -> VerifiersGenerationMetadata:
    """Turns 1 and 2's prefills ran tokens 14 (position 4) and 19 (position 9) forward."""
    rows = {
        (8, (18, 19)): (4, torch.tensor(_true_routing(4), dtype=torch.uint8)),
        (12, (22, 23)): (9, torch.tensor(_true_routing(9), dtype=torch.uint8)),
    }
    return VerifiersGenerationMetadata(
        min_policy_version=3,
        max_policy_version=3,
        metrics=[],
        routed_expert_boundary_rows={**rows, **boundary_rows},
        routed_experts_expected=True,
    )


def _routing(positions, *, branch: int = 0) -> list[list[list[int]]]:
    return [_true_routing(position, branch=branch) for position in positions]


def test_verifiers_trace_gives_each_turn_its_routed_experts_from_the_prefix_boundary() -> (
    None
):
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=_routed_three_turn_trace(),
        generation_metadata=_routed_metadata(),
        group_id=5,
        rollout_id=2,
    )

    assert [turn.prompt_prefix_len for turn in turns] == [0, 5, 10]
    # Rows from prompt_prefix_len - 1 to the completion's second-to-last token; positions 4
    # and 9 (each turn's last token) come from the next turn's prefill, not Verifiers' copy.
    assert turns[0].routed_expert_ids.tolist() == _routing(range(0, 4))
    assert turns[1].routed_expert_ids.tolist() == _routing(range(4, 9))
    assert turns[2].routed_expert_ids.tolist() == _routing(range(9, 13))
    # Each turn owns only its rows, so pickling it doesn't carry the branch.
    assert all(
        turn.routed_expert_ids.untyped_storage().nbytes()
        == turn.routed_expert_ids.numel()
        for turn in turns
    )
    assert not any(
        metric.key == "rollout/routed_experts_copied_boundary_rows"
        for turn in turns
        for metric in turn.metrics
    )


def test_verifiers_routed_experts_pack_one_row_per_trainer_input() -> None:
    from torchtitan.rl.components.batcher import Batcher
    from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
    from torchtitan.rl.rollout import Rollout, RolloutStatus

    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=_routed_three_turn_trace(),
        generation_metadata=_routed_metadata(),
        group_id=5,
        rollout_id=2,
    )
    rollout = Rollout(
        group_id=5,
        rollout_id=2,
        status=RolloutStatus.COMPLETED,
        turns=turns,
        reward=1.0,
        advantage=0.5,
    )
    [sample] = (
        TrainingSampleBuilder.Config().build().rollout_to_training_samples(rollout)
    )
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=16,
        max_context_length=16,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    microbatch = batcher._pack_training_samples([sample])

    assert sample.token_ids.tolist() == list(range(10, 24))
    assert len(sample.routed_expert_ids) == len(sample.token_ids) - 1
    packed = microbatch.model_kwargs["routed_expert_ids"]
    assert packed[:13].tolist() == _routing(range(13))
    assert packed[13:].count_nonzero() == 0


def test_verifiers_branch_takes_its_shared_prefix_routing_from_the_previous_sample() -> (
    None
):
    from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
    from torchtitan.rl.rollout import Rollout, RolloutStatus
    from verifiers.v1.trace import Branch

    # Branch A: user [10, 11]; turn 0 [12 | 13, 14]; tool [15, 16]; turn 1 [17 | 18, 19].
    # Branch B rewrites the tool reply to [15, 26]: its turn [27 | 28, 29] shares 6 tokens.
    user = _routed_node("user", [10, 11], 0)
    turn0 = _routed_node("assistant", [12, 13, 14], 2)
    branch_a = [user, turn0, _routed_node("tool", [15, 16], 5)]
    branch_a.append(_routed_node("assistant", [17, 18, 19], 7))
    tool_b = _routed_node("tool", [15, 26], 5)
    tool_b.routed_experts[1] = _true_routing(6, branch=1)
    branch_b = [
        user,
        turn0,
        tool_b,
        _routed_node("assistant", [27, 28, 29], 7, branch=1),
    ]
    trace = SimpleNamespace(
        nodes=[*branch_a, *branch_b[2:]],
        branches=[Branch(index=0, nodes=branch_a), Branch(index=1, nodes=branch_b)],
    )
    metadata = _routed_metadata()

    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=trace, generation_metadata=metadata, group_id=5, rollout_id=2
    )
    rollout = Rollout(
        group_id=5,
        rollout_id=2,
        status=RolloutStatus.COMPLETED,
        turns=turns,
        reward=1.0,
        advantage=0.5,
    )
    first, second = (
        TrainingSampleBuilder.Config().build().rollout_to_training_samples(rollout)
    )

    assert [turn.prompt_prefix_len for turn in turns] == [0, 5, 6]
    assert first.routed_expert_ids.tolist() == _routing(range(9))
    # Positions 0..4 from branch A's sample, 5 onward from branch B's own rows.
    assert second.token_ids.tolist() == [10, 11, 12, 13, 14, 15, 26, 27, 28, 29]
    assert second.routed_expert_ids.tolist() == [
        *_routing(range(6)),
        *_routing(range(6, 9), branch=1),
    ]


def test_verifiers_turn_without_routed_experts_fails_when_the_generator_returns_them() -> (
    None
):
    trace = _routed_three_turn_trace()
    trace.nodes[4].routed_experts = None  # e.g. a payload Verifiers could not attribute

    with pytest.raises(ValueError, match="lacks them"):
        VerifiersRollouter.trace_to_rollout_turns(
            trace=trace,
            generation_metadata=_routed_metadata(),
            group_id=5,
            rollout_id=2,
        )


def test_verifiers_turns_carry_no_routed_experts_unless_the_generator_returns_them() -> (
    None
):
    metadata = _routed_metadata()
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=_routed_three_turn_trace(),
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=3, max_policy_version=3, metrics=[]
        ),
        group_id=5,
        rollout_id=2,
    )

    assert metadata.routed_experts_expected
    assert all(turn.routed_expert_ids is None for turn in turns)


def test_verifiers_turn_without_its_boundary_row_keeps_verifiers_copy_and_counts_it() -> (
    None
):
    metadata = _routed_metadata()
    del metadata.routed_expert_boundary_rows[(8, (18, 19))]

    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=_routed_three_turn_trace(),
        generation_metadata=metadata,
        group_id=5,
        rollout_id=2,
    )

    # Position 4 keeps Verifiers' copy of position 3's row.
    assert turns[1].routed_expert_ids.tolist() == _routing([3, 5, 6, 7, 8])
    assert [
        metric.key
        for metric in turns[1].metrics
        if metric.key.startswith("rollout/routed_experts")
    ] == ["rollout/routed_experts_copied_boundary_rows"]


def test_verifiers_node_routed_experts_survive_the_env_server_wire() -> None:
    import msgpack
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.serve.encoding import msgpack_encoder

    node = _routed_node("assistant", [12, 13, 14], 2)
    wire = msgpack.packb(
        node.model_dump(mode="python"), default=msgpack_encoder, use_bin_type=True
    )
    decoded = MessageNode.model_validate(msgpack.unpackb(wire, raw=False))

    assert decoded.routed_experts.dtype == np.uint8
    assert decoded.routed_experts.tolist() == node.routed_experts.tolist()


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


def _post_one_generation(*, routed_expert_ids, prompt_start=None):
    """POST prompt [10, 11, 12] to a server whose generation completes [31, 32].

    Returns the status, the raw response body, and the SamplingConfig the generation got.
    """

    async def run_test():
        received = []

        async def generate_fn(prompt_token_ids, *, request_id, group_id, **kwargs):
            received.append(kwargs["sampling_config"])
            return Completion(
                min_policy_version=7,
                max_policy_version=7,
                request_id=request_id,
                token_ids=[31, 32],
                token_logprobs=[-0.1, -0.2],
                routed_expert_ids=routed_expert_ids,
                finish_reason="stop",
            )

        sampling_params = {"torchtitan_group_id": 1, "stop_token_ids": [99]}
        if prompt_start is not None:
            sampling_params["routed_experts_prompt_start"] = prompt_start
        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.generate_fns[1] = generate_fn
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11, 12],
                        "sampling_params": sampling_params,
                    },
                )
                return response.status, await response.read(), received
        finally:
            await server.close()

    return asyncio.run(run_test())


def _routed_rows(num_rows: int) -> torch.Tensor:
    """uint8 [num_rows, 2 layers, 2 experts per token]; row i holds 10 * i + [0..3]."""
    return (
        torch.arange(num_rows * 4, dtype=torch.uint8).view(num_rows, 2, 2)
        + torch.arange(num_rows, dtype=torch.uint8).view(num_rows, 1, 1) * 6
    )


def test_generation_server_returns_routed_experts_from_the_prompt_start() -> None:
    from renderers.client import parse_generate_response
    from verifiers.v1.graph import _attribute_routed_experts

    # Verifiers already holds positions 0..1, so it asks from position 2: tokens 12 and 31
    # (32, the last token, never ran forward).
    rows = _routed_rows(2)
    status, body, received = _post_one_generation(
        routed_expert_ids=rows, prompt_start=2
    )

    assert status == 200
    assert received[0].routed_experts_prompt_start == 2
    # The Verifiers client splices the base64 out of the raw bytes by this prefix.
    assert b'"routed_experts":{"data":"' in body
    payload = parse_generate_response(body)["choices"][0]["routed_experts"]
    assert (payload["shape"], payload["start"], payload["dtype"]) == (
        [2, 2, 2],
        2,
        "uint8",
    )

    # Verifiers' own attribution: the turn's new nodes tile positions 2.. ([12] then
    # [31, 32]); the final position gets the last row repeated.
    nodes = [
        SimpleNamespace(token_ids=[10, 11], routed_experts=None),
        SimpleNamespace(token_ids=[12], routed_experts=None),
        SimpleNamespace(token_ids=[31, 32], routed_experts=None),
    ]
    _attribute_routed_experts(SimpleNamespace(nodes=nodes), [1, 2], 2, payload)
    assert nodes[0].routed_experts is None
    assert nodes[1].routed_experts.tolist() == rows[:1].tolist()
    assert nodes[2].routed_experts.tolist() == [rows[1].tolist(), rows[1].tolist()]


def test_generation_server_trims_rows_a_generator_returned_from_position_zero() -> None:
    rows = _routed_rows(4)  # positions 0..3: a generator that ignored the start
    status, body, _ = _post_one_generation(routed_expert_ids=rows, prompt_start=2)

    assert status == 200
    payload = json.loads(body)["choices"][0]["routed_experts"]
    data = np.frombuffer(binascii.a2b_base64(payload["data"]), dtype=payload["dtype"])
    assert data.reshape(payload["shape"]).tolist() == rows[2:].tolist()


def test_generation_server_rejects_routed_experts_with_the_wrong_row_count() -> None:
    status, body, _ = _post_one_generation(
        routed_expert_ids=_routed_rows(3), prompt_start=2
    )

    assert status == 500
    assert "expected 2" in json.loads(body)["error"]


def test_generation_server_keeps_the_real_row_of_each_turn_boundary() -> None:
    # Turn 0: [10, 11] -> [12, 13]. Turn 1 (Verifiers bridged it, start 3):
    # [10..14] -> [15, 16]. Turn 2 (not bridged, no start): [10..17] -> [18].
    turns = [
        ([10, 11], [12, 13], None),
        ([10, 11, 12, 13, 14], [15, 16], 3),
        ([10, 11, 12, 13, 14, 15, 16, 17], [18], None),
    ]

    async def run_test():
        async def generate_fn(prompt_token_ids, *, request_id, group_id, **kwargs):
            start = kwargs["sampling_config"].routed_experts_prompt_start
            _, completion, _ = turns[int(request_id.rsplit("=", 1)[1])]
            num_rows = len(prompt_token_ids) + len(completion) - 1
            return Completion(
                min_policy_version=7,
                max_policy_version=7,
                request_id=request_id,
                token_ids=completion,
                token_logprobs=[-0.1] * len(completion),
                # Row of position i holds 10 * i + [0..3], whatever turn computed it.
                routed_expert_ids=_routed_rows(num_rows)[start:],
                finish_reason="stop",
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.generate_fns[1] = generate_fn
        await server.start()
        try:
            async with ClientSession() as session:
                for prompt, _, start in turns:
                    sampling_params = {"torchtitan_group_id": 1, "stop_token_ids": [99]}
                    if start is not None:
                        sampling_params["routed_experts_prompt_start"] = start
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": "group=1/rollout=2"},
                        json={"token_ids": prompt, "sampling_params": sampling_params},
                    )
                    assert response.status == 200
            return server.pop_generation_metadata("group=1/rollout=2")
        finally:
            await server.close()

    boundary_rows = asyncio.run(run_test()).routed_expert_boundary_rows

    rows = _routed_rows(8)
    assert set(boundary_rows) == {(5, (15, 16)), (8, (18,))}
    position, row = boundary_rows[(5, (15, 16))]
    assert (position, row.tolist()) == (3, rows[3].tolist())
    position, row = boundary_rows[(8, (18,))]
    assert (position, row.tolist()) == (6, rows[6].tolist())


def test_generation_server_omits_routed_experts_without_them() -> None:
    status, body, received = _post_one_generation(routed_expert_ids=None)

    assert status == 200
    assert received[0].routed_experts_prompt_start == 0
    assert "routed_experts" not in json.loads(body)["choices"][0]


def test_parse_sampling_config_rejects_a_prompt_start_past_the_prompt() -> None:
    with pytest.raises(ValueError, match="routed_experts_prompt_start"):
        _parse_sampling_config(
            {"stop_token_ids": [99], "routed_experts_prompt_start": 3},
            num_prompt_tokens=3,
        )


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
        _parse_sampling_config({"temperature": 1.0}, num_prompt_tokens=2)
