# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the thinking budget: forced end of thinking, merged completion, and loss mask."""

import asyncio
import math
from dataclasses import replace

import pytest
import torch

from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.generator import SamplingConfig
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.thinking_budget import ThinkingBudget
from torchtitan.rl.types import Completion, RolloutTurnID

THINK, END_THINK, END_OF_TURN, BRACE, BRACE_DOT = 1, 2, 3, 4, 5
FORCED = [90, 91, 92]


class _Tokenizer:
    """`<think>` and `</think>` are one token each; the forced text encodes to `FORCED`."""

    def token_to_id(self, token: str) -> int | None:
        return {"<think>": THINK, "</think>": END_THINK, "<|im_end|>": END_OF_TURN}.get(
            token
        )

    def encode(self, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
        return list(FORCED)

    def get_vocab_size(self) -> int:
        return 100

    def decode(self, token_ids: list[int]) -> str:
        return {BRACE: "}", BRACE_DOT: "}."}.get(token_ids[0], "x")


class _ScriptedGenerate:
    """Returns one scripted completion per call and records every request."""

    def __init__(self, *completions: Completion | None) -> None:
        self._completions = list(completions)
        self.calls: list[dict] = []

    async def __call__(self, prompt_token_ids, **kwargs) -> Completion | None:
        self.calls.append({"prompt_token_ids": prompt_token_ids, **kwargs})
        return self._completions.pop(0)


def _completion(token_ids, *, finish_reason, version=3) -> Completion:
    return Completion(
        min_policy_version=version,
        max_policy_version=version,
        request_id="r",
        token_ids=token_ids,
        token_logprobs=[-0.5] * len(token_ids),
        finish_reason=finish_reason,
    )


def _budget(max_thinking_tokens: int = 4) -> ThinkingBudget:
    return ThinkingBudget(
        ThinkingBudget.Config(max_thinking_tokens=max_thinking_tokens),
        tokenizer=_Tokenizer(),
    )


def _run(budget: ThinkingBudget, generate: _ScriptedGenerate, prompt, max_tokens=12):
    wrapped = budget.wrap(generate)
    return asyncio.run(
        wrapped(
            prompt,
            request_id="group=0/rollout=0/turn=0",
            group_id=0,
            routing_session_id="group=0/rollout=0",
            sampling_config=SamplingConfig(temperature=1.0, max_tokens=max_tokens),
        )
    )


def _forced_close_rate(completion: Completion) -> float:
    reduced = MetricsProcessor._aggregate_metrics(completion.metrics)
    return reduced["thinking_budget/forced_close_rate/mean"]


def test_reply_within_budget_is_returned_unchanged() -> None:
    reply = _completion([10, END_THINK, 11], finish_reason="stop")
    generate = _ScriptedGenerate(reply)
    completion = _run(_budget(), generate, prompt=[5, THINK])
    assert completion is reply
    assert completion.loss_mask is None
    assert len(generate.calls) == 1
    assert generate.calls[0]["sampling_config"].max_tokens == 4
    assert _forced_close_rate(completion) == 0.0


def test_reply_cut_while_thinking_gets_a_forced_close() -> None:
    thinking = _completion([10, 11, 12, 13], finish_reason="length", version=3)
    answer = _completion([20, 21], finish_reason="stop", version=4)
    generate = _ScriptedGenerate(thinking, answer)
    completion = _run(_budget(), generate, prompt=[5, THINK], max_tokens=12)

    first_call, second_call = generate.calls
    assert first_call["sampling_config"].max_tokens == 4
    assert second_call["prompt_token_ids"] == [5, THINK, 10, 11, 12, 13, *FORCED]
    # the turn stays within max_tokens: 12 - 4 thinking - 3 forced
    assert second_call["sampling_config"].max_tokens == 5
    assert second_call["request_id"] == "group=0/rollout=0/turn=0/answer"
    assert second_call["routing_session_id"] == "group=0/rollout=0"

    assert completion.token_ids == [10, 11, 12, 13, *FORCED, 20, 21]
    assert completion.loss_mask == [True] * 4 + [False] * 3 + [True] * 2
    assert all(math.isnan(lp) for lp in completion.token_logprobs[4:7])
    assert completion.token_logprobs[:4] == [-0.5] * 4
    assert (completion.min_policy_version, completion.max_policy_version) == (3, 4)
    assert completion.finish_reason == "stop"
    assert completion.request_id == "group=0/rollout=0/turn=0"
    assert _forced_close_rate(completion) == 1.0


def test_forced_answer_ends_the_turn_at_its_closing_brace() -> None:
    thinking = _completion([10, 11, 12, 13], finish_reason="length")
    answer = _completion([20, BRACE_DOT], finish_reason="stop")
    generate = _ScriptedGenerate(thinking, answer)
    budget = ThinkingBudget(
        ThinkingBudget.Config(max_thinking_tokens=4, answer_end_text="}"),
        tokenizer=_Tokenizer(),
    )
    completion = _run(budget, generate, [THINK])

    answer_sampling = generate.calls[1]["sampling_config"]
    assert answer_sampling.stop_token_ids == [BRACE, BRACE_DOT]
    assert (
        answer_sampling.max_tokens == 12 - 4 - len(FORCED) - 1
    )  # room for the end of turn
    assert completion.token_ids == [10, 11, 12, 13, *FORCED, 20, BRACE_DOT, END_OF_TURN]
    assert completion.loss_mask == [True] * 4 + [False] * 3 + [True] * 2 + [False]
    assert math.isnan(completion.token_logprobs[-1])
    assert completion.finish_reason == "stop"
    # the thinking call does not stop at braces
    assert generate.calls[0]["sampling_config"].stop_token_ids is None


def test_forced_close_joins_the_topk_rows_with_zero_rows_on_unsampled_tokens() -> None:
    # k=2: each sampled token's rows are [token, token + 50] with logprobs [-0.5, -3.0].
    def with_topk(completion: Completion) -> Completion:
        completion.topk_token_ids = torch.tensor(
            [[token_id, token_id + 50] for token_id in completion.token_ids],
            dtype=torch.int32,
        )
        completion.topk_logprobs = torch.tensor(
            [[-0.5, -3.0]] * len(completion.token_ids)
        )
        return completion

    thinking = with_topk(_completion([10, 11, 12, 13], finish_reason="length"))
    answer = with_topk(_completion([20, BRACE], finish_reason="stop"))
    budget = ThinkingBudget(
        ThinkingBudget.Config(max_thinking_tokens=4, answer_end_text="}"),
        tokenizer=_Tokenizer(),
    )
    completion = _run(budget, _ScriptedGenerate(thinking, answer), [THINK])

    # [10, 11, 12, 13, *FORCED, 20, BRACE, END_OF_TURN]: zero rows on the forced and end tokens
    assert completion.topk_token_ids.tolist() == (
        [[10, 60], [11, 61], [12, 62], [13, 63]]
        + [[0, 0]] * len(FORCED)
        + [[20, 70], [BRACE, BRACE + 50], [0, 0]]
    )
    torch.testing.assert_close(
        completion.topk_logprobs,
        torch.tensor(
            [[-0.5, -3.0]] * 4
            + [[0.0, 0.0]] * len(FORCED)
            + [[-0.5, -3.0]] * 2
            + [[0.0, 0.0]]
        ),
    )


def test_reply_cut_while_answering_continues_without_forcing() -> None:
    cut = _completion([10, END_THINK, 11, 12], finish_reason="length")
    rest = _completion([13], finish_reason="stop")
    generate = _ScriptedGenerate(cut, rest)
    completion = _run(_budget(), generate, prompt=[5, THINK])
    assert generate.calls[1]["prompt_token_ids"] == [5, THINK, 10, END_THINK, 11, 12]
    assert completion.token_ids == [10, END_THINK, 11, 12, 13]
    assert completion.loss_mask == [True] * 5
    assert _forced_close_rate(completion) == 0.0


@pytest.mark.parametrize(
    "prompt",
    [
        [
            5,
            THINK,
            6,
            END_THINK,
            7,
        ],  # thinking off: the generation prompt already closed it
        [5, 6, 7],  # a model that never thinks
    ],
)
def test_reply_that_is_not_thinking_is_never_force_closed(prompt) -> None:
    generate = _ScriptedGenerate(
        _completion([10, 11, 12, 13], finish_reason="length"),
        _completion([14], finish_reason="length"),
    )
    completion = _run(_budget(), generate, prompt=prompt)
    assert completion.token_ids == [10, 11, 12, 13, 14]
    assert completion.loss_mask == [True] * 5
    assert completion.finish_reason == "length"


@pytest.mark.parametrize("failed_call", [0, 1])
def test_a_failed_generation_returns_none(failed_call) -> None:
    completions = [_completion([10, 11, 12, 13], finish_reason="length"), None]
    if failed_call == 0:
        completions = [None]
    assert _run(_budget(), _ScriptedGenerate(*completions), prompt=[5, THINK]) is None


def test_budget_that_leaves_no_room_for_the_answer_raises() -> None:
    with pytest.raises(ValueError, match="must be below SamplingConfig.max_tokens"):
        _run(
            _budget(max_thinking_tokens=9),
            _ScriptedGenerate(),
            prompt=[5, THINK],
            max_tokens=12,
        )


def test_opening_turns_think_up_to_the_opening_budget() -> None:
    budget = ThinkingBudget(
        ThinkingBudget.Config(
            max_thinking_tokens=4, opening_max_thinking_tokens=6, opening_turns=2
        ),
        tokenizer=_Tokenizer(),
    )
    generate = _ScriptedGenerate(
        *(_completion([10, END_THINK, 11], finish_reason="stop") for _ in range(4))
    )

    async def play_rollout(num_turns: int) -> None:
        wrapped = budget.wrap(generate)
        for turn_idx in range(num_turns):
            await wrapped(
                [5, THINK],
                request_id=f"group=0/rollout=0/turn={turn_idx}",
                group_id=0,
                sampling_config=SamplingConfig(temperature=1.0, max_tokens=12),
            )

    asyncio.run(play_rollout(num_turns=3))
    asyncio.run(play_rollout(num_turns=1))
    # turns 0-1 think up to 6 tokens and turn 2 up to 4; the next rollout starts over
    thinking_caps = [call["sampling_config"].max_tokens for call in generate.calls]
    assert thinking_caps == [6, 6, 4, 6]


def test_opening_budget_that_leaves_no_room_for_the_answer_raises() -> None:
    # 4 thinking tokens + 3 forced fit in 12, but the opening turn's 9 + 3 do not
    budget = ThinkingBudget(
        ThinkingBudget.Config(
            max_thinking_tokens=4, opening_max_thinking_tokens=9, opening_turns=1
        ),
        tokenizer=_Tokenizer(),
    )
    with pytest.raises(ValueError, match="must be below SamplingConfig.max_tokens"):
        _run(budget, _ScriptedGenerate(), prompt=[5, THINK], max_tokens=12)


def test_delimiters_must_be_single_tokens() -> None:
    with pytest.raises(ValueError, match="must each be one token"):
        ThinkingBudget(
            ThinkingBudget.Config(max_thinking_tokens=4, think_end_token="</thinking>"),
            tokenizer=_Tokenizer(),
        )


def test_training_samples_skip_the_forced_tokens() -> None:
    # turn 0 was force-closed; turn 1 continues the same prefix and was not
    turn0 = RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        prompt_prefix_len=0,
        prompt_delta_token_ids=[5, THINK],
        completion_token_ids=[10, 11, *FORCED, 20],
        completion_logprobs=[-0.5, -0.5, math.nan, math.nan, math.nan, -0.5],
        completion_loss_mask=[True, True, False, False, False, True],
        min_policy_version=3,
        max_policy_version=3,
    )
    turn1 = replace(
        turn0,
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=1),
        # continues turn 0's prompt + completion with the env reply [6]
        prompt_prefix_len=len([5, THINK, 10, 11, *FORCED, 20]),
        prompt_delta_token_ids=[6],
        completion_token_ids=[30],
        completion_logprobs=[-0.5],
        completion_loss_mask=None,
    )
    rollout = Rollout(
        group_id=0,
        rollout_id=0,
        status=RolloutStatus.COMPLETED,
        turns=[turn0, turn1],
        advantage=1.0,
    )
    (sample,) = (
        TrainingSampleBuilder.Config().build().rollout_to_training_samples(rollout)
    )
    assert sample.token_ids.tolist() == [5, THINK, 10, 11, *FORCED, 20, 6, 30]
    assert sample.loss_mask.tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
