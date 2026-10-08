# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the rubric's forced-answer scale and length penalty, and their metrics."""

import asyncio
from dataclasses import dataclass, replace

import pytest

from torchtitan.rl.observability.controller import compute_rollout_metrics
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.rubric import CorrectLengthPenalty, RewardFn, Rubric
from torchtitan.rl.types import RolloutTurnID

RIGHT, WRONG = 7, 8


class _RewardLastToken(RewardFn):
    """1.0 when the completion ends with `RIGHT`."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        return float(rollout.turns[-1].completion_token_ids[-1] == RIGHT)


def _rollout(
    num_tokens: int,
    *,
    answer: int,
    forced: bool = False,
    loss_mask: list[bool] | None = None,
    status: RolloutStatus = RolloutStatus.COMPLETED,
) -> Rollout:
    """A one-turn rollout of `num_tokens` completion tokens ending in `answer`; a forced one
    has one appended token (`loss_mask` False) before the answer, as `ThinkingBudget` returns it."""
    token_ids = [1] * (num_tokens - 1) + [answer]
    turn = RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        prompt_token_ids=[0],
        completion_token_ids=token_ids,
        completion_logprobs=[-0.5] * num_tokens,
        completion_loss_mask=(
            [True] * (num_tokens - 2) + [False, True] if forced else loss_mask
        ),
    )
    return Rollout(group_id=0, rollout_id=0, status=status, turns=[turn])


def _score(rubric_config: Rubric.Config, rollouts: list[Rollout]) -> list[float]:
    outputs = asyncio.run(rubric_config.build().score_group(rollouts, None))
    return [output.reward for output in outputs]


def test_forced_answer_scale() -> None:
    rubric = Rubric.Config(
        reward_fns=[_RewardLastToken.Config()],
        truncation_reward=0.0,
        forced_answer_scale=0.5,
    )
    rollouts = [
        _rollout(10, answer=RIGHT),
        # Cut while answering: `ThinkingBudget` appended nothing, so the answer is not forced.
        _rollout(10, answer=RIGHT, loss_mask=[True] * 10),
        _rollout(10, answer=RIGHT, forced=True),
        _rollout(10, answer=WRONG, forced=True),
        # Forced, then cut in the answer reserve: truncated, never graded.
        _rollout(10, answer=RIGHT, forced=True, status=RolloutStatus.TRUNCATED_LENGTH),
    ]
    assert _score(rubric, rollouts) == [1.0, 1.0, 0.5, 0.0, 0.0]


def test_correct_length_penalty() -> None:
    rubric = Rubric.Config(
        reward_fns=[_RewardLastToken.Config()],
        forced_answer_scale=0.5,
        length_penalty=CorrectLengthPenalty.Config(max_tokens=131072),
    )
    # 12 of 16 correct; the median correct length is 20,000.
    lengths = [8000, 10000, 15000, 20000, 20000, 20000, 20000, 25000, 30000, 40000]
    rollouts = [_rollout(n, answer=RIGHT) for n in lengths]
    rollouts += [
        _rollout(75536, answer=RIGHT),
        _rollout(129024, answer=RIGHT, forced=True),
    ]
    rollouts += [_rollout(100000, answer=WRONG) for _ in range(4)]
    rewards = _score(rubric, rollouts)
    assert rewards[0] == 1.0  # shorter than the median: free
    # 0.1 * (75536 - 20000) / (131072 - 20000) = 0.05
    assert rewards[10] == pytest.approx(0.95)
    assert rewards[11] == pytest.approx(0.5 - 0.1 * 109024 / 111072)  # forced pays too
    assert rewards[12:] == [0.0] * 4  # wrong answers never pay

    # Exactly half solved pays; one correct fewer, or all correct, does not.
    half = [_rollout(10000, answer=RIGHT)] * 7 + [_rollout(131000, answer=RIGHT)]
    wrong = [_rollout(1000, answer=WRONG)] * 9
    assert _score(rubric, half + wrong[:8])[7] == pytest.approx(
        1 - 0.1 * 121000 / 121072
    )
    assert _score(rubric, half[1:] + wrong)[6] == 1.0
    assert _score(rubric, half) == [1.0] * 8

    # A correct reward below `max_penalty` pays at most itself, never dropping below a wrong one.
    tiny = replace(rubric, forced_answer_scale=0.05)
    assert _score(tiny, rollouts)[11] == 0.0


def test_kimi_length_reward() -> None:
    rubric = Rubric.Config(
        reward_fns=[_RewardLastToken.Config()],
        forced_answer_scale=0.5,
        length_reward_weight=0.1,
    )
    # lam = 0.5 - (len - 1000) / 2000: +0.5, 0, -0.5 at 1,000 / 2,000 / 3,000 tokens.
    rollouts = [
        _rollout(1000, answer=RIGHT),
        _rollout(2000, answer=RIGHT),
        _rollout(3000, answer=RIGHT, forced=True),
        _rollout(1000, answer=WRONG),  # a short wrong answer is not rewarded
        _rollout(3000, answer=WRONG),
    ]
    rewards = _score(rubric, rollouts)
    assert rewards == pytest.approx([1.05, 1.0, 0.45, 0.0, -0.05])
    # All lengths equal: no length reward.
    assert _score(rubric, [_rollout(10, answer=RIGHT), _rollout(10, answer=WRONG)]) == [
        1.0,
        0.0,
    ]


def test_rollout_metrics_split_forced_answers() -> None:
    rollouts = [
        _rollout(10, answer=RIGHT, forced=True),
        _rollout(10, answer=WRONG, forced=True),
        _rollout(10, answer=RIGHT, forced=True, status=RolloutStatus.TRUNCATED_LENGTH),
        _rollout(10, answer=RIGHT),
    ]
    for rollout, reward in zip(rollouts, [0.5, 0.0, 0.0, 1.0], strict=True):
        rollout.reward = reward
    metrics = MetricsProcessor._aggregate_metrics(
        compute_rollout_metrics(prefix="rollout", rollouts=rollouts)
    )
    assert metrics["rollout/forced_answer/count/sum"] == 3.0
    assert metrics["rollout/forced_answer/correct/mean"] == pytest.approx(1 / 3)
    assert metrics["rollout/forced_answer/truncation_rate/mean"] == pytest.approx(1 / 3)
    assert metrics["rollout/natural_answer/correct/mean"] == 1.0
