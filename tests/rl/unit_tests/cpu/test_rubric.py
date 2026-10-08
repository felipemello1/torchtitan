# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the rubric's forced-answer scale and length penalty, and their metrics."""

import asyncio
from dataclasses import dataclass

import pytest

from torchtitan.rl.observability.controller import compute_rollout_metrics
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.rubric import CorrectLengthPenalty, RewardFn, Rubric
from torchtitan.rl.types import RolloutTurnID

RIGHT, WRONG, FORCED = 7, 8, 9


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
    status: RolloutStatus = RolloutStatus.COMPLETED,
) -> Rollout:
    """A one-turn rollout of `num_tokens` completion tokens ending in `answer`; a forced one
    carries one appended token (`loss_mask` False), as `ThinkingBudget` returns it."""
    token_ids = [1] * (num_tokens - 2) + [FORCED, answer]
    turn = RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        prompt_token_ids=[0],
        completion_token_ids=token_ids,
        completion_logprobs=[-0.5] * num_tokens,
        completion_loss_mask=(
            [True] * (num_tokens - 2) + [False, True] if forced else None
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
        _rollout(10, answer=RIGHT, forced=True),
        _rollout(10, answer=WRONG, forced=True),
        # Forced, then cut in the answer reserve: truncated, never graded.
        _rollout(10, answer=RIGHT, forced=True, status=RolloutStatus.TRUNCATED_LENGTH),
    ]
    assert _score(rubric, rollouts) == [1.0, 0.5, 0.0, 0.0]


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
    assert rewards[10] == pytest.approx(
        0.95
    )  # 0.1 * (75536 - 20000) / (131072 - 20000)
    assert rewards[11] == pytest.approx(0.5 - 0.1 * 109024 / 111072)  # forced pays too
    assert rewards[12:] == [0.0] * 4  # wrong answers never pay

    # Solved by fewer than half, or by all: no penalty.
    hard = [_rollout(120000, answer=RIGHT)] + [_rollout(100, answer=WRONG)] * 15
    assert _score(rubric, hard) == [1.0] + [0.0] * 15
    solved = [_rollout(n, answer=RIGHT) for n in (1000, 120000)]
    assert _score(rubric, solved) == [1.0, 1.0]


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
    assert metrics["rollout/forced_answer/cut_in_reserve/mean"] == pytest.approx(1 / 3)
    assert metrics["rollout/natural_answer/correct/mean"] == 1.0
