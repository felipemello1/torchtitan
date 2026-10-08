# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the rubric's forced-answer scale and length reward, and their metrics."""

import asyncio
from dataclasses import dataclass

import pytest

from torchtitan.rl.observability.controller import compute_rollout_metrics
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.rubric import RewardFn, Rubric
from torchtitan.rl.types import RolloutTurnID

RIGHT, WRONG, PENALIZED = 7, 8, 9


class _RewardLastToken(RewardFn):
    """1.0 when the completion ends with `RIGHT`, -1.0 with `PENALIZED`, else 0.0."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        return {RIGHT: 1.0, PENALIZED: -1.0}.get(
            rollout.turns[-1].completion_token_ids[-1], 0.0
        )


def _rollout(
    num_tokens: int,
    *,
    answer: int,
    forced: bool = False,
    loss_mask: list[bool] | None = None,
    status: RolloutStatus = RolloutStatus.COMPLETED,
    prompt_len: int = 1,
) -> Rollout:
    """A one-turn rollout of `num_tokens` completion tokens ending in `answer`; a forced one
    has one appended token (`loss_mask` False) before the answer, as `ThinkingBudget` returns it."""
    token_ids = [1] * (num_tokens - 1) + [answer]
    turn = RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        prompt_token_ids=[0] * prompt_len,
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
        # Only a positive reward is scaled: a negative one would move toward 0.
        _rollout(10, answer=PENALIZED, forced=True),
    ]
    assert _score(rubric, rollouts) == [1.0, 1.0, 0.5, -1.0]


def test_kimi_length_reward() -> None:
    rubric = Rubric.Config(
        reward_fns=[_RewardLastToken.Config()],
        forced_answer_scale=0.5,
        length_reward_weight=0.1,
    )
    # lam = 0.5 - (len - 1000) / 2000: +0.5, 0, -0.5 at 1,000 / 2,000 / 3,000 tokens.
    rollouts = [
        _rollout(1000, answer=RIGHT),
        # A 700-token prompt does not count as response length.
        _rollout(2000, answer=RIGHT, prompt_len=700),
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
