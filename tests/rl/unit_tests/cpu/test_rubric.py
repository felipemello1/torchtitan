# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the rubric's length reward."""

import asyncio
from dataclasses import dataclass

import pytest

from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.rubric import RewardFn, Rubric
from torchtitan.rl.types import RolloutTurnID

RIGHT, WRONG = 7, 8


class _RewardLastToken(RewardFn):
    """1.0 when the completion ends with `RIGHT`, else 0.0."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        return float(rollout.turns[-1].completion_token_ids[-1] == RIGHT)


def _rollout(num_tokens: int, *, answer: int, prompt_len: int = 1) -> Rollout:
    """A one-turn rollout of `num_tokens` completion tokens ending in `answer`."""
    turn = RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        prompt_token_ids=[0] * prompt_len,
        completion_token_ids=[1] * (num_tokens - 1) + [answer],
        completion_logprobs=[-0.5] * num_tokens,
    )
    return Rollout(
        group_id=0, rollout_id=0, status=RolloutStatus.COMPLETED, turns=[turn]
    )


def _score(rubric_config: Rubric.Config, rollouts: list[Rollout]) -> list[float]:
    outputs = asyncio.run(rubric_config.build().score_group(rollouts, None))
    return [output.reward for output in outputs]


def test_kimi_length_reward() -> None:
    rubric = Rubric.Config(
        reward_fns=[_RewardLastToken.Config()],
        length_reward_weight=0.1,
    )
    # lam = 0.5 - (len - 1000) / 2000: +0.5, 0, -0.5 at 1,000 / 2,000 / 3,000 tokens.
    rollouts = [
        _rollout(1000, answer=RIGHT),
        # A 700-token prompt does not count as response length.
        _rollout(2000, answer=RIGHT, prompt_len=700),
        _rollout(3000, answer=RIGHT),
        _rollout(1000, answer=WRONG),  # a short wrong answer is not rewarded
        _rollout(3000, answer=WRONG),
    ]
    rewards = _score(rubric, rollouts)
    assert rewards == pytest.approx([1.05, 1.0, 0.95, 0.0, -0.05])
    # All lengths equal: no length reward.
    assert _score(rubric, [_rollout(10, answer=RIGHT), _rollout(10, answer=WRONG)]) == [
        1.0,
        0.0,
    ]
    # Weight 0 (the default): no `length_reward` in the breakdown, so no metric series for it.
    no_length_reward = Rubric.Config(reward_fns=[_RewardLastToken.Config()]).build()
    outputs = asyncio.run(no_length_reward.score_group(rollouts, None))
    assert all("length_reward" not in output.reward_breakdown for output in outputs)
