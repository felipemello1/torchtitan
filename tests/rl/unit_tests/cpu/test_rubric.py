# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Rubric's truncation handling: a fixed truncation_reward or an additive penalty."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchtitan.rl.rollout.rubric import RewardFn, Rubric
from torchtitan.rl.rollout.types import RolloutStatus


class _PartialCredit(RewardFn):
    """Scores the rollout's `credit` attribute, e.g. the fraction of tests it passed."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout, env_input) -> float:
        return rollout.credit


def _score(rubric: Rubric, *rollouts) -> list[tuple[float, dict[str, float]]]:
    outputs = asyncio.run(rubric.score_group(list(rollouts), env_input=None))
    return [(output.reward, output.reward_breakdown) for output in outputs]


def _rollout(credit: float, status: RolloutStatus = RolloutStatus.COMPLETED):
    return SimpleNamespace(credit=credit, status=status)


@pytest.mark.parametrize(
    "status",
    [
        RolloutStatus.TRUNCATED_MAX_TURNS,
        RolloutStatus.TRUNCATED_LENGTH,
        RolloutStatus.TRUNCATED_PROMPT_TOO_LONG,
    ],
)
def test_truncation_penalty_is_added_to_the_partial_credit(status) -> None:
    rubric = Rubric.Config(
        reward_fns=[_PartialCredit.Config(weight=2.0)], truncation_penalty=-0.2
    ).build()
    [(reward, breakdown)] = _score(rubric, _rollout(0.5, status))
    assert reward == pytest.approx(0.3)
    assert breakdown == {"_PartialCredit": 0.5, "truncation_penalty": -0.2}


def test_truncation_penalty_leaves_finished_and_errored_rollouts_alone() -> None:
    rubric = Rubric.Config(
        reward_fns=[_PartialCredit.Config()], truncation_penalty=-0.2
    ).build()
    finished, errored = _score(
        rubric,
        _rollout(0.5),
        _rollout(0.5, RolloutStatus.ERROR_TIMEOUT),
    )
    assert finished == (0.5, {"_PartialCredit": 0.5})
    assert errored == (0.5, {"_PartialCredit": 0.5})


def test_truncation_reward_still_short_circuits_the_reward_fns() -> None:
    rubric = Rubric.Config(
        reward_fns=[_PartialCredit.Config()], truncation_reward=0.0
    ).build()
    [(reward, breakdown)] = _score(
        rubric, _rollout(0.5, RolloutStatus.TRUNCATED_MAX_TURNS)
    )
    assert (reward, breakdown) == (0.0, {"truncated": 0.0})


def test_truncation_reward_and_penalty_are_exclusive() -> None:
    with pytest.raises(ValueError, match="not both"):
        Rubric.Config(
            reward_fns=[_PartialCredit.Config()],
            truncation_reward=0.0,
            truncation_penalty=-0.2,
        ).build()


def _spread_rubric() -> Rubric:
    return Rubric.Config(
        reward_fns=[_PartialCredit.Config()],
        truncation_penalty=-0.2,
        truncation_penalty_needs_spread=True,
    ).build()


@pytest.mark.parametrize(
    "status",
    [RolloutStatus.TRUNCATED_MAX_TURNS, RolloutStatus.TRUNCATED_REPETITION],
)
def test_spread_rule_waives_the_penalty_when_the_group_ties(status) -> None:
    finished, errored, truncated = _score(
        _spread_rubric(),
        _rollout(0.0),
        _rollout(0.0, RolloutStatus.ERROR),
        _rollout(0.0, status),
    )
    assert finished == (0.0, {"_PartialCredit": 0.0})
    assert errored == (0.0, {"_PartialCredit": 0.0})
    # Logged as 0.0, so the W&B component shows how often the rule waives the penalty.
    assert truncated == (0.0, {"_PartialCredit": 0.0, "truncation_penalty": 0.0})


def test_spread_rule_keeps_the_penalty_when_the_scores_differ() -> None:
    solved, truncated = _score(
        _spread_rubric(),
        _rollout(1.0),
        _rollout(0.0, RolloutStatus.TRUNCATED_REPETITION),
    )
    assert solved == (1.0, {"_PartialCredit": 1.0})
    assert truncated == (-0.2, {"_PartialCredit": 0.0, "truncation_penalty": -0.2})


def test_spread_rule_compares_scores_before_the_penalty() -> None:
    # 0.2 truncated would score 0.0, the same as its sibling, after the penalty.
    [_, (reward, _)] = _score(
        _spread_rubric(),
        _rollout(0.0),
        _rollout(0.2, RolloutStatus.TRUNCATED_LENGTH),
    )
    assert reward == pytest.approx(0.0)
    # A tie before the penalty waives it, even though the penalty would create spread.
    [_, (reward, _)] = _score(
        _spread_rubric(),
        _rollout(0.5),
        _rollout(0.5, RolloutStatus.TRUNCATED_LENGTH),
    )
    assert reward == 0.5


def test_spread_rule_keeps_the_penalty_for_a_lone_rollout() -> None:
    # A one-rollout group (e.g. a validation task with one sample) never trains; its
    # reward stays comparable to runs without the rule.
    [(reward, breakdown)] = _score(
        _spread_rubric(), _rollout(0.5, RolloutStatus.TRUNCATED_MAX_TURNS)
    )
    assert reward == pytest.approx(0.3)
    assert breakdown["truncation_penalty"] == -0.2


def test_spread_rule_ignores_nan_scores() -> None:
    [_, _, (reward, _)] = _score(
        _spread_rubric(),
        _rollout(float("nan")),
        _rollout(0.0),
        _rollout(0.0, RolloutStatus.TRUNCATED_MAX_TURNS),
    )
    assert reward == 0.0


def test_spread_rule_needs_a_truncation_penalty() -> None:
    with pytest.raises(ValueError, match="needs truncation_penalty"):
        Rubric.Config(
            reward_fns=[_PartialCredit.Config()], truncation_penalty_needs_spread=True
        ).build()
