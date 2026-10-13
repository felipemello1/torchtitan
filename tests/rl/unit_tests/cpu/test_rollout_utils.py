# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `TrainingSampleBuilder.rollout_to_training_samples`."""

from __future__ import annotations

import pytest

from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.observability.controller import compute_rollout_metrics
from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.types import RolloutTurnID

_GROUP_ID = "step=1/group=0"

# rollout_to_training_samples is the TrainingSampleBuilder override hook; the transform is config-independent.
rollout_to_training_samples = (
    TrainingSampleBuilder.Config().build().rollout_to_training_samples
)


def _turn(
    *,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    version: int,
    max_version: int | None = None,
    content: str = "x",
) -> RolloutTurn:
    return RolloutTurn(
        rollout_id=RolloutTurnID(group_id=_GROUP_ID, rollout_id=0, turn_id=0),
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        completion_logprobs=[-0.1] * len(completion_token_ids),
        min_policy_version=version,
        max_policy_version=version if max_version is None else max_version,
        completion_message={"role": "assistant", "content": content},
    )


def _scored_rollout(
    turns: list[RolloutTurn], *, reward: float, advantage: float
) -> Rollout:
    return Rollout(
        group_id=_GROUP_ID,
        rollout_id=0,
        status=RolloutStatus.COMPLETED,
        turns=turns,
        reward=reward,
        advantage=advantage,
    )


def test_single_turn_packs_one_training_sample() -> None:
    rollout = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=2)],
        reward=1.0,
        advantage=0.5,
    )
    [training_sample] = rollout_to_training_samples(rollout)
    assert training_sample.token_ids == [1, 2, 4, 5]
    assert training_sample.loss_mask == [False, False, True, True]
    assert training_sample.logprobs == [0.0, 0.0, -0.1, -0.1]
    assert training_sample.min_policy_version == 2
    # advantage on the two completion tokens, 0.0 on the prompt
    assert training_sample.advantage == [0.0, 0.0, 0.5, 0.5]
    assert training_sample.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )
    # single turn at version 2 -> min == max == 2
    assert training_sample.max_policy_version == 2


def test_packs_min_and_max_version_across_turns() -> None:
    # Two turns sampled at different versions (a weight pull landed between them); the packed
    # training_sample records the min (oldest, turn-0) and max (newest, turn-1) version.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=3),
            _turn(
                prompt_token_ids=[1, 2, 4, 5, 9], completion_token_ids=[7], version=4
            ),
        ],
        reward=1.0,
        advantage=0.5,
    )
    [training_sample] = rollout_to_training_samples(rollout)
    assert training_sample.token_ids == [1, 2, 4, 5, 9, 7]
    assert training_sample.min_policy_version == 3  # oldest (turn 0)
    assert training_sample.max_policy_version == 4  # newest (turn 1)


def test_multiturn_with_growing_prefix_packs_into_one_training_sample() -> None:
    # Each turn's prompt extends the prior turn's prompt+completion with the env reply
    # ([8], then [9]), so the whole trajectory packs into one training_sample.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=2),
            _turn(
                prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[5, 6], version=2
            ),
            _turn(
                prompt_token_ids=[1, 2, 4, 8, 5, 6, 9],
                completion_token_ids=[7],
                version=2,
            ),
        ],
        reward=0.8,
        advantage=-0.2,
    )
    [training_sample] = rollout_to_training_samples(rollout)
    #               P     P    a0    E1    a1    a1    E2    a2
    assert training_sample.token_ids == [1, 2, 4, 8, 5, 6, 9, 7]
    assert training_sample.loss_mask == [
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert training_sample.logprobs == [0.0, 0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1]
    # advantage broadcast onto every assistant token, 0.0 on prompt/env tokens
    assert training_sample.advantage == [0.0, 0.0, -0.2, 0.0, -0.2, -0.2, 0.0, -0.2]
    assert training_sample.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )


def test_history_edit_branches_into_separate_training_samples() -> None:
    # Turn 1's prompt does NOT extend turn 0's prompt+completion (the env rewrote history),
    # so the trajectory splits into two training_samples.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[90, 91], completion_token_ids=[5], version=1),
        ],
        reward=0.5,
        advantage=0.1,
    )
    first, second = rollout_to_training_samples(rollout)
    assert first.token_ids == [1, 2, 4]
    assert first.loss_mask == [False, False, True]
    # first segment opens at turn 0, the branch (history edit) opens at turn 1
    assert first.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )
    assert second.token_ids == [90, 91, 5]
    assert second.loss_mask == [False, False, True]
    assert second.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=1
    )
    assert first.advantage == [0.0, 0.0, 0.1]
    assert second.advantage == [0.0, 0.0, 0.1]


def test_empty_completion_on_a_later_turn_raises() -> None:
    # An empty completion is only expected on the first turn (initial prompt too long). On any later
    # turn it is an anomaly, so building the training samples raises.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[], version=1),
        ],
        reward=0.0,
        advantage=0.0,
    )
    with pytest.raises(ValueError, match="no completion"):
        rollout_to_training_samples(rollout)


def test_empty_completion_on_first_turn_is_skipped() -> None:
    # First turn with no completion = initial prompt too long; expected, so the rollout yields no samples.
    rollout = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[], version=1)],
        reward=0.0,
        advantage=0.0,
    )
    assert rollout_to_training_samples(rollout) == []


def test_group_filters_preserve_group_id_for_acknowledgement() -> None:
    builder = TrainingSampleBuilder.Config().build()

    failed = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=7, rollouts=[])
    )

    empty_completion = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[], version=1)],
        reward=0.0,
        advantage=0.0,
    )
    empty_completion.group_id = 8
    untrainable = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=8, rollouts=[empty_completion])
    )

    first = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[2], version=1)],
        reward=1.0,
        advantage=0.0,
    )
    second = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[3], version=1)],
        reward=1.0,
        advantage=0.0,
    )
    first.group_id = second.group_id = 9
    zero_std = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=9, rollouts=[first, second])
    )

    invalid_logprob = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[2], version=1)],
        reward=1.0,
        advantage=1.0,
    )
    invalid_logprob.group_id = 10
    invalid_logprob.turns[0].completion_logprobs = [float("nan")]
    no_valid_tokens = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=10, rollouts=[invalid_logprob])
    )

    timed_out = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[2], version=1)],
        reward=0.0,
        advantage=0.0,
    )
    timed_out.group_id = 11
    timed_out.status = RolloutStatus.ERROR_TIMEOUT
    all_timed_out = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=11, rollouts=[timed_out])
    )

    filtered_groups = [failed, untrainable, zero_std, no_valid_tokens, all_timed_out]
    assert [group.group_id for group in filtered_groups] == [7, 8, 9, 10, 11]
    assert all(not group.training_samples for group in filtered_groups)
    assert "training_sample_builder/num_groups_dropped_errored" in [
        metric.key for metric in all_timed_out.metrics
    ]


_COMPLETED = RolloutStatus.COMPLETED
_TIMEOUT = RolloutStatus.ERROR_TIMEOUT
_ERROR = RolloutStatus.ERROR


def _group(statuses: list[RolloutStatus], rewards: list[float]) -> RolloutGroup:
    """One single-turn group; rollout i completes with token 10 + i."""
    return RolloutGroup(
        group_id=0,
        rollouts=[
            Rollout(
                group_id=0,
                rollout_id=rollout_id,
                status=status,
                turns=[
                    _turn(
                        prompt_token_ids=[1],
                        completion_token_ids=[10 + rollout_id],
                        version=1,
                    )
                ],
                reward=reward,
            )
            for rollout_id, (status, reward) in enumerate(
                zip(statuses, rewards, strict=True)
            )
        ],
    )


@pytest.mark.parametrize(
    "should_std_normalize, expected",
    [
        # Mean of the 3 kept rewards is 2/3; the timed-out 0.0 doesn't drag it down.
        (False, [1 / 3, -2 / 3, 0.0, 1 / 3]),
        # Std of the kept rewards [1, 0, 1] is sqrt(2)/3.
        (True, [2**-0.5, -(2**0.5), 0.0, 2**-0.5]),
    ],
)
def test_timed_out_rollout_is_left_out_of_group_advantage_stats(
    should_std_normalize: bool, expected: list[float]
) -> None:
    estimator = AdvantageEstimator.Config(
        should_std_normalize=should_std_normalize
    ).build()
    group = _group(
        [_COMPLETED, _COMPLETED, _TIMEOUT, _COMPLETED], rewards=[1.0, 0.0, 0.0, 1.0]
    )

    assert estimator(group) == pytest.approx(expected, rel=1e-5)


def test_all_timed_out_group_gets_zero_advantages() -> None:
    estimator = AdvantageEstimator.Config(should_std_normalize=True).build()
    group = _group([_TIMEOUT, _TIMEOUT], rewards=[1.0, 0.0])

    assert estimator(group) == [0.0, 0.0]


def test_timed_out_rollout_is_dropped_from_training_samples() -> None:
    builder = TrainingSampleBuilder.Config().build()
    group = _group([_COMPLETED, _TIMEOUT, _COMPLETED], rewards=[1.0, 0.0, 0.0])
    for rollout, advantage in zip(
        group.rollouts, AdvantageEstimator.Config().build()(group), strict=True
    ):
        rollout.advantage = advantage

    output = builder.build_from_group(rollout_group=group)

    assert [sample.token_ids for sample in output.training_samples] == [
        [1, 10],
        [1, 12],
    ]
    assert [sample.advantage for sample in output.training_samples] == [
        [0.0, 0.5],
        [0.0, -0.5],
    ]


def test_zero_std_check_ignores_timed_out_rollouts() -> None:
    builder = TrainingSampleBuilder.Config().build()
    # Kept rewards are [1.0, 1.0]: no learning signal once the timeout is left out.
    group = _group([_TIMEOUT, _COMPLETED, _COMPLETED], rewards=[0.0, 1.0, 1.0])
    for rollout in group.rollouts:
        rollout.advantage = 0.0

    output = builder.build_from_group(rollout_group=group)

    assert output.training_samples == []
    assert "training_sample_builder/num_groups_dropped_zero_std" in [
        metric.key for metric in output.metrics
    ]


def test_errored_rollout_is_left_out_like_a_timeout() -> None:
    group = _group([_COMPLETED, _ERROR, _COMPLETED], rewards=[1.0, 0.0, 0.0])
    for rollout, advantage in zip(
        group.rollouts, AdvantageEstimator.Config().build()(group), strict=True
    ):
        rollout.advantage = advantage

    output = TrainingSampleBuilder.Config().build().build_from_group(rollout_group=group)

    assert [sample.advantage for sample in output.training_samples] == [
        [0.0, 0.5],
        [0.0, -0.5],
    ]
    assert "training_sample_builder/num_rollouts_dropped_errored" in [
        metric.key for metric in output.metrics
    ]


def test_group_with_one_clean_rollout_is_dropped() -> None:
    group = _group([_TIMEOUT, _ERROR, _COMPLETED], rewards=[0.0, 0.0, 1.0])
    for rollout in group.rollouts:
        rollout.advantage = 0.0

    dropped = TrainingSampleBuilder.Config().build().build_from_group(rollout_group=group)
    kept = (
        TrainingSampleBuilder.Config(drop_errored_rollouts=False)
        .build()
        .build_from_group(rollout_group=group)
    )

    assert dropped.training_samples == []
    assert "training_sample_builder/num_groups_dropped_errored" in [
        metric.key for metric in dropped.metrics
    ]
    assert len(kept.training_samples) == 3


def test_rollout_metrics_report_timeout_rate() -> None:
    group = _group(
        [_COMPLETED, _TIMEOUT, _COMPLETED, _TIMEOUT], rewards=[1.0, 0.0, 0.0, 0.0]
    )

    metrics = compute_rollout_metrics(prefix="validation", rollouts=group.rollouts)

    (timeout_rate,) = [
        metric.value for metric in metrics if metric.key == "validation/timeout_rate"
    ]
    assert m.Mean.reduce([timeout_rate]) == {"mean": 0.5}
