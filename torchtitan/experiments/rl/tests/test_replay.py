# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for rollout-to-replay conversion."""

from __future__ import annotations

import pytest

from torchtitan.experiments.rl.replay import rollouts_to_replay_samples
from torchtitan.experiments.rl.types import RolloutOutput, RolloutStatus, RolloutTurn


def _turn(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    logprobs: list[float] | None = None,
    policy_version: int = 7,
) -> RolloutTurn:
    return RolloutTurn(
        prompt_token_ids=prompt_ids,
        response_token_ids=response_ids,
        response_logprobs=(
            logprobs
            if logprobs is not None
            else [-0.1 * (idx + 1) for idx in range(len(response_ids))]
        ),
        policy_version=policy_version,
    )


def _rollout(
    *,
    group_id: str = "g0",
    sample_idx: int = 0,
    reward: float | None = 1.0,
    turns: list[RolloutTurn] | None = None,
) -> RolloutOutput:
    return RolloutOutput(
        group_id=group_id,
        sample_idx=sample_idx,
        status=RolloutStatus.COMPLETED,
        turns=turns
        if turns is not None
        else [_turn(prompt_ids=[1, 2], response_ids=[3, 4])],
        reward=reward,
        reward_components={"score": float(reward or 0.0)},
    )


def test_one_turn_rollout_becomes_one_replay_sample() -> None:
    samples = rollouts_to_replay_samples(
        [
            _rollout(group_id="g0", sample_idx=0, reward=1.0),
            _rollout(group_id="g0", sample_idx=1, reward=0.0),
        ]
    )

    assert len(samples) == 2
    assert sum(samples[0].loss_mask) == 2
    assert samples[0].token_ids == [1, 2, 3, 4]
    assert samples[0].ref_logprobs == [0.0, 0.0, -0.1, -0.2]
    assert [sample.advantage for sample in samples] == pytest.approx([1.0, -1.0])


def test_two_turn_prefix_continuous_rollout_coalesces() -> None:
    samples = rollouts_to_replay_samples(
        [
            _rollout(
                turns=[
                    _turn(prompt_ids=[1, 2], response_ids=[3]),
                    _turn(prompt_ids=[1, 2, 3, 4], response_ids=[5, 6]),
                ]
            )
        ]
    )

    assert len(samples) == 1
    assert samples[0].token_ids == [1, 2, 3, 4, 5, 6]
    assert samples[0].loss_mask == [0, 0, 1, 0, 1, 1]
    assert sum(samples[0].loss_mask) == 3
    assert samples[0].ref_logprobs[0:2] == [0.0, 0.0]
    assert samples[0].ref_logprobs[3] == 0.0


def test_prefix_break_flushes_current_row() -> None:
    samples = rollouts_to_replay_samples(
        [
            _rollout(
                turns=[
                    _turn(prompt_ids=[1, 2], response_ids=[3]),
                    _turn(prompt_ids=[9, 9], response_ids=[10]),
                ]
            )
        ]
    )

    assert [sample.token_ids for sample in samples] == [[1, 2, 3], [9, 9, 10]]
    assert [sum(sample.loss_mask) for sample in samples] == [1, 1]


def test_zero_variance_reward_group_keeps_zero_advantages() -> None:
    samples = rollouts_to_replay_samples(
        [
            _rollout(group_id="g0", sample_idx=0, reward=1.0),
            _rollout(group_id="g0", sample_idx=1, reward=1.0),
        ]
    )

    assert [sample.advantage for sample in samples] == [0.0, 0.0]
