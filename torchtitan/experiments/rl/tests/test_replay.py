# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for rollout-to-replay conversion."""

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.replay import (
    has_advantage_signal,
    ReplayBuffer,
    rollouts_to_replay_samples,
)
from torchtitan.experiments.rl.types import (
    ReplaySample,
    RolloutOutput,
    RolloutStatus,
    RolloutTurn,
)


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


def _sample(
    *,
    loss_tokens: int,
    behavior_version: int = 0,
    advantage: float = 1.0,
    sample_idx: int = 0,
) -> ReplaySample:
    return ReplaySample(
        token_ids=[1, *range(2, 2 + loss_tokens)],
        loss_mask=[0, *([1] * loss_tokens)],
        ref_logprobs=[0.0, *([-0.1] * loss_tokens)],
        advantage=advantage,
        group_id="g0",
        sample_idx=sample_idx,
        behavior_version=behavior_version,
        reward=1.0,
    )


def _metric_values(batch) -> dict[str, float]:
    return {metric.key: metric.value.value for metric in batch.metrics}


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


def test_has_advantage_signal_respects_epsilon() -> None:
    assert not has_advantage_signal([_sample(loss_tokens=1, advantage=0.0)])
    assert not has_advantage_signal(
        [_sample(loss_tokens=1, advantage=1e-13)],
        eps=1e-12,
    )
    assert has_advantage_signal(
        [_sample(loss_tokens=1, advantage=-1e-9)],
        eps=1e-12,
    )


def test_replay_buffer_batches_by_sample_budget() -> None:
    async def run() -> None:
        buffer = ReplayBuffer(max_samples=4)
        await buffer.put(
            [
                _sample(loss_tokens=1, sample_idx=0),
                _sample(loss_tokens=2, sample_idx=1),
                _sample(loss_tokens=1, sample_idx=2),
            ]
        )

        batch = await buffer.get_batch(min_samples=2, train_version=0)

        assert [sample.sample_idx for sample in batch.samples] == [0, 1]
        assert sum(sample.num_loss_tokens for sample in batch.samples) == 3
        metrics = _metric_values(batch)
        assert metrics["replay/num_samples"] == 2.0
        assert metrics["replay/num_loss_tokens"] == 3.0
        assert metrics["replay/buffer/depth_samples_pre_pull"] == 3.0
        assert metrics["replay/buffer/depth_samples_post_pull"] == 1.0

    asyncio.run(run())


def test_replay_buffer_requires_positive_sample_budget() -> None:
    async def run() -> None:
        buffer = ReplayBuffer(max_samples=4)
        with pytest.raises(ValueError, match="min_samples must be positive"):
            await buffer.get_batch(min_samples=0, train_version=0)

    asyncio.run(run())


def test_replay_buffer_drops_stale_samples_before_filling_batch() -> None:
    async def run() -> None:
        buffer = ReplayBuffer(max_samples=4, max_age_steps=1)
        await buffer.put(
            [
                _sample(loss_tokens=1, behavior_version=0, sample_idx=0),
                _sample(loss_tokens=1, behavior_version=2, sample_idx=1),
            ]
        )

        batch = await buffer.get_batch(min_samples=1, train_version=2)

        assert [sample.sample_idx for sample in batch.samples] == [1]
        assert [sample.sample_idx for sample in batch.dropped_samples] == [0]
        metrics = _metric_values(batch)
        assert metrics["replay/buffer/dropped_stale_samples"] == 1.0
        assert metrics["replay/buffer/max_observed_age_steps"] == 2.0

    asyncio.run(run())


def test_replay_buffer_close_unblocks_get_and_rejects_put() -> None:
    async def run() -> None:
        buffer = ReplayBuffer(max_samples=1)
        task = asyncio.create_task(buffer.get_batch(min_samples=1, train_version=0))
        await asyncio.sleep(0)

        await buffer.close()
        batch = await task

        assert batch.samples == []
        assert _metric_values(batch)["replay/num_samples"] == 0.0
        with pytest.raises(RuntimeError, match="closed ReplayBuffer"):
            await buffer.put(_sample(loss_tokens=1))

    asyncio.run(run())
