# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replay conversion and buffering for async rollout training."""

from __future__ import annotations

import asyncio
import statistics
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from torchtitan.experiments.rl.types import ReplaySample, RolloutOutput

_ADVANTAGE_STD_EPS = 1e-6


@dataclass(frozen=True, kw_only=True, slots=True)
class ReplayGroup:
    """Replay samples produced from one rollout group.

    ``behavior_version`` is the minimum policy version observed in the group
    and is used for conservative stale-drop decisions. ``max_behavior_version``
    keeps request-level weight-sync behavior visible when a multiturn group
    spans a policy update.
    """

    group_id: str
    samples: list[ReplaySample]
    rollouts: list[RolloutOutput]
    behavior_version: int
    max_behavior_version: int

    @classmethod
    def from_rollouts(
        cls,
        *,
        samples: list[ReplaySample],
        rollouts: list[RolloutOutput],
    ) -> "ReplayGroup":
        """Build a replay group and derive behavior-version bounds once."""
        if not rollouts:
            raise ValueError("replay group has no rollouts")
        group_id = rollouts[0].group_id
        mismatched_rollout_ids = sorted(
            {rollout.group_id for rollout in rollouts if rollout.group_id != group_id}
        )
        if mismatched_rollout_ids:
            raise ValueError(
                f"replay group {group_id!r} contains rollout group_ids "
                f"{mismatched_rollout_ids}"
            )
        mismatched_sample_ids = sorted(
            {sample.group_id for sample in samples if sample.group_id != group_id}
        )
        if mismatched_sample_ids:
            raise ValueError(
                f"replay group {group_id!r} contains sample group_ids "
                f"{mismatched_sample_ids}"
            )
        versioned_rollouts = [rollout for rollout in rollouts if rollout.turns]
        if not versioned_rollouts:
            raise ValueError(f"replay group {group_id!r} has no versioned rollouts")
        return cls(
            group_id=group_id,
            samples=samples,
            rollouts=rollouts,
            behavior_version=min(
                rollout.behavior_version for rollout in versioned_rollouts
            ),
            max_behavior_version=max(
                rollout.max_behavior_version for rollout in versioned_rollouts
            ),
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class ReplayBufferStats:
    """Counters returned with each replay batch."""

    num_groups: int
    num_samples: int
    num_loss_tokens: int
    num_dropped_stale_groups: int
    max_observed_age_steps: int
    depth_groups: int


@dataclass(frozen=True, kw_only=True, slots=True)
class ReplayBatch:
    """Replay rows consumed by one optimizer step."""

    groups: list[ReplayGroup]
    samples: list[ReplaySample]
    stats: ReplayBufferStats


def rollouts_to_replay_samples(rollouts: list[RolloutOutput]) -> list[ReplaySample]:
    """Convert a GRPO rollout group into token-aligned replay rows."""
    rewards = [float(r.reward) for r in rollouts if r.reward is not None]
    if not rewards:
        return []
    group_mean = sum(rewards) / len(rewards)
    group_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0

    samples: list[ReplaySample] = []
    for rollout in rollouts:
        if rollout.reward is None:
            continue
        advantage = 0.0
        if group_std > _ADVANTAGE_STD_EPS:
            advantage = (float(rollout.reward) - group_mean) / group_std
        samples.extend(
            _rollout_to_replay_samples(
                rollout,
                advantage=advantage,
            )
        )
    return samples


def has_advantage_signal(
    samples: Sequence[ReplaySample],
    *,
    eps: float = 1e-12,
) -> bool:
    """Return true when a replay group can produce a non-zero GRPO update."""
    return any(abs(sample.advantage) > eps for sample in samples)


def _rollout_to_replay_samples(
    rollout: RolloutOutput,
    *,
    advantage: float,
) -> list[ReplaySample]:
    rows: list[ReplaySample] = []
    token_ids: list[int] = []
    loss_mask: list[int] = []
    behavior_logprobs: list[float] = []

    def flush() -> None:
        nonlocal token_ids, loss_mask, behavior_logprobs
        if not token_ids or not any(loss_mask):
            token_ids, loss_mask, behavior_logprobs = [], [], []
            return
        rows.append(
            ReplaySample(
                token_ids=list(token_ids),
                loss_mask=list(loss_mask),
                behavior_logprobs=list(behavior_logprobs),
                advantage=advantage,
                group_id=rollout.group_id,
                sample_idx=rollout.sample_idx,
                behavior_version=rollout.behavior_version,
                reward=float(rollout.reward),
                reward_components=dict(rollout.reward_components),
            )
        )
        token_ids, loss_mask, behavior_logprobs = [], [], []

    for turn in rollout.turns:
        if not token_ids:
            _append_prompt(
                token_ids,
                loss_mask,
                behavior_logprobs,
                turn.prompt_token_ids,
            )
        elif _is_prefix(token_ids, turn.prompt_token_ids):
            prompt_tail = turn.prompt_token_ids[len(token_ids) :]
            _append_prompt(
                token_ids,
                loss_mask,
                behavior_logprobs,
                prompt_tail,
            )
        else:
            flush()
            _append_prompt(
                token_ids,
                loss_mask,
                behavior_logprobs,
                turn.prompt_token_ids,
            )

        token_ids.extend(turn.response_token_ids)
        loss_mask.extend([1] * len(turn.response_token_ids))
        behavior_logprobs.extend(turn.response_logprobs)

    flush()
    return rows


def _append_prompt(
    token_ids: list[int],
    loss_mask: list[int],
    behavior_logprobs: list[float],
    prompt_token_ids: list[int],
) -> None:
    token_ids.extend(prompt_token_ids)
    loss_mask.extend([0] * len(prompt_token_ids))
    behavior_logprobs.extend([0.0] * len(prompt_token_ids))


def _is_prefix(prefix: list[int], values: list[int]) -> bool:
    return values[: len(prefix)] == prefix


class ReplayBuffer:
    """Bounded FIFO buffer of completed rollout groups."""

    def __init__(self, *, max_groups: int, max_age_steps: int | None = None):
        if max_groups <= 0:
            raise ValueError(f"max_groups must be positive, got {max_groups}")
        self._max_groups = max_groups
        self._max_age_steps = max_age_steps
        self._groups: deque[ReplayGroup] = deque()
        self._condition = asyncio.Condition()
        self._closed = False

    async def put(self, group: ReplayGroup) -> None:
        """Append a group, blocking while the FIFO is full."""
        async with self._condition:
            while not self._closed and len(self._groups) >= self._max_groups:
                await self._condition.wait()
            if self._closed:
                raise RuntimeError("cannot put into a closed ReplayBuffer")
            self._groups.append(group)
            self._condition.notify_all()

    async def get_batch(
        self,
        *,
        min_groups: int,
        train_version: int,
    ) -> ReplayBatch:
        """Pop FIFO groups until at least ``min_groups`` are available.

        If the buffer is closed before enough groups arrive, this returns the
        valid groups consumed so far. Callers that require a full batch should
        enforce that postcondition.
        """
        if min_groups <= 0:
            raise ValueError(f"min_groups must be positive, got {min_groups}")

        return await self._consume(
            train_version=train_version,
            min_groups=min_groups,
        )

    async def _consume(
        self,
        *,
        train_version: int,
        min_groups: int,
    ) -> ReplayBatch:
        groups: list[ReplayGroup] = []
        samples: list[ReplaySample] = []
        num_dropped_stale_groups = 0
        max_age = 0

        async with self._condition:
            while True:
                if len(groups) >= min_groups:
                    break
                while not self._closed and not self._groups:
                    await self._condition.wait()
                if self._closed and not self._groups:
                    break

                group = self._groups.popleft()
                age = max(train_version - group.behavior_version, 0)
                max_age = max(max_age, age)
                if self._max_age_steps is not None and age > self._max_age_steps:
                    num_dropped_stale_groups += 1
                    self._condition.notify_all()
                    continue
                samples.extend(group.samples)
                groups.append(group)
                self._condition.notify_all()

            self._condition.notify_all()

        return ReplayBatch(
            groups=groups,
            samples=samples,
            stats=ReplayBufferStats(
                num_groups=len(groups),
                num_samples=len(samples),
                num_loss_tokens=sum(sample.num_loss_tokens for sample in samples),
                num_dropped_stale_groups=num_dropped_stale_groups,
                max_observed_age_steps=max_age,
                depth_groups=len(self._groups),
            ),
        )

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
