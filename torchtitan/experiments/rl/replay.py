# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replay conversion and FIFO queue for async rollout training."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

from torchtitan.experiments.rl.types import ReplaySample, RolloutOutput


@dataclass(frozen=True, kw_only=True, slots=True)
class ReplayGroup:
    """Replay samples produced from one rollout group."""

    group_id: str
    samples: list[ReplaySample]
    behavior_version: int
    train_step: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True, slots=True)
class QueueStats:
    """Counters returned when the trainer consumes the rollout queue."""

    num_groups: int
    num_samples: int
    num_loss_tokens: int
    num_dropped_stale_groups: int
    max_age_steps: int


def rollouts_to_replay_samples(rollouts: list[RolloutOutput]) -> list[ReplaySample]:
    """Convert a GRPO rollout group into token-aligned replay rows."""
    rewards = [float(r.reward) for r in rollouts if r.reward is not None]
    if not rewards:
        return []
    group_mean = sum(rewards) / len(rewards)

    samples: list[ReplaySample] = []
    for rollout in rollouts:
        if rollout.reward is None:
            continue
        samples.extend(
            _rollout_to_replay_samples(
                rollout,
                advantage=float(rollout.reward) - group_mean,
            )
        )
    return samples


def _rollout_to_replay_samples(
    rollout: RolloutOutput,
    *,
    advantage: float,
) -> list[ReplaySample]:
    rows: list[ReplaySample] = []
    token_ids: list[int] = []
    loss_mask: list[int] = []
    behavior_logprobs: list[float] = []
    advantages: list[float] = []

    def flush() -> None:
        nonlocal token_ids, loss_mask, behavior_logprobs, advantages
        if not token_ids or not any(loss_mask):
            token_ids, loss_mask, behavior_logprobs, advantages = [], [], [], []
            return
        rows.append(
            ReplaySample(
                token_ids=list(token_ids),
                loss_mask=list(loss_mask),
                behavior_logprobs=list(behavior_logprobs),
                advantages=list(advantages),
                group_id=rollout.group_id,
                sample_idx=rollout.sample_idx,
                behavior_version=rollout.behavior_version,
                reward=float(rollout.reward),
                reward_components=dict(rollout.reward_components),
                metrics=dict(rollout.metrics),
            )
        )
        token_ids, loss_mask, behavior_logprobs, advantages = [], [], [], []

    for turn in rollout.turns:
        if not token_ids:
            _append_prompt(
                token_ids,
                loss_mask,
                behavior_logprobs,
                advantages,
                turn.prompt_token_ids,
            )
        elif _is_prefix(token_ids, turn.prompt_token_ids):
            prompt_tail = turn.prompt_token_ids[len(token_ids) :]
            _append_prompt(
                token_ids,
                loss_mask,
                behavior_logprobs,
                advantages,
                prompt_tail,
            )
        else:
            flush()
            _append_prompt(
                token_ids,
                loss_mask,
                behavior_logprobs,
                advantages,
                turn.prompt_token_ids,
            )

        token_ids.extend(turn.response_token_ids)
        loss_mask.extend([1] * len(turn.response_token_ids))
        behavior_logprobs.extend(turn.response_logprobs)
        advantages.extend([advantage] * len(turn.response_token_ids))

    flush()
    return rows


def _append_prompt(
    token_ids: list[int],
    loss_mask: list[int],
    behavior_logprobs: list[float],
    advantages: list[float],
    prompt_token_ids: list[int],
) -> None:
    token_ids.extend(prompt_token_ids)
    loss_mask.extend([0] * len(prompt_token_ids))
    behavior_logprobs.extend([0.0] * len(prompt_token_ids))
    advantages.extend([0.0] * len(prompt_token_ids))


def _is_prefix(prefix: list[int], values: list[int]) -> bool:
    return values[: len(prefix)] == prefix


class RolloutQueue:
    """Bounded FIFO queue of completed rollout groups."""

    def __init__(self, *, max_groups: int, max_age_steps: int | None = None):
        if max_groups <= 0:
            raise ValueError(f"max_groups must be positive, got {max_groups}")
        self._max_groups = max_groups
        self._max_age_steps = max_age_steps
        self._groups: deque[ReplayGroup] = deque()
        self._condition = asyncio.Condition()
        self._closed = False

    def qsize(self) -> int:
        return len(self._groups)

    async def put(self, group: ReplayGroup) -> None:
        """Append a group, blocking while the FIFO is full."""
        async with self._condition:
            while not self._closed and len(self._groups) >= self._max_groups:
                await self._condition.wait()
            if self._closed:
                raise RuntimeError("cannot put into a closed RolloutQueue")
            self._groups.append(group)
            self._condition.notify_all()

    async def get_batch(
        self,
        *,
        min_samples: int,
        train_version: int,
    ) -> tuple[list[ReplaySample], QueueStats]:
        """Pop FIFO groups until at least ``min_samples`` are available.

        If the queue is closed before enough samples arrive, this returns the
        valid samples consumed so far. Callers that require a full batch should
        enforce that postcondition.
        """
        if min_samples <= 0:
            raise ValueError(f"min_samples must be positive, got {min_samples}")

        return await self._consume(
            train_version=train_version,
            min_samples=min_samples,
            until_closed=False,
        )

    async def get_all(
        self,
        *,
        train_version: int,
    ) -> tuple[list[ReplaySample], QueueStats]:
        """Pop FIFO groups until producers close the queue."""
        return await self._consume(
            train_version=train_version,
            min_samples=None,
            until_closed=True,
        )

    async def _consume(
        self,
        *,
        train_version: int,
        min_samples: int | None,
        until_closed: bool,
    ) -> tuple[list[ReplaySample], QueueStats]:
        samples: list[ReplaySample] = []
        num_groups = 0
        num_dropped_stale_groups = 0
        max_age = 0

        async with self._condition:
            while True:
                target_met = min_samples is not None and len(samples) >= min_samples
                if target_met and not until_closed:
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
                num_groups += 1
                self._condition.notify_all()

            self._condition.notify_all()

        return samples, QueueStats(
            num_groups=num_groups,
            num_samples=len(samples),
            num_loss_tokens=sum(sample.num_loss_tokens for sample in samples),
            num_dropped_stale_groups=num_dropped_stale_groups,
            max_age_steps=max_age,
        )

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
