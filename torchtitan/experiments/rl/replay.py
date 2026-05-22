# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replay-row conversion for rollout outputs."""

from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import ReplaySample, RolloutOutput

_ADVANTAGE_STD_EPS = 1e-6


@dataclass(frozen=True, kw_only=True, slots=True)
class ReplayBatch:
    """Replay samples plus buffer metrics for one optimizer step."""

    samples: list[ReplaySample]
    metrics: list[m.Metric]
    dropped_samples: list[ReplaySample] = field(default_factory=list)


def rollouts_to_replay_samples(rollouts: list[RolloutOutput]) -> list[ReplaySample]:
    """Convert rollout groups into token-aligned replay rows.

    Example::

        samples = rollouts_to_replay_samples([rollout0, rollout1])
        # Rewards [1.0, 0.0] in the same group become advantages [1.0, -1.0].
    """
    groups: dict[str, list[RolloutOutput]] = defaultdict(list)
    for rollout in rollouts:
        groups[rollout.group_id].append(rollout)

    samples: list[ReplaySample] = []
    for group in groups.values():
        rewards = [
            float(rollout.reward) for rollout in group if rollout.reward is not None
        ]
        if not rewards:
            continue
        mean = statistics.fmean(rewards)
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0

        for rollout in group:
            if rollout.reward is None:
                continue
            advantage = (
                (float(rollout.reward) - mean) / std
                if std > _ADVANTAGE_STD_EPS
                else 0.0
            )
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
    """Return whether any sample can contribute a non-zero policy update."""
    return any(abs(sample.advantage) > eps for sample in samples)


def _rollout_to_replay_samples(
    rollout: RolloutOutput,
    *,
    advantage: float,
) -> list[ReplaySample]:
    rows: list[ReplaySample] = []
    token_ids: list[int] = []
    loss_mask: list[int] = []
    ref_logprobs: list[float] = []

    def flush() -> None:
        nonlocal token_ids, loss_mask, ref_logprobs
        if not token_ids or not any(loss_mask):
            token_ids, loss_mask, ref_logprobs = [], [], []
            return
        rows.append(
            ReplaySample(
                token_ids=list(token_ids),
                loss_mask=list(loss_mask),
                ref_logprobs=list(ref_logprobs),
                advantage=advantage,
                group_id=rollout.group_id,
                sample_idx=rollout.sample_idx,
                behavior_version=rollout.behavior_version,
                reward=float(rollout.reward),
                reward_components=dict(rollout.reward_components),
            )
        )
        token_ids, loss_mask, ref_logprobs = [], [], []

    for turn in rollout.turns:
        if not token_ids:
            _append_prompt(token_ids, loss_mask, ref_logprobs, turn.prompt_token_ids)
        elif _is_prefix(token_ids, turn.prompt_token_ids):
            prompt_tail = turn.prompt_token_ids[len(token_ids) :]
            _append_prompt(token_ids, loss_mask, ref_logprobs, prompt_tail)
        else:
            flush()
            _append_prompt(token_ids, loss_mask, ref_logprobs, turn.prompt_token_ids)

        token_ids.extend(turn.response_token_ids)
        loss_mask.extend([1] * len(turn.response_token_ids))
        ref_logprobs.extend(turn.response_logprobs)

    flush()
    return rows


def _append_prompt(
    token_ids: list[int],
    loss_mask: list[int],
    ref_logprobs: list[float],
    prompt_token_ids: list[int],
) -> None:
    token_ids.extend(prompt_token_ids)
    loss_mask.extend([0] * len(prompt_token_ids))
    ref_logprobs.extend([0.0] * len(prompt_token_ids))


def _is_prefix(prefix: list[int], values: list[int]) -> bool:
    return values[: len(prefix)] == prefix


class ReplayBuffer:
    """Bounded FIFO of replay samples consumed by loss-token budget.

    Example::

        buffer = ReplayBuffer(max_samples=128, max_age_steps=1)
        await buffer.put(samples)
        batch = await buffer.get_batch(min_loss_tokens=8192, train_version=4)
    """

    def __init__(self, *, max_samples: int, max_age_steps: int | None = None):
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        self._max_samples = max_samples
        self._max_age_steps = max_age_steps
        self._samples: deque[ReplaySample] = deque()
        self._condition = asyncio.Condition()
        self._closed = False

    async def put(
        self,
        sample_or_samples: ReplaySample | Sequence[ReplaySample],
    ) -> None:
        """Append one sample or a sequence, blocking while the FIFO is full."""
        if isinstance(sample_or_samples, ReplaySample):
            samples: Sequence[ReplaySample] = (sample_or_samples,)
        else:
            samples = sample_or_samples

        for sample in samples:
            async with self._condition:
                while not self._closed and len(self._samples) >= self._max_samples:
                    await self._condition.wait()
                if self._closed:
                    raise RuntimeError("cannot put into a closed ReplayBuffer")
                self._samples.append(sample)
                self._condition.notify_all()

    async def get_batch(
        self,
        *,
        min_loss_tokens: int,
        train_version: int,
    ) -> ReplayBatch:
        """Pop FIFO samples until the loss-token budget is reached."""
        if min_loss_tokens <= 0:
            raise ValueError(f"min_loss_tokens must be positive, got {min_loss_tokens}")

        consumed: list[ReplaySample] = []
        dropped_samples: list[ReplaySample] = []
        consumed_loss_tokens = 0
        num_dropped_stale_samples = 0
        max_age = 0

        async with self._condition:
            pre_pull_depth_samples: int | None = None
            while consumed_loss_tokens < min_loss_tokens:
                while not self._closed and not self._samples:
                    await self._condition.wait()
                if self._closed and not self._samples:
                    break

                if pre_pull_depth_samples is None:
                    pre_pull_depth_samples = len(self._samples)
                sample = self._samples.popleft()
                age = max(train_version - sample.behavior_version, 0)
                max_age = max(max_age, age)
                if self._max_age_steps is not None and age > self._max_age_steps:
                    dropped_samples.append(sample)
                    num_dropped_stale_samples += 1
                    self._condition.notify_all()
                    continue

                consumed.append(sample)
                consumed_loss_tokens += sample.num_loss_tokens
                self._condition.notify_all()
            post_pull_depth_samples = len(self._samples)
            self._condition.notify_all()

        pre_pull_depth_samples = pre_pull_depth_samples or 0
        metrics = [
            m.Metric("replay/num_samples", m.NoReduce(float(len(consumed)))),
            m.Metric("replay/num_loss_tokens", m.NoReduce(float(consumed_loss_tokens))),
            m.Metric(
                "replay/buffer/depth_samples_pre_pull",
                m.NoReduce(float(pre_pull_depth_samples)),
            ),
            m.Metric(
                "replay/buffer/depth_samples_post_pull",
                m.NoReduce(float(post_pull_depth_samples)),
            ),
            m.Metric(
                "replay/buffer/dropped_stale_samples",
                m.NoReduce(float(num_dropped_stale_samples)),
            ),
            m.Metric(
                "replay/buffer/max_observed_age_steps",
                m.NoReduce(float(max_age)),
            ),
        ]
        return ReplayBatch(
            samples=consumed,
            metrics=metrics,
            dropped_samples=dropped_samples,
        )

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
