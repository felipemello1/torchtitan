# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replay-row conversion for rollout outputs."""

from __future__ import annotations

import statistics
from collections import defaultdict

from torchtitan.experiments.rl.types import ReplaySample, RolloutOutput

_ADVANTAGE_STD_EPS = 1e-6


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
