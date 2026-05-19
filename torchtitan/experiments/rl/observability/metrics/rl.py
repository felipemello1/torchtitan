# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metric builders for RL rollouts and replay batches."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.replay import ReplayBufferStats
from torchtitan.experiments.rl.types import ReplaySample, RolloutOutput, RolloutStatus


def rename_metric_prefix(
    metric: m.Metric,
    *,
    old_prefix: str,
    new_prefix: str,
) -> m.Metric:
    """Replace a metric key prefix while preserving its value object."""
    if metric.key.startswith(old_prefix):
        return m.Metric(new_prefix + metric.key[len(old_prefix) :], metric.value)
    if metric.key == "reward":
        return m.Metric(f"{new_prefix.rstrip('/')}/reward", metric.value)
    if metric.key.startswith("reward/"):
        return m.Metric(f"{new_prefix.rstrip('/')}/{metric.key}", metric.value)
    return metric


def build_replay_metrics(
    samples: Sequence[ReplaySample],
    replay_stats: ReplayBufferStats,
    *,
    dropped_empty_groups: int = 0,
    dropped_zero_advantage_groups: int = 0,
) -> list[m.Metric]:
    """Build replay-buffer and advantage metrics for one optimizer step."""
    advantages = [
        sample.advantage for sample in samples for _ in range(sample.num_loss_tokens)
    ]
    num_loss_tokens = sum(sample.num_loss_tokens for sample in samples)
    return [
        m.Metric("replay/num_samples", m.NoReduce(float(len(samples)))),
        m.Metric(
            "replay/num_loss_tokens",
            m.NoReduce(float(num_loss_tokens)),
        ),
        m.Metric("replay/buffer/groups", m.NoReduce(float(replay_stats.num_groups))),
        m.Metric(
            "replay/buffer/depth_groups",
            m.NoReduce(float(replay_stats.depth_groups)),
        ),
        m.Metric(
            "replay/buffer/dropped_stale_groups",
            m.NoReduce(float(replay_stats.num_dropped_stale_groups)),
        ),
        m.Metric(
            "rollout/dropped_empty_groups",
            m.NoReduce(float(dropped_empty_groups)),
        ),
        m.Metric(
            "rollout/dropped_zero_advantage_groups",
            m.NoReduce(float(dropped_zero_advantage_groups)),
        ),
        m.Metric(
            "replay/buffer/max_observed_age_steps",
            m.NoReduce(float(replay_stats.max_observed_age_steps)),
        ),
        m.Metric("advantage", m.SummaryStats.from_list(advantages)),
    ]


def build_rollout_metrics(
    rollouts: Sequence[RolloutOutput],
    *,
    generation_metrics: Sequence[m.Metric],
    prefix: str,
) -> list[m.Metric]:
    """Build rollout, reward, and generator metrics for a rollout set."""
    response_lens = [
        len(turn.response_token_ids) for rollout in rollouts for turn in rollout.turns
    ]
    prompt_lens = [
        len(turn.prompt_token_ids) for rollout in rollouts for turn in rollout.turns
    ]
    total_lens = [
        len(turn.prompt_token_ids) + len(turn.response_token_ids)
        for rollout in rollouts
        for turn in rollout.turns
    ]
    rewards = [
        float(rollout.reward) for rollout in rollouts if rollout.reward is not None
    ]
    truncated = [rollout.status == RolloutStatus.TRUNCATED for rollout in rollouts]
    errored = [rollout.status == RolloutStatus.ERROR for rollout in rollouts]

    metrics: list[m.Metric] = [
        m.Metric(f"{prefix}/response_length", m.Mean.from_list(response_lens)),
        m.Metric(f"{prefix}/response_length", m.Max.from_list(response_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Mean.from_list(prompt_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Max.from_list(prompt_lens)),
        m.Metric(f"{prefix}/total_length", m.Max.from_list(total_lens)),
        m.Metric(f"{prefix}/truncation_rate", m.Mean.from_list(truncated)),
        m.Metric(f"{prefix}/error_rate", m.Mean.from_list(errored)),
        m.Metric("reward", m.SummaryStats.from_list(rewards)),
    ]
    if rewards:
        by_group: dict[str, list[float]] = defaultdict(list)
        for rollout in rollouts:
            if rollout.reward is not None:
                by_group[rollout.group_id].append(float(rollout.reward))
        group_stds = [
            statistics.pstdev(group_rewards)
            for group_rewards in by_group.values()
            if group_rewards
        ]
        metrics.extend(
            [
                m.Metric("reward/group_std", m.Mean.from_list(group_stds)),
                m.Metric("reward/group_std", m.Max.from_list(group_stds)),
                m.Metric(
                    "reward/zero_std_frac",
                    m.NoReduce(
                        sum(1 for value in group_stds if value == 0.0) / len(group_stds)
                        if group_stds
                        else 0.0
                    ),
                ),
            ]
        )
    metrics += list(generation_metrics)
    metrics += _prepare_reward_metrics(
        prefix=f"{prefix}/reward/component",
        rollouts=rollouts,
    )
    return metrics


def _prepare_reward_metrics(
    prefix: str,
    rollouts: Sequence[RolloutOutput],
) -> list[m.Metric]:
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        for name, value in rollout.reward_components.items():
            values_by_name[name].append(float(value))
    return [
        m.Metric(f"{prefix}/{name}", m.Mean.from_list(values))
        for name, values in sorted(values_by_name.items())
    ]
