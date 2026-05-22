# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metric builders for RL rollouts."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import RolloutOutput, RolloutStatus


def rename_metric_prefix(
    metrics: Sequence[m.Metric],
    *,
    old_prefix: str,
    new_prefix: str,
) -> list[m.Metric]:
    """Replace a metric key prefix while preserving each value object.

    Example::

        rename_metric_prefix(
            [Metric("reward", Mean(1.0)), Metric("rollout/error_rate", Mean(0.0))],
            old_prefix="reward",
            new_prefix="validation/reward",
        )
        # -> ["validation/reward", "rollout/error_rate"]
    """
    out: list[m.Metric] = []
    for metric in metrics:
        if metric.key == old_prefix:
            out.append(m.Metric(new_prefix, metric.value))
        elif metric.key.startswith(f"{old_prefix}/"):
            out.append(
                m.Metric(
                    f"{new_prefix.rstrip('/')}/{metric.key[len(old_prefix) + 1:]}",
                    metric.value,
                )
            )
        else:
            out.append(metric)
    return out


def build_rollout_metrics(
    prefix: str,
    rollouts: Sequence[RolloutOutput],
    generation_metrics: Sequence[m.Metric] = (),
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
        zero_std_frac = (
            sum(1 for value in group_stds if value == 0.0) / len(group_stds)
            if group_stds
            else 0.0
        )
        metrics.extend(
            [
                m.Metric("reward/group_std", m.Mean.from_list(group_stds)),
                m.Metric("reward/group_std", m.Max.from_list(group_stds)),
                m.Metric("reward/zero_std_frac", m.Mean(zero_std_frac)),
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
