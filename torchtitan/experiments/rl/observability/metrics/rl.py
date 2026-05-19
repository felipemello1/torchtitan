# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metric builders for RL rollouts and replay batches."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.replay import ReplayBatch, ReplayBufferStats
from torchtitan.experiments.rl.types import ReplaySample, RolloutOutput, RolloutStatus


_TRACE_FWD_BWD_KEYS = (
    "loss/ratio/nonfinite_frac",
    "loss/logprob/policy_nonfinite_frac",
    "loss/logprob/behavior_nonfinite_frac",
    "bit_wise/nonfinite_logprob_frac",
)


def validate_train_step_fwd_bwd_metrics(
    fwd_bwd_metrics: Mapping[str, float],
) -> None:
    """Raise if train-step health metrics required by the controller are absent."""
    missing_trace_metrics = [
        key for key in _TRACE_FWD_BWD_KEYS if key not in fwd_bwd_metrics
    ]
    if missing_trace_metrics:
        raise KeyError(
            "fwd_bwd_metrics missing required train-step metrics: "
            f"{missing_trace_metrics}"
        )


@dataclass(frozen=True, slots=True)
class _WeightSyncTimings:
    """Wall-clock timing for one trainer-to-generator weight sync."""

    admission_drain_s: float
    push_s: float
    pull_s: float
    total_s: float


@dataclass(frozen=True, slots=True)
class _TrainStepTimings:
    """Wall-clock timings logged for one training step."""

    step_s: float
    replay_wait_s: float
    train_s: float
    checkpoint_s: float
    weight_sync: _WeightSyncTimings


def _zero_weight_sync_timings() -> _WeightSyncTimings:
    """Zero timings for train steps that do not publish new weights."""
    return _WeightSyncTimings(
        admission_drain_s=0.0,
        push_s=0.0,
        pull_s=0.0,
        total_s=0.0,
    )


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


def build_train_step_metrics(
    *,
    samples: list[ReplaySample],
    replay_batch: ReplayBatch,
    rollouts: list[RolloutOutput],
    live_generation_metrics: list[m.Metric],
    fwd_bwd_metrics: dict[str, float],
    optimizer_metrics: dict[str, float],
    checkpoint_saved: bool,
    timings: _TrainStepTimings,
    dropped_empty_groups: int,
    dropped_zero_advantage_groups: int,
    train_version: int,
    drop_metrics: Sequence[m.Metric] = (),
) -> tuple[list[m.Metric], dict[str, float]]:
    """Build metric records and structured scalars for one train step."""
    total_tokens = sum(len(sample.token_ids) for sample in samples)
    tokens_per_second = total_tokens / timings.step_s if timings.step_s > 0.0 else 0.0
    behavior_versions = [group.behavior_version for group in replay_batch.groups]
    max_behavior_versions = [
        group.max_behavior_version for group in replay_batch.groups
    ]
    behavior_version_min = min(behavior_versions) if behavior_versions else 0
    behavior_version_max = max(max_behavior_versions) if max_behavior_versions else 0

    metrics: list[m.Metric] = []
    metrics += build_rollout_metrics(
        rollouts,
        generation_metrics=[],
        prefix="rollout",
    )
    metrics += live_generation_metrics
    metrics += build_replay_metrics(
        samples,
        replay_batch.stats,
        dropped_empty_groups=dropped_empty_groups,
        dropped_zero_advantage_groups=dropped_zero_advantage_groups,
    )
    metrics += list(drop_metrics)
    metrics += [
        m.Metric("replay/policy_version/train", m.NoReduce(float(train_version))),
        m.Metric(
            "replay/policy_version/behavior_min",
            m.NoReduce(float(behavior_version_min)),
        ),
        m.Metric(
            "replay/policy_version/behavior_max",
            m.NoReduce(float(behavior_version_max)),
        ),
    ]
    metrics += [m.Metric(k, m.NoReduce(v)) for k, v in fwd_bwd_metrics.items()]
    metrics += [m.Metric(k, m.NoReduce(v)) for k, v in optimizer_metrics.items()]
    metrics.append(m.Metric("checkpoint/saved", m.NoReduce(float(checkpoint_saved))))
    for key, value in [
        ("timing/step", timings.step_s),
        ("timing/replay_wait", timings.replay_wait_s),
        ("timing/train", timings.train_s),
        ("timing/weight_sync/admission_drain", timings.weight_sync.admission_drain_s),
        ("timing/weight_sync/push", timings.weight_sync.push_s),
        ("timing/weight_sync/pull", timings.weight_sync.pull_s),
        ("timing/weight_sync/total", timings.weight_sync.total_s),
        ("timing/checkpoint", timings.checkpoint_s),
    ]:
        metrics.append(m.Metric(key, m.NoReduce(value)))
    metrics.append(
        m.Metric(
            "perf/tokens_per_second",
            m.NoReduce(tokens_per_second),
        )
    )

    trace_scalars = {
        "replay.buffer_depth_groups": replay_batch.stats.depth_groups,
        "replay.dropped_stale_groups": replay_batch.stats.num_dropped_stale_groups,
        "rollout.dropped_empty_groups": dropped_empty_groups,
        "rollout.dropped_zero_advantage_groups": dropped_zero_advantage_groups,
        "replay.train_version": train_version,
        "replay.behavior_version_min": behavior_version_min,
        "replay.behavior_version_max": behavior_version_max,
        "timing.replay_wait_ms": timings.replay_wait_s * 1000,
        "timing.weight_sync_admission_drain_ms": (
            timings.weight_sync.admission_drain_s * 1000
        ),
        "timing.weight_sync_push_ms": timings.weight_sync.push_s * 1000,
        "timing.weight_sync_pull_ms": timings.weight_sync.pull_s * 1000,
        "timing.checkpoint_ms": timings.checkpoint_s * 1000,
        "checkpoint.saved": float(checkpoint_saved),
    }
    validate_train_step_fwd_bwd_metrics(fwd_bwd_metrics)
    for key in _TRACE_FWD_BWD_KEYS:
        trace_scalars[key.replace("/", ".")] = fwd_bwd_metrics[key]
    return metrics, trace_scalars
