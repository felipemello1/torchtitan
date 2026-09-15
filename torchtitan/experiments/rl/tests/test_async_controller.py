# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the active-slot buffer backpressure,
the consume-time staleness invariant, the metrics timer drain, and RolloutTurnID."""

import asyncio
import json
import logging

import pytest

from torchtitan.experiments.rl.components.batcher import Batcher
from torchtitan.experiments.rl.components.work_buffer import (
    AdaptiveRolloutGroupWorkBuffer,
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
    StallDrivenDemand,
)
from torchtitan.experiments.rl.controller_metrics import (
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    MetricsTimer,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.experiments.rl.types import (
    RolloutTurnID,
    TrainingSample,
    TrainingSampleGroup,
)


def _training_sample(*, group_id: int, rollout_id: int) -> TrainingSample:
    return TrainingSample(
        min_policy_version=0,
        max_policy_version=0,
        rollout_id=RolloutTurnID(group_id=group_id, rollout_id=rollout_id, turn_id=0),
        token_ids=[1, 2, 3],
        loss_mask=[False, True, True],
        logprobs=[0.0, 0.1, 0.2],
        advantage=[0.0, 1.0, 1.0],
    )


def _trainable_group(group_id: int, *, num_samples: int) -> TrainingSampleGroup:
    return TrainingSampleGroup(
        group_id=group_id,
        training_samples=[
            _training_sample(group_id=group_id, rollout_id=i)
            for i in range(num_samples)
        ],
        metrics=[],
    )


def _untrainable_group(group_id: int) -> TrainingSampleGroup:
    return TrainingSampleGroup(group_id=group_id, training_samples=[], metrics=[])


def _build_batcher(*, num_prompts_per_train_step: int) -> Batcher:
    return Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=16384,
        max_context_length=2048,
        num_prompts_per_train_step=num_prompts_per_train_step,
        dp_degree=1,
        pad_id=0,
    )


def test_batcher_counts_trainable_groups_not_rollouts() -> None:
    # Target is 2 GROUPS. A single group with many rollouts is not a full batch; two groups are,
    # regardless of how many rollouts each contributes.
    batcher = _build_batcher(num_prompts_per_train_step=2)
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=8)
    )
    assert batch is None
    assert group_is_trainable
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=1)
    )
    assert batch is not None
    assert group_is_trainable
    assert batch.training_group_ids == [0, 1]


def test_batcher_carries_metric_only_groups_until_trainable_batch() -> None:
    # Metric-only (empty) groups do not count toward the target and cannot form a zero-token batch;
    # they ride along until a trainable group completes the batch.
    batcher = _build_batcher(num_prompts_per_train_step=1)
    metric_only = TrainingSampleGroup(group_id=0, training_samples=[], metrics=[])
    assert batcher.add_training_samples(training_sample_group=metric_only) == (
        None,
        False,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=2)
    )
    assert batch is not None
    assert group_is_trainable
    assert batch.num_global_valid_tokens > 0
    assert batch.training_group_ids == [1]


def test_batcher_warns_after_each_batch_of_untrainable_groups(
    caplog: pytest.LogCaptureFixture,
) -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)

    with caplog.at_level(logging.WARNING):
        batcher.add_training_samples(training_sample_group=_untrainable_group(0))
        batcher.add_training_samples(training_sample_group=_untrainable_group(1))

    assert (
        "Consecutive untrainable batches: 1/10 "
        "(2 rollout groups produced no trainable samples)."
    ) in caplog.text


def test_batcher_resets_no_progress_count_on_trainable_group(
    caplog: pytest.LogCaptureFixture,
) -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)

    with caplog.at_level(logging.WARNING):
        batcher.add_training_samples(training_sample_group=_untrainable_group(0))
        batcher.add_training_samples(training_sample_group=_untrainable_group(1))
        batcher.add_training_samples(
            training_sample_group=_trainable_group(2, num_samples=1)
        )
        caplog.clear()
        batcher.add_training_samples(training_sample_group=_untrainable_group(3))

    assert "zero-output batch equivalents" not in caplog.text


def test_batcher_raises_at_consecutive_untrainable_group_limit() -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)

    for group_id in range(19):
        batcher.add_training_samples(training_sample_group=_untrainable_group(group_id))

    with pytest.raises(RuntimeError, match="10 consecutive untrainable batches"):
        batcher.add_training_samples(training_sample_group=_untrainable_group(19))


def test_microbatch_grid_avoids_all_padding_cells_when_possible() -> None:
    # 5 real rows, 2 rows/rank, dp_degree=2 -> 4 cells x 2 = 8 rows (3 pad).
    # Redistributing one row into the fourth cell keeps every cell trainable.
    batcher = Batcher.Config(max_num_documents=4).build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=2,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=5)
    )
    assert batch is not None
    assert group_is_trainable
    cells = [microbatch for ranks in batch.microbatches for microbatch in ranks]
    assert len(cells) == 4  # 2 microbatches x 2 ranks
    for cell in cells:
        assert cell.loss_mask.any()
        assert cell.padding_mask.shape == cell.token_ids.shape
        assert not cell.padding_mask[cell.loss_mask].any()


def test_document_limit_applies_to_each_local_microbatch() -> None:
    # Each row holds two documents, but a local microbatch is capped at three.
    # The batcher must keep all five documents and split the rows 2 + 3.
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=5)
    )

    assert batch is not None
    assert group_is_trainable
    assert len(batch.microbatches) == 2
    num_documents = []
    for (microbatch,) in batch.microbatches:
        real_document_starts = (microbatch.positions == 0) & ~microbatch.padding_mask
        num_documents.append(int(real_document_starts.sum().item()))
    assert num_documents == [2, 3]
    assert sum(num_documents) == 5


def test_document_limit_can_be_smaller_than_rows_per_microbatch() -> None:
    batcher = Batcher.Config(max_num_documents=1).build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=2,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
    )
    batch, _ = batcher.add_training_samples(
        training_sample_group=_trainable_group(0, num_samples=2)
    )

    assert batch is not None
    assert len(batch.microbatches) == 2
    for (microbatch,) in batch.microbatches:
        real_document_starts = (microbatch.positions == 0) & ~microbatch.padding_mask
        assert int(real_document_starts.sum().item()) == 1


def test_batcher_filters_training_samples_longer_than_context() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
    )
    sample = _training_sample(group_id=0, rollout_id=0)
    sample.token_ids = list(range(6))
    sample.loss_mask = [False] * 6
    sample.logprobs = [0.0] * 6
    sample.advantage = [0.0] * 6

    pending, group_is_trainable = batcher.add_training_samples(
        training_sample_group=TrainingSampleGroup(
            group_id=0,
            training_samples=[sample],
            metrics=[],
        )
    )
    assert pending is None
    assert not group_is_trainable

    batch, _ = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=1)
    )
    assert batch is not None
    dropped_metric = next(
        metric
        for metric in batch.metrics
        if metric.key == "batcher/num_samples_dropped_oversized"
    )
    assert dropped_metric.value.value == 1


def test_batcher_requires_whole_rows_per_microbatch() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=5,
            max_context_length=3,
            num_prompts_per_train_step=1,
            dp_degree=1,
            pad_id=0,
        )


def test_compute_perf_ratio_metrics_reads_flushed_means() -> None:
    time_metrics = [
        m.Metric("timing/step/total", m.Mean.from_list([2.0])),
        m.Metric("timing/step/forward_backward", m.Mean.from_list([0.5])),
        m.Metric("timing/step/optim", m.Mean.from_list([0.5])),
    ]
    ratios = {
        metric.key: metric.value.value
        for metric in compute_perf_ratio_metrics(
            num_global_valid_tokens=100, time_metrics=time_metrics
        )
    }
    assert ratios["perf/trainer/tokens_per_second_full_step"] == 50.0
    assert ratios["perf/trainer/step_time_ratio/fwd_bwd"] == 0.5
    assert ratios["perf/trainer/tokens_per_second_fwd_bwd"] == 100.0


def test_compute_perf_ratio_metrics_skips_missing_spans() -> None:
    # Only `total` recorded -> emit the full-step throughput, skip every ratio whose span is absent.
    time_metrics = [m.Metric("timing/step/total", m.Mean.from_list([2.0]))]
    keys = {
        metric.key
        for metric in compute_perf_ratio_metrics(
            num_global_valid_tokens=100, time_metrics=time_metrics
        )
    }
    assert keys == {"perf/trainer/tokens_per_second_full_step"}


def test_compute_perf_ratio_metrics_returns_empty_without_total() -> None:
    assert (
        compute_perf_ratio_metrics(num_global_valid_tokens=100, time_metrics=[]) == []
    )


def test_metrics_timer_flush_drains() -> None:
    timer = MetricsTimer()
    with timer.record("timing/x"):
        pass
    assert timer.flush()  # non-empty on first read
    assert timer.flush() == []  # drained on the second read


def test_rollout_id_to_string_is_callable_and_uses_int_group_id() -> None:
    rollout_id = RolloutTurnID(group_id=5, rollout_id=2, turn_id=0)
    assert rollout_id.to_string() == "group=5/rollout=2/turn=0"
    assert rollout_id.to_string(include_turn=False) == "group=5/rollout=2"


def test_take_finalized_does_not_release_active_slot() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config(target_offpolicy_steps=0).build(
            num_prompts_per_train_step=1
        )
        reservation = await buffer.reserve_slot()
        if reservation is None:
            raise RuntimeError("buffer closed unexpectedly")
        await buffer.add_work(
            RolloutGroupWork(group_id=0, sample=object()),
            reservation=reservation,
        )
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        group = await buffer.take_finalized()
        assert group is not None
        await buffer.record_selected_outcomes([group.group_id], outcome="trained")

        waiter = asyncio.create_task(buffer.reserve_slot())
        await asyncio.sleep(0)
        assert not waiter.done()

        await buffer.release_active_groups(1, reason="trained")
        reservation = await waiter
        assert reservation is not None
        await buffer.cancel_reservation(reservation)

    asyncio.run(run())


def test_untrainable_group_releases_before_training() -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config(target_offpolicy_steps=0).build(
            num_prompts_per_train_step=1
        )
        batcher = Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=16384,
            max_context_length=2048,
            num_prompts_per_train_step=1,
            dp_degree=1,
            pad_id=0,
        )

        reservation = await buffer.reserve_slot()
        if reservation is None:
            raise RuntimeError("buffer closed unexpectedly")
        await buffer.add_work(
            RolloutGroupWork(group_id=0, sample=object()),
            reservation=reservation,
        )

        training_sample_group = TrainingSampleGroup(
            group_id=0, training_samples=[], metrics=[]
        )
        await buffer.release_active_groups(1, reason="untrainable_group")
        assert batcher.add_training_samples(
            training_sample_group=training_sample_group
        ) == (
            None,
            False,
        )

    asyncio.run(run())


def test_compute_policy_age_metrics_raises_on_consume_time_staleness() -> None:
    with pytest.raises(RuntimeError, match="admitted stale training data"):
        compute_policy_age_metrics(
            trainer_policy_version=4,
            min_policy_versions=[0],
            max_offpolicy_steps=3,
        )


def test_compute_policy_age_metrics_uses_hard_offpolicy_limit() -> None:
    metrics = compute_policy_age_metrics(
        trainer_policy_version=4,
        min_policy_versions=[0],
        max_offpolicy_steps=4,
    )
    assert any(metric.key == "train_batch/policy_age_max" for metric in metrics)

    with pytest.raises(RuntimeError, match="admitted stale training data"):
        compute_policy_age_metrics(
            trainer_policy_version=5,
            min_policy_versions=[0],
            max_offpolicy_steps=4,
        )


def _fifo_buffer(
    *, target_offpolicy_steps: int, window_fraction: float | None, num_prompts: int
) -> RolloutGroupWorkBuffer:
    return RolloutGroupWorkBuffer.Config(
        target_offpolicy_steps=target_offpolicy_steps, window_fraction=window_fraction
    ).build(num_prompts_per_train_step=num_prompts)


async def _admit(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    reservation = await buffer.reserve_slot()
    if reservation is None:
        raise RuntimeError("buffer closed unexpectedly")
    await buffer.add_work(
        RolloutGroupWork(group_id=group_id, sample=object()),
        reservation=reservation,
    )


async def _finalize(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    await buffer.finalize_work(RolloutGroup(group_id=group_id, rollouts=[]))


def test_windowed_fifo_takes_within_anchored_window() -> None:
    async def run() -> None:
        # capacity (1 + 1) * 4 = 8, window floor(0.5 * 8) = 4: [g0, g3]; g1/g2/g3 may bypass stuck g0; g4 remains blocked.
        buffer = _fifo_buffer(
            target_offpolicy_steps=1, window_fraction=0.5, num_prompts=4
        )
        for group_id in range(5):
            await _admit(buffer, group_id)
        await buffer.claim_next()  # g0 -> INFLIGHT and stuck
        for group_id in (1, 2, 3, 4):
            await _finalize(buffer, group_id)

        assert (await buffer.take_finalized()).group_id == 1
        assert (await buffer.take_finalized()).group_id == 2
        assert (await buffer.take_finalized()).group_id == 3

        taker = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not taker.done()  # g4 is finalized but outside the anchored window

        await _finalize(buffer, 0)
        assert (await taker).group_id == 0
        assert (await buffer.take_finalized()).group_id == 4

    asyncio.run(run())


def _adaptive_buffer(
    *,
    num_prompts: int,
    max_offpolicy_steps: int = 4,
    generation_capacity: int = 64,
    **kwargs
) -> AdaptiveRolloutGroupWorkBuffer:
    return AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=max_offpolicy_steps,
        generation_capacity=generation_capacity,
    ).build(num_prompts_per_train_step=num_prompts, **kwargs)


def _metric_value(metrics: list[m.Metric], key: str) -> float:
    return next(metric.value.value for metric in metrics if metric.key == key)


def test_stall_driven_demand_quantile_uses_every_point_of_the_lookback() -> None:
    from torchtitan.experiments.rl.components.adaptive_demand import (
        estimate_next_unavailable_upper_bound,
        mean_age_ceiling,
        prediction_multiplier,
        smooth_demand_toward_needed,
    )

    history = [44, 46, 41, 48, 45, 47, 50, 43, 46, 45]
    # mean 45.5, sd 2.55, Student-t prediction multiplier for 10 samples at 95% = 1.92 -> 50.4 -> 51 (above the max seen)
    assert estimate_next_unavailable_upper_bound(history=history, probability=0.95) == 51
    assert estimate_next_unavailable_upper_bound(history=[45, 45, 45], probability=0.95) == 45
    assert estimate_next_unavailable_upper_bound(history=[24], probability=0.95) == 24
    assert round(prediction_multiplier(samples=10, probability=0.95), 3) == 1.923
    assert round(prediction_multiplier(samples=1000, probability=0.95), 2) == 1.65
    assert mean_age_ceiling(
        num_prompts_per_train_step=8, mean_age_limit=4, groups_generating=40, untrainable_share=0.36
    ) == pytest.approx(54.4)
    # half the gap, the same up and down, rounded away from the current value
    assert smooth_demand_toward_needed(current_demand=64, demand_needed=59, damping_factor=0.5) == 61
    assert smooth_demand_toward_needed(current_demand=64, demand_needed=73, damping_factor=0.5) == 69
    assert smooth_demand_toward_needed(current_demand=64, demand_needed=65, damping_factor=0.5) == 65


def test_stall_driven_demand_starts_at_three_batches_and_moves_half_the_gap() -> None:
    demand = StallDrivenDemand(num_prompts_per_train_step=8, max_offpolicy_steps=10)
    assert demand.demand == 24
    # nothing ready: unavailable 24, one sample -> quantile 24, needed 8 + 24 + 1 = 33, half the gap up: 24 + 5
    assert demand.observe(step=1, ready=0, generating=24, completed=0, trainable=0) == 29
    assert demand.state == "ok"
    # a full shelf: unavailable max(0, 29 - 40) = 0; history [24, 0] -> mean 12, sd 17, two samples give a wide
    # margin (t = 7.5): quantile 128 -> needed 137. The ceiling at the max caps it: 10 x 8 + 8 + mean(24, 0) x
    # (1 - 40 / 40) = 88. Half the gap up from 29: 29 + 30
    assert demand.observe(step=2, ready=40, generating=0, completed=40, trainable=40) == 59
    assert demand.state == "age-limited"


def test_stall_driven_demand_is_capped_by_the_mean_age_ceiling() -> None:
    # target 4 with 40 generating and 36% rejected: ceiling 4 x 8 + 8 + 40 x 0.36 = 54.4 -> 54
    demand = StallDrivenDemand(
        num_prompts_per_train_step=8, max_offpolicy_steps=10, target_offpolicy_steps=4
    )
    path = [
        demand.observe(step=step, ready=0, generating=40, completed=25, trainable=16)
        for step in range(1, 9)
    ]
    assert path == [29, 42, 48, 51, 53, 54, 54, 54]
    assert demand.state == "age-limited"
    # no target: the same ceiling is applied at max_offpolicy_steps
    demand = StallDrivenDemand(num_prompts_per_train_step=8, max_offpolicy_steps=4)
    path = [
        demand.observe(step=step, ready=0, generating=40, completed=25, trainable=16)
        for step in range(1, 9)
    ]
    assert path == [29, 42, 48, 51, 53, 54, 54, 54]
    assert demand.state == "age-limited"
    # a ceiling that stops binding is reported: once the lookback holds only step starts with a full shelf
    # (unavailable 0), the need falls to P + 0 + 1 = 9 and the state returns to "ok"
    for step in range(9, 21):
        demand.observe(step=step, ready=100, generating=0, completed=8, trainable=8)
    assert demand.state == "ok"


def test_adaptive_buffer_takes_oldest_finalized_and_lets_slow_groups_keep_their_slot() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=2, generation_capacity=4)
        for group_id in range(4):
            await _admit(buffer, group_id)
        for _ in range(4):
            await buffer.claim_next()
        await _finalize(buffer, 2)
        await _finalize(buffer, 1)

        # g0 is still INFLIGHT: the batcher gets g1 then g2 without waiting for it
        assert (await buffer.take_finalized()).group_id == 1
        assert (await buffer.take_finalized()).group_id == 2
        taker = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not taker.done()

        await _finalize(buffer, 0)
        assert (await taker).group_id == 0
        # taking never frees a slot: 4 admitted -> 4 still active
        assert (
            _metric_value(buffer.metrics(), "rollout_buffer/active_slots_in_use_peak")
            == 4
        )

    asyncio.run(run())


def test_adaptive_buffer_drops_groups_past_max_offpolicy_steps() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=2, max_offpolicy_steps=1, generation_capacity=4
        )
        await _admit(buffer, 0)
        await _admit(buffer, 1)
        await buffer.claim_next()  # g0 generates under version 0
        await buffer.release_active_groups(0, reason="trained", policy_version=0)

        # The trainer starts step 2 (version 1): the batch being assembled trains at version 2 -> g0 would be 2 old
        await buffer.record_step_start(trainer_policy_version=1)
        await _finalize(buffer, 0)
        # Finalization alone cannot know whether the batcher is assembling the
        # current or prefetched batch, so it must not guess the consume version.
        metrics = buffer.metrics()
        assert _metric_value(metrics, "rollout_buffer/dropped_too_old") == 0
        assert _metric_value(metrics, "rollout_buffer/num_groups_finalized") == 1

        # g1 claimed after a pull to version 1 is fresh enough: age 2 - 1 = 1 <= 1
        await buffer.release_active_groups(0, reason="trained", policy_version=1)
        await buffer.claim_next()
        await _finalize(buffer, 1)
        selected = await buffer.take_finalized(consuming_policy_version=2)
        assert selected is not None
        assert selected.group_id == 1
        metrics = buffer.metrics()
        assert _metric_value(metrics, "rollout_buffer/dropped_too_old") == 1
        # g0's dropped slot is free (2 admitted, 1 dropped -> 1 active).
        assert (
            _metric_value(metrics, "rollout_buffer/available_active_slots")
            == buffer._active_group_limit() - 1
        )

    asyncio.run(run())


def test_adaptive_buffer_measures_the_shelf_and_moves_demand() -> None:
    async def run() -> None:
        # P=4: demand starts at three batches = 12; the shelf is what is finished and not held by the trainer.
        buffer = _adaptive_buffer(num_prompts=4, generation_capacity=40)
        assert buffer._active_group_limit() == 12
        for group_id in range(12):
            await _admit(buffer, group_id)
        for _ in range(12):
            await buffer.claim_next()

        # nothing finished: unavailable 12 -> needed 4 + 12 + 1 = 17 -> half the gap up: 12 + 3
        await buffer.record_step_start(trainer_policy_version=0)
        assert buffer._active_group_limit() == 15
        metrics = buffer.metrics()
        assert _metric_value(metrics, "rollout_buffer/ready_at_step_start") == 0
        assert _metric_value(metrics, "rollout_buffer/generating_at_step_start") == 12
        assert _metric_value(metrics, "rollout_buffer/demand_age_limited") == 0

        # ten groups finish; the batcher selects four and the trainer trains them: they leave the shelf
        for group_id in range(10):
            await _finalize(buffer, group_id)
        for _ in range(4):
            await buffer.take_finalized(consuming_policy_version=0)
        await buffer.record_selected_outcomes([0, 1, 2, 3], outcome="trained")
        await buffer.record_step_start(trainer_policy_version=1)
        metrics = buffer.metrics()
        assert _metric_value(metrics, "rollout_buffer/ready_at_step_start") == 6
        assert _metric_value(metrics, "rollout_buffer/generating_at_step_start") == 2
        # unavailable 15 - 6 = 9; history [12, 9] -> mean 10.5 + 4.46 x 2.12 -> 20 -> needed 25; ceiling
        # 4 x 4 + 4 + mean(12, 2) x (1 - 10 / 10) = 20 -> capped at 20 -> half the gap: 15 + 3
        assert buffer._active_group_limit() == 18
        assert _metric_value(buffer.metrics(), "rollout_buffer/demand_age_limited") == 1

    asyncio.run(run())


def test_adaptive_buffer_target_offpolicy_steps_caps_demand() -> None:
    # P=4, target 5 of max 10: the ceiling is 5 x 4 + 4 + generating x untrainable share
    config = AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=10, generation_capacity=40, target_offpolicy_steps=5
    )
    assert config.max_active_rollout_groups(num_prompts_per_train_step=4) == 84  # physical: 40 + 11 x 4
    buffer = config.build(num_prompts_per_train_step=4)
    assert buffer._active_group_limit() == 12  # three batches to start

    async def run() -> None:
        for _ in range(12):
            await buffer.record_step_start(trainer_policy_version=0)  # nothing generating, nothing completed
        # nothing generating and no completions: ceiling 20 + 4 + 0 = 24, however short the shelf
        assert buffer._active_group_limit() == 24
        assert buffer._demand.state == "age-limited"

    asyncio.run(run())
    with pytest.raises(ValueError, match="target_offpolicy_steps"):
        AdaptiveRolloutGroupWorkBuffer.Config(generation_capacity=40, target_offpolicy_steps=0)
    with pytest.raises(ValueError, match="target_offpolicy_steps"):
        AdaptiveRolloutGroupWorkBuffer.Config(
            max_offpolicy_steps=4, generation_capacity=40, target_offpolicy_steps=5
        )


def test_adaptive_buffer_never_exceeds_derived_max_demand() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=2, max_offpolicy_steps=1, generation_capacity=4
        )
        # C + (A + 1)P = 4 + 2*2 = 8.
        assert buffer.max_active_rollout_groups == 8
        for group_id in range(4):
            await _admit(buffer, group_id)
        for _ in range(4):
            await buffer.claim_next()
        for _ in range(10):
            await buffer.record_step_start(trainer_policy_version=0)
        # nothing has completed, so every generating group counts as untrainable: ceiling 1 x 2 + 2 + 4 x 1.0 = 8
        assert buffer._active_group_limit() == 8
        waiter = asyncio.create_task(buffer.reserve_slot())
        await asyncio.sleep(0)
        assert not waiter.done()
        await buffer.close()
        assert await waiter is None

    asyncio.run(run())


def test_adaptive_buffer_requires_explicit_generation_capacity() -> None:
    with pytest.raises(ValueError, match="generation_capacity"):
        AdaptiveRolloutGroupWorkBuffer.Config()


def test_adaptive_buffer_does_not_queue_ahead_of_generation() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=2, generation_capacity=2)
        await _admit(buffer, 0)
        await _admit(buffer, 1)
        await buffer.claim_next()
        await buffer.claim_next()

        third_slot = asyncio.create_task(buffer.reserve_slot())
        await asyncio.sleep(0)
        assert not third_slot.done()

        # Finalization releases a generation permit without releasing the
        # group's active demand slot.
        await _finalize(buffer, 0)
        reservation = await third_slot
        assert reservation is not None
        await buffer.cancel_reservation(reservation)
        await buffer.close()

    asyncio.run(run())


def test_admission_reservations_are_atomic_across_demand_drop() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=1, generation_capacity=4)
        reservation = await buffer.reserve_slot()
        assert reservation is not None

        # Model a controller update while data preparation is in progress.
        # The already granted reservation remains valid even below the new
        # target; future reservations wait until occupancy drains.
        async with buffer._condition:
            buffer._demand_target = 0
        assert await buffer.add_work(
            RolloutGroupWork(group_id=0, sample=object()),
            reservation=reservation,
        )
        assert buffer._active_rollout_groups == 1
        assert not buffer._admission_reservations

    asyncio.run(run())


def test_concurrent_admission_reservations_respect_generation_limit() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=1, generation_capacity=2)
        first = await buffer.reserve_slot()
        second = await buffer.reserve_slot()
        assert first is not None and second is not None
        third = asyncio.create_task(buffer.reserve_slot())
        await asyncio.sleep(0)
        assert not third.done()
        await buffer.cancel_reservation(first)
        replacement = await third
        assert replacement is not None
        assert len(buffer._admission_reservations) == 2
        await buffer.cancel_reservation(second)
        await buffer.cancel_reservation(replacement)

    asyncio.run(run())


def test_close_invalidates_uncommitted_admission_reservation() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=1, generation_capacity=2)
        reservation = await buffer.reserve_slot()
        assert reservation is not None
        await buffer.close()
        assert not await buffer.add_work(
            RolloutGroupWork(group_id=0, sample=object()),
            reservation=reservation,
        )
        assert buffer._active_rollout_groups == 0

    asyncio.run(run())


def test_work_buffer_writes_terminal_lifecycle_outcome(tmp_path) -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config(
            target_offpolicy_steps=0, window_fraction=None
        ).build(
            num_prompts_per_train_step=1,
            policy_version=3,
            lifecycle_log_dir=str(tmp_path),
        )
        await _admit(buffer, 0)
        await buffer.claim_next()
        await _finalize(buffer, 0)
        await buffer.record_step_start(trainer_policy_version=4)
        group = await buffer.take_finalized(consuming_policy_version=4)
        assert group is not None
        pending = buffer._selected_lifecycle_awaiting_outcome[group.group_id]
        assert not hasattr(pending, "rollout_group")
        assert buffer._active_rollout_groups == 1
        await buffer.record_selected_outcomes([group.group_id], outcome="trained")
        assert buffer._active_rollout_groups == 1

        lines = (tmp_path / "rollout_group_lifecycle.jsonl").read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["group_id"] == 0
        assert record["outcome"] == "trained"
        assert record["policy_version_at_admission"] == 3
        assert record["policy_version_at_claim"] == 3
        assert record["trainer_policy_version_at_exit"] == 4
        assert record["consuming_policy_version"] == 4
        assert (
            record["admitted_at"]
            <= record["claimed_at"]
            <= record["finalized_at"]
            <= record["left_at"]
        )

    asyncio.run(run())


def test_close_records_unresolved_selected_group_as_closed(tmp_path) -> None:
    async def run() -> None:
        buffer = RolloutGroupWorkBuffer.Config(
            target_offpolicy_steps=1, window_fraction=None
        ).build(
            num_prompts_per_train_step=2,
            lifecycle_log_dir=str(tmp_path),
        )
        for group_id in range(2):
            await _admit(buffer, group_id)
            await buffer.claim_next()
            await _finalize(buffer, group_id)
            selected = await buffer.take_finalized(consuming_policy_version=0)
            assert selected is not None

        await buffer.record_selected_outcomes([0], outcome="trained")
        assert buffer._active_rollout_groups == 2
        await buffer.close()

        records = [
            json.loads(line)
            for line in (tmp_path / "rollout_group_lifecycle.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [(record["group_id"], record["outcome"]) for record in records] == [
            (0, "trained"),
            (1, "closed"),
        ]
        assert buffer._active_rollout_groups == 2

    asyncio.run(run())
