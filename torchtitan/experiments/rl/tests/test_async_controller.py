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
    patience_steps: int = 20,
    **kwargs
) -> AdaptiveRolloutGroupWorkBuffer:
    return AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=max_offpolicy_steps,
        generation_capacity=generation_capacity,
        patience_steps=patience_steps,
    ).build(num_prompts_per_train_step=num_prompts, **kwargs)


def _metric_value(metrics: list[m.Metric], key: str) -> float:
    return next(metric.value.value for metric in metrics if metric.key == key)


def test_stall_driven_demand_rises_on_a_short_shelf_and_relaxes_with_patience() -> None:
    demand = StallDrivenDemand(
        num_prompts_per_train_step=8,
        ceiling=168,
        generation_capacity=128,
        patience_steps=3,
    )
    assert demand.demand == 40  # five batches
    assert (
        demand.observe(ready=5, inflight=30, completed=12, trainable=8, dropped=0) == 44
    )  # short shelf: + P // 2
    assert (
        demand.observe(ready=8, inflight=30, completed=12, trainable=8, dropped=0) == 44
    )  # one batch: hold
    for _ in range(2):
        assert (
            demand.observe(ready=16, inflight=30, completed=12, trainable=8, dropped=0)
            == 44
        )
    assert (
        demand.observe(ready=16, inflight=30, completed=12, trainable=8, dropped=0)
        == 43
    )  # third comfortable step: -1
    assert (
        demand.observe(ready=9, inflight=30, completed=12, trainable=8, dropped=0) == 43
    )  # comfort streak reset
    assert demand.state == "ok"


def test_stall_driven_demand_stays_within_bounds() -> None:
    demand = StallDrivenDemand(
        num_prompts_per_train_step=2, ceiling=8, generation_capacity=4
    )
    for _ in range(10):
        demand.observe(ready=0, inflight=2, completed=1, trainable=1, dropped=0)
    assert demand.demand == 8
    demand = StallDrivenDemand(
        num_prompts_per_train_step=2, ceiling=8, generation_capacity=4, patience_steps=1
    )
    for _ in range(20):
        demand.observe(ready=4, inflight=2, completed=1, trainable=1, dropped=0)
    assert demand.demand == 4  # floor: two batches


def test_stall_driven_demand_refuses_growth_when_age_limited_or_generation_bound() -> None:
    demand = StallDrivenDemand(
        num_prompts_per_train_step=8,
        ceiling=168,
        generation_capacity=128,
        guard_window_steps=2,
    )
    demand.observe(
        ready=0, inflight=10, completed=10, trainable=6, dropped=4
    )  # window not full yet: grows
    assert demand.demand == 44 and demand.state == "ok"
    demand.observe(
        ready=0, inflight=10, completed=10, trainable=6, dropped=4
    )  # 8 of 20 dropped > 15%
    assert demand.demand == 44 and demand.state == "age-limited"
    demand.observe(
        ready=0, inflight=10, completed=10, trainable=6, dropped=0
    )  # 4 of 20 still > 15%
    assert demand.demand == 44 and demand.state == "age-limited"
    # drops gone; permits full and only 6 trainable per step for a batch of 8: a bigger shelf cannot be filled
    demand.observe(ready=0, inflight=128, completed=10, trainable=6, dropped=0)
    assert demand.demand == 44 and demand.state == "generation-bound"
    # permits full but 10 trainable per step: extra demand buffers the surplus, so it grows
    demand.observe(ready=0, inflight=128, completed=14, trainable=10, dropped=0)
    assert demand.demand == 48 and demand.state == "ok"


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
        # P=4: demand starts at five batches = 20; the shelf is what is finished and not held by the trainer.
        buffer = _adaptive_buffer(
            num_prompts=4, generation_capacity=40, patience_steps=2
        )
        assert buffer._active_group_limit() == 20
        for group_id in range(20):
            await _admit(buffer, group_id)
        for _ in range(20):
            await buffer.claim_next()

        # nothing finished: shelf 0 -> +2 per step start
        await buffer.record_step_start(trainer_policy_version=0)
        assert buffer._active_group_limit() == 22
        assert (
            _metric_value(buffer.metrics(), "rollout_buffer/ready_at_step_start") == 0
        )

        # ten groups finish; the batcher selects four and the trainer trains them: they leave the shelf
        for group_id in range(10):
            await _finalize(buffer, group_id)
        for _ in range(4):
            await buffer.take_finalized(consuming_policy_version=0)
        await buffer.record_selected_outcomes([0, 1, 2, 3], outcome="trained")
        await buffer.record_step_start(trainer_policy_version=1)
        assert (
            _metric_value(buffer.metrics(), "rollout_buffer/ready_at_step_start") == 6
        )
        assert buffer._active_group_limit() == 22  # one batch on the shelf: hold

        # the pull releases the trained batch; the shelf still holds six, plus four more finish -> two batches
        await buffer.release_active_groups(4, reason="trained", policy_version=1)
        for group_id in range(10, 14):
            await _finalize(buffer, group_id)
        await buffer.record_step_start(trainer_policy_version=1)
        await buffer.record_step_start(trainer_policy_version=1)
        assert buffer._active_group_limit() == 21  # two comfortable steps: -1

    asyncio.run(run())


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
