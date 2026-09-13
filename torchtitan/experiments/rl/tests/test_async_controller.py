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
        if not await buffer.wait_for_slot():
            raise RuntimeError("buffer closed unexpectedly")
        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        await buffer.take_finalized()

        waiter = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not waiter.done()

        await buffer.release_active_groups(1, reason="trained")
        assert await waiter

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

        if not await buffer.wait_for_slot():
            raise RuntimeError("buffer closed unexpectedly")
        await buffer.add_work(RolloutGroupWork(group_id=0, sample=object()))

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
    if not await buffer.wait_for_slot():
        raise RuntimeError("buffer closed unexpectedly")
    await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))


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
    capacity_ceiling: int = 128,
    memory_steps: int = 300,
    **kwargs
) -> AdaptiveRolloutGroupWorkBuffer:
    return AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=max_offpolicy_steps,
        capacity_ceiling=capacity_ceiling,
        memory_steps=memory_steps,
    ).build(num_prompts_per_train_step=num_prompts, **kwargs)


def _metric_value(metrics: list[m.Metric], key: str) -> float:
    return next(metric.value.value for metric in metrics if metric.key == key)


def test_adaptive_buffer_takes_oldest_finalized_and_lets_slow_groups_keep_their_slot() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=2, capacity_ceiling=8)
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
            num_prompts=2, max_offpolicy_steps=1, capacity_ceiling=8
        )
        await _admit(buffer, 0)
        await _admit(buffer, 1)
        await buffer.claim_next()  # g0 generates under version 0
        await buffer.release_active_groups(0, reason="trained", policy_version=0)

        # The trainer starts step 2 (version 1): the batch being assembled trains at version 2 -> g0 would be 2 old
        buffer.record_step_start(trainer_policy_version=1)
        await _finalize(buffer, 0)
        metrics = buffer.metrics()
        assert _metric_value(metrics, "rollout_buffer/dropped_too_old") == 1
        assert _metric_value(metrics, "rollout_buffer/num_groups_finalized") == 0
        # the dropped group's slot is free at once (2 admitted, 1 dropped -> 1 active)
        assert (
            _metric_value(metrics, "rollout_buffer/available_active_slots")
            == buffer._active_capacity() - 1
        )

        # g1 claimed after a pull to version 1 is fresh enough: age 2 - 1 = 1 <= 1
        await buffer.release_active_groups(0, reason="trained", policy_version=1)
        await buffer.claim_next()
        await _finalize(buffer, 1)
        assert (await buffer.take_finalized()).group_id == 1

    asyncio.run(run())


def test_adaptive_buffer_capacity_follows_the_unavailable_count() -> None:
    async def run() -> None:
        # P=4: starts at 5 * P = 20 slots
        buffer = _adaptive_buffer(num_prompts=4, capacity_ceiling=64, memory_steps=3)
        assert buffer._active_capacity() == 20
        for group_id in range(20):
            await _admit(buffer, group_id)
        for _ in range(20):
            await buffer.claim_next()

        # 20 unavailable -> need = 2 * 4 + 20 + 1 = 29; up at most P // 2 = 2 per step
        capacities = []
        for _ in range(6):
            buffer.record_step_start(trainer_policy_version=0)
            capacities.append(buffer._active_capacity())
        assert capacities == [22, 24, 26, 28, 29, 29]
        assert _metric_value(buffer.metrics(), "rollout_buffer/capacity") == 29

        # everything finishes: after the 3-step memory flushes, need = 2 * 4 + 0 + 1 = 9; down by halving the gap
        for group_id in range(20):
            await _finalize(buffer, group_id)
        capacities = []
        for _ in range(8):
            buffer.record_step_start(trainer_policy_version=0)
            capacities.append(buffer._active_capacity())
        assert capacities == [29, 29, 19, 14, 12, 11, 10, 9]

    asyncio.run(run())


def test_adaptive_buffer_never_exceeds_its_ceiling() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=2, capacity_ceiling=6)
        assert buffer.max_active_rollout_groups == 6
        for group_id in range(6):
            await _admit(buffer, group_id)
        for _ in range(6):
            await buffer.claim_next()
        for _ in range(10):
            buffer.record_step_start(trainer_policy_version=0)
        assert buffer._active_capacity() == 6
        waiter = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not waiter.done()
        await buffer.close()
        assert not await waiter

    asyncio.run(run())


def test_adaptive_buffer_rejects_a_ceiling_below_two_batches() -> None:
    with pytest.raises(ValueError, match="capacity_ceiling"):
        _adaptive_buffer(num_prompts=4, capacity_ceiling=7)


def test_work_buffer_writes_one_lifecycle_line_per_group(tmp_path) -> None:
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
        buffer.record_step_start(trainer_policy_version=4)
        await buffer.take_finalized()

        lines = (tmp_path / "rollout_group_lifecycle.jsonl").read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["group_id"] == 0
        assert record["outcome"] == "taken"
        assert record["policy_version_at_admission"] == 3
        assert record["policy_version_at_claim"] == 3
        assert record["trainer_policy_version_at_exit"] == 4
        assert (
            record["admitted_at"]
            <= record["claimed_at"]
            <= record["finalized_at"]
            <= record["left_at"]
        )

    asyncio.run(run())
