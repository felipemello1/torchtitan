# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the active-slot buffer backpressure,
finish-order consumption, the policy-age metrics, the metrics timer drain, and RolloutTurnID."""

import asyncio
import logging

import pytest

from torchtitan.experiments.rl.components.batcher import Batcher
from torchtitan.experiments.rl.components.work_buffer import (
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
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=1)
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
        buffer = RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=1)
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


def _metric_value(metrics: list[m.Metric], key: str) -> float:
    (metric,) = [metric for metric in metrics if metric.key == key]
    return metric.value.value


def test_compute_policy_age_metrics_trains_over_target_age_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # target S=3; one straggler sample comes back at age S+3=6. It is counted and warned, not raised.
    with caplog.at_level(logging.WARNING):
        metrics = compute_policy_age_metrics(
            trainer_policy_version=10,
            min_policy_versions=[4, 9, 8],
            target_offpolicy_steps=3,
        )

    assert _metric_value(metrics, "train_batch/policy_age_max") == 6
    assert _metric_value(metrics, "train_batch/num_samples_over_target_age") == 1
    assert "1 training samples are older than target_offpolicy_steps=3" in caplog.text
    assert "Expected for straggler groups" in caplog.text


def test_compute_policy_age_metrics_is_quiet_within_target(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        metrics = compute_policy_age_metrics(
            trainer_policy_version=10,
            min_policy_versions=[7, 9],
            target_offpolicy_steps=3,
        )

    assert _metric_value(metrics, "train_batch/policy_age_max") == 3
    assert _metric_value(metrics, "train_batch/num_samples_over_target_age") == 0
    assert caplog.text == ""


def _buffer(*, capacity: int) -> RolloutGroupWorkBuffer:
    return RolloutGroupWorkBuffer.Config().build(max_active_rollout_groups=capacity)


async def _admit(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    if not await buffer.wait_for_slot():
        raise RuntimeError("buffer closed unexpectedly")
    await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))


async def _finalize(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    await buffer.finalize_work(RolloutGroup(group_id=group_id, rollouts=[]))


def test_take_finalized_follows_finish_order_not_group_id() -> None:
    async def run() -> None:
        buffer = _buffer(capacity=4)
        for group_id in range(4):
            await _admit(buffer, group_id)
        for group_id in (3, 1, 2):
            await _finalize(buffer, group_id)

        assert (await buffer.take_finalized()).group_id == 3
        assert (await buffer.take_finalized()).group_id == 1
        assert (await buffer.take_finalized()).group_id == 2

    asyncio.run(run())


def test_finish_order_trains_younger_groups_while_head_never_finishes() -> None:
    async def run() -> None:
        # S=1, P=4 -> 8 slots. g0 is claimed and never finishes; g1..g4 finish.
        buffer = _buffer(capacity=8)
        for group_id in range(8):
            await _admit(buffer, group_id)
        await buffer.claim_next()  # g0 -> INFLIGHT, stuck
        for group_id in (1, 2, 3, 4):
            await _finalize(buffer, group_id)

        # One full batch of P=4 younger groups is taken without waiting on g0.
        for expected_group_id in (1, 2, 3, 4):
            taker = asyncio.create_task(buffer.take_finalized())
            await asyncio.sleep(0)
            assert taker.done()
            assert taker.result().group_id == expected_group_id

        # The remaining wait is for generation (g0, g5..g7 unfinished), not for the head.
        taker = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not taker.done()
        await _finalize(buffer, 6)
        assert (await taker).group_id == 6

    asyncio.run(run())
