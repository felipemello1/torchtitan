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
from types import SimpleNamespace

import pytest
import torch

from torchtitan.rl.components.batcher import Batcher
from torchtitan.rl.components.work_buffer import (
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.rl.controller import Controller
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.observability.controller import (
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    MetricsTimer,
)
from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.types import RolloutTurnID, TrainingSample, TrainingSampleGroup


def test_controller_config_maybe_log(tmp_path, caplog) -> None:
    from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_5_debug_varlen

    config = rl_grpo_qwen3_5_debug_varlen(seq_len=128)
    assert config.generator.max_num_batched_tokens == 128
    config.dump_folder = str(tmp_path)
    config.trainer.debug.print_config = True
    config.trainer.debug.save_config_file = "config.json"

    with caplog.at_level(logging.INFO, logger="torchtitan.rl.controller"):
        config.maybe_log()

    assert "Running with configs:" in caplog.text
    with open(tmp_path / "config.json") as file:
        saved_config = json.load(file)
    assert saved_config["trainer"]["training"]["max_context_length"] == 128


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


def _variable_length_group(
    group_id: int, *, token_lengths: list[int]
) -> TrainingSampleGroup:
    samples = []
    for rollout_id, token_length in enumerate(token_lengths):
        samples.append(
            TrainingSample(
                min_policy_version=0,
                max_policy_version=0,
                rollout_id=RolloutTurnID(
                    group_id=group_id,
                    rollout_id=rollout_id,
                    turn_id=0,
                ),
                token_ids=list(range(token_length)),
                loss_mask=[False] + [True] * (token_length - 1),
                logprobs=[0.0] * token_length,
                advantage=[0.0] + [1.0] * (token_length - 1),
            )
        )
    return TrainingSampleGroup(group_id=group_id, training_samples=samples, metrics=[])


def _metric_value(batch, key: str) -> float:
    metric = next(metric for metric in batch.metrics if metric.key == key)
    assert isinstance(metric.value, m.NoReduce)
    return metric.value.value


def _untrainable_group(group_id: int) -> TrainingSampleGroup:
    return TrainingSampleGroup(group_id=group_id, training_samples=[], metrics=[])


def _build_batcher(
    *, num_prompts_per_train_step: int, num_mtp_layers: int = 0
) -> Batcher:
    config = Batcher.Config()
    config.num_mtp_layers = num_mtp_layers
    return config.build(
        num_tokens_per_microbatch_per_dp_rank=16384,
        max_context_length=2048,
        num_prompts_per_train_step=num_prompts_per_train_step,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
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


def test_batcher_packs_groups_in_id_order_regardless_of_arrival() -> None:
    batcher = _build_batcher(num_prompts_per_train_step=2)
    late_group = _trainable_group(7, num_samples=1)
    early_group = _trainable_group(3, num_samples=1)
    late_group.training_samples[0].min_policy_version = 7
    early_group.training_samples[0].min_policy_version = 3

    # g7 finishes first; the packed batch still lists g3 before g7.
    batcher.add_training_samples(training_sample_group=late_group)
    batch, _ = batcher.add_training_samples(training_sample_group=early_group)

    assert batch is not None
    assert batch.min_policy_versions == [3, 7]


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
    assert batch.global_loss_token_counts[0] > 0
    assert batch.global_routing_token_counts.shape == (1,)


def test_batcher_prepares_per_depth_mtp_token_counts() -> None:
    batcher = _build_batcher(num_prompts_per_train_step=1, num_mtp_layers=2)
    batch, _ = batcher.add_training_samples(
        training_sample_group=_trainable_group(1, num_samples=2)
    )

    assert batch is not None
    assert batch.global_loss_token_counts.shape == (3,)
    assert batch.global_routing_token_counts.shape == (3,)


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


def test_dp_assignment_avoids_all_padding_ranks_when_possible() -> None:
    # Five two-token samples need four rank inputs across two microbatches.
    # Redistributing one sample keeps every rank input trainable.
    batcher = Batcher.Config(max_num_documents=4).build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=2,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
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
        assert cell.padding_mask.shape == cell.input.shape
        assert not cell.padding_mask[cell.loss_mask].any()


def test_batcher_uses_flat_rank_capacity_and_reports_padding() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_variable_length_group(
            0,
            token_lengths=[4, 4, 3],
        )
    )

    assert batch is not None
    assert group_is_trainable
    assert len(batch.microbatches) == 1
    microbatch = batch.microbatches[0][0]
    assert microbatch.positions.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    assert not microbatch.padding_mask.any()
    torch.testing.assert_close(microbatch.loss_token_counts, torch.tensor([8]))
    assert _metric_value(batch, "train_batch/padding_frac") == 0.0


def test_flat_rank_packing_preserves_padding_mask() -> None:
    batcher = Batcher.Config(per_sample_pad_multiple=4).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    batch, _ = batcher.add_training_samples(
        training_sample_group=_variable_length_group(0, token_lengths=[4])
    )

    assert batch is not None
    microbatch = batch.microbatches[0][0]
    assert microbatch.positions.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    torch.testing.assert_close(microbatch.loss_token_counts, torch.tensor([3]))
    assert microbatch.padding_mask.tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_batcher_balances_packing_across_dp_ranks() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    batch, group_is_trainable = batcher.add_training_samples(
        training_sample_group=_variable_length_group(
            0,
            token_lengths=[5, 4, 4, 3],
        )
    )

    assert batch is not None
    assert group_is_trainable
    assert [
        [int((~rank.padding_mask).sum().item()) for rank in microbatch]
        for microbatch in batch.microbatches
    ] == [[6, 6]]


def test_batcher_fills_new_bin_from_multiple_heaviest_bins() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=40,
        max_context_length=40,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _variable_length_group(
        0,
        token_lengths=[11] * 12,
    ).training_samples
    bins = [samples[:4], samples[4:8], samples[8:]]

    batcher._expand_bins_by_splitting(bins, target_num_bins=4)

    assert [len(bin_) for bin_ in bins] == [3, 3, 3, 3]
    assert [batcher._attention_workload(bin_) for bin_ in bins] == [400] * 4


def test_batcher_pads_when_no_bin_can_donate_a_sample() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _variable_length_group(0, token_lengths=[6, 6]).training_samples
    bins = [[samples[0]], [samples[1]]]

    batcher._expand_bins_by_splitting(bins, target_num_bins=4)

    assert bins == [[samples[0]], [samples[1]], [], []]


def test_batcher_splits_sorts_and_zigzags_by_attention_workload() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _variable_length_group(
        0,
        # Effective lengths are [6, 6, 6, 4, 4, 4]. FFD finds three bins,
        # then LPT repacks into four DP inputs.
        token_lengths=[7, 7, 7, 5, 5, 5],
    ).training_samples

    assignments = batcher._assign_training_samples_to_microbatches(samples)
    workloads = [
        [batcher._attention_workload(rank_samples) for rank_samples in microbatch]
        for microbatch in assignments
    ]

    # Global sorting puts similarly expensive bins in each concurrent DP group.
    # Reversing the second group pairs its lighter bin with the first rank.
    assert workloads == [[52, 52], [36, 52]]


def test_batcher_zigzags_workloads_across_dp_ranks() -> None:
    batcher = Batcher.Config(max_num_documents=1).build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _variable_length_group(
        0,
        # Effective lengths are [10, 8, 6, 4]. The document limit forces each
        # sample into its own bin, with padding-aware workloads [100, 68, 52, 52].
        token_lengths=[11, 9, 7, 5],
    ).training_samples

    assignments = batcher._assign_training_samples_to_microbatches(samples)
    workloads = [
        [batcher._attention_workload(rank_samples) for rank_samples in microbatch]
        for microbatch in assignments
    ]

    assert workloads == [[100, 68], [52, 52]]
    assert [sum(rank_workloads) for rank_workloads in zip(*workloads)] == [152, 120]


def test_batcher_reports_padding_when_document_limit_blocks_greedy_order() -> None:
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=6,
        max_context_length=3,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    batch, _ = batcher.add_training_samples(
        training_sample_group=_variable_length_group(
            0,
            token_lengths=[2, 2, 4, 2, 2, 4],
        )
    )

    assert batch is not None
    assert len(batch.microbatches) == 3
    assert all(
        microbatch.padding_mask.numel() == 6 for (microbatch,) in batch.microbatches
    )
    assert _metric_value(batch, "train_batch/padding_frac") == pytest.approx(4 / 9)


def test_document_limit_applies_to_each_local_microbatch() -> None:
    # A local microbatch is capped at three documents. The batcher must keep all
    # five documents and distribute them 2 + 3 across two microbatches.
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
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
    assert sorted(num_documents) == [2, 3]
    assert sum(num_documents) == 5


def test_document_limit_can_be_smaller_than_rows_per_microbatch() -> None:
    batcher = Batcher.Config(max_num_documents=1).build(
        num_tokens_per_microbatch_per_dp_rank=4,
        max_context_length=2,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
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
        temperature=1.0,
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
            temperature=1.0,
        )


def test_compute_perf_ratio_metrics_reads_flushed_means() -> None:
    time_metrics = [
        m.Metric("timing/step/total", m.Mean.from_list([2.0])),
        m.Metric("timing/step/forward_backward", m.Mean.from_list([0.5])),
        m.Metric("timing/step/optimizer", m.Mean.from_list([0.5])),
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
        buffer = RolloutGroupWorkBuffer.Config().build(
            max_active_rollout_groups=1, window_size=None
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
        buffer = RolloutGroupWorkBuffer.Config().build(
            max_active_rollout_groups=1, window_size=None
        )
        batcher = Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=16384,
            max_context_length=2048,
            num_prompts_per_train_step=1,
            dp_degree=1,
            pad_id=0,
            temperature=1.0,
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


def test_compute_policy_age_metrics_raises_beyond_cap() -> None:
    # cap 4 (S=3, windowed_fifo_batches=1): age 4 passes, age 5 raises
    metrics = compute_policy_age_metrics(
        trainer_policy_version=4,
        min_policy_versions=[0],
        target_offpolicy_steps=3,
        max_offpolicy_steps=4,
    )
    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert aggregated["train_batch/policy_age_max"] == 4
    assert aggregated["train_batch/pct_samples_over_target_age"] == 100.0

    with pytest.raises(RuntimeError, match="admitted stale training data"):
        compute_policy_age_metrics(
            trainer_policy_version=5,
            min_policy_versions=[0],
            target_offpolicy_steps=3,
            max_offpolicy_steps=4,
        )


def test_compute_policy_age_metrics_uncapped_trains_over_target_age_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # no cap; one sample comes back at age S+3=6: counted and warned, not raised
    with caplog.at_level(logging.WARNING):
        metrics = compute_policy_age_metrics(
            trainer_policy_version=10,
            min_policy_versions=[4, 9, 8],
            target_offpolicy_steps=3,
            max_offpolicy_steps=None,
        )

    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert aggregated["train_batch/policy_age/mean"] == 3
    assert aggregated["train_batch/policy_age_max"] == 6
    assert aggregated["train_batch/pct_samples_over_target_age"] == pytest.approx(
        100 / 3
    )
    assert "1 samples (33.3%) older than target_offpolicy_steps=3" in caplog.text


def test_compute_policy_age_metrics_uncapped_is_quiet_within_target(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        metrics = compute_policy_age_metrics(
            trainer_policy_version=10,
            min_policy_versions=[7, 9],
            target_offpolicy_steps=3,
            max_offpolicy_steps=None,
        )

    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert aggregated["train_batch/policy_age/mean"] == 2
    assert aggregated["train_batch/pct_samples_over_target_age"] == 0.0
    assert caplog.text == ""


def _buffer(*, capacity: int, window_size: int | None) -> RolloutGroupWorkBuffer:
    return RolloutGroupWorkBuffer.Config().build(
        max_active_rollout_groups=capacity, window_size=window_size
    )


async def _admit(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    if not await buffer.wait_for_slot():
        raise RuntimeError("buffer closed unexpectedly")
    await buffer.add_work(RolloutGroupWork(group_id=group_id, sample=object()))


async def _finalize(buffer: RolloutGroupWorkBuffer, group_id: int) -> None:
    await buffer.finalize_work(RolloutGroup(group_id=group_id, rollouts=[]))


def test_windowed_fifo_takes_within_anchored_window() -> None:
    async def run() -> None:
        # P=4, windowed_fifo_batches=1 -> window of 4 ids [g0, g3]: g1/g2/g3 may bypass stuck g0; g4 waits.
        buffer = _buffer(capacity=8, window_size=4)
        for group_id in range(5):
            await _admit(buffer, group_id)
        claimed_group_ids = [(await buffer.claim_next()).group_id for _ in range(5)]
        assert claimed_group_ids == [0, 1, 2, 3, 4]
        for group_id in (1, 2, 3, 4):
            await _finalize(buffer, group_id)

        assert (await buffer.take_finalized()).group_id == 1
        assert (await buffer.take_finalized()).group_id == 2
        assert (await buffer.take_finalized()).group_id == 3

        taker = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not taker.done()  # g4 is finalized but outside the anchored window

        await _finalize(buffer, 0)
        await asyncio.sleep(0)
        assert taker.done()
        assert taker.result().group_id == 0
        assert (await buffer.take_finalized()).group_id == 4

    asyncio.run(run())


def test_no_window_takes_oldest_ready_group_past_a_stuck_head() -> None:
    async def run() -> None:
        # S=1, P=4 -> 8 slots, no window. g0..g4 INFLIGHT; g0 never finishes; g1..g4 finish out of id order.
        buffer = _buffer(capacity=8, window_size=None)
        for group_id in range(8):
            await _admit(buffer, group_id)
        claimed_group_ids = [(await buffer.claim_next()).group_id for _ in range(5)]
        assert claimed_group_ids == [0, 1, 2, 3, 4]
        for group_id in (3, 1, 4, 2):
            await _finalize(buffer, group_id)

        # A full batch of P=4 is taken oldest-ready first, without waiting on g0.
        for expected_group_id in (1, 2, 3, 4):
            taker = asyncio.create_task(buffer.take_finalized())
            await asyncio.sleep(0)
            assert taker.done()
            assert taker.result().group_id == expected_group_id

        # The remaining wait is for generation (nothing finalized), not for the head g0.
        taker = asyncio.create_task(buffer.take_finalized())
        await asyncio.sleep(0)
        assert not taker.done()
        await buffer.claim_next()  # g5 -> INFLIGHT
        await buffer.claim_next()  # g6 -> INFLIGHT
        await _finalize(buffer, 6)
        await asyncio.sleep(0)
        assert taker.done()
        assert taker.result().group_id == 6

    asyncio.run(run())


class _FakeTrainingSampleBuilder:
    """Builds one trainable sample per group, or none for `untrainable_group_ids`."""

    def __init__(self, *, untrainable_group_ids: tuple[int, ...]) -> None:
        self._untrainable_group_ids = untrainable_group_ids

    def build_from_group(self, *, rollout_group: RolloutGroup) -> TrainingSampleGroup:
        if rollout_group.group_id in self._untrainable_group_ids:
            return _untrainable_group(rollout_group.group_id)
        return _trainable_group(rollout_group.group_id, num_samples=1)


async def _pack_on_event_loop(func, /, *args, **kwargs):
    """Stands in for `asyncio.to_thread`, so a fixed number of turns settles the batcher loop.
    Other tasks still run for one turn while it packs, as they would during the thread."""
    await asyncio.sleep(0)
    return func(*args, **kwargs)


async def _settle() -> None:
    """Give the batcher loop enough event loop turns to take every finished group and block."""
    for _ in range(50):
        await asyncio.sleep(0)


def _start_batcher_loop(
    *,
    buffer: RolloutGroupWorkBuffer,
    num_prompts_per_train_step: int,
    untrainable_group_ids: tuple[int, ...] = (),
) -> tuple[Controller, asyncio.Queue, asyncio.Task]:
    """Run the controller's `_batcher_loop` on `buffer` with a real `Batcher` and a size-1 queue."""
    # _batcher_loop reads only num_prompts_per_train_step and its counter; skip the actor setup.
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            num_prompts_per_train_step=num_prompts_per_train_step
        )
    )
    controller._num_groups_in_batcher = 0
    training_batch_queue = asyncio.Queue(maxsize=1)
    batcher_task = asyncio.create_task(
        controller._batcher_loop(
            group_buffer=buffer,
            training_sample_builder=_FakeTrainingSampleBuilder(
                untrainable_group_ids=untrainable_group_ids
            ),
            batcher=_build_batcher(
                num_prompts_per_train_step=num_prompts_per_train_step
            ),
            training_batch_queue=training_batch_queue,
        )
    )
    return controller, training_batch_queue, batcher_task


def _slot_gauges(
    buffer: RolloutGroupWorkBuffer,
    controller: Controller,
    training_batch_queue: asyncio.Queue,
) -> dict[str, float]:
    """The trainer loop's `rollout_buffer` gauges, without the per-flush peak."""
    num_prompts_per_train_step = controller.config.async_loop.num_prompts_per_train_step
    metrics = buffer.metrics(
        num_groups_in_batcher=controller._num_groups_in_batcher,
        num_groups_in_queue=training_batch_queue.qsize() * num_prompts_per_train_step,
    )
    return {
        metric.key: metric.value.value
        for metric in metrics
        if metric.key != "rollout_buffer/active_slots_in_use_peak"
    }


def _expected_gauges(
    *,
    waiting: int = 0,
    inflight: int = 0,
    finalized: int = 0,
    in_batcher: int = 0,
    in_queue: int = 0,
    in_trainer: int = 0,
    available: int = 0,
) -> dict[str, float]:
    return {
        "rollout_buffer/num_groups_waiting": waiting,
        "rollout_buffer/num_groups_inflight": inflight,
        "rollout_buffer/num_groups_finalized": finalized,
        "rollout_buffer/num_groups_in_batcher": in_batcher,
        "rollout_buffer/num_groups_in_queue": in_queue,
        "rollout_buffer/num_groups_in_trainer": in_trainer,
        "rollout_buffer/available_active_slots": available,
    }


def test_buffer_gauges_account_for_every_slot(monkeypatch) -> None:
    monkeypatch.setattr(asyncio, "to_thread", _pack_on_event_loop)

    async def run() -> None:
        # S=2, P=2 -> 6 slots. g1 has no trainable samples.
        buffer = _buffer(capacity=6, window_size=None)
        controller, training_batch_queue, batcher_task = _start_batcher_loop(
            buffer=buffer, num_prompts_per_train_step=2, untrainable_group_ids=(1,)
        )

        def read_gauges() -> dict[str, float]:
            return _slot_gauges(buffer, controller, training_batch_queue)

        async def expect(**counts: int) -> None:
            await _settle()
            assert read_gauges() == _expected_gauges(**counts)

        # The trainer could log on any event loop turn, including while the batcher packs or
        # waits on put. On every turn, in_trainer must be what this test's trainer holds.
        num_groups_held_by_trainer = 0
        wrong_readings: list[dict[str, float]] = []

        async def check_every_turn() -> None:
            while True:
                gauges = read_gauges()
                if (
                    gauges["rollout_buffer/num_groups_in_trainer"]
                    != num_groups_held_by_trainer
                ):
                    wrong_readings.append(gauges)
                await asyncio.sleep(0)

        checker_task = asyncio.create_task(check_every_turn())

        for group_id in range(6):
            await _admit(buffer, group_id)
        for _ in range(5):
            await buffer.claim_next()
        await expect(waiting=1, inflight=5)

        # The batcher takes g0 and waits for a second group.
        await _finalize(buffer, 0)
        await expect(waiting=1, inflight=4, in_batcher=1)

        # g1 is untrainable: the batcher frees its slot at once.
        await _finalize(buffer, 1)
        await expect(waiting=1, inflight=3, in_batcher=1, available=1)

        # g0 + g2 make a batch; the queue has room.
        await _finalize(buffer, 2)
        await expect(waiting=1, inflight=2, in_queue=2, available=1)

        # g3 + g4 make a batch; put waits for room in the queue.
        await _finalize(buffer, 3)
        await _finalize(buffer, 4)
        await expect(waiting=1, in_batcher=2, in_queue=2, available=1)

        # The batcher is blocked on put, so g5 stays in the buffer.
        await buffer.claim_next()
        await _finalize(buffer, 5)
        await expect(finalized=1, in_batcher=2, in_queue=2, available=1)

        # The trainer takes g0 + g2; the batcher queues g3 + g4, then takes g5.
        await training_batch_queue.get()
        num_groups_held_by_trainer = 2
        await expect(in_batcher=1, in_queue=2, in_trainer=2, available=1)

        # The weight pull after the trainer's optimizer step frees its batch's slots.
        await buffer.release_active_groups(2, reason="trained")
        num_groups_held_by_trainer = 0
        await expect(in_batcher=1, in_queue=2, available=3)

        assert wrong_readings == []
        await buffer.close()
        for task in (checker_task, batcher_task):
            task.cancel()
        await asyncio.gather(checker_task, batcher_task, return_exceptions=True)

    asyncio.run(run())


def test_group_taken_by_batcher_is_counted(monkeypatch) -> None:
    monkeypatch.setattr(asyncio, "to_thread", _pack_on_event_loop)

    async def run() -> None:
        # 2 slots, 2 groups per batch: the batcher takes finished g0 and waits for g1.
        buffer = _buffer(capacity=2, window_size=None)
        controller, training_batch_queue, batcher_task = _start_batcher_loop(
            buffer=buffer, num_prompts_per_train_step=2
        )
        for group_id in range(2):
            await _admit(buffer, group_id)
            await buffer.claim_next()
        await _finalize(buffer, 0)
        await _settle()

        gauges = _slot_gauges(buffer, controller, training_batch_queue)
        assert gauges == _expected_gauges(inflight=1, in_batcher=1)
        # The gauges that existed before num_groups_in_batcher count 1 of the 2 slots.
        old_keys = (
            "rollout_buffer/num_groups_waiting",
            "rollout_buffer/num_groups_inflight",
            "rollout_buffer/num_groups_finalized",
            "rollout_buffer/available_active_slots",
        )
        assert sum(gauges[key] for key in old_keys) == 1
        assert sum(gauges.values()) == 2

        await buffer.close()
        batcher_task.cancel()
        await asyncio.gather(batcher_task, return_exceptions=True)

    asyncio.run(run())
