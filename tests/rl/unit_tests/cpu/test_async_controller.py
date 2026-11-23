# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for async-controller pieces: batcher group-counting, the active-slot buffer backpressure,
the consume-time staleness invariant, the metrics timer drain, and RolloutTurnID."""

import asyncio
import dataclasses
import itertools
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch
from torchtitan.rl.controller.adaptive_demand import (
    estimate_next_unavailable_upper_bound,
    mean_age_ceiling,
    prediction_multiplier,
    smooth_demand_toward_needed,
    StallDrivenDemand,
)
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.controller.metrics import (
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    MetricsTimer,
)
from torchtitan.rl.controller.work_buffer import (
    AdaptiveRolloutGroupWorkBuffer,
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.rollout.types import (
    Prompt,
    Rollout,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)

from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)


def test_controller_config_maybe_log(tmp_path, caplog) -> None:
    from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_5_debug_varlen

    config = rl_grpo_qwen3_5_debug_varlen(seq_len=128)
    assert config.generator.max_num_batched_tokens == 128
    config.dump_folder = str(tmp_path)
    config.trainer.debug.print_config = True
    config.trainer.debug.save_config_file = "config.json"

    with caplog.at_level(logging.INFO, logger="torchtitan.rl.controller.controller"):
        config.maybe_log()

    assert "Running with configs:" in caplog.text
    with open(tmp_path / "config.json") as file:
        saved_config = json.load(file)
    assert saved_config["trainer"]["model"]["max_context_length"] == 128


def _controller_config_with_groups(
    *,
    num_groups: int,
    group_size: int,
    rollouts_per_worker: int = 128,
    target_offpolicy_steps: int = 0,
    max_concurrent_rollout_groups: int | None = None,
) -> Controller.Config:
    """A CI recipe with `(target_offpolicy_steps + 1) * num_groups` active slots of `group_size` rollouts."""
    from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_5_debug_varlen

    config = rl_grpo_qwen3_5_debug_varlen(seq_len=128)
    async_loop = dataclasses.replace(
        config.async_loop,
        target_offpolicy_steps=target_offpolicy_steps,
        num_prompts_per_train_step=num_groups,
        num_samples_per_prompt=group_size,
        max_concurrent_rollout_groups=max_concurrent_rollout_groups,
    )
    rollouter = dataclasses.replace(
        config.rollouter, rollouts_per_worker=rollouts_per_worker
    )
    # `replace` reruns `__post_init__`, so the group-size check runs.
    return dataclasses.replace(config, async_loop=async_loop, rollouter=rollouter)


def test_num_rollout_workers_rounds_up_training_rollouts_over_rollouts_per_worker() -> (
    None
):
    # 20 groups of 16: 320 rollouts in flight / 128 per worker = 2.5
    config = _controller_config_with_groups(num_groups=20, group_size=16)
    assert config.num_rollout_workers == 3


@pytest.mark.parametrize(
    "max_concurrent_rollout_groups, expected_workers",
    [
        # 180 active slots x 16 = 2,880 rollouts / 128 per worker = 22.5
        (None, 23),
        # 8 concurrent groups x 16 = 128 rollouts / 128 per worker = 1
        (8, 1),
    ],
)
def test_num_rollout_workers_counts_only_the_concurrent_groups(
    max_concurrent_rollout_groups, expected_workers
) -> None:
    # S = 8, P = 20: 180 active slots
    config = _controller_config_with_groups(
        num_groups=20,
        group_size=16,
        target_offpolicy_steps=8,
        max_concurrent_rollout_groups=max_concurrent_rollout_groups,
    )
    assert config.num_rollout_workers == expected_workers


def test_a_concurrent_group_cap_above_the_active_slots_raises() -> None:
    # S = 8, P = 32: 288 active slots
    with pytest.raises(ValueError, match=r"\[1, \(S \+ 1\) \* P = 288\], got 289"):
        AsyncLoopConfig(
            num_prompts_per_train_step=32,
            target_offpolicy_steps=8,
            max_concurrent_rollout_groups=289,
        )
    # The adaptive buffer's slots follow its demand, so its loops may outnumber (S + 1) * P.
    config = AsyncLoopConfig(
        num_prompts_per_train_step=32,
        target_offpolicy_steps=8,
        max_concurrent_rollout_groups=289,
        group_buffer=AdaptiveRolloutGroupWorkBuffer.Config(),
    )
    assert config.num_concurrent_rollout_groups == 289


# rollouts_per_worker=0 must raise this ValueError too: `num_rollout_workers` divides by it.
@pytest.mark.parametrize("group_size, rollouts_per_worker", [(256, 128), (8, 0)])
def test_a_group_larger_than_rollouts_per_worker_raises(
    group_size, rollouts_per_worker
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"num_samples_per_prompt \({group_size}\) is larger than "
        rf"rollouter.rollouts_per_worker \({rollouts_per_worker}\)",
    ):
        _controller_config_with_groups(
            num_groups=16,
            group_size=group_size,
            rollouts_per_worker=rollouts_per_worker,
        )


def test_rollout_loop_keeps_the_metrics_a_worker_sent_with_its_group() -> None:
    """A rollout worker adds its own metrics to each training group (e.g. its event-loop lag).
    The controller adds the rollout metrics next to them."""
    work = SimpleNamespace(sample="prompt", group_id=7)
    claims = [work, None]
    finalized: list[RolloutGroup] = []

    async def run_group_rollouts(*, group_id: int, **kwargs) -> RolloutGroup:
        worker_metric = m.Metric("rollout_worker/event_loop_lag_ms", m.Max(5.0))
        return RolloutGroup(group_id=group_id, rollouts=[], metrics=[worker_metric])

    async def claim_next():
        return claims.pop(0)

    async def finalize_work(group: RolloutGroup) -> None:
        finalized.append(group)

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(num_samples_per_prompt=1)
    )
    controller._sampling = SamplingConfig()
    controller._rollouter = SimpleNamespace(run_group_rollouts=run_group_rollouts)
    controller.rollout_recorder = SimpleNamespace(record=lambda **kwargs: None)
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=AsyncMock())
    )
    group_buffer = SimpleNamespace(
        claim_next=claim_next,
        track_generation=lambda work, task: None,
        finalize_work=finalize_work,
    )

    asyncio.run(
        controller._rollout_loop(
            group_buffer=group_buffer, generate_fn=None, release_session_fn=None
        )
    )

    keys = m.MetricsProcessor._aggregate_metrics(finalized[0].metrics)
    assert keys["rollout_worker/event_loop_lag_ms/max"] == 5.0


def _training_sample(*, group_id: int, rollout_id: int) -> TrainingSample:
    return TrainingSample(
        min_policy_version=0,
        max_policy_version=0,
        rollout_id=RolloutTurnID(group_id=group_id, rollout_id=rollout_id, turn_id=0),
        token_ids=np.array([1, 2, 3]),
        loss_mask=np.array([False, True, True]),
        logprobs=np.array([0.0, 0.1, 0.2], dtype=np.float32),
        advantage=np.array([0.0, 1.0, 1.0], dtype=np.float32),
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
                token_ids=np.arange(token_length),
                loss_mask=np.array([False] + [True] * (token_length - 1)),
                logprobs=np.zeros(token_length, dtype=np.float32),
                advantage=np.array(
                    [0.0] + [1.0] * (token_length - 1), dtype=np.float32
                ),
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
    assert batch.group_ids == [3, 7]


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
    assert batch.group_ids == [0, 1]


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
    cells = [
        microbatch
        for group in batch.microbatch_groups
        for ranks in group
        for microbatch in ranks
    ]
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
    assert len(batch.microbatch_groups) == 1
    microbatch = batch.microbatch_groups[0][0][0]
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
    microbatch = batch.microbatch_groups[0][0][0]
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
        for (microbatch,) in batch.microbatch_groups
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
    assert len(batch.microbatch_groups) == 3
    assert all(
        microbatch.padding_mask.numel() == 6
        for ((microbatch,),) in batch.microbatch_groups
    )
    assert _metric_value(batch, "train_batch/padding_frac") == pytest.approx(4 / 9)
    # 18 token slots, 8 of them padding.
    assert _metric_value(batch, "train_batch/num_tokens") == 10


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
    assert len(batch.microbatch_groups) == 2
    num_documents = []
    for ((microbatch,),) in batch.microbatch_groups:
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
    assert len(batch.microbatch_groups) == 2
    for ((microbatch,),) in batch.microbatch_groups:
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
    sample.token_ids = np.arange(6)
    sample.loss_mask = np.zeros(6, dtype=bool)
    sample.logprobs = np.zeros(6, dtype=np.float32)
    sample.advantage = np.zeros(6, dtype=np.float32)

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
    assert batch.group_ids == [0, 1]
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
        await buffer.take_finalized(consuming_policy_version=0)

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

        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 1
        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 2
        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 3

        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=0))
        await asyncio.sleep(0)
        assert not taker.done()  # g4 is finalized but outside the anchored window

        await _finalize(buffer, 0)
        await asyncio.sleep(0)
        assert taker.done()
        assert taker.result().group_id == 0
        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 4

    asyncio.run(run())


def test_no_window_takes_oldest_ready_group_past_a_stuck_head() -> None:
    async def run() -> None:
        # S=1, P=4 -> 8 slots, no window. g0..g4 CLAIMED; g0 never finishes; g1..g4 finish out of id order.
        buffer = _buffer(capacity=8, window_size=None)
        for group_id in range(8):
            await _admit(buffer, group_id)
        claimed_group_ids = [(await buffer.claim_next()).group_id for _ in range(5)]
        assert claimed_group_ids == [0, 1, 2, 3, 4]
        for group_id in (3, 1, 4, 2):
            await _finalize(buffer, group_id)

        # A full batch of P=4 is taken oldest-ready first, without waiting on g0.
        for expected_group_id in (1, 2, 3, 4):
            taker = asyncio.create_task(
                buffer.take_finalized(consuming_policy_version=0)
            )
            await asyncio.sleep(0)
            assert taker.done()
            assert taker.result().group_id == expected_group_id

        # The remaining wait is for generation (nothing finalized), not for the head g0.
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=0))
        await asyncio.sleep(0)
        assert not taker.done()
        await buffer.claim_next()  # g5 -> CLAIMED
        await buffer.claim_next()  # g6 -> CLAIMED
        await _finalize(buffer, 6)
        await asyncio.sleep(0)
        assert taker.done()
        assert taker.result().group_id == 6

    asyncio.run(run())


async def _block_forever(*args, **kwargs) -> None:
    await asyncio.Event().wait()


async def _train_one_step(*args, **kwargs) -> None:
    await asyncio.sleep(0)


def _run_with_generator_watcher(
    watcher,
    *,
    trainer_loop=_block_forever,
    rollout_loop=_block_forever,
    validate=None,
    validation=None,
    on_close=None,
    max_active_rollout_groups=1,
    num_concurrent_rollout_groups=1,
    group_buffer=None,
    start_step=0,
) -> Controller:
    """Run Controller.run with stub loops: `watcher` waits on the engine loops, `trainer_loop` trains, the rest block."""
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            num_training_steps=1,
            max_active_rollout_groups=max_active_rollout_groups,
            num_concurrent_rollout_groups=num_concurrent_rollout_groups,
            window_size=None,
            target_offpolicy_steps=0,
            windowed_fifo_batches=None,
            max_offpolicy_steps=None,
            num_prompts_per_train_step=1,
            group_buffer=group_buffer
            or SimpleNamespace(
                build=lambda **kwargs: SimpleNamespace(
                    close=AsyncMock(side_effect=on_close)
                )
            ),
            validation=validation or ValidationConfig(),
        ),
        trainer=SimpleNamespace(
            training=SimpleNamespace(num_tokens_per_microbatch_per_dp_rank=8)
        ),
    )
    controller.start_step = start_step
    controller._validation_tasks = []
    controller._has_pending_validation = False
    controller.metrics_processor = Mock()
    controller.trainer = None
    controller.trainer_dp_degree = 1
    controller.tokenizer = SimpleNamespace(eos_id=0)
    controller._sampling = SimpleNamespace(temperature=1.0)
    controller.generator_router = SimpleNamespace(
        wait_engine_loop=SimpleNamespace(call_one=watcher)
    )
    with (
        patch.multiple(
            Controller,
            validate=validate or AsyncMock(return_value=[]),
            _make_generate_fn=lambda self: None,
            _data_input_loop=_block_forever,
            _rollout_loop=rollout_loop,
            _batcher_loop=_block_forever,
            _trainer_loop=trainer_loop,
        ),
    ):
        asyncio.run(asyncio.wait_for(controller.run(), timeout=10))
    return controller


def test_run_starts_one_rollout_loop_per_concurrent_group() -> None:
    # 18 active slots (S = 8, P = 2), 3 of them may roll out at once
    started_loops = []

    async def rollout_loop(self, **kwargs) -> None:
        started_loops.append(asyncio.current_task().get_name())
        await asyncio.Event().wait()

    _run_with_generator_watcher(
        _block_forever,
        trainer_loop=_train_one_step,
        rollout_loop=rollout_loop,
        max_active_rollout_groups=18,
        num_concurrent_rollout_groups=3,
    )
    assert sorted(started_loops) == [
        "rollout_loop_0",
        "rollout_loop_1",
        "rollout_loop_2",
    ]


def test_run_fails_when_an_engine_loop_crashes() -> None:
    async def crashed() -> None:
        raise RuntimeError("engine loop crashed")

    with pytest.raises(RuntimeError, match="engine loop crashed"):
        _run_with_generator_watcher(crashed)


def test_run_fails_when_the_engine_loops_exit_mid_run() -> None:
    async def exited() -> None:
        return None

    with pytest.raises(
        RuntimeError, match="generator_wait_engine_loop exited unexpectedly"
    ):
        _run_with_generator_watcher(exited)


@pytest.mark.parametrize(
    "step_0_validation, validate_after_training, started_steps, logged_steps",
    [
        # Ends before training does: the call after training logs it at step 1.
        ("ends", False, [0], [1]),
        # Ends while run()'s finally awaits group_buffer.close(): also logged at step 1.
        ("ends_during_close", False, [0], [1]),
        # Still running: run() cancels it (else run() hangs) and logs nothing.
        ("runs", False, [0], []),
        # Cancelled, then the validation after training runs and is logged at step 1.
        ("runs", True, [0, 1], [1]),
        # Failed: the call after training raises its error.
        ("fails", False, [0], None),
    ],
)
def test_run_ends_validation(
    step_0_validation, validate_after_training, started_steps, logged_steps
) -> None:
    """1 training step: what run() does with the step-0 validation and the validation after training."""
    started = []
    close_started = asyncio.Event()

    async def validate(self, *, step: int) -> list[m.Metric]:
        started.append(step)
        if step == 0 and step_0_validation == "ends_during_close":
            await close_started.wait()
        if step == 0 and step_0_validation == "runs":
            await asyncio.Event().wait()
        if step == 0 and step_0_validation == "fails":
            raise RuntimeError("validation failed")
        return [m.Metric(f"started_{step}", m.Mean(1.0))]

    async def close() -> None:
        close_started.set()
        await asyncio.sleep(0)  # the validation returns

    def run() -> Controller:
        return _run_with_generator_watcher(
            _block_forever,
            trainer_loop=_train_one_step,
            validate=validate,
            validation=ValidationConfig(
                validate_after_training=validate_after_training
            ),
            on_close=close,
        )

    if step_0_validation == "fails":
        with pytest.raises(RuntimeError, match="validation failed"):
            run()
    else:
        controller = run()
        assert [
            call.kwargs["step"]
            for call in controller.metrics_processor.log.call_args_list
        ] == logged_steps
    assert started == started_steps


@pytest.mark.parametrize(
    "start_step, validate_at_step_0, defer_new_if_not_finished, started_steps, logged",
    [
        # The validation started at 3 came due at 2; the one started at 6 covers steps 4 and 6.
        (0, True, True, [0, 3, 6], [(3, "started_0"), (6, "started_3")]),
        (
            0,
            True,
            False,
            [0, 2, 4, 6],
            [(3, "started_0"), (5, "started_2"), (7, "started_4")],
        ),
        (0, False, True, [2, 5], [(5, "started_2"), (8, "started_5")]),
        # Resumed at step 2, a multiple of freq: a validation starts there. validate_at_step_0 is for step 0 only.
        (2, False, True, [2, 5], [(5, "started_2"), (8, "started_5")]),
        # Resumed at step 3: the first validation starts at 4, also with validate_at_step_0.
        (3, True, True, [4, 7], [(7, "started_4")]),
    ],
)
def test_validation_schedule(
    start_step, validate_at_step_0, defer_new_if_not_finished, started_steps, logged
) -> None:
    """freq=2, 8 steps, each validation ends 3 steps after it starts and is logged at that step.

    No validation starts at step 8, the last step.
    """
    end_events: dict[int, asyncio.Event] = {}

    async def validate(*, step: int) -> list[m.Metric]:
        end_events[step] = asyncio.Event()
        await end_events[step].wait()
        return [m.Metric(f"started_{step}", m.Mean(1.0))]

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            num_training_steps=8,
            validation=ValidationConfig(
                freq=2,
                validate_at_step_0=validate_at_step_0,
                defer_new_if_not_finished=defer_new_if_not_finished,
            ),
        )
    )
    controller._validation_tasks = []
    controller._has_pending_validation = False
    controller.validate = validate
    controller.metrics_processor = Mock()

    async def run_steps() -> None:
        for step in range(start_step, 9):
            if step - 3 in end_events:
                end_events[step - 3].set()
                await asyncio.sleep(0)  # the validation returns
            controller._maybe_start_validation(step=step)
            await asyncio.sleep(0)  # a new validation starts

    asyncio.run(run_steps())

    assert list(end_events) == started_steps
    assert [
        (call.kwargs["step"], call.kwargs["metrics"][0].key)
        for call in controller.metrics_processor.log.call_args_list
    ] == logged


def test_generate_fn_forwards_the_session_and_releases_on_failure() -> None:
    """A failed call releases its reserved load, or the router would keep counting it."""

    async def generate_call(*args, **kwargs) -> None:
        # The load stays reserved while the generator runs the call.
        release.assert_not_called()
        raise RuntimeError("generator failed")

    generate_call = AsyncMock(side_effect=generate_call)
    reserve = AsyncMock(
        return_value=SimpleNamespace(generate=SimpleNamespace(call_one=generate_call))
    )
    release = Mock()
    controller = object.__new__(Controller)
    controller.generator_router = SimpleNamespace(
        reserve=SimpleNamespace(call_one=reserve),
        release_reservation=SimpleNamespace(broadcast=release),
    )
    generate = controller._make_generate_fn()
    prompt = Prompt(token_ids=[1, 2])

    with pytest.raises(RuntimeError, match="generator failed"):
        asyncio.run(
            generate(prompt, request_id="r0", group_id=0, routing_session_id="s0")
        )

    # The session reaches both routing layers: the router and the generator's DP routing.
    reserve.assert_awaited_once_with("r0", routing_session_id="s0")
    generate_call.assert_awaited_once_with(
        prompt,
        request_id="r0",
        group_id=0,
        routing_session_id="s0",
        sampling_config=None,
    )
    release.assert_called_once_with("r0")


def test_validation_renames_turn_metrics_and_uses_fresh_group_ids() -> None:
    """Validation prefixes its turns' metrics with `validation_` (`generator/x` -> `validation_generator/x`,
    `math_verify/x` -> `validation_math_verify/x`). Two validations never share a group id."""

    async def run_group_rollouts(*, group_id: int, **kwargs) -> RolloutGroup:
        turn = RolloutTurn(
            rollout_id=RolloutTurnID(group_id=group_id, rollout_id=0, turn_id=0),
            prompt_prefix_len=0,
            prompt_delta_token_ids=np.array([1], dtype=np.int32),
            completion_token_ids=np.array([2], dtype=np.int32),
            completion_logprobs=np.array([-0.1], dtype=np.float32),
            min_policy_version=0,
            max_policy_version=0,
            metrics=[
                m.Metric("generator/queue_time_ms", m.Mean(2.0)),
                m.Metric("math_verify/timeouts", m.Sum(1.0)),
            ],
        )
        rollout = Rollout(
            group_id=group_id,
            rollout_id=0,
            status=RolloutStatus.COMPLETED,
            reward=0.0,
            turns=[turn],
        )
        return RolloutGroup(group_id=group_id, rollouts=[rollout])

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(validation=ValidationConfig())
    )
    controller._validation_group_ids = itertools.count(start=-1, step=-1)
    controller._make_generate_fn = lambda: None
    controller._make_release_session_fn = lambda: None
    controller._rollouter = SimpleNamespace(run_group_rollouts=run_group_rollouts)
    release_groups = AsyncMock()
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=release_groups)
    )

    async def run_two_validations() -> list[m.Metric]:
        for step in (0, 2):
            _, metrics = await controller._collect_validation_rollouts(
                samples=["prompt 1", "prompt 2"], sampling=SamplingConfig(), step=step
            )
        return metrics

    keys = m.MetricsProcessor._aggregate_metrics(asyncio.run(run_two_validations()))
    assert [call.args[0] for call in release_groups.call_args_list] == [
        [-1, -2],
        [-3, -4],
    ]
    assert keys["validation_generator/queue_time_ms/mean"] == 2.0
    assert keys["validation_math_verify/timeouts/sum"] == 2.0
    assert "generator/queue_time_ms/mean" not in keys
    assert "math_verify/timeouts/sum" not in keys


def _scored_group(group_id: int, rewards: list[float | None]) -> RolloutGroup:
    return RolloutGroup(
        group_id=group_id,
        rollouts=[
            Rollout(
                group_id=group_id,
                rollout_id=rollout_id,
                status=RolloutStatus.COMPLETED,
                reward=reward,
            )
            for rollout_id, reward in enumerate(rewards)
        ],
    )


@pytest.mark.parametrize(
    "num_samples_per_prompt, rewards, expected_mean, expected_best_of",
    [
        (1, [[1.0], [0.0]], 0.5, None),
        (2, [[1.0, 0.0], [0.0, 0.0]], 0.25, 0.5),
        # Unscored rollouts (reward None) are skipped; the prompt with none scored is left out.
        (2, [[None, 1.0], [None, None]], 1.0, 1.0),
    ],
)
def test_validate_rolls_out_each_prompt_n_times(
    num_samples_per_prompt: int,
    rewards: list[list[float | None]],
    expected_mean: float,
    expected_best_of: float | None,
) -> None:
    """2 prompts x n rollouts at T=1: the mean covers all rollouts; best_of_n, logged only for
    n > 1, the best rollout per prompt."""
    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            validation=ValidationConfig(
                steps=2,
                num_samples_per_prompt=num_samples_per_prompt,
                top_p=0.95,
                top_k=20,
            )
        )
    )
    controller._sampling = SamplingConfig(temperature=0.8, seed=7, stop_token_ids=[1])
    controller._validation_group_ids = itertools.count(start=-1, step=-1)
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=AsyncMock())
    )
    controller.rollout_recorder = SimpleNamespace(record=lambda **kwargs: None)
    run_group_rollouts = AsyncMock(
        side_effect=[
            _scored_group(-(i + 1), group_rewards)
            for i, group_rewards in enumerate(rewards)
        ]
    )
    controller._rollouter = SimpleNamespace(
        get_validation_samples=lambda steps: ["a", "b"],
        run_group_rollouts=run_group_rollouts,
    )

    metrics = asyncio.run(controller.validate(step=0))

    calls = run_group_rollouts.call_args_list
    assert [call.kwargs["group_size"] for call in calls] == [num_samples_per_prompt] * 2
    # Validation's own temperature, top_p and top_k; the trainer never scores these tokens.
    expected_sampling = SamplingConfig(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        seed=7,
        stop_token_ids=[1],
        scored_by_trainer=False,
    )
    assert [call.kwargs["sampling"] for call in calls] == [expected_sampling] * 2
    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert aggregated["validation_reward/_mean"] == expected_mean
    best_of_key = f"validation_reward/best_of_{num_samples_per_prompt}/mean"
    assert aggregated.get(best_of_key) == expected_best_of


def test_validation_config_rejects_bad_sampling() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ValidationConfig(temperature=-1.0)
    with pytest.raises(ValueError, match="num_samples_per_prompt"):
        ValidationConfig(num_samples_per_prompt=0)
    with pytest.raises(ValueError, match="greedy"):
        ValidationConfig(temperature=0.0, num_samples_per_prompt=4)
    with pytest.raises(ValueError, match="top_p"):
        ValidationConfig(top_p=0.0)
    with pytest.raises(ValueError, match="top_k"):
        ValidationConfig(top_k=0)


def test_fixed_buffer_ignores_the_adaptive_hooks() -> None:
    """The default buffer takes and meters the same with the adaptive buffer's hooks called."""
    config = AsyncLoopConfig(
        num_prompts_per_train_step=4, target_offpolicy_steps=1, windowed_fifo_batches=1
    )
    assert type(config.group_buffer) is RolloutGroupWorkBuffer.Config
    assert config.num_concurrent_rollout_groups == config.max_active_rollout_groups == 8
    assert config.max_offpolicy_steps == 2

    async def run(*, with_hooks: bool) -> list:
        buffer = _buffer(capacity=8, window_size=4)
        for group_id in range(8):
            await _admit(buffer, group_id)
        for _ in range(8):
            await buffer.claim_next()
        for group_id in (3, 1, 2, 5, 0, 7):
            await _finalize(buffer, group_id)
        log = []
        for version in range(2):
            if with_hooks:
                await buffer.record_step_start(trainer_policy_version=version)
            for _ in range(3):
                group = await buffer.take_finalized(consuming_policy_version=version)
                log.append(group.group_id)
            if with_hooks:
                assert buffer.pop_dropped_group_ids() == []
            await buffer.release_active_groups(3, reason="trained")
            log.append({metric.key: metric.value.value for metric in buffer.metrics()})
        return log

    log = asyncio.run(run(with_hooks=True))
    assert log == asyncio.run(run(with_hooks=False))
    assert [entry for entry in log if isinstance(entry, int)] == [0, 1, 2, 3, 5, 7]


def _adaptive_buffer(
    *,
    num_prompts: int,
    max_offpolicy_steps: int | None = 4,
    generation_capacity: int = 64,
    **kwargs,
) -> AdaptiveRolloutGroupWorkBuffer:
    return AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=max_offpolicy_steps
    ).build(
        num_prompts_per_train_step=num_prompts,
        generation_capacity=generation_capacity,
        **kwargs,
    )


def _buffer_metric(buffer: RolloutGroupWorkBuffer, key: str) -> float:
    return next(metric.value.value for metric in buffer.metrics() if metric.key == key)


def test_stall_driven_demand_quantile_uses_every_point_of_the_lookback() -> None:
    history = [44, 46, 41, 48, 45, 47, 50, 43, 46, 45]
    # mean 45.5, sd 2.55, Student-t prediction multiplier for 10 samples at 95% = 1.92 -> 50.4 -> 51 (above the max seen)
    assert (
        estimate_next_unavailable_upper_bound(history=history, probability=0.95) == 51
    )
    assert (
        estimate_next_unavailable_upper_bound(history=[45, 45, 45], probability=0.95)
        == 45
    )
    assert estimate_next_unavailable_upper_bound(history=[24], probability=0.95) == 24
    assert round(prediction_multiplier(samples=10, probability=0.95), 3) == 1.923
    assert round(prediction_multiplier(samples=1000, probability=0.95), 2) == 1.65
    assert mean_age_ceiling(
        num_prompts_per_train_step=8,
        mean_age_limit=4,
        groups_generating=40,
        untrainable_share=0.36,
    ) == pytest.approx(54.4)
    # half the gap, the same up and down, rounded away from the current value
    assert (
        smooth_demand_toward_needed(
            current_demand=64, demand_needed=59, damping_factor=0.5
        )
        == 61
    )
    assert (
        smooth_demand_toward_needed(
            current_demand=64, demand_needed=73, damping_factor=0.5
        )
        == 69
    )
    assert (
        smooth_demand_toward_needed(
            current_demand=64, demand_needed=65, damping_factor=0.5
        )
        == 65
    )


def test_stall_driven_demand_starts_at_three_batches_and_moves_half_the_gap() -> None:
    demand = StallDrivenDemand(num_prompts_per_train_step=8, max_offpolicy_steps=10)
    assert demand.demand == 24
    # nothing ready: unavailable 24, one sample -> quantile 24, needed 8 + 24 + 1 = 33, half the gap up: 24 + 5
    assert (
        demand.observe(step=1, ready=0, generating=24, completed=0, trainable=0) == 29
    )
    assert demand.state == "ok"
    # a full shelf: unavailable max(0, 29 - 40) = 0; history [24, 0] -> mean 12, sd 17, two samples give a wide
    # margin (t = 7.5): quantile 128 -> needed 137. The ceiling at the max caps it: 10 x 8 + 8 + mean(24, 0) x
    # (1 - 40 / 40) = 88. Half the gap up from 29: 29 + 30
    assert (
        demand.observe(step=2, ready=40, generating=0, completed=40, trainable=40) == 59
    )
    assert demand.state == "age-limited"


def test_stall_driven_demand_moves_down_one_group_per_step() -> None:
    for damping_factor in (0.5, 1.0):
        demand = StallDrivenDemand(
            num_prompts_per_train_step=8, start_batches=8, damping_factor=damping_factor
        )
        # a full shelf: unavailable 0 -> needed 8 + 0 + 1 = 9; one group down from 64, whatever the damping
        assert (
            demand.observe(step=1, ready=64, generating=0, completed=64, trainable=64)
            == 63
        )
        assert demand.state == "ok"


def test_stall_driven_demand_above_the_ceiling_moves_down_half_the_gap() -> None:
    # ceiling with a max of 1: 1 x 8 + 8 + generating x untrainable share = 16 + 0 in both cases below
    # nothing ready: needed 8 + 64 + 1 = 73, capped at 16; half the gap down: 64 - 24
    demand = StallDrivenDemand(
        num_prompts_per_train_step=8, max_offpolicy_steps=1, start_batches=8
    )
    assert demand.observe(step=1, ready=0, generating=0, completed=0, trainable=0) == 40
    assert demand.state == "age-limited"
    # a full shelf: needed 8 + 4 + 1 = 13 is under the ceiling, but demand 64 is above it; half the gap down too
    demand = StallDrivenDemand(
        num_prompts_per_train_step=8, max_offpolicy_steps=1, start_batches=8
    )
    assert (
        demand.observe(step=1, ready=60, generating=4, completed=60, trainable=60) == 40
    )
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


def test_stall_driven_demand_without_a_max_has_no_ceiling_and_drops_nothing() -> None:
    demand = StallDrivenDemand(num_prompts_per_train_step=2, max_offpolicy_steps=None)
    for step in range(1, 11):
        demand.observe(step=step, ready=0, generating=6, completed=0, trainable=0)
    # with a max of 1 the ceiling would have held it at 8
    assert demand.demand > 8 and demand.state == "ok"

    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=2, max_offpolicy_steps=None, generation_capacity=4
        )
        await _admit(buffer, 0)
        await buffer.claim_next()  # g0 generates under version 0
        await _finalize(buffer, 0)
        # 50 versions old: kept
        selected = await buffer.take_finalized(consuming_policy_version=50)
        assert selected is not None and selected.group_id == 0
        assert _buffer_metric(buffer, "rollout_buffer/dropped_too_old") == 0

    asyncio.run(run())
    with pytest.raises(ValueError, match="target_offpolicy_steps"):
        AdaptiveRolloutGroupWorkBuffer.Config(
            max_offpolicy_steps=None, target_offpolicy_steps=0
        )
    # a target without a max is allowed: the ceiling holds the mean age, nothing is dropped
    config = AsyncLoopConfig(
        group_buffer=AdaptiveRolloutGroupWorkBuffer.Config(
            max_offpolicy_steps=None, target_offpolicy_steps=3
        )
    )
    assert config.max_offpolicy_steps is None


def test_adaptive_buffer_takes_oldest_finalized_and_lets_slow_groups_keep_their_slot() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=2, generation_capacity=4)
        for group_id in range(4):
            await _admit(buffer, group_id)
        for _ in range(4):
            await buffer.claim_next()
        await _finalize(buffer, 2)
        await _finalize(buffer, 1)

        # g0 is still CLAIMED: the batcher gets g1 then g2 without waiting for it
        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 1
        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 2
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=0))
        await asyncio.sleep(0)
        assert not taker.done()

        await _finalize(buffer, 0)
        assert (await taker).group_id == 0
        # taking never frees a slot: 4 admitted -> 4 still active
        assert _buffer_metric(buffer, "rollout_buffer/active_slots_in_use_peak") == 4

    asyncio.run(run())


def test_adaptive_buffer_drops_groups_past_max_offpolicy_steps() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=2, max_offpolicy_steps=1, generation_capacity=4
        )
        for group_id in range(4):
            await _admit(buffer, group_id)
        for _ in range(4):
            await buffer.claim_next()  # g0..g3 generate under version 0

        # Step 1 trains g2 and g3 at version 0; the pull of version 1 releases their slots
        for group_id in (2, 3):
            await _finalize(buffer, group_id)
            await buffer.take_finalized(consuming_policy_version=0)
        await buffer.release_active_groups(2, reason="trained")
        await _admit(buffer, 4)
        await buffer.claim_next()  # g4 generates under version 1

        # The trainer starts step 2 (version 1); g0 and g1 finish
        await buffer.record_step_start(trainer_policy_version=1)
        await _finalize(buffer, 1)
        await _finalize(buffer, 0)
        # Finalization alone cannot know whether the batcher is assembling the
        # current or prefetched batch, so it must not guess the consume version.
        assert _buffer_metric(buffer, "rollout_buffer/num_groups_finalized") == 2

        # The batch being assembled trains at version 2: g0 and g1 would be 2 old -> dropped; g4 is 1 old
        await _finalize(buffer, 4)
        selected = await buffer.take_finalized(consuming_policy_version=2)
        assert selected is not None
        assert selected.group_id == 4
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/dropped_too_old"] == 2
        # The controller commits dropped ids once, so the dataloader does not replay them
        assert buffer.pop_dropped_group_ids() == [0, 1]
        assert buffer.pop_dropped_group_ids() == []
        # Dropped slots are free at once: 5 admitted, 2 trained, 2 dropped -> 1 active
        assert metrics["rollout_buffer/available_active_slots"] == (
            metrics["rollout_buffer/demand_target_groups"] - 1
        )

    asyncio.run(run())


def test_adaptive_buffer_cancels_claimed_groups_past_max_offpolicy_steps() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )
        for group_id in range(3):
            await _admit(buffer, group_id)
        works = [await buffer.claim_next() for _ in range(3)]  # g0..g2 under version 0
        hung = asyncio.create_task(asyncio.sleep(3600))
        buffer.track_generation(works[0], hung)

        # Step 1 trains g2 at version 0; the pull of version 1 releases its slot
        await _finalize(buffer, 2)
        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 2
        await buffer.release_active_groups(1, reason="trained")

        # The batch for version 1 makes g0/g1 1 old: still trainable, nothing cancelled
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=1))
        await asyncio.sleep(0)
        assert not hung.cancelled()
        await _finalize(buffer, 1)
        assert (await taker).group_id == 1

        # The batch for version 2 makes g0 2 old: it can never train, so it is cancelled mid-generation
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        with pytest.raises(asyncio.CancelledError):
            await hung
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/cancelled_too_old"] == 1
        assert metrics["rollout_buffer/dropped_too_old"] == 0
        # Counted as dropped at once: the dataloader commits it at the next optimizer step
        assert buffer.pop_dropped_group_ids() == [0]
        # Its slot stays charged until the rollout loop has stopped it on its worker and releases it
        assert metrics["rollout_buffer/num_groups_claimed"] == 1
        available = metrics["rollout_buffer/available_active_slots"]
        await buffer.release_cancelled(0)
        assert _buffer_metric(buffer, "rollout_buffer/num_groups_claimed") == 0
        assert (
            _buffer_metric(buffer, "rollout_buffer/available_active_slots")
            == available + 1
        )
        await buffer.close()
        assert await taker is None

    asyncio.run(run())


def test_adaptive_buffer_cancel_counts_like_a_finalized_too_old_drop() -> None:
    """A group cancelled while generating moves the demand counters exactly as the same group would if it
    finished and were dropped too old, so the demand ceiling's untrainable share does not change."""

    async def counters(*, finish_first: bool) -> tuple:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )
        for group_id in range(2):
            await _admit(buffer, group_id)
        work0, _ = [await buffer.claim_next() for _ in range(2)]  # under version 0
        buffer.track_generation(work0, asyncio.create_task(asyncio.sleep(3600)))
        await _finalize(buffer, 1)
        if finish_first:
            await _finalize(buffer, 0)
        # The batch for version 2: g0 is 2 old (dropped or cancelled); g1 is too, so nothing is returned
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        await asyncio.sleep(0)
        if not finish_first:
            await asyncio.sleep(0)  # let the cancelled task finish
            await buffer.release_cancelled(0)
        result = (
            buffer._completed_since_step_start,
            buffer._dropped_since_step_start,
            buffer.pop_dropped_group_ids(),
            _buffer_metric(buffer, "rollout_buffer/available_active_slots"),
        )
        await buffer.close()
        assert await taker is None
        return result

    async def run() -> None:
        dropped = await counters(finish_first=True)
        cancelled = await counters(finish_first=False)
        # completed 2, dropped 2 (g0 and g1 both too old), the same committed ids and free slots
        assert dropped == cancelled
        assert sorted(dropped[2]) == [0, 1]

    asyncio.run(run())


def test_adaptive_buffer_cancel_racing_the_last_rollout_releases_the_slot_once() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )
        await _admit(buffer, 0)
        await _admit(buffer, 1)
        work0 = await buffer.claim_next()  # g0 under version 0
        # g0's generation already ended, but its result is not stored yet: the cancel is a no-op
        finished = asyncio.create_task(asyncio.sleep(0))
        await finished
        buffer.track_generation(work0, finished)
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        await asyncio.sleep(0)
        assert _buffer_metric(buffer, "rollout_buffer/cancelled_too_old") == 1
        available = _buffer_metric(buffer, "rollout_buffer/available_active_slots")

        # The late result is dropped and frees the slot, exactly once
        assert not await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        assert (
            _buffer_metric(buffer, "rollout_buffer/available_active_slots")
            == available + 1
        )
        await buffer.release_cancelled(0)  # entry already gone: no second release
        assert (
            _buffer_metric(buffer, "rollout_buffer/available_active_slots")
            == available + 1
        )
        # g1 was never claimed (WAITING): never cancelled
        assert _buffer_metric(buffer, "rollout_buffer/num_groups_waiting") == 1
        assert buffer.pop_dropped_group_ids() == [0]
        await buffer.close()
        assert await taker is None

    asyncio.run(run())


def _controller_for_rollout_loops(
    *, run_group_rollouts, cancel_group, events: list
) -> Controller:
    async def release_groups(group_ids: list[int]) -> None:
        events.append(("release_groups", group_ids))

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(num_samples_per_prompt=2)
    )
    controller._sampling = None
    controller.generator_router = SimpleNamespace(
        release_groups=SimpleNamespace(call_one=release_groups)
    )
    controller.rollout_recorder = SimpleNamespace(record=lambda **kwargs: None)
    controller._rollouter = SimpleNamespace(
        run_group_rollouts=run_group_rollouts, cancel_group=cancel_group
    )
    return controller


def test_rollout_loop_stops_a_group_the_buffer_cancels_on_its_worker() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )
        events = []
        cancel_drained = asyncio.Event()

        async def run_group_rollouts(*, group_id: int, **kwargs) -> RolloutGroup:
            # g0 is a slow episode; a cancel stops only this wait, as the rollouter's shield does
            if group_id == 0:
                await asyncio.sleep(3600)
            return RolloutGroup(group_id=group_id, rollouts=[])

        async def cancel_group(group_id: int) -> None:
            # The rollouter returns once the group's run_group ended on its worker.
            events.append(("cancel_group", group_id))
            await cancel_drained.wait()

        controller = _controller_for_rollout_loops(
            run_group_rollouts=run_group_rollouts,
            cancel_group=cancel_group,
            events=events,
        )
        loops = [
            asyncio.create_task(
                controller._rollout_loop(
                    group_buffer=buffer, generate_fn=None, release_session_fn=None
                )
            )
            for _ in range(2)
        ]
        await _admit(buffer, 0)
        await _admit(buffer, 1)
        # g1 finishes and trains at version 0; g0 is slow
        assert (await buffer.take_finalized(consuming_policy_version=0)).group_id == 1
        await buffer.release_active_groups(1, reason="trained")

        # The batch for version 2 cancels g0; its slot is held until its worker stopped it
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        for _ in range(3):
            await asyncio.sleep(0)
        assert ("cancel_group", 0) in events
        assert ("release_groups", [0]) not in events
        assert _buffer_metric(buffer, "rollout_buffer/num_groups_claimed") == 1
        cancel_drained.set()
        for _ in range(5):
            await asyncio.sleep(0)
        assert _buffer_metric(buffer, "rollout_buffer/num_groups_claimed") == 0
        assert ("release_groups", [0]) in events

        # Both loops keep serving: g2 is generated and taken
        await _admit(buffer, 2)
        assert (await taker).group_id == 2
        assert not any(loop.done() for loop in loops)

        await buffer.close()
        await asyncio.gather(*loops)

    asyncio.run(run())


def test_rollout_loop_propagates_a_cancel_the_buffer_did_not_make() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )
        events = []

        async def run_group_rollouts(*, group_id: int, **kwargs) -> RolloutGroup:
            raise asyncio.CancelledError  # e.g. an awaited call cancelled from elsewhere

        controller = _controller_for_rollout_loops(
            run_group_rollouts=run_group_rollouts,
            cancel_group=AsyncMock(),
            events=events,
        )
        loop = asyncio.create_task(
            controller._rollout_loop(
                group_buffer=buffer, generate_fn=None, release_session_fn=None
            )
        )
        await _admit(buffer, 0)
        # The loop exits with the CancelledError instead of treating it as a buffer cancel.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(loop, timeout=5)
        controller._rollouter.cancel_group.assert_not_awaited()

    asyncio.run(run())


def test_rollout_loop_exits_when_shutdown_lands_with_a_buffer_cancel() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=1, max_offpolicy_steps=1, generation_capacity=4
        )
        events = []
        group_started = asyncio.Event()

        async def run_group_rollouts(*, group_id: int, **kwargs) -> RolloutGroup:
            group_started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                events.append(("generation_cancelled", group_id))
                raise

        controller = _controller_for_rollout_loops(
            run_group_rollouts=run_group_rollouts,
            cancel_group=AsyncMock(),
            events=events,
        )
        loop = asyncio.create_task(
            controller._rollout_loop(
                group_buffer=buffer, generate_fn=None, release_session_fn=None
            )
        )
        await _admit(buffer, 0)
        await asyncio.wait_for(group_started.wait(), timeout=5)

        # The batch for version 2 cancels g0 (2 versions old, max 1); shutdown cancels the loop before it wakes
        taker = asyncio.create_task(buffer.take_finalized(consuming_policy_version=2))
        await asyncio.sleep(0)
        assert events == []
        loop.cancel()

        # The loop exits without cancelling g0 on its worker: `Rollouter.close` waits for that call
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(loop, timeout=5)
        assert events == [("generation_cancelled", 0)]
        controller._rollouter.cancel_group.assert_not_awaited()

        await buffer.close()
        assert await taker is None

    asyncio.run(run())


def test_adaptive_buffer_measures_the_shelf_and_moves_demand() -> None:
    async def run() -> None:
        # P=4: demand starts at three batches = 12; the shelf is what is finished and not held by the trainer.
        buffer = _adaptive_buffer(num_prompts=4, generation_capacity=40)
        assert _buffer_metric(buffer, "rollout_buffer/demand_target_groups") == 12
        for group_id in range(12):
            await _admit(buffer, group_id)
        for _ in range(12):
            await buffer.claim_next()

        # nothing finished: unavailable 12 -> needed 4 + 12 + 1 = 17 -> half the gap up: 12 + 3
        await buffer.record_step_start(trainer_policy_version=0)
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/demand_target_groups"] == 15
        assert metrics["rollout_buffer/ready_at_step_start"] == 0
        assert metrics["rollout_buffer/generating_at_step_start"] == 12
        assert metrics["rollout_buffer/unavailable_at_step_start"] == 12
        assert metrics["rollout_buffer/demand_age_limited"] == 0

        # ten groups finish; the batcher selects four and the trainer trains them at version 0. The pull
        # of version 1 has not released them yet, so they leave the shelf
        for group_id in range(10):
            await _finalize(buffer, group_id)
        for _ in range(4):
            await buffer.take_finalized(consuming_policy_version=0)
        await buffer.record_step_start(trainer_policy_version=1)
        metrics = {metric.key: metric.value.value for metric in buffer.metrics()}
        assert metrics["rollout_buffer/ready_at_step_start"] == 6
        assert metrics["rollout_buffer/generating_at_step_start"] == 2
        assert metrics["rollout_buffer/unavailable_at_step_start"] == 9
        # unavailable 15 - 6 = 9; history [12, 9] -> mean 10.5 + 4.46 x 2.12 -> 20 -> needed 25; ceiling
        # 4 x 4 + 4 + mean(12, 2) x (1 - 10 / 10) = 20 -> capped at 20 -> half the gap: 15 + 3
        assert metrics["rollout_buffer/demand_target_groups"] == 18
        assert metrics["rollout_buffer/demand_age_limited"] == 1

    asyncio.run(run())


def test_adaptive_buffer_target_offpolicy_steps_caps_demand() -> None:
    # P=4, target 5 of max 10: the ceiling is 5 x 4 + 4 + generating x untrainable share
    buffer = AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=10, target_offpolicy_steps=5
    ).build(num_prompts_per_train_step=4, generation_capacity=40)
    assert _buffer_metric(buffer, "rollout_buffer/demand_target_groups") == 12

    async def run() -> None:
        for _ in range(12):
            # nothing generating, nothing completed
            await buffer.record_step_start(trainer_policy_version=0)
        # nothing generating and no completions: ceiling 20 + 4 + 0 = 24, however short the shelf
        assert _buffer_metric(buffer, "rollout_buffer/demand_target_groups") == 24
        assert _buffer_metric(buffer, "rollout_buffer/demand_age_limited") == 1

    asyncio.run(run())
    with pytest.raises(ValueError, match="target_offpolicy_steps"):
        AdaptiveRolloutGroupWorkBuffer.Config(target_offpolicy_steps=0)
    with pytest.raises(ValueError, match="target_offpolicy_steps"):
        AdaptiveRolloutGroupWorkBuffer.Config(
            max_offpolicy_steps=4, target_offpolicy_steps=5
        )


def test_adaptive_buffer_demand_stays_under_the_mean_age_ceiling() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(
            num_prompts=2, max_offpolicy_steps=1, generation_capacity=4
        )
        for group_id in range(4):
            await _admit(buffer, group_id)
        for _ in range(4):
            await buffer.claim_next()
        for _ in range(10):
            await buffer.record_step_start(trainer_policy_version=0)
        # nothing has completed, so every generating group counts as untrainable: ceiling 1 x 2 + 2 + 4 x 1.0 = 8
        assert _buffer_metric(buffer, "rollout_buffer/demand_target_groups") == 8
        # demand has room, generation does not
        waiter = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not waiter.done()
        await buffer.close()
        assert await waiter is False

    asyncio.run(run())


def test_adaptive_buffer_does_not_queue_ahead_of_generation() -> None:
    async def run() -> None:
        buffer = _adaptive_buffer(num_prompts=2, generation_capacity=2)
        await _admit(buffer, 0)
        await _admit(buffer, 1)
        await buffer.claim_next()
        await buffer.claim_next()

        third_slot = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not third_slot.done()

        # Finalization releases a generation permit without releasing the
        # group's active demand slot.
        await _finalize(buffer, 0)
        assert await third_slot
        assert (
            _buffer_metric(buffer, "rollout_buffer/available_generation_permits") == 1
        )
        await buffer.close()

    asyncio.run(run())


def test_adaptive_buffer_admits_a_granted_slot_after_the_demand_drops() -> None:
    async def run() -> None:
        # P=2: demand 6. Five groups finished and none trained: a full shelf.
        buffer = _adaptive_buffer(num_prompts=2, generation_capacity=8)
        for group_id in range(5):
            await _admit(buffer, group_id)
            await buffer.claim_next()
            await _finalize(buffer, group_id)
        assert await buffer.wait_for_slot()

        # While the data input loop reads the sample, a step start lowers the demand:
        # unavailable 6 - 5 = 1 -> needed 2 + 1 + 1 = 4 -> one group down to 5, which the 5 groups fill.
        await buffer.record_step_start(trainer_policy_version=0)
        assert _buffer_metric(buffer, "rollout_buffer/demand_target_groups") == 5
        # The granted slot is still admitted; the next one waits.
        await buffer.add_work(RolloutGroupWork(group_id=5, sample=object()))
        assert _buffer_metric(buffer, "rollout_buffer/active_slots_in_use_peak") == 6
        waiter = asyncio.create_task(buffer.wait_for_slot())
        await asyncio.sleep(0)
        assert not waiter.done()
        await buffer.close()

    asyncio.run(run())


def test_batcher_passes_the_version_that_trains_each_group() -> None:
    """Batches train in order, one per version, so the batch a group joins sets its consuming version."""
    # Groups 0, 2, 3, 4 are trainable; group 1 is not and joins no batch's count.
    trainable = {0: True, 1: False, 2: True, 3: True, 4: True}
    consuming_policy_versions = []

    class _Buffer:
        async def take_finalized(self, *, consuming_policy_version):
            consuming_policy_versions.append(consuming_policy_version)
            group_id = len(consuming_policy_versions) - 1
            if group_id == len(trainable):
                return None
            return RolloutGroup(group_id=group_id, rollouts=[])

        async def release_active_groups(self, count, *, reason):
            assert reason == "untrainable_group"

    trainable_seen = []

    async def add_group(*, rollout_group):
        # A batch of 2 trainable groups, as the batcher actor returns it.
        if not trainable[rollout_group.group_id]:
            return None, False
        trainable_seen.append(rollout_group.group_id)
        return ("batch" if len(trainable_seen) % 2 == 0 else None), True

    controller = SimpleNamespace(
        start_step=5,
        _group_buffer=_Buffer(),
        batcher=SimpleNamespace(add_group=SimpleNamespace(call_one=add_group)),
    )
    queue: asyncio.Queue = asyncio.Queue()
    asyncio.run(Controller._batcher_loop(controller, training_batch_queue=queue))
    # [0, 1, 2] trains at version 5, [3, 4] at 6; the last call returns None
    assert consuming_policy_versions == [5, 5, 5, 6, 6, 7]
    assert queue.qsize() == 3  # two batches and the None sentinel


def test_run_builds_the_adaptive_buffer_with_one_generating_group_per_rollout_loop() -> None:
    """The adaptive buffer admits at most one generating group per rollout loop, and starts at the resumed version."""
    controller = _run_with_generator_watcher(
        _block_forever,
        trainer_loop=_train_one_step,
        num_concurrent_rollout_groups=3,
        group_buffer=AdaptiveRolloutGroupWorkBuffer.Config(max_offpolicy_steps=8),
        start_step=5,
    )
    buffer = controller._group_buffer
    assert isinstance(buffer, AdaptiveRolloutGroupWorkBuffer)
    assert _buffer_metric(buffer, "rollout_buffer/generation_capacity_groups") == 3
    # Resumed at step 5: a group claimed now is 0 versions old for the first batch, which trains at 5.
    assert buffer._generator_policy_version == buffer._consuming_policy_version == 5
