# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for fixed-bin LPT packing and its FFD fallback."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchtitan.rl.rollout.types import RolloutTurnID

from torchtitan.rl.trainer.batcher import Batcher, TrainerStepBatch
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)


def _make_samples(lengths: list[int]) -> list[TrainingSample]:
    return [
        TrainingSample(
            min_policy_version=0,
            max_policy_version=0,
            rollout_id=RolloutTurnID(group_id=0, rollout_id=index, turn_id=0),
            token_ids=np.arange(length + 1),
            loss_mask=np.array([False] + [True] * length),
            logprobs=np.zeros(length + 1, dtype=np.float32),
            advantage=np.zeros(length + 1, dtype=np.float32),
        )
        for index, length in enumerate(lengths)
    ]


def test_padding_workload_matches_existing_full_length_segments() -> None:
    batcher = Batcher.Config(max_num_documents=20).build(
        num_tokens_per_microbatch_per_dp_rank=60,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    long_samples = _make_samples([8] * 6)
    short_samples = _make_samples([1] * 20)

    assert batcher._attention_workload(long_samples) == 6 * 8**2 + 10**2 + 2**2
    assert batcher._attention_workload(short_samples) == 20 + 4 * 10**2
    assert batcher._attention_workload([]) == 6 * 10**2

    for samples in (long_samples, short_samples, []):
        microbatch = batcher._pack_training_samples(samples)
        num_padding_segments = int(
            ((microbatch.positions == 0) & microbatch.padding_mask).sum().item()
        )
        assert num_padding_segments <= 6


def test_lpt_uses_padding_workload_for_full_length_documents() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=48,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([8] * 3 + [2] * 16)

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert [
        batcher._attention_workload(rank_samples)
        for row in assignments
        for rank_samples in row
    ] == [288, 264]
    assert sorted(
        sum(map(batcher.num_tokens_to_pack, rank_samples))
        for row in assignments
        for rank_samples in row
    ) == [26, 30]


def test_lpt_does_not_leave_a_full_length_document_in_an_expensive_bin() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=20,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([10, 5, 5, 5, 5])

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert [
        batcher._attention_workload(rank_samples)
        for row in assignments
        for rank_samples in row
    ] == [150, 100]


def test_lpt_rebalances_ffd_bins_with_document_limit() -> None:
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=4,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([4, 4, 3, 3, 3, 3, 2, 2])

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert [
        [batcher._attention_workload(rank_samples) for rank_samples in microbatch]
        for microbatch in assignments
    ] == [[26, 26], [22, 22]]
    assert all(
        len(rank_samples) <= 3
        and sum(map(batcher.num_tokens_to_pack, rank_samples)) <= 8
        for microbatch in assignments
        for rank_samples in microbatch
    )
    assert sorted(
        sample.rollout_id.rollout_id
        for microbatch in assignments
        for rank_samples in microbatch
        for sample in rank_samples
    ) == list(range(len(samples)))


def test_lpt_falls_back_when_ffd_fits_but_lpt_cannot() -> None:
    batcher = Batcher.Config(max_num_documents=3).build(
        num_tokens_per_microbatch_per_dp_rank=6,
        max_context_length=3,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([3] * 8 + [2] * 9)
    assert batcher._pack_with_lpt(samples, target_num_bins=7) is None

    assignments = batcher._assign_training_samples_to_microbatches(samples)

    assert len(assignments) == 7
    assert [
        sum(batcher.num_tokens_to_pack(sample) for sample in rank_samples)
        for (rank_samples,) in assignments
    ] == [6] * 7
    assert (
        sorted(len(rank_samples) for (rank_samples,) in assignments)
        == [2] * 4 + [3] * 3
    )


def test_three_pp_microbatches_balance_rank_costs_within_step() -> None:
    batcher = Batcher.Config(max_num_documents=1).build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([10, 9, 8, 7, 6, 5])

    assignments = batcher._assign_training_samples_to_microbatches(
        samples, num_pp_microbatches=3
    )

    assert len(assignments) == 3
    workloads = [
        [batcher._attention_workload(rank_samples) for rank_samples in microbatch]
        for microbatch in assignments
    ]
    assert workloads == [[100, 82], [58, 68], [50, 52]]
    assert [sum(rank_workloads) for rank_workloads in zip(*workloads)] == [208, 202]


def test_packing_preserves_all_samples_and_capacity_constraints() -> None:
    rng = random.Random(42)
    for _ in range(40):
        seq_len = rng.randint(3, 12)
        num_rows = rng.randint(1, 3)
        dp_degree = rng.randint(1, 4)
        num_pp_microbatches = rng.choice([1, 3])
        max_num_documents = rng.choice([None, 1, 3, 6])
        batcher = Batcher.Config(max_num_documents=max_num_documents).build(
            num_tokens_per_microbatch_per_dp_rank=num_rows * seq_len,
            max_context_length=seq_len,
            num_prompts_per_train_step=1,
            dp_degree=dp_degree,
            pad_id=0,
            temperature=1.0,
        )
        samples = _make_samples(
            [rng.randint(1, seq_len) for _ in range(rng.randint(1, 30))]
        )

        assignments = batcher._assign_training_samples_to_microbatches(
            samples, num_pp_microbatches=num_pp_microbatches
        )

        assert len(assignments) % num_pp_microbatches == 0
        assert all(len(row) == dp_degree for row in assignments)
        for row in assignments:
            for rank_samples in row:
                assert sum(map(batcher.num_tokens_to_pack, rank_samples)) <= (
                    num_rows * seq_len
                )
                if max_num_documents is not None:
                    assert len(rank_samples) <= max_num_documents
        assert sorted(
            sample.rollout_id.rollout_id
            for row in assignments
            for rank_samples in row
            for sample in rank_samples
        ) == list(range(len(samples)))


def test_microbatch_carries_the_sampling_temperature_into_the_loss() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=0.5,
    )
    microbatch = batcher._pack_training_samples(_make_samples([3]))

    temperature = microbatch.as_input_dict()["loss_kwargs"]["temperature"]
    assert temperature.dtype == torch.float32
    assert temperature.tolist() == [0.5] * 8


def test_microbatch_moves_loss_kwargs_with_the_model_inputs() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    microbatch = batcher._pack_training_samples(_make_samples([3]))

    input_dict = microbatch.to_input_dict("meta")

    assert input_dict["input"].device.type == "meta"
    assert set(input_dict["loss_kwargs"]) == {
        "generator_logprobs",
        "temperature",
        "loss_mask",
        "advantages",
    }
    for value in input_dict["loss_kwargs"].values():
        assert value.device.type == "meta"


class _SizedSample:
    """Stands in for a TrainingSample in packing-only tests: packs `num_tokens` tokens."""

    __slots__ = ("num_tokens",)

    def __init__(self, num_tokens: int) -> None:
        self.num_tokens = num_tokens


def _record_streamed_step(
    batcher: Batcher,
    groups: list[list[_SizedSample]],
    *,
    rng: random.Random | None = None,
) -> tuple[list[list[list[_SizedSample]]], list[list[list[_SizedSample]]]]:
    """Add one step's groups to a streaming batcher of `_SizedSample`s; return the streamed batches'
    rows and the step's last batch's rows, each `[num_rows][dp_degree]` bins, as packing assigns them.

    The trainer has nothing queued, or with `rng`, a random number of microbatch groups queued at each
    group, and the controller sometimes asks for microbatches between groups.
    """
    num_pp_microbatches = batcher._num_pp_microbatches
    streamed_rows: list[list[list[_SizedSample]]] = []
    last_rows: list[list[list[_SizedSample]]] = []

    def record_pack(*, streamed_bins=None):
        if streamed_bins is None:
            batcher._take_groups()
            last_rows.extend(
                batcher._assign_training_samples_to_microbatches(
                    batcher._unsent_training_samples,
                    num_pp_microbatches=num_pp_microbatches,
                )
            )
            batcher._unsent_training_samples = []
            return TrainerStepBatch(
                microbatch_groups=[],
                global_loss_token_counts=torch.zeros(1),
                global_routing_token_counts=torch.zeros(1),
                metrics=[],
                group_ids=[],
                min_policy_versions=[],
            )
        assert len(streamed_bins) % (num_pp_microbatches * batcher._dp_degree) == 0
        streamed_rows.extend(
            batcher._assign_bins_to_dp_ranks(
                streamed_bins, num_pp_microbatches=num_pp_microbatches
            )
        )
        return TrainerStepBatch(
            microbatch_groups=[],
            global_loss_token_counts=torch.zeros(1),
            global_routing_token_counts=torch.zeros(1),
            metrics=[],
            group_ids=[],
            min_policy_versions=[],
            ends_step=False,
        )

    def num_queued() -> int:
        return (
            0
            if rng is None
            else rng.randint(0, batcher._stream_queued_microbatches + 1)
        )

    batcher.num_tokens_to_pack = lambda sample: sample.num_tokens
    batcher._pack_one_training_batch = record_pack
    for group_id, samples in enumerate(groups):
        batch, _ = batcher.add_training_samples(
            training_sample_group=TrainingSampleGroup(
                group_id=group_id, training_samples=samples, metrics=[]
            ),
            num_queued_microbatch_groups=num_queued(),
        )
        is_last_group = group_id == len(groups) - 1
        assert batch is None or batch.ends_step == is_last_group
        assert (batch is not None) or not is_last_group
        if rng is not None and not is_last_group and rng.random() < 0.5:
            batch = batcher.pack_ready_microbatches(
                num_queued_microbatch_groups=num_queued()
            )
            assert batch is None or not batch.ends_step
    return streamed_rows, last_rows


def _streaming_batcher(
    *,
    num_tokens_per_microbatch: int,
    max_context_length: int,
    num_groups: int,
    dp_degree: int,
    num_pp_microbatches: int = 1,
    max_num_documents: int | None = None,
) -> Batcher:
    return Batcher.Config(
        stream_microbatches=True, max_num_documents=max_num_documents
    ).build(
        num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
        max_context_length=max_context_length,
        num_prompts_per_train_step=num_groups,
        dp_degree=dp_degree,
        num_pp_microbatches=num_pp_microbatches,
        pad_id=0,
        temperature=1.0,
    )


def _document_attention_imbalance(row: list[list[_SizedSample]]) -> float:
    """Max over mean, across one row's DP ranks, of the sum of squared sample lengths."""
    works = [sum(sample.num_tokens**2 for sample in bin_) for bin_ in row]
    return max(works) / (sum(works) / len(works))


def test_streamed_rows_balance_attention_on_realistic_lengths() -> None:
    """A 16-rank step of agentic rollouts, lengths as in a 64K-context run (log-normal, median ~24K,
    correlated within a group), 2 contexts per microbatch: each streamed row's ranks get about the same
    attention work, every streamed bin is at least 98% full, and the step needs about as many rows as
    packing it at once.

    Before streamed rows were built this way, each row repacked FFD's longest unsent samples into
    exactly 16 bins, which rarely fit; the FFD fallback averaged 1.51x max/mean attention work on
    these lengths (1.11x now).
    """
    rng = np.random.default_rng(0)
    num_tokens_per_microbatch, max_context_length, dp_degree = 131_072, 65_536, 16
    groups = []
    for _ in range(96):
        center = rng.lognormal(mean=10.08, sigma=0.45)
        lengths = center * rng.lognormal(mean=0.0, sigma=0.23, size=26)
        groups.append(
            [_SizedSample(int(n)) for n in np.clip(lengths, 1_000, max_context_length)]
        )
    batcher = _streaming_batcher(
        num_tokens_per_microbatch=num_tokens_per_microbatch,
        max_context_length=max_context_length,
        num_groups=len(groups),
        dp_degree=dp_degree,
    )
    streamed_rows, last_rows = _record_streamed_step(batcher, groups)

    samples = [sample for group in groups for sample in group]
    whole_step_rows = batcher._assign_training_samples_to_microbatches(samples)
    assert len(streamed_rows) >= 0.8 * len(whole_step_rows)
    assert len(streamed_rows) + len(last_rows) <= len(whole_step_rows) + 1
    assert all(
        sum(sample.num_tokens for sample in bin_) >= 0.98 * num_tokens_per_microbatch
        for row in streamed_rows
        for bin_ in row
    )
    imbalances = [_document_attention_imbalance(row) for row in streamed_rows]
    assert sum(imbalances) / len(imbalances) < 1.15
    assert max(imbalances) < 1.5


@pytest.mark.parametrize("num_pp_microbatches", [1, 2])
def test_streamed_packing_sends_each_sample_once_per_step(
    num_pp_microbatches: int,
) -> None:
    rng = random.Random(0)
    for _ in range(30):
        seq_len = rng.randint(4, 12)
        num_tokens_per_microbatch = seq_len * rng.randint(1, 3)
        dp_degree = rng.randint(1, 4)
        max_num_documents = rng.choice([None, 2, 4])
        num_groups = rng.randint(1, 6)
        batcher = _streaming_batcher(
            num_tokens_per_microbatch=num_tokens_per_microbatch,
            max_context_length=seq_len,
            num_groups=num_groups,
            dp_degree=dp_degree,
            num_pp_microbatches=num_pp_microbatches,
            max_num_documents=max_num_documents,
        )
        # Two steps on one batcher: the second starts with nothing unsent.
        for _step in range(2):
            groups = [
                [
                    _SizedSample(rng.randint(1, seq_len))
                    for _ in range(rng.randint(1, 8))
                ]
                for _ in range(num_groups)
            ]
            streamed_rows, last_rows = _record_streamed_step(batcher, groups, rng=rng)
            rows = streamed_rows + last_rows
            assert len(streamed_rows) % num_pp_microbatches == 0
            assert len(last_rows) % num_pp_microbatches == 0
            assert all(len(row) == dp_degree for row in rows)
            for row in rows:
                for bin_ in row:
                    assert sum(s.num_tokens for s in bin_) <= num_tokens_per_microbatch
                    if max_num_documents is not None:
                        assert len(bin_) <= max_num_documents
            packed = [id(s) for row in rows for bin_ in row for s in bin_]
            assert sorted(packed) == sorted(id(s) for group in groups for s in group)
            assert batcher._unsent_training_samples == []
            assert batcher._groups_for_next_batch == []


def test_streaming_leaves_samples_that_cannot_fill_a_row_to_the_last_batch() -> None:
    # 10 tokens per bin and samples of 6 tokens: no bin can reach 98%, so nothing streams, and the
    # step's last batch packs every sample, one per bin.
    batcher = _streaming_batcher(
        num_tokens_per_microbatch=10, max_context_length=10, num_groups=3, dp_degree=2
    )
    groups = [[_SizedSample(6) for _ in range(3)] for _ in range(3)]
    streamed_rows, last_rows = _record_streamed_step(batcher, groups)
    assert streamed_rows == []
    assert len(last_rows) == 5
    assert sorted(len(bin_) for row in last_rows for bin_ in row) == [0] + [1] * 9


def test_streaming_sends_full_rows_and_keeps_the_rest_unsent() -> None:
    batcher = _streaming_batcher(
        num_tokens_per_microbatch=10, max_context_length=10, num_groups=2, dp_degree=2
    )
    nine_a, nine_b, nine_c, one_a, one_b = (_SizedSample(n) for n in (9, 9, 9, 1, 1))
    streamed_rows, last_rows = _record_streamed_step(
        batcher, [[nine_a, nine_b, nine_c, one_a, one_b], [_SizedSample(1)]]
    )
    # The first group fills both bins ([9+1] [9+1]); the third 9 waits for the step's last batch.
    assert [
        sorted(s.num_tokens for s in bin_) for row in streamed_rows for bin_ in row
    ] == [
        [1, 9],
        [1, 9],
    ]
    assert sorted(s.num_tokens for row in last_rows for bin_ in row for s in bin_) == [
        1,
        9,
    ]


def test_streamed_bins_balance_document_attention() -> None:
    batcher = _streaming_batcher(
        num_tokens_per_microbatch=10, max_context_length=10, num_groups=1, dp_degree=2
    )
    batcher.num_tokens_to_pack = lambda sample: sample.num_tokens
    samples = [_SizedSample(n) for n in (9, 6, 5, 4, 3)]
    bins, unplaced = batcher._pack_streamed_bins(samples, num_bins=2)
    assert [[s.num_tokens for s in bin_] for bin_ in bins] == [[9], [6, 4]]
    assert [s.num_tokens for s in unplaced] == [5, 3]


def test_streaming_packs_only_the_microbatch_groups_the_trainer_is_missing() -> None:
    batcher = _streaming_batcher(
        num_tokens_per_microbatch=10, max_context_length=10, num_groups=2, dp_degree=2
    )
    assert batcher._stream_queued_microbatches == 3
    packed_bins = []

    def record_pack(*, streamed_bins=None):
        packed_bins.append(streamed_bins)
        return SimpleNamespace(ends_step=streamed_bins is None)

    batcher.num_tokens_to_pack = lambda sample: sample.num_tokens
    batcher._pack_one_training_batch = record_pack
    # Enough samples for 3 microbatch groups of 2 bins: 6 x (9 + 1).
    group = TrainingSampleGroup(
        group_id=0,
        training_samples=[_SizedSample(n) for n in [9] * 6 + [1] * 6],
        metrics=[],
    )
    # The trainer has its 3 queued: keep every sample unsent.
    assert batcher.add_training_samples(
        training_sample_group=group, num_queued_microbatch_groups=3
    ) == (None, True)
    assert batcher.pack_ready_microbatches(num_queued_microbatch_groups=4) is None
    assert len(batcher._unsent_training_samples) == 12
    # It finished one and has 2: pack 1 microbatch group (2 bins), keep the rest unsent.
    assert not batcher.pack_ready_microbatches(num_queued_microbatch_groups=2).ends_step
    assert [sorted(s.num_tokens for s in bin_) for bin_ in packed_bins[-1]] == [
        [1, 9],
        [1, 9],
    ]
    # Its queue is empty: pack the 2 groups left.
    batcher.pack_ready_microbatches(num_queued_microbatch_groups=0)
    assert len(packed_bins[-1]) == 4
    assert batcher._unsent_training_samples == []


def test_stream_queued_microbatches_must_be_positive() -> None:
    with pytest.raises(ValueError, match="stream_queued_microbatches must be >= 1"):
        Batcher.Config(stream_microbatches=True, stream_queued_microbatches=0)
