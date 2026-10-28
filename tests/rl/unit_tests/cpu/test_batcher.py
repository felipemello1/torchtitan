# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for fixed-bin LPT packing and its FFD fallback."""

import asyncio
import copy
import random
import weakref
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


def _make_samples(
    lengths: list[int], first_version: int | None = None
) -> list[TrainingSample]:
    """Samples of `lengths`; min_policy_version is 0, or first_version + index when given."""
    return [
        TrainingSample(
            min_policy_version=0 if first_version is None else first_version + index,
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

    temperature = microbatch.loss_kwargs()["temperature"]
    assert temperature.dtype == torch.float32
    assert temperature.tolist() == [0.5] * 8


def test_routed_expert_ids_follow_input_ids_with_zero_padding() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([2, 3])
    for sample, first_id in zip(samples, (10, 20), strict=True):
        # One row per input token; uint16, as vLLM returns above 256 experts.
        rows = first_id + np.arange(len(sample.token_ids) - 1, dtype=np.uint16)
        sample.routed_expert_ids = rows.reshape(-1, 1, 1)

    microbatch = batcher._pack_training_samples(samples)

    routed_expert_ids = microbatch.model_kwargs["routed_expert_ids"]
    # Torch can't unpickle uint16 tensors, so the batcher ships int16.
    assert routed_expert_ids.dtype == torch.int16
    assert routed_expert_ids.flatten().tolist() == [10, 11, 20, 21, 22, 0, 0, 0]


def test_microbatch_ships_topk_rows_for_loss_tokens_only() -> None:
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
        num_topk_logprobs=2,
    )
    first, second = _make_samples([3, 3])
    # Prompt, completion, env reply, completion: two loss tokens, so two top-k rows.
    first.loss_mask = np.array([False, True, False, True])
    first.topk_token_ids = np.array([[5, 6], [9, 4]], dtype=np.int32)
    first.topk_logprobs = np.array([[-0.1, -2.0], [-0.3, -4.0]], dtype=np.float32)
    second.loss_mask = np.array([False, False, True, True])
    second.topk_token_ids = np.array([[1, 2], [3, 7]], dtype=np.int32)
    second.topk_logprobs = np.array([[-0.5, -1.0], [-0.6, -1.1]], dtype=np.float32)

    microbatch = batcher._pack_training_samples([first, second])
    loss_kwargs = microbatch.to_loss_kwargs("cpu")

    assert microbatch.generator_topk_token_ids.shape == (4, 2)
    # Shifted like generator_logprobs: row t holds the top-k that sampled labels[t].
    assert loss_kwargs["generator_topk_token_ids"].tolist() == (
        [[5, 6], [0, 0], [9, 4]] + [[0, 0], [1, 2], [3, 7]] + [[0, 0]] * 2
    )
    assert torch.equal(
        loss_kwargs["generator_topk_logprobs"],
        torch.tensor(
            [[-0.1, -2.0], [0.0, 0.0], [-0.3, -4.0]]
            + [[0.0, 0.0], [-0.5, -1.0], [-0.6, -1.1]]
            + [[0.0, 0.0]] * 2
        ),
    )
    # An empty bin ships no rows but still gives the loss [T, k], so every rank calls it alike.
    empty_microbatch = batcher._pack_training_samples([])
    assert empty_microbatch.generator_topk_token_ids.shape == (0, 2)
    empty_loss_kwargs = empty_microbatch.to_loss_kwargs("cpu")
    assert empty_loss_kwargs["generator_topk_token_ids"].tolist() == [[0, 0]] * 8
    # A wrong row count shifts later rows, so packing checks each sample: 3 + 1 rows for
    # 2 + 2 loss tokens has the right total but is refused.
    first.topk_token_ids = np.array([[5, 6], [9, 4], [8, 8]], dtype=np.int32)
    first.topk_logprobs = np.zeros((3, 2), dtype=np.float32)
    second.topk_token_ids = second.topk_token_ids[:1]
    second.topk_logprobs = second.topk_logprobs[:1]
    with pytest.raises(ValueError, match="3 top-k rows for 2 loss tokens"):
        batcher._pack_training_samples([first, second])


def test_microbatch_has_no_topk_kwargs_by_default() -> None:
    # DAPOLoss and GRPOLoss take no top-k arguments.
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    loss_kwargs = batcher._pack_training_samples(_make_samples([3])).loss_kwargs()
    assert "generator_topk_token_ids" not in loss_kwargs


def test_token_budget_need_not_be_a_multiple_of_the_context() -> None:
    """A 2.5-context budget packs whole samples; padding positions wrap at the context."""
    from torchtitan.models.common.attention.attention import (
        create_varlen_metadata_for_document,
    )

    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=25,
        max_context_length=10,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
        temperature=1.0,
    )
    samples = _make_samples([10, 10, 4])

    assignments = batcher._assign_training_samples_to_microbatches(samples)
    microbatch = batcher._pack_training_samples(samples)
    metadata = create_varlen_metadata_for_document(
        microbatch.positions, padding_mask=microbatch.padding_mask
    )

    assert [len(rank_samples) for row in assignments for rank_samples in row] == [3]
    assert microbatch.input.shape == (25,)
    assert microbatch.positions.tolist()[20:] == [0, 1, 2, 3, 0]
    assert microbatch.padding_mask.tolist()[24] is True
    assert metadata.cu_seq_q.tolist() == [0, 10, 20, 24, 25]


def test_token_budget_must_fit_a_full_context_sample() -> None:
    with pytest.raises(ValueError, match="must be at least max_context_length"):
        Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=9,
            max_context_length=10,
            num_prompts_per_train_step=1,
            dp_degree=1,
            pad_id=0,
            temperature=1.0,
        )


def _two_rank_batcher() -> Batcher:
    return Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=20,
        max_context_length=10,
        num_prompts_per_train_step=2,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
    )


_GROUP_LENGTHS = [[9, 5, 3, 7], [8, 6, 2, 2, 9]]


def _group_samples(distinct_versions: bool) -> list[list[TrainingSample]]:
    """`_GROUP_LENGTHS` as samples; with distinct_versions, min_policy_version counts up across groups."""
    groups, first_version = [], 0
    for lengths in _GROUP_LENGTHS:
        groups.append(
            _make_samples(lengths, first_version if distinct_versions else None)
        )
        first_version += len(lengths)
    return groups


def _queue_groups(
    batcher: Batcher, distinct_versions: bool = False
) -> list[weakref.ref]:
    """Queue `_GROUP_LENGTHS` as groups held only by the batcher; return weakrefs to each sample's tokens."""
    token_refs = []
    for group_id, samples in enumerate(_group_samples(distinct_versions)):
        token_refs += [weakref.ref(sample.token_ids) for sample in samples]
        batcher._groups_for_next_batch.append(
            TrainingSampleGroup(group_id=group_id, training_samples=samples, metrics=[])
        )
    return token_refs


def test_packed_batch_equals_packing_each_assigned_microbatch() -> None:
    batcher = _two_rank_batcher()
    samples = [s for group in _group_samples(distinct_versions=True) for s in group]
    sample_copies = copy.deepcopy(samples)
    assignments = batcher._assign_training_samples_to_microbatches(sample_copies)
    expected = [
        [batcher._pack_training_samples(rank_samples) for rank_samples in row]
        for row in assignments
    ]
    expected_packing_metrics = batcher._packing_metrics(
        len(assignments), sample_copies, len(_GROUP_LENGTHS), 0
    )
    expected_microbatches = [mb for row in expected for mb in row]
    expected_loss_counts = sum(mb.loss_token_counts for mb in expected_microbatches)
    expected_routing_counts = sum(
        mb.routing_token_counts for mb in expected_microbatches
    )
    num_response_tokens = sum(int(mb.loss_mask.sum()) for mb in expected_microbatches)
    expected_nan_frac = (
        num_response_tokens - int(expected_loss_counts.reshape(-1)[0])
    ) / max(num_response_tokens, 1)
    _queue_groups(batcher, distinct_versions=True)

    batch = batcher._pack_one_training_batch()

    packed = [row for group in batch.microbatch_groups for row in group]
    assert len(packed) == len(expected) > 1
    for row, expected_row in zip(packed, expected):
        for microbatch, expected_microbatch in zip(row, expected_row):
            for key, value in expected_microbatch.as_input_dict().items():
                if isinstance(value, dict):
                    for loss_key, loss_value in value.items():
                        assert torch.equal(
                            microbatch.as_input_dict()[key][loss_key], loss_value
                        )
                else:
                    assert torch.equal(microbatch.as_input_dict()[key], value)
    assert torch.equal(batch.global_loss_token_counts, expected_loss_counts)
    assert torch.equal(batch.global_routing_token_counts, expected_routing_counts)
    # Distinct versions per sample, so a change in order would show here.
    assert batch.min_policy_versions == [s.min_policy_version for s in samples]
    # Every metric, in order: the groups' own (none here), the NaN fraction, then packing.
    assert [(metric.key, metric.value.value) for metric in batch.metrics] == [
        ("loss/generator_logprob_nan_frac", expected_nan_frac),
        *[(metric.key, metric.value.value) for metric in expected_packing_metrics],
    ]


def test_packing_releases_each_microbatchs_samples_once_packed() -> None:
    """The batcher never holds a packed microbatch and its samples at once."""
    batcher = _two_rank_batcher()
    token_refs = _queue_groups(batcher)
    pack_training_samples = batcher._pack_training_samples
    num_alive_before_each_pack: list[int] = []
    num_samples_per_pack: list[int] = []

    def counting_pack(training_samples: list[TrainingSample]):
        num_alive_before_each_pack.append(sum(ref() is not None for ref in token_refs))
        num_samples_per_pack.append(len(training_samples))
        return pack_training_samples(training_samples)

    batcher._pack_training_samples = counting_pack
    batch = batcher._pack_one_training_batch()

    num_samples = len(token_refs)
    num_packed_before = np.cumsum([0, *num_samples_per_pack[:-1]]).tolist()
    assert len(num_samples_per_pack) > 2
    # Before each microbatch is packed, every earlier microbatch's samples are gone.
    assert num_alive_before_each_pack == [num_samples - n for n in num_packed_before]
    assert all(ref() is None for ref in token_refs)
    assert batch.min_policy_versions == [0] * num_samples


def _replay_groups() -> list[TrainingSampleGroup]:
    """Three groups whose samples carry router-replay rows equal to their input tokens, and
    top-k rows equal to their labels; every input token is distinct across samples."""
    groups = []
    # The first group fills a full microbatch group ([9+1] [7+3]), so it streams.
    for group_id, lengths in enumerate([[9, 5, 3, 7, 1], [8, 6, 2], [4, 9, 1, 6]]):
        samples = _make_samples(lengths)
        for index, sample in enumerate(samples):
            first_token = 100 * (5 * group_id + index) + 1
            sample.token_ids = first_token + np.arange(len(sample.token_ids))
            rows = sample.token_ids[:-1].astype(np.uint16)
            sample.routed_expert_ids = rows.reshape(-1, 1, 1)
            sample.topk_token_ids = sample.token_ids[1:, None].astype(np.int32)
            sample.topk_logprobs = np.zeros(sample.topk_token_ids.shape, np.float32)
        groups.append(
            TrainingSampleGroup(group_id=group_id, training_samples=samples, metrics=[])
        )
    return groups


def _replay_batcher(
    *, stream_microbatches: bool, register_rdma_microbatch=None, num_topk_logprobs=0
):
    return Batcher.Config(
        stream_microbatches=stream_microbatches,
        rdma_min_tensor_bytes=None if register_rdma_microbatch is None else 1,
    ).build(
        num_tokens_per_microbatch_per_dp_rank=10,
        max_context_length=10,
        num_prompts_per_train_step=3,
        dp_degree=2,
        pad_id=0,
        temperature=1.0,
        register_rdma_microbatch=register_rdma_microbatch,
        replay_routed_experts=True,
        num_topk_logprobs=num_topk_logprobs,
    )


def _add_groups(batcher: Batcher, groups: list[TrainingSampleGroup]) -> list:
    batches = []
    for group in groups:
        batch, _ = batcher.add_training_samples(training_sample_group=group)
        if batch is not None:
            batches.append(batch)
    return batches


def test_streamed_microbatches_carry_each_tokens_router_replay_row() -> None:
    """Router replay with streamed microbatches: each early batch's rows follow its own tokens,
    and the step's batches together send every token's row once, as the whole step does."""
    streamed = _add_groups(_replay_batcher(stream_microbatches=True), _replay_groups())
    [whole_step] = _add_groups(
        _replay_batcher(stream_microbatches=False), _replay_groups()
    )
    assert len(streamed) > 1 and not streamed[0].ends_step

    def token_rows(batches) -> list[int]:
        rows = []
        for batch in batches:
            for group in batch.microbatch_groups:
                for row in group:
                    for microbatch in row:
                        routed = microbatch.model_kwargs["routed_expert_ids"].flatten()
                        real = ~microbatch.padding_mask
                        assert torch.equal(routed[real], microbatch.input[real].short())
                        assert not routed[~real].any()
                        rows += routed[real].tolist()
        return rows

    streamed_rows = token_rows(streamed)
    assert len(streamed_rows) == len(set(streamed_rows))
    assert sorted(streamed_rows) == sorted(token_rows([whole_step]))


@pytest.mark.parametrize("stream_microbatches", [True, False])
def test_microbatches_carry_each_loss_tokens_topk_row(stream_microbatches) -> None:
    """Score centering's top-k rows land on their own loss tokens, streamed or not."""
    batcher = _replay_batcher(
        stream_microbatches=stream_microbatches, num_topk_logprobs=1
    )
    for batch in _add_groups(batcher, _replay_groups()):
        for group in batch.microbatch_groups:
            for row in group:
                for microbatch in row:
                    loss_kwargs = microbatch.to_loss_kwargs("cpu")
                    topk = loss_kwargs["generator_topk_token_ids"][:, 0]
                    loss_mask = loss_kwargs["loss_mask"]
                    assert torch.equal(
                        topk[loss_mask], microbatch.labels[loss_mask].int()
                    )
                    assert not topk[~loss_mask].any()


def test_streamed_batches_register_rdma_microbatches_per_batch() -> None:
    """Each streamed batch registers its own RDMA batch: batch indices count the batches in
    training order (what the trainer checks), microbatch indices restart per batch, and releases
    drop the oldest batch first."""
    registered = []
    dropped = []

    class _FakeRDMAMicrobatch:
        def __init__(self, microbatch, *, batch_index, dp_rank, microbatch_index):
            self.identity = (batch_index, dp_rank, microbatch_index)

        async def drop(self) -> None:
            dropped.append(self.identity)

    def register(microbatch, *, min_bytes, batch_index, dp_rank, microbatch_index):
        registered.append((batch_index, dp_rank, microbatch_index))
        return _FakeRDMAMicrobatch(
            microbatch,
            batch_index=batch_index,
            dp_rank=dp_rank,
            microbatch_index=microbatch_index,
        )

    batcher = _replay_batcher(
        stream_microbatches=True, register_rdma_microbatch=register
    )
    batches = _add_groups(batcher, _replay_groups())
    assert len(batches) > 1
    expected = []
    for batch_index, batch in enumerate(batches):
        rows = [row for group in batch.microbatch_groups for row in group]
        expected += [
            (batch_index, dp_rank, microbatch_index)
            for microbatch_index, row in enumerate(rows)
            for dp_rank in range(len(row))
        ]
        assert [[microbatch.identity for microbatch in row] for row in rows] == [
            [(batch_index, dp_rank, index) for dp_rank in range(2)]
            for index in range(len(rows))
        ]
    assert registered == expected

    for _ in batches:
        asyncio.run(batcher.maybe_release_oldest_batch())
    assert dropped == expected


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
    correlated within a group): each streamed row's ranks get about the same attention work, every
    streamed bin is at least 98% full, and the step needs about as many rows as packing it at once.

    Before streamed rows were built this way, each row repacked FFD's longest unsent samples into
    exactly 16 bins, which rarely fit; the FFD fallback averaged 1.47x max/mean attention work on
    these lengths (1.09x now).
    """
    rng = np.random.default_rng(0)
    num_tokens_per_microbatch, max_context_length, dp_degree = 163_840, 65_536, 16
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
    assert sum(imbalances) / len(imbalances) < 1.12
    assert max(imbalances) < 1.3


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


def test_streaming_sends_bins_at_their_document_limit() -> None:
    # Two documents per bin: four 1-token samples fill both bins' document limits, though not their
    # tokens, so they stream.
    batcher = _streaming_batcher(
        num_tokens_per_microbatch=10,
        max_context_length=10,
        num_groups=2,
        dp_degree=2,
        max_num_documents=2,
    )
    streamed_rows, last_rows = _record_streamed_step(
        batcher, [[_SizedSample(1) for _ in range(4)], [_SizedSample(5)]]
    )
    assert [[len(bin_) for bin_ in row] for row in streamed_rows] == [[2, 2]]
    assert sorted(s.num_tokens for row in last_rows for bin_ in row for s in bin_) == [
        5
    ]
