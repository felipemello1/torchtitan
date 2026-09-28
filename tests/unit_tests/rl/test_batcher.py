# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for fixed-bin LPT packing and its FFD fallback."""

import random

import pytest

from torchtitan.rl.components.batcher import Batcher
from torchtitan.rl.types import RolloutTurnID, TrainingSample


def _make_samples(lengths: list[int]) -> list[TrainingSample]:
    return [
        TrainingSample(
            min_policy_version=0,
            max_policy_version=0,
            rollout_id=RolloutTurnID(group_id=0, rollout_id=index, turn_id=0),
            token_ids=list(range(length + 1)),
            loss_mask=[False] + [True] * length,
            logprobs=[0.0] * (length + 1),
            advantage=[0.0] * (length + 1),
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
    )
    samples = _make_samples([3] * 8 + [2] * 9)
    chunks = [[sample] for sample in samples]
    assert batcher._pack_with_lpt(chunks, target_num_bins=7) is None

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


def _make_prompt_group(
    group_id: int, prompt: list[int], completions: list[list[int]]
) -> list[TrainingSample]:
    return [
        TrainingSample(
            min_policy_version=0,
            max_policy_version=0,
            rollout_id=RolloutTurnID(group_id=group_id, rollout_id=index, turn_id=0),
            token_ids=prompt + completion,
            loss_mask=[False] * len(prompt) + [True] * len(completion),
            logprobs=[0.0] * len(prompt) + [-0.5 - index] * len(completion),
            advantage=[0.0] * len(prompt) + [float(index + 1)] * len(completion),
        )
        for index, completion in enumerate(completions)
    ]


def _trained_tokens(microbatch) -> list[tuple]:
    """(input, label, position, logprob, advantage) of every trained token."""
    return [
        (
            int(microbatch.input[index]),
            int(microbatch.labels[index]),
            int(microbatch.positions[index]),
            float(microbatch.generator_logprobs[index]),
            float(microbatch.advantages[index]),
        )
        for index in microbatch.loss_mask.nonzero().flatten().tolist()
    ]


def test_share_prompt_packs_prompt_once_and_duplicates_its_last_token() -> None:
    batcher = Batcher.Config(share_prompt=True).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
    )
    samples = _make_prompt_group(0, [10, 11, 12], [[20, 21], [30, 31]])

    microbatch = batcher._pack_training_samples(samples)

    assert microbatch.input.tolist() == [10, 11, 12, 20, 12, 30, 0, 0]
    assert microbatch.labels.tolist() == [11, 12, 20, 21, 30, 31, 0, 0]
    assert microbatch.positions.tolist() == [0, 1, 2, 3, 2, 3, 0, 1]
    assert microbatch.loss_mask.tolist() == [False, False] + [True] * 4 + [False] * 2
    assert microbatch.padding_mask.tolist() == [False] * 6 + [True] * 2


def test_share_prompt_keeps_every_trained_token_of_the_unshared_layout() -> None:
    rng = random.Random(0)
    samples = []
    for group_id in range(6):
        prompt = [rng.randrange(1, 1000) for _ in range(rng.randint(2, 12))]
        completions = [
            [rng.randrange(1, 1000) for _ in range(rng.randint(1, 10))]
            for _ in range(rng.randint(1, 5))
        ]
        samples += _make_prompt_group(group_id, prompt, completions)

    microbatches = {}
    for share_prompt in (False, True):
        batcher = Batcher.Config(share_prompt=share_prompt).build(
            num_tokens_per_microbatch_per_dp_rank=48,
            max_context_length=24,
            num_prompts_per_train_step=1,
            dp_degree=2,
            pad_id=0,
        )
        assignments = batcher._assign_training_samples_to_microbatches(samples)
        microbatches[share_prompt] = [
            batcher._pack_training_samples(rank_samples)
            for row in assignments
            for rank_samples in row
        ]

    # Same trained tokens, inputs, labels, and positions; fewer packed tokens.
    assert sorted(
        token for mb in microbatches[True] for token in _trained_tokens(mb)
    ) == sorted(token for mb in microbatches[False] for token in _trained_tokens(mb))
    num_real_tokens = {
        share_prompt: sum(int((~mb.padding_mask).sum()) for mb in mbs)
        for share_prompt, mbs in microbatches.items()
    }
    assert num_real_tokens[True] < num_real_tokens[False]


def test_share_prompt_splits_a_group_into_chunks_that_fit_a_microbatch() -> None:
    batcher = Batcher.Config(share_prompt=True).build(
        num_tokens_per_microbatch_per_dp_rank=16,
        max_context_length=16,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
    )
    # 6-token prompt: 5 shared tokens; 11 tokens left per microbatch.
    samples = _make_prompt_group(
        0, list(range(1, 7)), [[7] * 8, [8] * 6, [9] * 4, [10] * 1]
    )

    chunks = batcher._split_into_chunks(samples)

    assert [[len(s.token_ids) - 6 for s in chunk] for chunk in chunks] == [
        [8, 1],
        [6, 4],
    ]
    assert [batcher._num_tokens_to_pack_chunk(chunk) for chunk in chunks] == [14, 15]


def test_share_prompt_only_shares_identical_prompts() -> None:
    batcher = Batcher.Config(share_prompt=True).build(
        num_tokens_per_microbatch_per_dp_rank=32,
        max_context_length=32,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
    )
    # Same group, but the second prompt edited history (a multi-turn branch).
    samples = _make_prompt_group(0, [10, 11, 12], [[20, 21]]) + _make_prompt_group(
        0, [10, 11, 13], [[30, 31]]
    )

    microbatch = batcher._pack_training_samples(samples)

    assert microbatch.input.tolist()[:8] == [10, 11, 12, 20, 10, 11, 13, 30]
    assert microbatch.positions.tolist()[:8] == [0, 1, 2, 3, 0, 1, 2, 3]


def test_share_prompt_packs_samples_without_a_prompt_unshared() -> None:
    batcher = Batcher.Config(share_prompt=True).build(
        num_tokens_per_microbatch_per_dp_rank=8,
        max_context_length=8,
        num_prompts_per_train_step=1,
        dp_degree=1,
        pad_id=0,
    )
    samples = _make_prompt_group(0, [], [[20, 21, 22], [30, 31]])

    microbatch = batcher._pack_training_samples(samples)

    assert microbatch.input.tolist() == [20, 21, 30, 0, 0, 0, 0, 0]
    assert microbatch.positions.tolist() == [0, 1, 0, 0, 1, 2, 3, 4]


def test_share_prompt_rejects_fixed_shape_metadata() -> None:
    with pytest.raises(ValueError, match="share_prompt"):
        Batcher.Config(share_prompt=True, max_num_documents=4)
    with pytest.raises(ValueError, match="share_prompt"):
        Batcher.Config(share_prompt=True, per_sample_pad_multiple=8)
