# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for AlphabetSort data, grading, env, and recipes."""

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.alphabet_sort import (
    AlphabetSortBuilder,
    AlphabetSortDataset,
    AlphabetSortExample,
    data as alphabet_data,
)
from torchtitan.experiments.rl.alphabet_sort.grading import (
    aggregate_turn_scores,
    score_turn_similarity,
)
from torchtitan.experiments.rl.types import RolloutStatus


def test_alphabet_sort_dataset_is_deterministic(monkeypatch) -> None:
    names = tuple(
        [
            "AliceZephyr",
            "BobYellow",
            "CarolXavier",
            "DinaWest",
            "EvanVale",
            "FayeUnderwood",
            "GusTeller",
            "HanaStone",
        ]
    )
    monkeypatch.setattr(alphabet_data, "_load_names", lambda: names)
    dataset = AlphabetSortDataset(
        AlphabetSortDataset.Config(
            seed=123,
            min_turns=3,
            max_turns=3,
            min_names_per_turn=2,
            max_names_per_turn=2,
        )
    )

    first = dataset.sample_group(sample_step=7, group_idx=2)
    second = dataset.sample_group(sample_step=7, group_idx=2)

    assert first == second
    assert first.group_id == "alphabet_sort/step=7/group=2"
    assert isinstance(first.payload, AlphabetSortExample)
    assert first.payload.num_turns == 3


def test_score_turn_similarity_orders_exact_partial_and_missing() -> None:
    expected = ["AliceZephyr", "BobYellow", "CarolXavier"]
    exact = """<alphabetical_sorted>
AliceZephyr
BobYellow
CarolXavier
</alphabetical_sorted>"""
    swapped = """<alphabetical_sorted>
AliceZephyr
CarolXavier
BobYellow
</alphabetical_sorted>"""
    missing = "AliceZephyr, BobYellow, CarolXavier"

    assert score_turn_similarity(exact, expected=expected, turn_idx=0) == 1.0
    assert score_turn_similarity(
        exact, expected=expected, turn_idx=0
    ) > score_turn_similarity(swapped, expected=expected, turn_idx=0)
    assert score_turn_similarity(missing, expected=expected, turn_idx=0) == 0.0


def test_aggregate_turn_scores_power_modes() -> None:
    similarities = [0.5, 1.0]

    assert aggregate_turn_scores(
        similarities, similarity_power=2, power_per_turn=False
    ) == pytest.approx(0.75**2)
    assert aggregate_turn_scores(
        similarities, similarity_power=2, power_per_turn=True
    ) == pytest.approx((0.5**2 + 1.0**2) / 2)
    assert aggregate_turn_scores([], similarity_power=2, power_per_turn=False) == 0.0


def test_alphabet_sort_env_progresses_to_final_reward() -> None:
    async def scenario():
        example = AlphabetSortExample(
            initial_prompt="sort AliceZephyr and BobYellow",
            follow_ups=("add CarolXavier",),
            ground_truths=(
                ("AliceZephyr", "BobYellow"),
                ("AliceZephyr", "BobYellow", "CarolXavier // new name!"),
            ),
            turn_names=(("AliceZephyr", "BobYellow"), ("CarolXavier",)),
            sort_by_first=True,
        )
        builder = AlphabetSortBuilder(
            AlphabetSortBuilder.Config(similarity_power=1, power_per_turn=False)
        )
        env = builder.build(
            example=alphabet_data.EnvExample(
                group_id="g0",
                sample_step=0,
                group_idx=0,
                payload=example,
            )
        )
        reset = await env.reset()
        first = await env.step(
            {
                "role": "assistant",
                "content": """<alphabetical_sorted>
AliceZephyr
BobYellow
</alphabetical_sorted>""",
            }
        )
        final = await env.step(
            {
                "role": "assistant",
                "content": """<combined_alphabetical_sorted>
AliceZephyr
BobYellow
CarolXavier // new name!
</combined_alphabetical_sorted>""",
            }
        )
        await env.close()
        return reset, first, final

    reset, first, final = asyncio.run(scenario())

    assert reset.messages[0]["content"] == "sort AliceZephyr and BobYellow"
    assert first.done is False
    assert first.messages[0]["content"] == "add CarolXavier"
    assert final.status == RolloutStatus.COMPLETED
    assert final.reward == pytest.approx(1.0)
    assert final.reward_components["alphabet_sort"] == pytest.approx(1.0)


def test_alphabet_sort_recipe_shapes() -> None:
    from torchtitan.experiments.rl.config_registry import (
        rl_dapo_qwen3_0_6b_alphabet_sort,
        rl_dapo_qwen3_1_7b_alphabet_sort_2gpu,
        rl_dapo_qwen3_1_7b_alphabet_sort_2gpu_acceptance,
        rl_dapo_qwen3_1_7b_alphabet_sort_3gpu_multigen,
    )

    small = rl_dapo_qwen3_0_6b_alphabet_sort()
    two_gpu = rl_dapo_qwen3_1_7b_alphabet_sort_2gpu()
    three_gpu_multigen = rl_dapo_qwen3_1_7b_alphabet_sort_3gpu_multigen()
    acceptance = rl_dapo_qwen3_1_7b_alphabet_sort_2gpu_acceptance()

    assert small.max_rollout_turns == 5
    assert small.train_dataset.min_turns == 3
    assert small.train_dataset.max_turns == 5
    assert small.batcher.batch.local_batch_size == 8
    assert small.batcher.batch.global_batch_size == 64
    assert two_gpu.batcher.batch.local_batch_size == 4
    assert two_gpu.batcher.batch.global_batch_size == 128
    assert three_gpu_multigen.num_generator_instances == 2
    assert three_gpu_multigen.generator.parallelism.tensor_parallel_degree == 1
    assert acceptance.batcher.batch.local_batch_size == 4
    assert acceptance.batcher.batch.global_batch_size == 256
