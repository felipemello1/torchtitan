# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.rollouts.types import DatasetOutput, Rollout, RolloutTurn
from torchtitan.experiments.rl.tasks.alphabet_sort import (
    AlphabetSortDataset,
    AlphabetSortInput,
    AlphabetSortReward,
    AlphabetSortTask,
    data as alphabet_data,
)
from torchtitan.experiments.rl.tasks.alphabet_sort.env import AlphabetSortEnv
from torchtitan.experiments.rl.tasks.alphabet_sort.grader import score_sorted_list


_NAMES = (
    "AliceZephyr",
    "BobYang",
    "CarolXu",
    "DanWalsh",
    "EveVance",
    "FrankStone",
    "GraceReyes",
    "HeidiQuinn",
)


def _patch_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alphabet_data, "_load_names", lambda *a, **k: _NAMES)


def _assistant_turn(content: str) -> RolloutTurn:
    return RolloutTurn(
        prompt_token_ids=[],
        response_token_ids=[],
        response_logprobs=[],
        policy_version=0,
        assistant_message={"role": "assistant", "content": content},
    )


# Dataset


def test_dataset_rejects_invalid_num_turns() -> None:
    with pytest.raises(ValueError, match="num_turns must be >= 1"):
        AlphabetSortDataset.Config(num_turns=0)


def test_dataset_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_names(monkeypatch)
    a = AlphabetSortDataset(AlphabetSortDataset.Config(seed=7))
    b = AlphabetSortDataset(AlphabetSortDataset.Config(seed=7))
    assert next(a) == next(b)


def test_state_dict_resumes_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_names(monkeypatch)
    dataset = AlphabetSortDataset(AlphabetSortDataset.Config(seed=3))
    next(dataset)  # advance past a couple of episodes
    next(dataset)
    checkpoint = dataset.state_dict()
    expected_after = [next(dataset), next(dataset)]

    resumed = AlphabetSortDataset(AlphabetSortDataset.Config(seed=3))
    resumed.load_state_dict(checkpoint)
    assert [next(resumed), next(resumed)] == expected_after


def test_one_turn_sorts_by_first_or_last(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_names(monkeypatch)
    dataset = AlphabetSortDataset(AlphabetSortDataset.Config(seed=0))
    seen_first = seen_last = False
    for _ in range(50):
        episode = next(dataset).env_input
        assert len(episode.expected_lines) == 1
        part = 0 if episode.sort_by_first else 1
        expected = sorted(
            episode.presented_names[0], key=lambda n: alphabet_data._split_name(n)[part]
        )
        assert list(episode.expected_lines[0]) == expected
        seen_first |= episode.sort_by_first
        seen_last |= not episode.sort_by_first
    assert seen_first and seen_last


def test_multiturn_targets_are_cumulative_and_tagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_names(monkeypatch)
    dataset = AlphabetSortDataset(
        AlphabetSortDataset.Config(
            seed=11, num_turns=3, min_names_per_turn=2, max_names_per_turn=2
        )
    )
    episode = next(dataset).env_input

    # The cumulative answer grows by the names introduced each turn.
    assert [len(lines) for lines in episode.expected_lines] == [2, 4, 6]
    # Turn 0 has no "// new name!" tags; later turns tag exactly the names new that turn.
    assert all("// new name!" not in line for line in episode.expected_lines[0])
    assert sum("// new name!" in line for line in episode.expected_lines[1]) == 2


def test_split_name_handles_multi_part_names() -> None:
    assert alphabet_data._split_name("MarcChardin") == ("marc", "chardin")
    assert alphabet_data._split_name("VanDerBerg") == ("van", "derberg")
    assert alphabet_data._split_name("Plato") == ("plato", "")


# Grader


def test_score_exact_swapped_missing_and_garbage() -> None:
    correct = ("AliceZephyr", "BobYang", "CarolXu")
    exact = (
        "<alphabetical_sorted>\nAliceZephyr\nBobYang\nCarolXu\n</alphabetical_sorted>"
    )
    swapped = (
        "<alphabetical_sorted>\nAliceZephyr\nCarolXu\nBobYang\n</alphabetical_sorted>"
    )
    garbage = "<alphabetical_sorted>\nName1\nName2\n</alphabetical_sorted>"

    def score(text: str) -> float:
        return score_sorted_list(
            text,
            correct_lines=correct,
            xml_tag="alphabetical_sorted",
            similarity_power=4,
        )

    assert score(exact) == 1.0
    assert score(exact) > score(swapped)
    assert score("no block here") == 0.0
    assert score(garbage) < 0.05  # echoed template names are nothing like the answer


def test_score_uses_the_last_block() -> None:
    correct = ("AliceZephyr", "BobYang")
    text = (
        "<alphabetical_sorted>\nBobYang\nAliceZephyr\n</alphabetical_sorted>\n"
        "<alphabetical_sorted>\nAliceZephyr\nBobYang\n</alphabetical_sorted>"
    )
    assert (
        score_sorted_list(
            text,
            correct_lines=correct,
            xml_tag="alphabetical_sorted",
            similarity_power=4,
        )
        == 1.0
    )


def test_followup_turns_use_the_combined_tag() -> None:
    correct = ("AliceZephyr", "BobYang", "CarolXu // new name!")
    text = (
        "<combined_alphabetical_sorted>\n"
        "AliceZephyr\nBobYang\nCarolXu // new name!\n"
        "</combined_alphabetical_sorted>"
    )
    # Read with the combined tag -> match; with the turn-0 tag -> block not found.
    assert (
        score_sorted_list(
            text,
            correct_lines=correct,
            xml_tag="combined_alphabetical_sorted",
            similarity_power=4,
        )
        == 1.0
    )
    assert (
        score_sorted_list(
            text,
            correct_lines=correct,
            xml_tag="alphabetical_sorted",
            similarity_power=4,
        )
        == 0.0
    )


def test_reward_averages_every_turn() -> None:
    rollout = Rollout(
        group_id="g",
        sample_idx=0,
        turns=[
            _assistant_turn(
                "<alphabetical_sorted>\nAliceZephyr\nBobYang\n</alphabetical_sorted>"
            ),
            _assistant_turn(
                "<combined_alphabetical_sorted>\n"
                "AliceZephyr\nBobYang\nCarolXu // new name!\n"
                "</combined_alphabetical_sorted>"
            ),
        ],
    )
    env_input = AlphabetSortInput(
        presented_names=(("BobYang", "AliceZephyr"), ("CarolXu",)),
        expected_lines=(
            ("AliceZephyr", "BobYang"),
            ("AliceZephyr", "BobYang", "CarolXu // new name!"),
        ),
        sort_by_first=True,
    )
    reward = AlphabetSortReward(similarity_power=4)
    assert asyncio.run(reward(rollout, env_input)) == pytest.approx(1.0)


# Env


def test_env_single_turn_finishes_after_first_answer() -> None:
    env = AlphabetSortEnv(
        env_input=AlphabetSortInput(
            presented_names=(("BobYang", "AliceZephyr"),),
            expected_lines=(("AliceZephyr", "BobYang"),),
            sort_by_first=True,
        )
    )
    reset = asyncio.run(env.reset())
    prompt = reset.messages[0]["content"]
    assert "BobYang" in prompt and "AliceZephyr" in prompt
    assert "<alphabetical_sorted>" in prompt

    step = asyncio.run(env.step_message({"role": "assistant", "content": "x"}))
    assert step.done is True


def test_env_walks_through_follow_up_turns() -> None:
    env = AlphabetSortEnv(
        env_input=AlphabetSortInput(
            presented_names=(("CarolXu", "AliceZephyr"), ("BobYang",)),
            expected_lines=(
                ("AliceZephyr", "CarolXu"),
                ("AliceZephyr", "BobYang // new name!", "CarolXu"),
            ),
            sort_by_first=True,
        )
    )
    assert "<alphabetical_sorted>" in asyncio.run(env.reset()).messages[0]["content"]

    follow_up = asyncio.run(env.step_message({"role": "assistant", "content": "x"}))
    assert follow_up.done is False
    second_prompt = follow_up.messages[0]["content"]
    assert "<combined_alphabetical_sorted>" in second_prompt
    assert "// new name!" in second_prompt and "BobYang" in second_prompt

    done = asyncio.run(env.step_message({"role": "assistant", "content": "y"}))
    assert done.done is True
    assert done.messages == []


# Task / recipe


def test_task_builds_one_env_per_group_member() -> None:
    task = AlphabetSortTask(AlphabetSortTask.Config())
    example = DatasetOutput(
        task_name="alphabet_sort",
        env_input=AlphabetSortInput(
            presented_names=(("BobYang",),),
            expected_lines=(("BobYang",),),
            sort_by_first=True,
        ),
    )
    envs = task.make_envs(example=example, group_size=3, renderer=None)
    assert len(envs) == 3


def test_alphabet_sort_recipe_shape() -> None:
    from torchtitan.experiments.rl.config_registry import (
        rl_grpo_qwen3_0_6b_alphabet_sort,
    )

    cfg = rl_grpo_qwen3_0_6b_alphabet_sort()
    assert cfg.num_steps == 20
    assert set(cfg.tasks) == {"alphabet_sort"}
    assert isinstance(cfg.train_dataset, AlphabetSortDataset.Config)
    assert isinstance(cfg.validation_dataset, AlphabetSortDataset.Config)
    # The recipe stays single-turn (what the controller trains today).
    assert cfg.train_dataset.num_turns == 1
    assert cfg.validation_dataset.num_turns == 1
