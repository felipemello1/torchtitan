# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AlphabetSort multiturn environment."""

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.config import Configurable
from torchtitan.experiments.rl.alphabet_sort.data import AlphabetSortExample
from torchtitan.experiments.rl.alphabet_sort.grading import (
    aggregate_turn_scores,
    score_turn_similarity,
)
from torchtitan.experiments.rl.envs import EnvExample, EnvReset, EnvStep
from torchtitan.experiments.rl.types import RolloutStatus


class AlphabetSortBuilder(Configurable):
    """Builds single-use :class:`AlphabetSortEnv` instances from dataset rows."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        similarity_power: int = 8
        power_per_turn: bool = False

        def __post_init__(self) -> None:
            if self.similarity_power <= 0:
                raise ValueError(
                    "similarity_power must be positive, " f"got {self.similarity_power}"
                )

    def __init__(self, config: Config):
        self.config = config

    def build(self, *, example: EnvExample) -> "AlphabetSortEnv":
        return AlphabetSortEnv(
            episode=_episode_from_payload(example),
            similarity_power=self.config.similarity_power,
            power_per_turn=self.config.power_per_turn,
        )


def _episode_from_payload(example: EnvExample) -> AlphabetSortExample:
    payload = example.payload
    initial_prompt = payload.get("initial_prompt")
    follow_ups = payload.get("follow_ups")
    ground_truths = payload.get("ground_truths")
    turn_names = payload.get("turn_names")
    sort_by_first = payload.get("sort_by_first")

    if not isinstance(initial_prompt, str):
        raise ValueError("AlphabetSort payload must contain string 'initial_prompt'")
    if not (
        isinstance(follow_ups, list)
        and all(isinstance(item, str) for item in follow_ups)
    ):
        raise ValueError("AlphabetSort payload must contain list[str] 'follow_ups'")
    if not _is_list_of_string_lists(ground_truths):
        raise ValueError(
            "AlphabetSort payload must contain list[list[str]] 'ground_truths'"
        )
    if not _is_list_of_string_lists(turn_names):
        raise ValueError(
            "AlphabetSort payload must contain list[list[str]] 'turn_names'"
        )
    if not isinstance(sort_by_first, bool):
        raise ValueError("AlphabetSort payload must contain bool 'sort_by_first'")

    return AlphabetSortExample(
        initial_prompt=initial_prompt,
        follow_ups=list(follow_ups),
        ground_truths=[list(row) for row in ground_truths],
        turn_names=[list(row) for row in turn_names],
        sort_by_first=sort_by_first,
    )


def _is_list_of_string_lists(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(row, list) and all(isinstance(item, str) for item in row)
        for row in value
    )


class AlphabetSortEnv:
    """Multi-turn cumulative name-sorting task."""

    def __init__(
        self,
        *,
        episode: AlphabetSortExample,
        similarity_power: int,
        power_per_turn: bool,
    ):
        self._example = episode
        self._similarity_power = similarity_power
        self._power_per_turn = power_per_turn
        self._turn_idx = 0
        self._turn_similarities: list[float] = []

    async def reset(self) -> EnvReset:
        return EnvReset(
            messages=[{"role": "user", "content": self._example.initial_prompt}],
        )

    async def step(self, assistant_message: Message) -> EnvStep:
        completion = str(assistant_message.get("content") or "")
        similarity = score_turn_similarity(
            completion,
            expected=self._example.ground_truths[self._turn_idx],
            turn_idx=self._turn_idx,
        )
        self._turn_similarities.append(similarity)
        self._turn_idx += 1

        if self._turn_idx >= self._example.num_turns:
            mean_similarity = sum(self._turn_similarities) / len(
                self._turn_similarities
            )
            reward = aggregate_turn_scores(
                self._turn_similarities,
                similarity_power=self._similarity_power,
                power_per_turn=self._power_per_turn,
            )
            return EnvStep(
                reward=reward,
                reward_components={
                    "alphabet_sort": reward,
                    "mean_turn_similarity": mean_similarity,
                    "last_turn_similarity": similarity,
                },
                done=True,
                status=RolloutStatus.COMPLETED,
            )

        return EnvStep(
            messages=[
                {
                    "role": "user",
                    "content": self._example.follow_ups[self._turn_idx - 1],
                }
            ],
            reward=None,
            done=False,
        )

    async def close(self) -> None:
        return None
