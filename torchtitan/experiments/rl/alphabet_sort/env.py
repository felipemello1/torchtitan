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
from torchtitan.experiments.rl.alphabet_sort.data import build_example, NAMES
from torchtitan.experiments.rl.alphabet_sort.grading import (
    aggregate_turn_scores,
    score_turn_similarity,
)
from torchtitan.experiments.rl.envs import EnvExample, EnvReset, EnvStep
from torchtitan.experiments.rl.types import RolloutStatus


class AlphabetSortEnv(Configurable):
    """Multi-turn cumulative name-sorting task."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 1337420
        min_turns: int = 3
        max_turns: int = 3
        min_names_per_turn: int = 1
        max_names_per_turn: int = 4
        similarity_power: int = 8
        power_per_turn: bool = False

        def __post_init__(self) -> None:
            if self.min_turns <= 0:
                raise ValueError(f"min_turns must be positive, got {self.min_turns}")
            if self.max_turns < self.min_turns:
                raise ValueError(
                    "max_turns must be greater than or equal to "
                    f"min_turns, got {self.max_turns} < {self.min_turns}"
                )
            if self.min_names_per_turn <= 0:
                raise ValueError(
                    "min_names_per_turn must be positive, "
                    f"got {self.min_names_per_turn}"
                )
            if self.max_names_per_turn < self.min_names_per_turn:
                raise ValueError(
                    "max_names_per_turn must be greater than or equal to "
                    "min_names_per_turn, got "
                    f"{self.max_names_per_turn} < {self.min_names_per_turn}"
                )
            if self.max_turns * self.max_names_per_turn > len(NAMES):
                raise ValueError(
                    "max_turns * max_names_per_turn must not exceed the "
                    f"{len(NAMES)} local names"
                )
            if self.similarity_power <= 0:
                raise ValueError(
                    "similarity_power must be positive, " f"got {self.similarity_power}"
                )

    def __init__(
        self,
        config: Config,
        *,
        example: EnvExample | None = None,
        step: int = 0,
        group_idx: int = 0,
    ):
        self._config = config
        if example is not None:
            step = example.step
            group_idx = example.group_idx
        self._example = build_example(
            seed=config.seed,
            step=step,
            group_idx=group_idx,
            min_turns=config.min_turns,
            max_turns=config.max_turns,
            min_names_per_turn=config.min_names_per_turn,
            max_names_per_turn=config.max_names_per_turn,
        )
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
                similarity_power=self._config.similarity_power,
                power_per_turn=self._config.power_per_turn,
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
