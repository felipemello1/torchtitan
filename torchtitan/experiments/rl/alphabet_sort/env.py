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
from torchtitan.experiments.rl.alphabet_sort.data import build_example
from torchtitan.experiments.rl.alphabet_sort.grading import score_completion
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

    def __init__(
        self,
        config: Config,
        *,
        example: EnvExample | None = None,
        step: int = 0,
        group_idx: int = 0,
        sample_idx: int = 0,
    ):
        self._config = config
        self._sample_idx = sample_idx
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
        self._turn_scores: list[float] = []

    async def reset(self) -> EnvReset:
        return EnvReset(
            messages=[{"role": "user", "content": self._example.initial_prompt}],
            metadata={
                "sample_idx": self._sample_idx,
                "num_turns": self._example.num_turns,
                "sort_by": "first" if self._example.sort_by_first else "last",
            },
        )

    async def step(self, assistant_message: Message) -> EnvStep:
        completion = str(assistant_message.get("content") or "")
        score = score_completion(
            completion,
            expected=self._example.ground_truths[self._turn_idx],
            turn_idx=self._turn_idx,
            similarity_power=self._config.similarity_power,
        )
        self._turn_scores.append(score)
        self._turn_idx += 1

        if self._turn_idx >= self._example.num_turns:
            reward = sum(self._turn_scores) / len(self._turn_scores)
            return EnvStep(
                reward=reward,
                reward_components={
                    "alphabet_sort": reward,
                    "last_turn_similarity": score,
                },
                done=True,
                status=RolloutStatus.COMPLETED,
                metrics={"num_turns": float(self._example.num_turns)},
            )

        return EnvStep(
            messages=[
                {
                    "role": "user",
                    "content": self._example.follow_ups[self._turn_idx - 1],
                }
            ],
            reward=None,
            reward_components={"turn_similarity": score},
            metrics={"turn_similarity": score},
            done=False,
        )

    async def close(self) -> None:
        return None
