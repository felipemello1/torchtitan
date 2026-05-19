# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from renderers import Message

from torchtitan.config import Configurable
from torchtitan.experiments.rl.envs import EnvExample, EnvReset, EnvStep
from torchtitan.experiments.rl.types import RolloutStatus


SYSTEM_PROMPT = """\
You are a helpful assistant. Solve the problem step by step.
When you have your final answer, state it as [ANSWER] <number>.

Example:
User: What is the total digit sum of [12, 345, 67]?
Assistant: Break each number into digits:
12 -> 1, 2
345 -> 3, 4, 5
67 -> 6, 7
Sum all digits: 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
[ANSWER] 28"""


class SumDigitsDataset(Configurable):
    """Deterministic synthetic dataset for single-turn digit-sum tasks."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 42
        """Seed mixed with ``(step, group_idx)`` for deterministic rows."""

        min_values: int = 2
        """Minimum number of integers in one prompt."""

        max_values: int = 4
        """Maximum number of integers in one prompt."""

        min_value: int = 10
        """Minimum sampled integer, inclusive."""

        max_value: int = 99
        """Maximum sampled integer, inclusive."""

        def __post_init__(self) -> None:
            if self.min_values <= 0:
                raise ValueError(f"min_values must be positive, got {self.min_values}")
            if self.max_values < self.min_values:
                raise ValueError(
                    "max_values must be greater than or equal to min_values, "
                    f"got {self.max_values} < {self.min_values}"
                )
            if self.max_value < self.min_value:
                raise ValueError(
                    "max_value must be greater than or equal to min_value, "
                    f"got {self.max_value} < {self.min_value}"
                )

    def __init__(self, config: Config):
        self.config = config

    def sample_group(self, *, step: int, group_idx: int) -> EnvExample:
        cfg = self.config
        rng = random.Random(f"{cfg.seed}:{step}:{group_idx}")
        num_values = rng.randint(cfg.min_values, cfg.max_values)
        values = [rng.randint(cfg.min_value, cfg.max_value) for _ in range(num_values)]
        target = sum(int(digit) for value in values for digit in str(abs(value)))
        return EnvExample(
            group_id=f"sum_digits/step={step}/group={group_idx}",
            step=step,
            group_idx=group_idx,
            payload={"values": values, "target": target},
        )


class SumDigitsBuilder(Configurable):
    """Builds single-use :class:`SumDigitsEnv` instances from dataset rows."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        correctness_reward: float = 1.0
        """Reward for a response containing ``[ANSWER] <target>``."""

        format_reward: float = 0.3
        """Reward bonus for any ``[ANSWER] <number>`` tag in the response."""

        system_prompt: str = SYSTEM_PROMPT
        """System prompt rendered before each digit-sum question."""

    def __init__(self, config: Config):
        self.config = config

    def build(self, *, example: EnvExample) -> "SumDigitsEnv":
        values = example.payload.get("values")
        if not (
            isinstance(values, list) and all(isinstance(value, int) for value in values)
        ):
            raise ValueError(
                f"SumDigits example payload must contain list[int] 'values', "
                f"got {values!r}"
            )
        target = example.payload.get("target")
        if not isinstance(target, int):
            raise ValueError(
                f"SumDigits example payload must contain int 'target', got {target!r}"
            )
        return SumDigitsEnv(
            values=values,
            target=target,
            correctness_reward=self.config.correctness_reward,
            format_reward=self.config.format_reward,
            system_prompt=self.config.system_prompt,
        )


class SumDigitsEnv:
    """Single-turn, single-use sum-of-digits task."""

    def __init__(
        self,
        *,
        values: list[int],
        target: int,
        correctness_reward: float,
        format_reward: float,
        system_prompt: str,
    ):
        self._values = list(values)
        self._correctness_reward_value = correctness_reward
        self._format_reward_value = format_reward
        self._system_prompt = system_prompt
        self._target = target
        self._question = f"What is the total digit sum of {self._values}?"

    async def reset(self) -> EnvReset:
        return EnvReset(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._question},
            ],
        )

    async def step(self, assistant_message: Message) -> EnvStep:
        completion = str(assistant_message.get("content") or "")
        reward_components = {
            "correctness": self._correctness_reward(completion),
            "format": self._format_reward(completion),
        }
        return EnvStep(
            reward=sum(reward_components.values()),
            reward_components=reward_components,
            done=True,
            status=RolloutStatus.COMPLETED,
        )

    async def close(self) -> None:
        return None

    def _correctness_reward(self, completion: str) -> float:
        matches = re.findall(r"\[ANSWER\]\s*(-?\d+)", completion)
        correct = bool(matches) and int(matches[-1]) == self._target
        return self._correctness_reward_value if correct else 0.0

    def _format_reward(self, completion: str) -> float:
        if re.search(r"\[ANSWER\]\s*-?\d+", completion):
            return self._format_reward_value
        return 0.0
