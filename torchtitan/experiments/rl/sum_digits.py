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


class SumDigitsEnv(Configurable):
    """Single-turn, single-use sum-of-digits task."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        correctness_reward: float = 1.0
        """Reward for a response containing ``[ANSWER] <target>``."""

        format_reward: float = 0.3
        """Reward bonus for any ``[ANSWER] <number>`` tag in the response."""

        seed: int = 42
        """Seed mixed with ``(step, group_idx)`` to generate deterministic tasks."""

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
        rng = random.Random(f"{config.seed}:{step}:{group_idx}")
        values_from_payload = (
            example.payload.get("values") if example is not None else None
        )
        if isinstance(values_from_payload, list) and all(
            isinstance(value, int) for value in values_from_payload
        ):
            values = values_from_payload
        else:
            num_values = rng.randint(2, 4)
            values = [rng.randint(10, 99) for _ in range(num_values)]

        target_from_payload = (
            example.payload.get("target") if example is not None else None
        )
        if isinstance(target_from_payload, int):
            self._target = target_from_payload
        else:
            self._target = sum(int(digit) for value in values for digit in str(value))
        self._question = f"What is the total digit sum of {values}?"

    async def reset(self) -> EnvReset:
        return EnvReset(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
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
        return self._config.correctness_reward if correct else 0.0

    def _format_reward(self, completion: str) -> float:
        if re.search(r"\[ANSWER\]\s*-?\d+", completion):
            return self._config.format_reward
        return 0.0
