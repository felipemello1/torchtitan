# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollouts.types import DatasetOutput


@dataclass(frozen=True, kw_only=True, slots=True)
class SumDigitsInput:
    """Typed payload for one SumDigits problem."""

    numbers: list[int]  # [N_numbers]
    """Numbers the model must digit-sum."""

    target: int
    """Ground-truth total digit sum."""


class SumDigitsDataset(Configurable):
    """Endless seeded stream of SumDigits problems.

    Iterate for problems; `state_dict`/`load_state_dict` snapshot the RNG so a run
    can resume mid-stream.

        dataset = SumDigitsDataset(SumDigitsDataset.Config(seed=42))
        problem = next(iter(dataset))   # problem.task_name == "sum_digits"
    """

    TASK_NAME = "sum_digits"

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 42

    def __init__(self, config: Config) -> None:
        self._rng = random.Random(config.seed)

    def __iter__(self) -> Iterator[DatasetOutput]:
        return self

    def __next__(self) -> DatasetOutput:
        return self._sample_problem()

    def state_dict(self) -> dict:
        """Snapshot the RNG so a run can resume at the same point in the stream."""
        return {"rng_state": self._rng.getstate()}

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])

    def _sample_problem(self) -> DatasetOutput:
        n = self._rng.randint(2, 4)
        numbers = [self._rng.randint(10, 99) for _ in range(n)]
        target = sum(int(d) for num in numbers for d in str(num))
        return DatasetOutput(
            task_name=self.TASK_NAME,
            env_input=SumDigitsInput(numbers=numbers, target=target),
        )
