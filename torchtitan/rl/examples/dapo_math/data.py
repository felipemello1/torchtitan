# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass

from datasets import concatenate_datasets, load_dataset

from torchtitan.config import Configurable

_MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: \\boxed{{$Answer}}, where $Answer is the answer "
    "to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line as "Answer: \\boxed{{...}}".'
)

# Options `A) 12 B) 14 C) 15` or `\\textbf{a)}\\ 12` in a question: the gold is the option's value,
# so a boxed letter scores 0.
_MULTIPLE_CHOICE = re.compile(r"(?:\bA|\{a)\).*(?:\bB|\{b)\).*(?:\bC|\{c)\)", re.S)
# A one-letter gold like ` n `: mostly "for which n ...?" questions, where it names the variable.
_ONE_LETTER = re.compile(r"\s*[A-Za-z]\s*")


@dataclass(frozen=True, kw_only=True, slots=True)
class DapoMathSample:
    """A math prompt paired with its expected final answer."""

    prompt: str
    ground_truth: str


# TODO: Share this cycling iterator with other RL datasets instead of keeping
# per-environment implementations.
class _CyclingDataset(Configurable):
    """Provides an endless, resumable stream over a finite sample list."""

    def __init__(
        self,
        samples: list[DapoMathSample],
        *,
        seed: int,
        shuffle: bool,
    ) -> None:
        if not samples:
            raise ValueError("math dataset must contain at least one sample")
        self._samples = samples
        self._rng = random.Random(seed)
        self._shuffle = shuffle
        self._order = list(range(len(samples)))
        if shuffle:
            self._rng.shuffle(self._order)
        self._position = 0

    def __iter__(self) -> Iterator[DapoMathSample]:
        return self

    def __next__(self) -> DapoMathSample:
        if self._position == len(self._order):
            # Rollout production consumes an endless stream; crossing the dataset
            # boundary starts a new epoch. Training reshuffles; validation does not.
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._position = 0
        sample_index = self._order[self._position]
        self._position += 1
        return self._samples[sample_index]

    def state_dict(self) -> dict:
        """Snapshot row order and position so resume continues the same stream."""
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "position": self._position,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state returned by `state_dict`."""
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._position = state_dict["position"]


class DapoMathDataset(_CyclingDataset):
    """Provides filtered DAPO-Math problems in the original `Answer:` format."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        repo_id: str = "hamishivi/DAPO-Math-17k-Processed_filtered"
        split: str = "train"
        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        dataset = load_dataset(config.repo_id, split=config.split)
        samples: list[DapoMathSample] = []
        for row in dataset:
            prompt_messages = row["source_prompt"]
            if len(prompt_messages) != 1 or prompt_messages[0]["role"] != "user":
                raise ValueError("DAPO-Math rows must contain exactly one user prompt")
            samples.append(
                DapoMathSample(
                    # `prompt` is the raw question without answer-format instructions.
                    prompt=_MATH_PROMPT_TEMPLATE.format(problem=row["prompt"]),
                    ground_truth=str(row["ground_truth"]),
                )
            )
        super().__init__(samples, seed=config.seed, shuffle=config.shuffle)


class Intellect3MathDataset(_CyclingDataset):
    """INTELLECT-3 RL math problems that Qwen3-4B-Thinking-2507 solves in some, not all, of 8 tries.

    Harder than DAPO-Math-17k for a strong base model: the dataset already drops problems
    solved 8/8, and the default bounds drop the 0/8 ones too (21,161 -> 10,805 rows). It also
    drops 148 rows with options `A) B) C)` or a one-letter gold, whose golds mis-score answers,
    leaving 10,657.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        repo_id: str = "PrimeIntellect/INTELLECT-3-RL"
        subset: str = "math"
        split: str = "train"
        min_pass_rate: float = 0.125
        """Keep rows whose `avg@8_qwen3_4b_thinking_2507` is at least this."""
        max_pass_rate: float = 0.875
        """Keep rows whose `avg@8_qwen3_4b_thinking_2507` is at most this."""
        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        dataset = load_dataset(config.repo_id, config.subset, split=config.split)
        samples = [
            DapoMathSample(
                prompt=_MATH_PROMPT_TEMPLATE.format(problem=row["question"]),
                # Some golds are JSON-escaped (`\\frac`), and Math-Verify reads `\\` before `\text` or `\{`
                # as a line break, so `0 \\text{ or } 5` parses as 5. No gold holds a real line break.
                ground_truth=str(row["answer"]).replace("\\\\", "\\"),
            )
            for row in dataset
            if config.min_pass_rate
            <= row["avg@8_qwen3_4b_thinking_2507"]
            <= config.max_pass_rate
            and not _MULTIPLE_CHOICE.search(row["question"])
            and not _ONE_LETTER.fullmatch(str(row["answer"]))
        ]
        super().__init__(samples, seed=config.seed, shuffle=config.shuffle)


class AIME2025Dataset(_CyclingDataset):
    """Provides AIME 2025 I+II problems using the DAPO answer format."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        repo_id: str = "opencompass/AIME2025"
        subsets: tuple[str, ...] = ("AIME2025-I", "AIME2025-II")
        split: str = "test"
        seed: int = 99
        shuffle: bool = False
        num_samples: int = 30

    def __init__(self, config: Config) -> None:
        dataset = concatenate_datasets(
            [
                load_dataset(config.repo_id, subset, split=config.split)
                for subset in config.subsets
            ]
        ).select(range(config.num_samples))
        samples = [
            DapoMathSample(
                prompt=_MATH_PROMPT_TEMPLATE.format(problem=row["question"]),
                ground_truth=str(row["answer"]),
            )
            for row in dataset
        ]
        super().__init__(samples, seed=config.seed, shuffle=config.shuffle)
