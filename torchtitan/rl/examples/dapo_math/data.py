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

from torchtitan.rl.components.data import RLDataset

_MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: \\boxed{{$Answer}}, where $Answer is the answer "
    "to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line as "Answer: \\boxed{{...}}".'
)

# Options `A) 12 B) 14 C) 15` or `\textbf{a)}\ 12` in a question: the gold is the option's value,
# so a boxed letter scores 0.
_MULTIPLE_CHOICE = re.compile(r"(?:\bA|\{a)\).*(?:\bB|\{b)\).*(?:\bC|\{c)\)", re.S)
# A one-letter gold like ` n `: mostly "for which n ...?" questions, where it names the variable.
_ONE_LETTER = re.compile(r"\s*[A-Za-z]\s*")
# A question asking for two values, as in `find $f(2)$ and $f(-2)$` or `find the maximum and
# minimum values`: with a one-value gold, the full answer `0, -6` scores 0.
_TWO_VALUES = re.compile(
    r"\b(?:find|determine|calculate|compute|evaluate)\s+(?:the\s+(?:values?|lengths?)\s+of\s+)?"
    r"(?:\$|\\\()[^$]{1,30}(?:\$|\\\))\s*,?\s*and\s+(?:\$|\\\()[^$]{1,30}(?:\$|\\\))\s*[.?;]"
    r"|\b(?:find|determine)\s+(?:the\s+)?(?:maximum|minimum|largest|smallest|greatest|least)\s+"
    r"(?:value\s+)?and\s+(?:the\s+)?(?:maximum|minimum|largest|smallest|greatest|least)\b",
    re.IGNORECASE,
)
# A gold with two values: `m=6, n=9`, `1 \text{ and } -1`, `11 \pm 2\sqrt{3}`.
_TWO_VALUE_GOLD = re.compile(r"[,;]|\band\b|\\pm")


@dataclass(frozen=True, kw_only=True, slots=True)
class DapoMathSample:
    """A math prompt paired with its expected final answer."""

    prompt: str
    ground_truth: str


# TODO: Share this cycling iterator with other RL datasets instead of keeping
# per-environment implementations.
class _CyclingDataset(RLDataset):
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
        # The saved order indexes the rows at save time; against another row count it would
        # pick wrong rows, then raise IndexError a few steps later.
        if len(state_dict["order"]) != len(self._samples):
            raise ValueError(
                f"Saved dataset state covers {len(state_dict['order'])} rows, but the dataset "
                f"has {len(self._samples)}; start the data stream fresh."
            )
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._position = state_dict["position"]


class DapoMathDataset(_CyclingDataset):
    """Provides filtered DAPO-Math problems in the original `Answer:` format."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
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
    """Provides INTELLECT-3-RL math problems that Qwen3-4B-Thinking-2507 solves in 1-7 of 8 tries.

    Harder than `DapoMathDataset` for a strong base model, so more groups mix right and wrong
    answers; only those groups train. INTELLECT-3-RL has no 8/8 rows; the default
    `min_pass_rate` drops the 0/8 ones, keeping 10,805 of 21,161 rows. It also drops 173 rows
    whose golds mis-score answers (options `A) B) C)`, a one-letter gold, or two values asked
    and one given), leaving 10,632.

    Example:
        config = rl_dapo_qwen3_4b_math_32k()
        config.rollouter.training_dataloader.dataset = Intellect3MathDataset.Config()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        repo_id: str = "PrimeIntellect/INTELLECT-3-RL"
        split: str = "train"
        min_pass_rate: float = 0.125
        """Keep problems solved in at least this fraction of the 8 tries; 0.125 is 1 of 8."""
        max_pass_rate: float = 0.875
        """Keep problems solved in at most this fraction of the 8 tries; 0.875 is 7 of 8."""
        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        dataset = load_dataset(config.repo_id, "math", split=config.split)
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
            and not (
                _TWO_VALUES.search(row["question"])
                and not _TWO_VALUE_GOLD.search(str(row["answer"]))
            )
        ]
        super().__init__(samples, seed=config.seed, shuffle=config.shuffle)


class AIME2025Dataset(_CyclingDataset):
    """Provides AIME 2025 I+II problems using the DAPO answer format."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
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
