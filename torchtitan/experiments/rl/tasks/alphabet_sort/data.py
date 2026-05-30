# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollouts.types import DatasetOutput


@dataclass(frozen=True, kw_only=True, slots=True)
class AlphabetSortInput:
    """One episode, served to the model turn by turn.

    `presented_names[i]` are the (shuffled) names shown on turn `i`; `expected_lines[i]`
    is the correct answer for that turn (every name so far, sorted, with the names new
    that turn suffixed `// new name!`).

    Example (2 turns, by FIRST name):

        presented_names = (("MarcChardin", "AnaChardin"), ("BobBeck",))
        expected_lines  = (("AnaChardin", "MarcChardin"),
                           ("AnaChardin", "BobBeck // new name!", "MarcChardin"))
    """

    presented_names: tuple[tuple[str, ...], ...]  # [num_turns][names_shown]
    expected_lines: tuple[tuple[str, ...], ...]  # [num_turns][correct_answer_lines]
    sort_by_first: bool
    """Order names by first name (True) or last name (False)."""


class AlphabetSortDataset(Configurable):
    """Endless stream of name-sorting episodes (the prime-rl AlphabetSort task).

    Names are arXiv authors in CamelCase (e.g. "MarcChardin"), drawn from a Hugging
    Face dataset. Each episode shows a few names and asks for them alphabetically
    sorted; multi-turn episodes add names each turn and ask for the whole list
    re-sorted, marking the names that are new.

    Iterate for episodes; `state_dict`/`load_state_dict` snapshot the RNG so a run
    can resume mid-stream. Training currently runs one turn per rollout, so
    `num_turns` defaults to 1.

        dataset = AlphabetSortDataset(AlphabetSortDataset.Config(seed=42))
        episode = next(iter(dataset))   # episode.task_name == "alphabet_sort"
    """

    TASK_NAME = "alphabet_sort"

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 1337420
        num_turns: int = 1
        min_names_per_turn: int = 2
        max_names_per_turn: int = 5
        hf_dataset: str = "kalomaze/alphabetic-arxiv-authors-it1"
        hf_split: str = "train"

        def __post_init__(self) -> None:
            if self.num_turns < 1:
                raise ValueError(f"num_turns must be >= 1; got {self.num_turns}")
            if not 1 <= self.min_names_per_turn <= self.max_names_per_turn:
                raise ValueError(
                    "need 1 <= min_names_per_turn <= max_names_per_turn; "
                    f"got {self.min_names_per_turn}, {self.max_names_per_turn}"
                )

    def __init__(self, config: Config) -> None:
        self._config = config
        self._names = _load_names(config.hf_dataset, config.hf_split)
        self._rng = random.Random(config.seed)

    def __iter__(self) -> Iterator[DatasetOutput]:
        return self

    def __next__(self) -> DatasetOutput:
        return self._sample_episode()

    def state_dict(self) -> dict:
        """Snapshot the RNG so a run can resume at the same point in the stream."""
        return {"rng_state": self._rng.getstate()}

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])

    def _sample_episode(self) -> DatasetOutput:
        cfg = self._config

        # Pick how many names each turn introduces, then draw them all at once so a
        # name never repeats across turns.
        names_per_turn = [
            self._rng.randint(cfg.min_names_per_turn, cfg.max_names_per_turn)
            for _ in range(cfg.num_turns)
        ]
        episode_names = self._rng.sample(self._names, k=sum(names_per_turn))
        sort_by_first = self._rng.choice([True, False])
        part = 0 if sort_by_first else 1  # which half of (first, last) to sort on

        # Build each turn. `names_so_far` is the cumulative list; `new_names` are the
        # names introduced this turn. The answer is `names_so_far` sorted, with the
        # new names marked (except on turn 0, where every name is new).
        presented_names: list[tuple[str, ...]] = []
        expected_lines: list[tuple[str, ...]] = []
        names_so_far: list[str] = []
        start = 0
        for count in names_per_turn:
            new_names = episode_names[start : start + count]
            start += count
            names_so_far.extend(new_names)

            is_first_turn = not presented_names
            new = set(new_names)
            answer = tuple(
                name if (is_first_turn or name not in new) else f"{name} // new name!"
                for name in sorted(names_so_far, key=lambda n: _split_name(n)[part])
            )

            shown = list(new_names)
            self._rng.shuffle(shown)
            presented_names.append(tuple(shown))
            expected_lines.append(answer)

        return DatasetOutput(
            task_name=self.TASK_NAME,
            env_input=AlphabetSortInput(
                presented_names=tuple(presented_names),
                expected_lines=tuple(expected_lines),
                sort_by_first=sort_by_first,
            ),
        )


def _load_names(hf_dataset: str, hf_split: str) -> tuple[str, ...]:
    """Load and dedupe the CamelCase author names (e.g. "Marc Chardin" -> "MarcChardin")."""
    dataset = load_dataset(hf_dataset, split=hf_split)
    seen: set[str] = set()
    names: list[str] = []
    for row in dataset:
        for raw_name in row["names"]:
            name = raw_name.replace(" ", "")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return tuple(names)


def _split_name(name: str) -> tuple[str, str]:
    """Split a CamelCase author name into lowercased (first, last) at the first
    internal capital.

        _split_name("MarcChardin")  # -> ("marc", "chardin")
        _split_name("VanDerBerg")   # -> ("van", "derberg")
        _split_name("Plato")        # -> ("plato", "")
    """
    for idx in range(1, len(name)):
        if name[idx].isupper():
            return name[:idx].lower(), name[idx:].lower()
    return name.lower(), ""
