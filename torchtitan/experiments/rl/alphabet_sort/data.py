# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data generation for AlphabetSort rollouts."""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.envs import EnvExample

logger = logging.getLogger(__name__)

_HF_DATASET_NAME = "kalomaze/alphabetic-arxiv-authors-it1"
_HF_DATASET_SPLIT = "train"
_names_cache: tuple[str, ...] | None = None
_names_cache_lock = threading.Lock()


def _load_names() -> tuple[str, ...]:
    """Load and cache a deduped CamelCase name pool."""
    global _names_cache
    if _names_cache is not None:
        return _names_cache
    with _names_cache_lock:
        if _names_cache is not None:
            return _names_cache
        from datasets import load_dataset

        ds = load_dataset(_HF_DATASET_NAME, split=_HF_DATASET_SPLIT)
        seen: set[str] = set()
        names: list[str] = []
        for row in ds:
            for raw in row["names"]:
                combined = raw.replace(" ", "")
                if combined and combined not in seen:
                    seen.add(combined)
                    names.append(combined)
        _names_cache = tuple(names)
        logger.info(
            "Loaded %d unique CamelCase names from %s",
            len(_names_cache),
            _HF_DATASET_NAME,
        )
    return _names_cache


@dataclass(frozen=True, slots=True)
class AlphabetSortExample:
    """One multi-turn cumulative name-sorting task."""

    initial_prompt: str
    follow_ups: tuple[str, ...]
    ground_truths: tuple[tuple[str, ...], ...]
    turn_names: tuple[tuple[str, ...], ...]
    sort_by_first: bool

    @property
    def num_turns(self) -> int:
        return len(self.turn_names)


def build_example(
    *,
    seed: int,
    step: int,
    group_idx: int,
    min_turns: int,
    max_turns: int,
    min_names_per_turn: int,
    max_names_per_turn: int,
) -> AlphabetSortExample:
    """Build one deterministic episode from the name pool."""
    rng = random.Random(f"{seed}:{step}:{group_idx}")
    num_turns = rng.randint(min_turns, max_turns)
    names_per_turn = [
        rng.randint(min_names_per_turn, max_names_per_turn) for _ in range(num_turns)
    ]
    names_needed = sum(names_per_turn)
    selected_names = rng.sample(_load_names(), k=names_needed)
    sort_by_first = rng.choice([True, False])

    turn_names: list[list[str]] = []
    offset = 0
    for count in names_per_turn:
        turn_names.append(selected_names[offset : offset + count])
        offset += count

    ground_truths: list[list[str]] = []
    cumulative: list[str] = []
    for turn_idx, names in enumerate(turn_names):
        cumulative.extend(names)
        sorted_names = sorted(
            cumulative,
            key=_extract_first_name if sort_by_first else _extract_last_name,
        )
        if turn_idx == 0:
            ground_truths.append(sorted_names)
        else:
            ground_truths.append(
                [
                    f"{name} // new name!" if name in names else name
                    for name in sorted_names
                ]
            )

    sort_key = "FIRST" if sort_by_first else "LAST"
    shuffled_first = list(turn_names[0])
    rng.shuffle(shuffled_first)
    first_names = ", ".join(shuffled_first)
    initial_prompt = f"""Sort these names in alphabetical order by {sort_key} name: {first_names}

Use exactly this format:
<alphabetical_sorted>
Name1
Name2
</alphabetical_sorted>"""

    follow_ups = []
    for turn_idx, names in enumerate(turn_names[1:], start=1):
        shuffled = list(names)
        rng.shuffle(shuffled)
        name_list = ", ".join(shuffled)
        new_name_instruction = (
            "These are in addition to the prior list. Mark any NEW names "
            "(that weren't in the prior list) with `// new name!` at the end."
        )
        if turn_idx == 1:
            follow_ups.append(
                f"""New names to add to the prior list: {name_list}

Sort the COMPLETE cumulative list alphabetically by {sort_key} name.

{new_name_instruction}

Use exactly this format:
<combined_alphabetical_sorted>
Name1
Name2 // new name!
</combined_alphabetical_sorted>"""
            )
        else:
            follow_ups.append(
                f"""New names to add to the prior list: {name_list}

Sort the COMPLETE cumulative list alphabetically by {sort_key} name.

{new_name_instruction} Follow the same format as before."""
            )

    return AlphabetSortExample(
        initial_prompt=initial_prompt,
        follow_ups=tuple(follow_ups),
        ground_truths=tuple(tuple(row) for row in ground_truths),
        turn_names=tuple(tuple(row) for row in turn_names),
        sort_by_first=sort_by_first,
    )


class AlphabetSortDataset(Configurable):
    """Deterministic dataset of concrete AlphabetSort episodes."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 1337420
        min_turns: int = 3
        max_turns: int = 5
        min_names_per_turn: int = 1
        max_names_per_turn: int = 4

        def __post_init__(self) -> None:
            if self.min_turns <= 0:
                raise ValueError(f"min_turns must be positive, got {self.min_turns}")
            if self.max_turns < self.min_turns:
                raise ValueError(
                    "max_turns must be greater than or equal to min_turns, "
                    f"got {self.max_turns} < {self.min_turns}"
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

    def __init__(self, config: Config):
        self.config = config

    def sample_group(self, *, sample_step: int, group_idx: int) -> EnvExample:
        episode = build_example(
            seed=self.config.seed,
            step=sample_step,
            group_idx=group_idx,
            min_turns=self.config.min_turns,
            max_turns=self.config.max_turns,
            min_names_per_turn=self.config.min_names_per_turn,
            max_names_per_turn=self.config.max_names_per_turn,
        )
        return EnvExample(
            group_id=f"alphabet_sort/step={sample_step}/group={group_idx}",
            sample_step=sample_step,
            group_idx=group_idx,
            payload=episode,
        )


def _extract_first_name(name: str) -> str:
    for idx, char in enumerate(name[1:], start=1):
        if char.isupper():
            return name[:idx].lower()
    return name.lower()


def _extract_last_name(name: str) -> str:
    for idx, char in enumerate(name[1:], start=1):
        if char.isupper():
            return name[idx:].lower()
    return ""
