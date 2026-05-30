# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from torchtitan.experiments.rl.rollouts import Rollout, RolloutTurn
from torchtitan.experiments.rl.rubrics import RewardFn, Rubric
from torchtitan.experiments.rl.tasks.alphabet_sort.data import AlphabetSortInput


def _xml_tag_for_turn(turn_idx: int) -> str:
    """The XML tag the model is asked to use on a given turn.

    Turn 0 is a plain sort (`<alphabetical_sorted>`); later turns re-sort the whole
    list and mark new names, so they use a distinct `<combined_alphabetical_sorted>`
    tag (matching the different request the env makes on follow-up turns).
    """
    return "alphabetical_sorted" if turn_idx == 0 else "combined_alphabetical_sorted"


def _answer_lines(text: str, *, xml_tag: str) -> list[str]:
    """The non-empty lines inside the model's last `<xml_tag>` block, or [] if absent."""
    blocks = re.findall(
        rf"<\s*{xml_tag}\s*>(.*?)</\s*{xml_tag}\s*>", text, re.DOTALL | re.IGNORECASE
    )
    if not blocks:
        return []
    return [line.strip() for line in blocks[-1].splitlines() if line.strip()]


def score_sorted_list(
    response_text: str,
    *,
    correct_lines: Sequence[str],
    xml_tag: str,
    similarity_power: int,
) -> float:
    """How close the model's sorted list is to the correct one, as a score in [0, 1].

    Pulls the lines from the model's `<xml_tag>` block and compares them — as
    newline-joined, lowercased text — to `correct_lines` with difflib's ratio, then
    raises the result to `similarity_power` so only near-perfect orderings score
    high. Returns 0.0 if the model produced no block.

    Example — the model returns:

        <alphabetical_sorted>
        AnaChardin
        MarcChardin
        </alphabetical_sorted>

    score_sorted_list(response, correct_lines=("AnaChardin", "MarcChardin"),
                      xml_tag="alphabetical_sorted", similarity_power=4)  # -> 1.0

    A swapped order scores below 1.0; a missing block scores 0.0.
    """
    predicted = _answer_lines(response_text, xml_tag=xml_tag)
    if not predicted:
        return 0.0
    predicted_text = "\n".join(line.lower() for line in predicted)
    correct_text = "\n".join(line.lower() for line in correct_lines)
    similarity = difflib.SequenceMatcher(None, predicted_text, correct_text).ratio()
    return similarity**similarity_power


def _assistant_text(turn: RolloutTurn) -> str:
    message = turn.assistant_message
    return (message.get("content") or "") if message else ""


@dataclass(frozen=True, slots=True)
class AlphabetSortReward:
    """Reward = the average, over the episode's turns, of how well each turn was sorted.

    Each turn is scored by `score_sorted_list` against that turn's correct lines.
    """

    similarity_power: int

    __name__ = "reward_alphabet_sort"  # the reward-component metric key

    async def __call__(self, rollout: Rollout, env_input: AlphabetSortInput) -> float:
        turn_scores = [
            score_sorted_list(
                _assistant_text(turn),
                correct_lines=correct_lines,
                xml_tag=_xml_tag_for_turn(turn_idx),
                similarity_power=self.similarity_power,
            )
            for turn_idx, (turn, correct_lines) in enumerate(
                zip(rollout.turns, env_input.expected_lines)
            )
        ]
        return sum(turn_scores) / len(turn_scores) if turn_scores else 0.0


class AlphabetSortRubric(Rubric):
    """Scores AlphabetSort rollouts with a single sequence-similarity reward."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rubric.Config):
        similarity_power: int = 4

        def __post_init__(self) -> None:
            if self.similarity_power <= 0:
                raise ValueError(
                    f"similarity_power must be positive; got {self.similarity_power}"
                )

    def register_funcs(self) -> list[RewardFn]:
        return [
            RewardFn(
                fn=AlphabetSortReward(similarity_power=self._config.similarity_power),
                weight=1.0,
            )
        ]


__all__ = ["AlphabetSortReward", "AlphabetSortRubric", "score_sorted_list"]
