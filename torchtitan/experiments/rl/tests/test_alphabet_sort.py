# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.experiments.rl.alphabet_sort.env import AlphabetSortEnv
from torchtitan.experiments.rl.alphabet_sort.grading import (
    aggregate_turn_scores,
    extract_names,
    score_turn_similarity,
)


def test_score_turn_similarity_returns_raw_similarity():
    raw = score_turn_similarity(
        "<alphabetical_sorted>\nAlice\nBob\n</alphabetical_sorted>",
        expected=["Alice", "Beatrice"],
        turn_idx=0,
    )

    assert 0.0 < raw < 1.0


def test_aggregate_turn_scores_can_power_after_averaging():
    similarities = [1.0, 0.5]

    assert aggregate_turn_scores(
        similarities,
        similarity_power=2,
        power_per_turn=False,
    ) == pytest.approx(0.75**2)
    assert aggregate_turn_scores(
        similarities,
        similarity_power=2,
        power_per_turn=True,
    ) == pytest.approx((1.0**2 + 0.5**2) / 2)


def test_extract_names_accepts_case_and_list_prefixes():
    completion = """<COMBINED_ALPHABETICAL_SORTED>
1. Bob // new name!
- Alice
Name1
</COMBINED_ALPHABETICAL_SORTED>"""

    assert extract_names(completion, turn_idx=1) == [
        "Bob // new name!",
        "Alice",
    ]


def test_alphabet_sort_config_validates_ranges():
    with pytest.raises(ValueError, match="max_turns"):
        AlphabetSortEnv.Config(min_turns=4, max_turns=3)
