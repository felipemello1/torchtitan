# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward helpers for AlphabetSort."""

from __future__ import annotations

import difflib
import re


def score_completion(
    completion: str,
    *,
    expected: list[str],
    turn_idx: int,
    similarity_power: int,
) -> float:
    """Sequence-similarity reward for one AlphabetSort turn."""
    predicted = extract_names(completion, turn_idx=turn_idx)
    if not predicted or not expected:
        return 0.0
    pred_text = "\n".join(item.strip().lower() for item in predicted)
    expected_text = "\n".join(item.strip().lower() for item in expected)
    similarity = difflib.SequenceMatcher(None, pred_text, expected_text).ratio()
    return similarity**similarity_power


def extract_names(completion: str, *, turn_idx: int) -> list[str]:
    """Extract sorted names from the expected XML-ish tag."""
    tag = "alphabetical_sorted" if turn_idx == 0 else "combined_alphabetical_sorted"
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", completion, re.DOTALL)
    if not matches:
        return []
    body = matches[-1]
    return [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("Name")
    ]
