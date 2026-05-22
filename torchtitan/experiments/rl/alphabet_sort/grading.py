# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward helpers for AlphabetSort."""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence

_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[\).\s]+)")
_PLACEHOLDER_RE = re.compile(r"^name\s*\d+(?:\s*//.*)?$", re.IGNORECASE)


def score_turn_similarity(
    completion: str,
    *,
    expected: Sequence[str],
    turn_idx: int,
) -> float:
    """Score one turn by sequence similarity against the expected list."""
    predicted = extract_names(completion, turn_idx=turn_idx)
    if not predicted or not expected:
        return 0.0
    pred_text = "\n".join(item.strip().lower() for item in predicted)
    expected_text = "\n".join(item.strip().lower() for item in expected)
    return difflib.SequenceMatcher(None, pred_text, expected_text).ratio()


def aggregate_turn_scores(
    similarities: Sequence[float],
    *,
    similarity_power: int,
    power_per_turn: bool,
) -> float:
    """Aggregate per-turn similarities into one episode reward."""
    if not similarities:
        return 0.0
    mean_similarity = sum(similarities) / len(similarities)
    if not power_per_turn:
        return mean_similarity**similarity_power
    powered = [similarity**similarity_power for similarity in similarities]
    return sum(powered) / len(powered)


def extract_names(completion: str, *, turn_idx: int) -> list[str]:
    """Extract sorted names from the expected XML-style block."""
    tag = "alphabetical_sorted" if turn_idx == 0 else "combined_alphabetical_sorted"
    matches = re.findall(
        rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>",
        completion,
        re.DOTALL | re.IGNORECASE,
    )
    if not matches:
        return []
    body = matches[-1]
    names: list[str] = []
    for line in body.splitlines():
        name = _LIST_PREFIX_RE.sub("", line).strip()
        if not name or _PLACEHOLDER_RE.fullmatch(name):
            continue
        names.append(name)
    return names
