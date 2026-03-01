# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollout logger for RL training data (PR6).

No enforced schema. Takes list[dict], writes one JSON line per dict.
Single write call for the whole batch. Separate from experiment JSONL
(not consumed by DefaultAggregator).

Query with:
    jq 'select(.reward < 0.1)' rollouts.jsonl
    # or DuckDB: SELECT * FROM read_json('rollouts.jsonl') WHERE policy_version = 5
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable


class RolloutLogger:
    """Logs rollout data as JSONL for offline analysis.

    No enforced schema — takes any list[dict]. Step is added to each record.
    Supports optional filtering (e.g., keep top/bottom-k by reward).

    Args:
        output_dir: Directory for rollout files.
        filename: Name of the JSONL file (default: rollouts.jsonl).
    """

    def __init__(self, output_dir: str, filename: str = "rollouts.jsonl"):
        os.makedirs(output_dir, exist_ok=True)
        self._filepath = os.path.join(output_dir, filename)
        self._file = open(self._filepath, "a")

    def log(
        self,
        records: list[dict],
        step: int,
        filter_fn: Callable[[list[dict]], list[dict]] | None = None,
    ) -> None:
        """Write rollout dicts as JSON lines. Single write call.

        Args:
            records: List of rollout dicts. No schema enforced.
            step: Training step (added to each record).
            filter_fn: Optional filter (e.g., keep top/bottom-k by reward).
        """
        if not records:
            return
        if filter_fn is not None:
            records = filter_fn(records)
        self._file.write(
            "\n".join(json.dumps({**r, "step": step}) for r in records) + "\n"
        )
        self._file.flush()

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def filter_top_bottom(
    records: list[dict], key: str = "reward", k: int = 5
) -> list[dict]:
    """Keep top-k and bottom-k records by a key.

    Useful for logging only the most/least rewarded completions.

    Args:
        records: List of rollout dicts.
        key: Key to sort by (default: "reward").
        k: Number of records to keep from each end.

    Returns:
        Bottom-k + top-k records. If fewer than 2*k records, returns all
        without duplicates.
    """
    sorted_recs = sorted(records, key=lambda r: r.get(key, 0))
    k = min(k, len(sorted_recs) // 2) if sorted_recs else 0
    if k == 0:
        return sorted_recs
    return sorted_recs[:k] + sorted_recs[-k:]
