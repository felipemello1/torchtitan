# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollout logger for RL training data.

No enforced schema. Takes list[dict], writes one JSON line per dict.
Separate from experiment JSONL (not consumed by DefaultAggregator).

Query with:
    jq 'select(.reward < 0.1)' rollouts.jsonl
    # or DuckDB: SELECT * FROM read_json('rollouts.jsonl') WHERE policy_version = 5
"""

import json
import os
from collections.abc import Callable


class RolloutLogger:
    """Logs rollout data as JSONL for offline analysis.

    No enforced schema — takes any list[dict]. Step is added to each record.
    Optional filter_fn can be set at init or overridden per log() call.

    Args:
        output_dir: Directory for rollout files.
        filename: Name of the JSONL file (default: rollouts.jsonl).
        filter_fn: Optional default filter applied to every log() call.
    """

    def __init__(
        self,
        output_dir: str,
        filename: str = "rollouts.jsonl",
        filter_fn: Callable[[list[dict]], list[dict]] | None = None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self._filepath = os.path.join(output_dir, filename)
        # File kept open for the lifetime of the logger. Closing and reopening
        # per flush would incur two OS syscalls per write and reset the kernel
        # buffer. Data is durable after each flush() call regardless.
        self._file = open(self._filepath, "a")
        self._filter_fn = filter_fn

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
            filter_fn: Override the default filter for this call.
        """
        if not records:
            return
        fn = filter_fn if filter_fn is not None else self._filter_fn
        if fn is not None:
            records = fn(records)
        self._file.write(
            "\n".join(json.dumps({**r, "step": step}) for r in records) + "\n"
        )
        self._file.flush()

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def filter_top_bottom(
    records: list[dict], key: str = "reward", k: int = 1
) -> list[dict]:
    """Keep top-k and bottom-k records by a key.

    Args:
        records: List of rollout dicts.
        key: Key to sort by (default: "reward").
        k: Number of records to keep from each end (default: 1).

    Returns:
        Bottom-k + top-k records. If fewer than 2*k records, clamps k
        to len//2 to avoid duplicates.
    """
    sorted_recs = sorted(records, key=lambda r: r.get(key, 0))
    k = min(k, len(sorted_recs) // 2) if sorted_recs else 0
    if k == 0:
        return sorted_recs
    return sorted_recs[:k] + sorted_recs[-k:]
