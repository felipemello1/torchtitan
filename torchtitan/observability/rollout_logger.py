# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class RolloutOutput:
    """One prompt+completion pair from the generator.

    Token fields are used for training. Text fields are for logging
    and human inspection only. In a real pipeline, text fields would
    be populated by the tokenizer's decode method.

    Example::

        rollout = RolloutOutput(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[4, 5, 6],
            prompt_text="What is 2+2?",
            completion_text="The answer is 4.",
        )
        rollout.reward = 1.0
        rollout.to_logging_dict()
        # {"prompt": "What is 2+2?", "completion": "The answer is 4.", "reward": 1.0}
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    prompt_text: str
    completion_text: str
    reward: float | None = None

    def to_logging_dict(self) -> dict:
        """Convert to a dict suitable for RolloutLogger."""
        d = {"prompt": self.prompt_text, "completion": self.completion_text}
        if self.reward is not None:
            d["reward"] = self.reward
        return d


class RolloutLogger:
    """Logs rollout data as JSONL for offline analysis.

    Takes any list[dict] (typically from ``RolloutOutput.to_logging_dict()``).
    An optional ``filter_fn`` selects which records to keep, e.g.
    ``filter_top_bottom`` to log only the best and worst rollouts.

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
        self._file = open(self._filepath, "a")  # kept open for lifetime
        self._filter_fn = filter_fn

    def log(
        self,
        records: list[dict],
        metadata: dict | None = None,
        filter_fn: Callable[[list[dict]], list[dict]] | None = None,
    ) -> None:
        """Write rollout dicts as JSON lines.

        Args:
            records: List of rollout dicts. No schema enforced.
            metadata: Extra fields merged into each record (e.g. {"step": 1}).
            filter_fn: Override the default filter for this call.
        """
        if not records:
            return
        fn = filter_fn if filter_fn is not None else self._filter_fn
        if fn is not None:
            records = fn(records)
        extra = metadata or {}
        self._file.write(
            "\n".join(json.dumps({**r, **extra}) for r in records) + "\n"
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def filter_top_bottom(
    records: list[dict], key: str = "reward", k: int = 1
) -> list[dict]:
    """Keep top-k and bottom-k records by a key.

    If fewer than 2*k records, returns all records.

    Args:
        records: List of rollout dicts.
        key: Key to sort by (default: "reward").
        k: Number of records to keep from each end (default: 1).

    Returns:
        Bottom-k + top-k records sorted by key.
    """
    sorted_recs = sorted(records, key=lambda r: r.get(key, 0))
    k = min(k, len(sorted_recs) // 2) if sorted_recs else 0
    if k == 0:
        return sorted_recs
    return sorted_recs[:k] + sorted_recs[-k:]
