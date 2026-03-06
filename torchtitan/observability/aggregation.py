# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FileWatcher + aggregate: collect and reduce experiment metrics from JSONL.

FileWatcher runs a background thread that tails JSONL files written by
record_metric / log_reduced_metrics. drain(step) returns all entries for
a step. aggregate() reduces them to a dict[str, float] for logging.
"""

import glob
import json
import os
import time
from collections import defaultdict
from threading import Lock, Thread
from typing import IO

from torchtitan.observability.metrics import REDUCE_REGISTRY


class FileWatcher:
    """Background thread tails JSONL files, buffers entries by step.

    The poll thread reads continuously during training. drain(step) does a final
    catch-up read under the lock to get entries written just before barrier.
    """

    def __init__(self, log_dir: str, poll_interval: float = 0.05):
        self._log_dir = log_dir
        self._poll_interval = poll_interval
        self._offsets: dict[str, int] = {}
        self._handles: dict[str, IO] = {}
        self._buffer: dict[int, list[dict]] = defaultdict(list)
        self._lock = Lock()
        self._running = True

        # Skip historical data.
        for fp in glob.glob(os.path.join(log_dir, "*.jsonl")):
            self._offsets[fp] = os.path.getsize(fp)

        self._thread = Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            with self._lock:
                self._read_new()
            time.sleep(self._poll_interval)

    def _read_new(self):
        """Tail all JSONL files. Must be called under self._lock."""
        for fp in glob.glob(os.path.join(self._log_dir, "*.jsonl")):
            if fp not in self._handles:
                try:
                    f = open(fp)
                    f.seek(self._offsets.get(fp, 0))
                    self._handles[fp] = f
                except FileNotFoundError:
                    continue
            f = self._handles[fp]
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = entry.get("step")
                if step is not None:
                    self._buffer[step].append(entry)
            self._offsets[fp] = f.tell()

    def drain(self, step: int) -> list[dict]:
        """Return entries for step. Catch-up read + purge stale."""
        with self._lock:
            self._read_new()
            entries = self._buffer.pop(step, [])
            for s in [s for s in self._buffer if s < step]:
                del self._buffer[s]
        return entries

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        for f in self._handles.values():
            f.close()


def aggregate(entries: list[dict]) -> dict[str, float]:
    """Reduce a list of metric entries to a single dict. Pure function.

    Handles two entry types:
    - all_reduced=True: pass through (tensor metrics from replicate_to_host)
    - all_reduced absent: reduce via REDUCE_REGISTRY (e.g., MeanMetric -> sum/weight)
    """
    if not entries:
        return {}
    result: dict[str, float] = {}
    by_key: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        if entry.get("all_reduced"):
            result[entry["key"]] = entry["value"]
        else:
            by_key[entry["key"]].append(entry)
    for key, key_entries in by_key.items():
        cls = REDUCE_REGISTRY[key_entries[0]["reduce"]]
        result[key] = cls.get_reduced_value_from_states(key_entries)
    return result
