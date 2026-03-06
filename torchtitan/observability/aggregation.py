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

# How often the poll thread re-globs for new files (in number of poll ticks).
# At 50ms poll interval, 100 ticks = ~5 seconds.
_REGLOB_EVERY_N_POLLS = 100


class FileWatcher:
    """Background thread tails JSONL files, buffers entries by step.

    The poll thread reads known file handles every tick (cheap).
    New files are discovered via glob periodically (~5s) and on drain().
    drain(step) does a catch-up read to get entries written just before barrier.
    """

    def __init__(self, log_dir: str, poll_interval: float = 0.05):
        self._log_dir = log_dir
        self._poll_interval = poll_interval
        self._offsets: dict[str, int] = {}
        self._handles: dict[str, IO] = {}
        self._buffer: dict[int, list[dict]] = defaultdict(list)
        self._lock = Lock()
        self._running = True

        # Skip historical data by recording end-of-file offsets for existing files.
        for fp in glob.glob(os.path.join(log_dir, "*.jsonl")):
            self._offsets[fp] = os.path.getsize(fp)

        self._thread = Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        polls_since_glob = 0
        while self._running:
            with self._lock:
                if polls_since_glob >= _REGLOB_EVERY_N_POLLS:
                    self._discover_new_files()
                    polls_since_glob = 0
                self._read_known_handles()
                polls_since_glob += 1
            time.sleep(self._poll_interval)

    def _discover_new_files(self):
        """Glob for new JSONL files and open them. Must be called under self._lock."""
        for fp in glob.glob(os.path.join(self._log_dir, "*.jsonl")):
            if fp not in self._handles:
                try:
                    f = open(fp)
                    f.seek(self._offsets.get(fp, 0))
                    self._handles[fp] = f
                except FileNotFoundError:
                    continue
                # If this is a brand-new file we haven't seen before, skip to end
                # (historical data). If we have an offset, seek was already done.
                if fp not in self._offsets:
                    self._offsets[fp] = f.tell()

    def _read_known_handles(self):
        """Read new lines from all open file handles. Must be called under self._lock."""
        for fp, f in self._handles.items():
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
        """Return entries for step. Discovers new files, catch-up reads, purges stale."""
        with self._lock:
            self._discover_new_files()
            self._read_known_handles()
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
