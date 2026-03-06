# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for FileWatcher + aggregate."""

import json
import os
import time

import pytest

from torchtitan.observability.aggregation import aggregate, FileWatcher


def _write_entry(filepath: str, entry: dict) -> None:
    """Append a JSON entry to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(entry) + "\n")


class TestFileWatcher:
    def test_comprehensive_scenario(self, tmp_path):
        """Tests basic drain, stale purge, late file, multi-file, incomplete line, close."""
        log_dir = str(tmp_path / "experiment_logs")
        os.makedirs(log_dir)
        file_a = os.path.join(log_dir, "rank_0.jsonl")

        # 1. Create watcher on empty directory.
        watcher = FileWatcher(log_dir, poll_interval=0.5)

        # 2. Write entries for steps 1, 2, 3 to file A.
        for step in [1, 2, 3]:
            _write_entry(file_a, {"key": "loss", "value": step * 0.1, "step": step})

        # 3. drain(1) — should get step 1 entry.
        entries = watcher.drain(1)
        assert len(entries) == 1
        assert entries[0]["step"] == 1
        assert entries[0]["value"] == 0.1

        # drain(1) again — step 1 was already drained, should be empty.
        assert watcher.drain(1) == []

        # 4. drain(3) — should get step 3. Step 2 is stale (< 3) and purged.
        entries = watcher.drain(3)
        assert len(entries) == 1
        assert entries[0]["step"] == 3

        # Verify step 2 was purged.
        assert watcher.drain(2) == []

        # 5. Create file B (late-arriving file). Write step 4 to both A and B.
        file_b = os.path.join(log_dir, "rank_1.jsonl")
        _write_entry(file_a, {"key": "loss", "value": 0.4, "step": 4})
        _write_entry(file_b, {"key": "loss", "value": 0.5, "step": 4})

        # drain(4) discovers file B and reads both.
        entries = watcher.drain(4)
        assert len(entries) == 2
        values = sorted(e["value"] for e in entries)
        assert values == [0.4, 0.5]

        # 6. Write entries for step 5 and verify rapid drain works (catch-up read).
        _write_entry(file_a, {"key": "loss", "value": 0.6, "step": 5})
        entries = watcher.drain(5)
        assert len(entries) == 1
        assert entries[0]["value"] == 0.6

        # 7. Malformed JSON is silently skipped.
        with open(file_a, "a") as f:
            f.write("not valid json\n")
        _write_entry(file_a, {"key": "loss", "value": 0.7, "step": 6})
        entries = watcher.drain(6)
        assert len(entries) == 1
        assert entries[0]["value"] == 0.7

        # 8. Close — should join thread and close handles.
        watcher.close()

    def test_skips_historical_data(self, tmp_path):
        """FileWatcher constructed after data exists should skip it."""
        log_dir = str(tmp_path / "experiment_logs")
        os.makedirs(log_dir)
        filepath = os.path.join(log_dir, "rank_0.jsonl")

        # Write data BEFORE creating watcher.
        _write_entry(filepath, {"key": "old", "value": 1.0, "step": 1})

        watcher = FileWatcher(log_dir, poll_interval=0.5)

        # drain(1) should NOT return the historical entry.
        entries = watcher.drain(1)
        assert len(entries) == 0

        # Write new data AFTER watcher creation.
        _write_entry(filepath, {"key": "new", "value": 2.0, "step": 2})

        entries = watcher.drain(2)
        assert len(entries) == 1
        assert entries[0]["key"] == "new"

        watcher.close()


class TestAggregate:
    def test_mean_metric_across_ranks(self):
        """MeanMetric entries from multiple ranks are reduced correctly."""
        entries = [
            {"key": "loss", "reduce": "MeanMetric", "sum": 3.0, "weight": 1.0, "step": 1, "rank": 0},
            {"key": "loss", "reduce": "MeanMetric", "sum": 5.0, "weight": 1.0, "step": 1, "rank": 1},
        ]
        result = aggregate(entries)
        assert result["loss"] == 4.0  # (3 + 5) / (1 + 1)

    def test_max_metric(self):
        entries = [
            {"key": "mem", "reduce": "MaxMetric", "value": 10.0, "step": 1, "rank": 0},
            {"key": "mem", "reduce": "MaxMetric", "value": 15.0, "step": 1, "rank": 1},
        ]
        result = aggregate(entries)
        assert result["mem"] == 15.0

    def test_all_reduced_passthrough(self):
        """Entries with all_reduced=True pass through without reduction."""
        entries = [
            {"key": "loss", "value": 2.5, "all_reduced": True, "step": 1},
            {"key": "grad", "reduce": "MaxMetric", "value": 1.0, "step": 1, "rank": 0},
        ]
        result = aggregate(entries)
        assert result["loss"] == 2.5
        assert result["grad"] == 1.0

    def test_empty(self):
        assert aggregate([]) == {}
