# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test for PR1: validates toy_spmd.py JSONL outputs.

These tests read the actual output directory produced by running toy_spmd.py.
They verify that structured logging produces correct per-rank JSONL files
with expected events, structure, and step context.

Prerequisites:
    torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd
"""

import json
import os

import pytest

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "torchtitan", "experiments", "observability", "outputs", "toy_spmd",
)


@pytest.fixture
def sys_log_dir():
    d = os.path.join(OUTPUT_DIR, "system_logs")
    assert os.path.exists(d), (
        f"Output not found at {d}. Run first:\n"
        "  torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd"
    )
    return d


@pytest.fixture
def rank0_events(sys_log_dir):
    """Load all events from rank 0's system JSONL."""
    files = sorted(f for f in os.listdir(sys_log_dir) if "rank_0" in f and f.endswith(".jsonl"))
    assert files, "No rank 0 JSONL file found"
    with open(os.path.join(sys_log_dir, files[0])) as f:
        return [json.loads(line) for line in f if line.strip()]


NUM_STEPS = 20


class TestPR1Integration:
    def test_system_jsonl_files_exist(self, sys_log_dir):
        """4 ranks → 4 per-rank JSONL files."""
        files = [f for f in os.listdir(sys_log_dir) if f.endswith(".jsonl")]
        assert len(files) == 4, f"Expected 4 per-rank files, got {len(files)}"

    def test_jsonl_has_4_column_structure(self, rank0_events):
        """Each JSONL record should have int/normal/double/normvector columns."""
        first = rank0_events[0]
        for key in ("int", "normal", "double", "normvector"):
            assert key in first, f"Missing '{key}' in JSONL structure"

    def test_jsonl_has_rank(self, rank0_events):
        assert "rank" in rank0_events[0]["int"]

    def test_jsonl_has_source(self, rank0_events):
        assert rank0_events[0]["normal"]["source"] == "trainer"

    def test_record_span_produces_start_end(self, rank0_events):
        """record_span should emit _start and _end events."""
        event_types = [e["normal"].get("log_type_name") for e in rank0_events]
        assert "fwd_bwd_start" in event_types, "Missing fwd_bwd_start event"
        assert "fwd_bwd_end" in event_types, "Missing fwd_bwd_end event"
        assert "optim_start" in event_types, "Missing optim_start event"
        assert "optim_end" in event_types, "Missing optim_end event"

    def test_end_events_have_duration(self, rank0_events):
        """END events should have a positive duration value (ms)."""
        end_events = [
            e for e in rank0_events
            if "end" in str(e["normal"].get("log_type_name", ""))
        ]
        assert end_events, "No END events found"
        for e in end_events:
            assert "value" in e["double"], f"END event missing duration: {e}"
            assert e["double"]["value"] > 0, "Duration should be positive"

    def test_record_event_writes_metrics(self, rank0_events):
        """record_event should produce metric_value events for train.loss and train.grad_norm."""
        metric_events = [
            e for e in rank0_events
            if e["normal"].get("log_type_name") == "metric_value"
        ]
        assert len(metric_events) > 0, "No metric_value events from record_event"
        event_names = {e["normal"].get("event_name") for e in metric_events}
        assert "train.loss" in event_names
        assert "train.grad_norm" in event_names

    def test_step_context_in_events(self, rank0_events):
        """Events should have step numbers from set_step()."""
        events_with_step = [e for e in rank0_events if "step" in e["int"]]
        assert len(events_with_step) > 0, "No events have step context"
        steps = {e["int"]["step"] for e in events_with_step}
        assert 1 in steps, "Step 1 should be present"
        assert NUM_STEPS in steps, "Last step should be present"
