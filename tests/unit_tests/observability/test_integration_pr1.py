# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test for PR1: validates toy_spmd.py outputs.

Prerequisites:
    torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py

This test reads the output directory and validates that PR1's structured
logging produces correct JSONL files.
"""

import json
import os

import pytest

OUTPUT_DIR = "/tmp/toy_spmd_output"


@pytest.fixture
def sys_log_dir():
    d = os.path.join(OUTPUT_DIR, "system_logs")
    if not os.path.exists(d):
        pytest.skip("Run toy_spmd.py first: torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py")
    return d


@pytest.fixture
def losses():
    p = os.path.join(OUTPUT_DIR, "losses.json")
    if not os.path.exists(p):
        pytest.skip("Run toy_spmd.py first")
    with open(p) as f:
        return json.load(f)


@pytest.fixture
def rank0_events(sys_log_dir):
    """Load all events from rank 0's system JSONL."""
    files = sorted(f for f in os.listdir(sys_log_dir) if f.endswith(".jsonl"))
    assert files, "No JSONL files found"
    with open(os.path.join(sys_log_dir, files[0])) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestPR1Integration:
    def test_loss_decreases(self, losses):
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"

    def test_system_jsonl_files_exist(self, sys_log_dir):
        files = [f for f in os.listdir(sys_log_dir) if f.endswith(".jsonl")]
        assert len(files) == 4, f"Expected 4 per-rank files, got {len(files)}"

    def test_jsonl_has_4_column_structure(self, rank0_events):
        first = rank0_events[0]
        for key in ("int", "normal", "double", "normvector"):
            assert key in first, f"Missing '{key}' in JSONL structure"

    def test_jsonl_has_rank(self, rank0_events):
        assert "rank" in rank0_events[0]["int"]

    def test_jsonl_has_source(self, rank0_events):
        assert rank0_events[0]["normal"]["source"] == "trainer"

    def test_record_span_produces_start_end(self, rank0_events):
        event_types = [e["normal"].get("log_type_name") for e in rank0_events]
        assert "fwd_bwd_start" in event_types, "Missing fwd_bwd_start event"
        assert "fwd_bwd_end" in event_types, "Missing fwd_bwd_end event"
        assert "optim_start" in event_types, "Missing optim_start event"
        assert "optim_end" in event_types, "Missing optim_end event"

    def test_end_events_have_duration(self, rank0_events):
        end_events = [
            e for e in rank0_events
            if "end" in str(e["normal"].get("log_type_name", ""))
        ]
        assert end_events, "No END events found"
        for e in end_events:
            assert "value" in e["double"], f"END event missing duration: {e}"
            assert e["double"]["value"] > 0, "Duration should be positive"

    def test_record_event_writes_metrics(self, rank0_events):
        metric_events = [
            e for e in rank0_events
            if e["normal"].get("log_type_name") == "metric_value"
        ]
        assert len(metric_events) > 0, "No metric_value events from record_event"
        # Check that train.loss and train.grad_norm are present
        event_names = {e["normal"].get("event_name") for e in metric_events}
        assert "train.loss" in event_names
        assert "train.grad_norm" in event_names

    def test_step_context_in_events(self, rank0_events):
        """Events should have step numbers from set_step()."""
        events_with_step = [e for e in rank0_events if "step" in e["int"]]
        assert len(events_with_step) > 0, "No events have step context"
        steps = {e["int"]["step"] for e in events_with_step}
        assert 1 in steps, "Step 1 should be present"
        assert NUM_STEPS in steps or 20 in steps, "Last step should be present"


NUM_STEPS = 20
