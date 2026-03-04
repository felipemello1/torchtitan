# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test: validates system JSONL outputs from toy_spmd.py (PR2).

Reads system JSONL files produced by running toy_spmd.py with 4 GPUs.
Checks that:
- System JSONL files exist (one per rank)
- record_span events have correct structure (FWD_BWD, OPTIM)
- record_event metric values are present (train.loss, train.grad_norm)
- Loss decreases over training (verifies learning)

Prerequisites:
    torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd
"""

import json
import os

# Output dir matches toy_spmd.py:
# OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "toy_spmd")
OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "torchtitan", "experiments", "observability", "outputs", "toy_spmd",
))

SYS_LOG_DIR = os.path.join(OUTPUT_DIR, "system_logs")


def _load_system_records(rank: int = 0) -> list[dict]:
    """Load all system JSONL records for a given rank."""
    path = os.path.join(SYS_LOG_DIR, f"trainer_rank_{rank}_system.jsonl")
    assert os.path.exists(path), (
        f"System JSONL not found: {path}. "
        "Run: torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd"
    )
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestSystemJSONL:
    def test_system_jsonl_files_exist(self):
        """All 4 ranks should produce system JSONL files."""
        assert os.path.isdir(SYS_LOG_DIR), (
            f"system_logs dir not found: {SYS_LOG_DIR}. "
            "Run: torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd"
        )
        files = [f for f in os.listdir(SYS_LOG_DIR) if f.endswith(".jsonl")]
        assert len(files) == 4, f"Expected 4 system JSONL files (one per rank), got {len(files)}: {files}"

    def test_records_have_rank_and_source(self):
        records = _load_system_records(rank=0)
        assert len(records) > 0
        for r in records:
            assert r["int"]["rank"] == 0
            assert r["normal"]["source"] == "trainer"

    def test_fwd_bwd_spans_present(self):
        """record_span("Forward/Backward", EventType.FWD_BWD) should produce start+end events."""
        records = _load_system_records(rank=0)
        event_types = [r["normal"].get("log_type_name", "") for r in records]
        assert "fwd_bwd_start" in event_types, f"Missing fwd_bwd_start. Events: {set(event_types)}"
        assert "fwd_bwd_end" in event_types, f"Missing fwd_bwd_end. Events: {set(event_types)}"

    def test_optim_spans_present(self):
        records = _load_system_records(rank=0)
        event_types = [r["normal"].get("log_type_name", "") for r in records]
        assert "optim_start" in event_types
        assert "optim_end" in event_types

    def test_metric_value_events_present(self):
        """record_event should produce metric_value events with train.loss and train.grad_norm."""
        records = _load_system_records(rank=0)
        metric_events = [
            r for r in records
            if r["normal"].get("log_type_name") == "metric_value"
        ]
        assert len(metric_events) > 0, "No metric_value events found"
        event_names = {r["normal"].get("event_name") for r in metric_events}
        assert "train.loss" in event_names, f"Missing train.loss. Found: {event_names}"
        assert "train.grad_norm" in event_names, f"Missing train.grad_norm. Found: {event_names}"


class TestLossDecreases:
    def test_loss_decreases_over_training(self):
        """Verify the model actually learns — loss at end should be lower than start."""
        records = _load_system_records(rank=0)
        loss_events = [
            r for r in records
            if r["normal"].get("log_type_name") == "metric_value"
            and r["normal"].get("event_name") == "train.loss"
        ]
        assert len(loss_events) >= 2, f"Need at least 2 loss events, got {len(loss_events)}"
        first_loss = loss_events[0]["double"]["value"]
        last_loss = loss_events[-1]["double"]["value"]
        assert last_loss < first_loss, (
            f"Loss should decrease: first={first_loss:.4f}, last={last_loss:.4f}"
        )
