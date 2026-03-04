# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test: validates EveryNSteps schedule + writer lifecycle.

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
    rank0_file = sorted(f for f in os.listdir(sys_log_dir) if "rank_0" in f)[0]
    with open(os.path.join(sys_log_dir, rank0_file)) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestScheduleIntegration:
    def test_system_jsonl_still_works(self, sys_log_dir):
        files = [f for f in os.listdir(sys_log_dir) if f.endswith(".jsonl")]
        assert len(files) == 4

    def test_record_event_fires_every_step(self, rank0_events):
        """record_event fires every step (not gated by schedule)."""
        metric_events = [
            e for e in rank0_events
            if e["normal"].get("log_type_name") == "metric_value"
        ]
        assert len(metric_events) > 0
        steps = {e["int"]["step"] for e in metric_events if "step" in e["int"]}
        assert len(steps) == 20, f"Expected 20 steps, got {len(steps)}: {sorted(steps)}"

    def test_fwd_bwd_spans_every_step(self, rank0_events):
        """record_span fires every step."""
        fwd_bwd_starts = [
            e for e in rank0_events if e["normal"].get("log_type_name") == "fwd_bwd_start"
        ]
        assert len(fwd_bwd_starts) == 20


class TestScheduleUnit:
    def test_every_n_steps_basic(self):
        from torchtitan.observability import EveryNSteps
        schedule = EveryNSteps(every_n=5)
        assert schedule(5) is True
        assert schedule(10) is True
        assert schedule(3) is False

    def test_every_n_steps_with_additional(self):
        from torchtitan.observability import EveryNSteps
        schedule = EveryNSteps(every_n=5, additional_steps={1})
        assert schedule(1) is True
        assert schedule(2) is False
        assert schedule(5) is True
