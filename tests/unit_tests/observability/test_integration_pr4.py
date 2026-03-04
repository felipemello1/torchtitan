# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test: validates experiment JSONL + aggregation.

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
def exp_log_dir():
    d = os.path.join(OUTPUT_DIR, "experiment_logs")
    assert os.path.exists(d), (
        f"Output not found at {d}. Run first:\n"
        "  torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd"
    )
    return d


class TestExperimentJSONLIntegration:
    def test_experiment_jsonl_files_exist(self, exp_log_dir):
        """record_metric should produce per-rank experiment JSONL files."""
        files = [f for f in os.listdir(exp_log_dir) if f.endswith(".jsonl")]
        assert len(files) >= 1, "No experiment JSONL files found"

    def test_experiment_jsonl_has_cpu_metrics(self, exp_log_dir):
        """Experiment JSONL should contain record_metric entries."""
        files = [f for f in os.listdir(exp_log_dir) if f.endswith(".jsonl")]
        with open(os.path.join(exp_log_dir, files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) > 0, "Experiment JSONL is empty"
        keys = {entry.get("key") for entry in lines if "key" in entry}
        assert "learning_rate" in keys, f"Missing learning_rate. Keys: {keys}"
        assert "step_time_ms" in keys, f"Missing step_time_ms. Keys: {keys}"

    def test_experiment_jsonl_has_all_reduced(self, exp_log_dir):
        """log_reduced_metrics should produce entries with all_reduced=True."""
        files = [f for f in os.listdir(exp_log_dir) if f.endswith(".jsonl")]
        with open(os.path.join(exp_log_dir, files[0])) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        all_reduced = [entry for entry in lines if entry.get("all_reduced")]
        assert len(all_reduced) > 0, "No all_reduced entries from log_reduced_metrics"

    def test_system_jsonl_still_works(self):
        """System JSONL should still exist after adding experiment metrics."""
        sys_dir = os.path.join(OUTPUT_DIR, "system_logs")
        assert os.path.exists(sys_dir), f"system_logs missing at {sys_dir}"
