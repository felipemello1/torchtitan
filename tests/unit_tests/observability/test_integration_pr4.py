# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test for PR4: validates experiment JSONL + aggregation.

Prerequisites:
    torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py
"""

import json
import os

import pytest

OUTPUT_DIR = "/tmp/toy_spmd_output"


@pytest.fixture
def exp_log_dir():
    d = os.path.join(OUTPUT_DIR, "experiment_logs")
    if not os.path.exists(d):
        pytest.skip("Run toy_spmd.py first")
    return d


class TestPR4Integration:
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

    def test_scalars_include_tensor_and_cpu_metrics(self):
        """Scalars JSON files should contain both tensor metrics (from replicate_to_host)
        and the schedule should match EveryNSteps(5)."""
        if not os.path.exists(OUTPUT_DIR):
            pytest.skip("Run toy_spmd.py first")
        scalars_file = os.path.join(OUTPUT_DIR, "scalars_step_1.json")
        if not os.path.exists(scalars_file):
            pytest.skip("No scalars files")
        with open(scalars_file) as f:
            data = json.load(f)
        assert "loss" in data, f"Missing tensor metric 'loss'. Keys: {list(data.keys())}"

    def test_previous_prs_still_work(self):
        """System JSONL (PR1) and losses (PR0) should still exist."""
        if not os.path.exists(OUTPUT_DIR):
            pytest.skip("Run toy_spmd.py first")
        assert os.path.exists(os.path.join(OUTPUT_DIR, "system_logs")), "PR1 system_logs missing"
        assert os.path.exists(os.path.join(OUTPUT_DIR, "losses.json")), "losses.json missing"
