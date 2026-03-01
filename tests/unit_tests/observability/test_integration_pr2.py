# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test for PR2: validates tensor metric outputs from toy_spmd.py.

Prerequisites:
    torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py
"""

import json
import os

import pytest

OUTPUT_DIR = "/tmp/toy_spmd_output"


@pytest.fixture
def scalars_files():
    files = sorted(
        f for f in os.listdir(OUTPUT_DIR)
        if f.startswith("scalars_step_") and f.endswith(".json")
    ) if os.path.exists(OUTPUT_DIR) else []
    if not files:
        pytest.skip("Run toy_spmd.py first")
    return files


class TestPR2Integration:
    def test_scalars_files_exist(self, scalars_files):
        """replicate_to_host should produce scalar JSON files on logging steps."""
        assert len(scalars_files) >= 2, f"Expected multiple scalar files, got {len(scalars_files)}"

    def test_scalars_are_floats(self, scalars_files):
        """Each scalar file should contain a dict of string → float."""
        for f in scalars_files:
            with open(os.path.join(OUTPUT_DIR, f)) as fh:
                data = json.load(fh)
            assert isinstance(data, dict), f"{f}: expected dict, got {type(data)}"
            for key, val in data.items():
                assert isinstance(key, str), f"{f}: key {key!r} should be string"
                assert isinstance(val, (int, float)), f"{f}: value for {key!r} should be numeric, got {type(val)}"

    def test_loss_scalar_present(self, scalars_files):
        """The 'loss' key should be present (from record_tensor_metric)."""
        with open(os.path.join(OUTPUT_DIR, scalars_files[0])) as f:
            data = json.load(f)
        assert "loss" in data, f"Missing 'loss' key. Keys: {list(data.keys())}"

    def test_loss_scalar_is_reasonable(self, scalars_files):
        """Loss value should be a positive float (not NaN, not zero)."""
        with open(os.path.join(OUTPUT_DIR, scalars_files[0])) as f:
            data = json.load(f)
        loss = data["loss"]
        assert loss > 0, f"Loss should be positive, got {loss}"
        assert loss < 100, f"Loss suspiciously high: {loss}"

    def test_dtensor_metric_present(self, scalars_files):
        """DTensor metric (layer_0/w1_norm) should be present from child_context."""
        with open(os.path.join(OUTPUT_DIR, scalars_files[0])) as f:
            data = json.load(f)
        assert "layer_0/w1_norm" in data, f"Missing DTensor metric. Keys: {list(data.keys())}"
        assert data["layer_0/w1_norm"] > 0, "DTensor metric should be positive"

    def test_grad_norm_present(self, scalars_files):
        """grad_norm should be present (from record_tensor_metric with MaxTMetric)."""
        with open(os.path.join(OUTPUT_DIR, scalars_files[0])) as f:
            data = json.load(f)
        assert "grad_norm" in data, f"Missing grad_norm. Keys: {list(data.keys())}"

    def test_pr1_still_works(self):
        """System JSONL should still exist (PR1 not broken by PR2)."""
        sys_dir = os.path.join(OUTPUT_DIR, "system_logs")
        if not os.path.exists(sys_dir):
            pytest.skip("Run toy_spmd.py first")
        files = [f for f in os.listdir(sys_dir) if f.endswith(".jsonl")]
        assert len(files) == 4, f"Expected 4 system JSONL files, got {len(files)}"
