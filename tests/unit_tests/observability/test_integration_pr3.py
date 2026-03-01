# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test for PR3: validates EveryNSteps schedule + writer lifecycle.

Prerequisites:
    torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py
"""

import json
import os

import pytest

OUTPUT_DIR = "/tmp/toy_spmd_output"


@pytest.fixture
def scalars_files():
    """Collect all scalars_step_N.json files."""
    if not os.path.exists(OUTPUT_DIR):
        pytest.skip("Run toy_spmd.py first")
    files = {}
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("scalars_step_") and f.endswith(".json"):
            step = int(f.replace("scalars_step_", "").replace(".json", ""))
            with open(os.path.join(OUTPUT_DIR, f)) as fh:
                files[step] = json.load(fh)
    if not files:
        pytest.skip("No scalars files found — run toy_spmd.py first")
    return files


class TestPR3Integration:
    def test_schedule_logged_on_correct_steps(self, scalars_files):
        """EveryNSteps(5) + additional_steps={1} → steps 1, 5, 10, 15, 20."""
        expected_steps = {1, 5, 10, 15, 20}
        actual_steps = set(scalars_files.keys())
        assert actual_steps == expected_steps, f"Expected {expected_steps}, got {actual_steps}"

    def test_scalars_are_floats(self, scalars_files):
        """All values in scalars should be Python floats."""
        for step, scalars in scalars_files.items():
            for key, val in scalars.items():
                assert isinstance(val, (int, float)), (
                    f"Step {step}, key {key}: expected numeric, got {type(val)}"
                )

    def test_scalars_have_loss_key(self, scalars_files):
        """The 'loss' metric should be present."""
        first_step = min(scalars_files.keys())
        assert "loss" in scalars_files[first_step]

    def test_pr1_and_pr2_still_work(self):
        """Previous PRs' outputs should still exist."""
        if not os.path.exists(OUTPUT_DIR):
            pytest.skip("Run toy_spmd.py first")
        assert os.path.exists(os.path.join(OUTPUT_DIR, "system_logs")), "PR1 system_logs missing"
        assert os.path.exists(os.path.join(OUTPUT_DIR, "losses.json")), "losses.json missing"
