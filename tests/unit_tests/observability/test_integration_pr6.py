# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test for PR6: validates toy_rl.py outputs.

Prerequisites:
    python torchtitan/experiments/observability/toy_rl.py
"""

import json
import os

import pytest

OUTPUT_DIR = "/tmp/toy_rl_output"


@pytest.fixture
def output_exists():
    if not os.path.exists(OUTPUT_DIR):
        pytest.skip("Run toy_rl.py first")


class TestPR6Integration:
    def test_system_jsonl_for_all_actors(self, output_exists):
        sys_dir = os.path.join(OUTPUT_DIR, "system_logs")
        files = sorted(os.listdir(sys_dir))
        # Should have: controller, reward, 4 trainer ranks
        assert any("controller" in f for f in files)
        assert any("reward" in f for f in files)
        assert any("trainer" in f for f in files)
        assert len(files) >= 6

    def test_experiment_jsonl_for_all_actors(self, output_exists):
        exp_dir = os.path.join(OUTPUT_DIR, "experiment_logs")
        files = sorted(os.listdir(exp_dir))
        assert any("reward" in f for f in files), "Reward actor should write experiment JSONL"
        assert any("trainer" in f for f in files), "Trainer actor should write experiment JSONL"

    def test_rollout_jsonl_exists(self, output_exists):
        rollout_file = os.path.join(OUTPUT_DIR, "rollouts", "rollouts.jsonl")
        assert os.path.exists(rollout_file)
        with open(rollout_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) > 0
        assert "reward" in lines[0]
        assert "step" in lines[0]
        assert "policy_version" in lines[0]

    def test_tensor_metrics_in_experiment_jsonl(self, output_exists):
        """Tensor metrics (loss, grad_norm) should be written via log_reduced_metrics."""
        exp_dir = os.path.join(OUTPUT_DIR, "experiment_logs")
        trainer_files = [f for f in os.listdir(exp_dir) if "trainer_rank_0" in f]
        assert trainer_files, "No trainer rank 0 experiment JSONL"
        with open(os.path.join(exp_dir, trainer_files[0])) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        keys = {l.get("key") for l in lines if "key" in l}
        all_reduced = [l for l in lines if l.get("all_reduced")]
        assert all_reduced, "No all_reduced entries — tensor metrics not bridged to JSONL"

    def test_reward_metrics_in_experiment_jsonl(self, output_exists):
        """Reward actor should write reward/mean and reward/max via record_metric."""
        exp_dir = os.path.join(OUTPUT_DIR, "experiment_logs")
        reward_files = [f for f in os.listdir(exp_dir) if "reward" in f]
        assert reward_files
        with open(os.path.join(exp_dir, reward_files[0])) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        keys = {l.get("key") for l in lines if "key" in l}
        assert "reward/mean" in keys, f"Missing reward/mean. Keys: {keys}"
        assert "reward/max" in keys, f"Missing reward/max. Keys: {keys}"
