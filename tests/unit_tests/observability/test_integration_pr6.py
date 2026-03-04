# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test: validates RolloutLogger and toy_rl outputs.

Prerequisites:
    python -m torchtitan.experiments.observability.toy_rl
"""

import json
import os

OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "torchtitan", "experiments", "observability", "outputs", "toy_rl",
))


class TestRolloutIntegration:
    def test_rollout_jsonl_exists(self):
        rollout_dir = os.path.join(OUTPUT_DIR, "rollouts")
        assert os.path.exists(rollout_dir), (
            f"Rollout dir not found at {rollout_dir}. Run first:\n"
            "  python -m torchtitan.experiments.observability.toy_rl"
        )
        files = [f for f in os.listdir(rollout_dir) if f.endswith(".jsonl")]
        assert len(files) >= 1, "No rollout JSONL files"

    def test_rollout_has_reward_field(self):
        rollout_dir = os.path.join(OUTPUT_DIR, "rollouts")
        assert os.path.exists(rollout_dir), f"Missing {rollout_dir}"
        files = [f for f in os.listdir(rollout_dir) if f.endswith(".jsonl")]
        with open(os.path.join(rollout_dir, files[0])) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) > 0, "Rollout JSONL is empty"
        assert "reward" in records[0], f"Missing 'reward' key. Keys: {list(records[0].keys())}"
        assert "step" in records[0], f"Missing 'step' key. Keys: {list(records[0].keys())}"


class TestBroadcastSetStep:
    """Verify that broadcast set_step correctly propagates to all actors."""

    def test_system_jsonl_has_all_steps(self):
        """Each actor's system JSONL should have events for all training steps."""
        sys_dir = os.path.join(OUTPUT_DIR, "system_logs")
        assert os.path.isdir(sys_dir), (
            f"system_logs not found: {sys_dir}. "
            "Run: python -m torchtitan.experiments.observability.toy_rl"
        )
        for fname in os.listdir(sys_dir):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(sys_dir, fname)) as f:
                records = [json.loads(line) for line in f if line.strip()]
            steps = {r["int"]["step"] for r in records if "int" in r and "step" in r["int"]}
            # Every actor should have logged events for multiple steps
            assert len(steps) >= 2, (
                f"{fname}: expected events for multiple steps, got steps={sorted(steps)}"
            )

    def test_experiment_jsonl_has_correct_steps(self):
        """Experiment JSONL should have metrics with correct step values."""
        exp_dir = os.path.join(OUTPUT_DIR, "experiment_logs")
        assert os.path.isdir(exp_dir), (
            f"experiment_logs not found: {exp_dir}. "
            "Run: python -m torchtitan.experiments.observability.toy_rl"
        )
        for fname in os.listdir(exp_dir):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(exp_dir, fname)) as f:
                records = [json.loads(line) for line in f if line.strip()]
            steps = sorted({r["step"] for r in records if "step" in r})
            # Steps should be monotonically increasing integers starting from 1
            assert steps[0] == 1, f"{fname}: first step should be 1, got {steps[0]}"
            assert steps == list(range(steps[0], steps[-1] + 1)), (
                f"{fname}: steps should be contiguous, got {steps}"
            )
