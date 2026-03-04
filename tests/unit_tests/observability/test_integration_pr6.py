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
