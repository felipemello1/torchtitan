# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for rollout_logger.py."""

import json
import os

import pytest

from torchtitan.observability.rollout_logger import (
    filter_top_bottom,
    RolloutLogger,
    RolloutOutput,
)


class TestRolloutOutput:
    def test_to_logging_dict_with_reward(self):
        r = RolloutOutput(
            prompt_tokens=[1, 2, 3],
            completion_tokens=[4, 5, 6],
            prompt_text="What is 2+2?",
            completion_text="The answer is 4.",
            reward=1.0,
        )
        d = r.to_logging_dict()
        assert d == {"prompt": "What is 2+2?", "completion": "The answer is 4.", "reward": 1.0}

    def test_to_logging_dict_without_reward(self):
        r = RolloutOutput(
            prompt_tokens=[1, 2],
            completion_tokens=[3, 4],
            prompt_text="hello",
            completion_text="world",
        )
        d = r.to_logging_dict()
        assert d == {"prompt": "hello", "completion": "world"}
        assert "reward" not in d

    def test_reward_can_be_set_after_init(self):
        r = RolloutOutput(
            prompt_tokens=[1],
            completion_tokens=[2],
            prompt_text="p",
            completion_text="c",
        )
        assert r.reward is None
        r.reward = 0.5
        assert r.to_logging_dict()["reward"] == 0.5


class TestRolloutLogger:
    def test_writes_jsonl(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        logger.log(
            [{"prompt": "hello", "reward": 0.5}, {"prompt": "world", "reward": 0.3}],
            metadata={"step": 1},
        )
        logger.close()

        with open(tmp_path / "rollouts.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["prompt"] == "hello"
        assert lines[0]["step"] == 1

    def test_metadata_merged_into_records(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        logger.log([{"x": 1}], metadata={"step": 42, "epoch": 3})
        logger.close()

        with open(tmp_path / "rollouts.jsonl") as f:
            record = json.loads(f.readline())
        assert record["step"] == 42
        assert record["epoch"] == 3
        assert record["x"] == 1

    def test_no_metadata(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        logger.log([{"x": 1}])
        logger.close()

        with open(tmp_path / "rollouts.jsonl") as f:
            record = json.loads(f.readline())
        assert record == {"x": 1}

    def test_empty_records_noop(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        logger.log([], metadata={"step": 1})
        logger.close()

        filepath = tmp_path / "rollouts.jsonl"
        assert filepath.stat().st_size == 0

    def test_filter_fn_applied(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        records = [{"id": i, "reward": i * 0.1} for i in range(10)]
        logger.log(records, metadata={"step": 1}, filter_fn=lambda r: r[:3])
        logger.close()

        with open(tmp_path / "rollouts.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 3

    def test_custom_filename(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path), filename="custom.jsonl")
        logger.log([{"x": 1}], metadata={"step": 1})
        logger.close()

        assert os.path.exists(tmp_path / "custom.jsonl")

    def test_append_across_calls(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        logger.log([{"step_data": "a"}], metadata={"step": 1})
        logger.log([{"step_data": "b"}], metadata={"step": 2})
        logger.close()

        with open(tmp_path / "rollouts.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2


class TestFilterTopBottom:
    def test_basic(self):
        records = [{"reward": i} for i in range(10)]
        result = filter_top_bottom(records, k=2)
        rewards = [r["reward"] for r in result]
        assert rewards == [0, 1, 8, 9]

    def test_small_list_returns_all(self):
        records = [{"reward": 1}, {"reward": 2}]
        result = filter_top_bottom(records, k=5)
        assert len(result) == 2

    def test_k_equals_half(self):
        records = [{"reward": i} for i in range(6)]
        result = filter_top_bottom(records, k=3)
        rewards = [r["reward"] for r in result]
        assert rewards == [0, 1, 2, 3, 4, 5]  # 3 bottom + 3 top = all 6

    def test_odd_list_no_duplicates(self):
        records = [{"reward": i} for i in range(3)]
        result = filter_top_bottom(records, k=5)
        # k >> len, k clamped to len//2 = 1, keeps bottom-1 + top-1
        assert len(result) == 2
        rewards = [r["reward"] for r in result]
        assert rewards == [0, 2]  # extremes kept, middle dropped

    def test_custom_key(self):
        records = [{"score": 3}, {"score": 1}, {"score": 2}]
        result = filter_top_bottom(records, key="score", k=1)
        scores = [r["score"] for r in result]
        assert scores == [1, 3]

    def test_single_element(self):
        records = [{"reward": 5}]
        result = filter_top_bottom(records, k=1)
        assert len(result) == 1
        assert result[0]["reward"] == 5

    def test_empty_list(self):
        result = filter_top_bottom([], k=1)
        assert result == []

    def test_missing_key_defaults_to_zero(self):
        records = [{"reward": 5}, {"no_reward": True}, {"reward": 3}]
        result = filter_top_bottom(records, k=1)
        assert len(result) == 2
