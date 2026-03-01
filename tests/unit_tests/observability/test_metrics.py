# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for metrics.py (PR4 — novel code).

Tests: MetricValue types, record_metric, log_reduced_metrics,
ExperimentJSONFormatter, DefaultAggregator, init_observability.
"""

import json
import logging
import os

import pytest

from torchtitan.observability.structured_logging import _STEP, set_step
from torchtitan.observability.structured_logging import init_observability
from torchtitan.observability.metrics import (
    DefaultAggregator,
    ExperimentJSONFormatter,
    ExperimentLoggingHandler,
    ExperimentMetricsFilter,
    log_reduced_metrics,
    MaxMetric,
    MeanMetric,
    MinMetric,
    record_metric,
    REDUCE_REGISTRY,
    SumMetric,
)


@pytest.fixture(autouse=True)
def reset_step():
    _STEP.set(None)
    yield
    _STEP.set(None)


@pytest.fixture
def exp_logger():
    """Provide a clean experiment logger."""
    from torchtitan.observability.structured_logging import EXPERIMENT_LOGGER_NAME
    logger = logging.getLogger(EXPERIMENT_LOGGER_NAME)
    original_handlers = logger.handlers[:]
    original_level = logger.level
    yield logger
    logger.handlers = original_handlers
    logger.level = original_level


# ---------------------------------------------------------------------------
# MetricValue types
# ---------------------------------------------------------------------------


class TestMeanMetric:
    def test_from_value(self):
        m = MeanMetric(value=2.5)
        state = m.get_state()
        assert state["reduce"] == "MeanMetric"
        assert state["sum"] == 2.5
        assert state["weight"] == 1.0

    def test_from_sum_weight(self):
        m = MeanMetric(sum=10.0, weight=4.0)
        state = m.get_state()
        assert state["sum"] == 10.0
        assert state["weight"] == 4.0

    def test_reduce(self):
        states = [
            {"sum": 6.0, "weight": 3.0},
            {"sum": 4.0, "weight": 2.0},
        ]
        result = MeanMetric.get_reduced_value_from_states(states)
        assert result == pytest.approx(2.0)  # 10/5

    def test_both_value_and_sum_raises(self):
        with pytest.raises(ValueError):
            MeanMetric(value=1.0, sum=2.0)


class TestMaxMetric:
    def test_basic(self):
        assert MaxMetric(5.0).get_state()["value"] == 5.0

    def test_reduce(self):
        states = [{"value": 3.0}, {"value": 7.0}, {"value": 1.0}]
        assert MaxMetric.get_reduced_value_from_states(states) == 7.0


class TestMinMetric:
    def test_reduce(self):
        states = [{"value": 3.0}, {"value": 7.0}, {"value": 1.0}]
        assert MinMetric.get_reduced_value_from_states(states) == 1.0


class TestSumMetric:
    def test_reduce(self):
        states = [{"value": 3.0}, {"value": 7.0}, {"value": 1.0}]
        assert SumMetric.get_reduced_value_from_states(states) == 11.0


class TestReduceRegistry:
    def test_all_types_registered(self):
        assert "MeanMetric" in REDUCE_REGISTRY
        assert "MaxMetric" in REDUCE_REGISTRY
        assert "MinMetric" in REDUCE_REGISTRY
        assert "SumMetric" in REDUCE_REGISTRY


# ---------------------------------------------------------------------------
# ExperimentJSONFormatter
# ---------------------------------------------------------------------------


class TestExperimentJSONFormatter:
    def test_formats_metric_entry(self):
        fmt = ExperimentJSONFormatter(rank=0, source="trainer")
        set_step(5)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=42, msg="experiment_metric", args=None, exc_info=None,
        )
        record._metric_entry = {"key": "reward", "reduce": "MaxMetric", "value": 0.95}
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["key"] == "reward"
        assert parsed["step"] == 5
        assert parsed["rank"] == 0
        assert parsed["source"] == "trainer"
        assert "caller" in parsed
        assert "timestamp" in parsed

    def test_formats_reduced_metrics(self):
        fmt = ExperimentJSONFormatter(rank=0, source="trainer")
        set_step(10)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="reduced_metrics", args=None, exc_info=None,
        )
        record._reduced_metrics = {"loss": 2.5, "grad_norm": 1.1}
        output = fmt.format(record)
        # Multiple lines (one per key)
        lines = [json.loads(line) for line in output.strip().split("\n")]
        assert len(lines) == 2
        keys = {l["key"] for l in lines}
        assert keys == {"loss", "grad_norm"}
        assert all(l["all_reduced"] is True for l in lines)

    def test_raises_without_step(self):
        fmt = ExperimentJSONFormatter(rank=0, source="trainer")
        # step not set
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test", args=None, exc_info=None,
        )
        record._metric_entry = {"key": "x", "reduce": "MaxMetric", "value": 1.0}
        with pytest.raises(ValueError, match="No step"):
            fmt.format(record)


# ---------------------------------------------------------------------------
# record_metric + log_reduced_metrics (end-to-end with file)
# ---------------------------------------------------------------------------


class TestRecordMetricEndToEnd:
    def test_writes_to_experiment_jsonl(self, tmp_path, exp_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(42)

        record_metric("reward", MaxMetric(0.95))
        record_metric("lr", MeanMetric(value=1e-4))

        # Flush
        for h in exp_logger.handlers:
            h.flush()

        exp_dir = tmp_path / "experiment_logs"
        jsonl_files = list(exp_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        with open(jsonl_files[0]) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        keys = {l["key"] for l in lines}
        assert keys == {"reward", "lr"}

    def test_log_reduced_metrics(self, tmp_path, exp_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(10)

        log_reduced_metrics({"loss": 2.5, "tps": 1000.0})

        for h in exp_logger.handlers:
            h.flush()

        exp_dir = tmp_path / "experiment_logs"
        jsonl_files = list(exp_dir.glob("*.jsonl"))
        with open(jsonl_files[0]) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        all_reduced = [l for l in lines if l.get("all_reduced")]
        assert len(all_reduced) == 2


# ---------------------------------------------------------------------------
# DefaultAggregator
# ---------------------------------------------------------------------------


class TestDefaultAggregator:
    def test_aggregates_from_jsonl(self, tmp_path):
        """Create aggregator first, then write JSONL, then aggregate."""
        log_dir = tmp_path / "experiment_logs"
        log_dir.mkdir()

        # Create aggregator BEFORE writing (simulates real usage: init at startup)
        agg = DefaultAggregator(experiment_log_dir=str(log_dir))

        # Write entries for step 5
        entries = [
            {"key": "reward", "reduce": "MaxMetric", "value": 0.8, "step": 5},
            {"key": "reward", "reduce": "MaxMetric", "value": 0.95, "step": 5},
            {"key": "loss", "reduce": "MeanMetric", "sum": 6.0, "weight": 3.0, "step": 5},
        ]
        with open(log_dir / "rank_0.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = agg.collect_and_aggregate(step=5)

        assert result["reward"] == pytest.approx(0.95)  # max
        assert result["loss"] == pytest.approx(2.0)  # 6/3

    def test_passes_through_all_reduced(self, tmp_path):
        log_dir = tmp_path / "experiment_logs"
        log_dir.mkdir()

        agg = DefaultAggregator(experiment_log_dir=str(log_dir))

        entries = [
            {"key": "loss", "value": 2.5, "all_reduced": True, "step": 10},
            {"key": "tps", "value": 1000.0, "all_reduced": True, "step": 10},
        ]
        with open(log_dir / "rank_0.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        result = agg.collect_and_aggregate(step=10)

        assert result["loss"] == 2.5
        assert result["tps"] == 1000.0

    def test_empty_step_returns_empty(self, tmp_path):
        log_dir = tmp_path / "experiment_logs"
        log_dir.mkdir()
        agg = DefaultAggregator(experiment_log_dir=str(log_dir))
        assert agg.collect_and_aggregate(step=99) == {}

    def test_mixed_reduce_types_raises(self, tmp_path):
        log_dir = tmp_path / "experiment_logs"
        log_dir.mkdir()

        agg = DefaultAggregator(experiment_log_dir=str(log_dir))

        entries = [
            {"key": "x", "reduce": "MaxMetric", "value": 1.0, "step": 1},
            {"key": "x", "reduce": "MinMetric", "value": 0.5, "step": 1},
        ]
        with open(log_dir / "rank_0.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        with pytest.raises(ValueError, match="mixed reduce types"):
            agg.collect_and_aggregate(step=1)

    def test_no_memory_leak_for_non_requested_steps(self, tmp_path):
        """Entries for steps older than the requested step should be purged."""
        log_dir = tmp_path / "experiment_logs"
        log_dir.mkdir()

        agg = DefaultAggregator(experiment_log_dir=str(log_dir))

        # Write entries for steps 1 through 20
        with open(log_dir / "rank_0.jsonl", "w") as f:
            for step in range(1, 21):
                f.write(json.dumps({"key": "lr", "reduce": "MeanMetric",
                                    "sum": 0.001, "weight": 1.0, "step": step}) + "\n")

        # Only aggregate step 10 — steps 1-9 should be purged
        result = agg.collect_and_aggregate(step=10)
        assert "lr" in result

        # Steps 1-9 should have been purged, steps 11-20 remain
        assert all(s > 10 for s in agg._pending), (
            f"Stale steps not purged: {sorted(agg._pending.keys())}"
        )
