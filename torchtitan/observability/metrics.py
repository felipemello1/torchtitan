# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Non-tensor experiment metrics: record_metric → JSONL → DefaultAggregator → backends.

record_metric is fire-and-forget. The formatter adds step (ContextVar) and
rank/source (self attributes) automatically.
"""

import glob
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from torchtitan.observability.common import (
    _METRIC_ENTRY,
    _REDUCED_METRICS,
    _STEP,
    EXPERIMENT_LOGGER_NAME,
)


# ---------------------------------------------------------------------------
# Non-tensor metric types (fire-and-forget). For tensor metrics, use TMetricValue.
# ---------------------------------------------------------------------------


class MetricValue(ABC):
    """Base for CPU metric types. Each knows its own state + reduction."""

    reduce_name: str

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return serializable state dict for JSONL. Must include 'reduce' key."""
        ...

    @classmethod
    @abstractmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        """Reduce multiple states (from multiple ranks/calls) to a single float."""
        ...


class MeanMetric(MetricValue):
    """Weighted mean. Stores sum AND weight, reduces as (Σsum)/(Σweight).

    Example:
        MeanMetric(sum=0.5)              # single value, weight=1
        MeanMetric(sum=3.0, weight=6)    # weighted: mean = 3.0 / 6 = 0.5
    """

    reduce_name = "MeanMetric"

    def __init__(self, *, sum: float, weight: float = 1.0) -> None:  # noqa: A002
        self._sum = sum
        self._weight = weight

    def get_state(self) -> dict[str, Any]:
        return {"reduce": self.reduce_name, "sum": self._sum, "weight": self._weight}

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        total_sum = sum(s["sum"] for s in states)
        total_weight = sum(s["weight"] for s in states)
        return total_sum / total_weight if total_weight > 0 else 0.0


class MaxMetric(MetricValue):
    """Maximum value across all entries."""

    reduce_name = "MaxMetric"

    def __init__(self, value: float) -> None:
        self._value = value

    def get_state(self) -> dict[str, Any]:
        return {"reduce": self.reduce_name, "value": self._value}

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        return max(s["value"] for s in states)


class MinMetric(MetricValue):
    """Minimum value across all entries."""

    reduce_name = "MinMetric"

    def __init__(self, value: float) -> None:
        self._value = value

    def get_state(self) -> dict[str, Any]:
        return {"reduce": self.reduce_name, "value": self._value}

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        return min(s["value"] for s in states)


class SumMetric(MetricValue):
    """Sum of values across all entries."""

    reduce_name = "SumMetric"

    def __init__(self, value: float) -> None:
        self._value = value

    def get_state(self) -> dict[str, Any]:
        return {"reduce": self.reduce_name, "value": self._value}

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        return sum(s["value"] for s in states)


# ---------------------------------------------------------------------------
# REDUCE_REGISTRY — extensible type registry for aggregation
# ---------------------------------------------------------------------------

REDUCE_REGISTRY: dict[str, type[MetricValue]] = {
    "MeanMetric": MeanMetric,
    "MaxMetric": MaxMetric,
    "MinMetric": MinMetric,
    "SumMetric": SumMetric,
}


# ---------------------------------------------------------------------------
# Experiment logger plumbing
# ---------------------------------------------------------------------------

_experiment_logger = logging.getLogger(EXPERIMENT_LOGGER_NAME)


# ---------------------------------------------------------------------------
# record_metric — fire-and-forget public API
# ---------------------------------------------------------------------------


def record_metric(key: str, value: MetricValue) -> None:
    """Record a CPU metric to experiment JSONL.

    Formatter adds step (ContextVar), rank/source (self) automatically.
    Caller file:line captured via stacklevel=2 (free — logging already does this).

    Args:
        key: Metric name (e.g., "reward", "learning_rate").
        value: MetricValue instance (MeanMetric, MaxMetric, etc.).

    Raises:
        ValueError: If step not set (call set_step first).
    """
    if _STEP.get() is None:
        raise ValueError("No step in context. Call set_step() before record_metric().")
    state = value.get_state()
    state["key"] = key
    _experiment_logger.info(
        "experiment_metric",
        extra={_METRIC_ENTRY: state},
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# log_reduced_metrics — bridge for tensor metrics → JSONL
# ---------------------------------------------------------------------------


def log_reduced_metrics(metrics: dict[str, float]) -> None:
    """Write already-reduced tensor metrics to experiment JSONL.

    Called after replicate_to_host() to persist tensor metrics in the
    same JSONL files as record_metric entries. The aggregator checks
    all_reduced=True and passes these values through.

    Step from ContextVar. Rank/source from formatter instance attributes.
    """
    if not metrics:
        return
    _experiment_logger.info(
        "reduced_metrics",
        extra={_REDUCED_METRICS: metrics},
    )


# ---------------------------------------------------------------------------
# ExperimentJSONFormatter
# ---------------------------------------------------------------------------


class ExperimentJSONFormatter(logging.Formatter):
    """Formats experiment metric records as flat JSONL.

    Rank/source from self (constants), step from ContextVar (per-task).
    rank/source on self (set once), step from ContextVar (per-task).
    Handles both record_metric entries and log_reduced_metrics entries.
    """

    def __init__(self, rank: int, source: str):
        super().__init__()
        self.rank = rank
        self.source = source

    def format(self, record: logging.LogRecord) -> str:
        step = _STEP.get()
        if step is None:
            raise ValueError(
                "No step in context. Call set_step() before record_metric()."
            )

        # Handle record_metric entries (copy to avoid mutating shared state)
        state = getattr(record, _METRIC_ENTRY, None)
        if state is not None:
            state = dict(state)
            state["step"] = step
            state["rank"] = self.rank
            state["source"] = self.source
            state["caller"] = f"{record.filename}:{record.lineno}"
            state["timestamp"] = time.time()
            return json.dumps(state)

        # Handle log_reduced_metrics entries (all_reduced=True)
        reduced = getattr(record, _REDUCED_METRICS, None)
        if reduced is not None:
            lines = []
            for key, val in reduced.items():
                lines.append(json.dumps({
                    "key": key, "value": val, "all_reduced": True,
                    "step": step, "rank": self.rank, "source": self.source,
                    "timestamp": time.time(),
                }))
            return "\n".join(lines)

        return ""


# ---------------------------------------------------------------------------
# ExperimentMetricsFilter + ExperimentLoggingHandler
# ---------------------------------------------------------------------------


class ExperimentMetricsFilter(logging.Filter):
    """Only pass records with metric or reduced_metrics data."""

    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, _METRIC_ENTRY) or hasattr(record, _REDUCED_METRICS)


class ExperimentLoggingHandler(logging.FileHandler):
    """Writes experiment metrics to per-rank JSONL files."""

    def __init__(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        super().__init__(filename=filepath)
        self.addFilter(ExperimentMetricsFilter())


# ---------------------------------------------------------------------------
# DefaultAggregator — reads JSONL, buffers by step, reduces via registry
# ---------------------------------------------------------------------------


class DefaultAggregator:
    """Buffered file-offset tracking aggregator.

    Reads all new lines from experiment JSONL files, buffers by step.
    Handles two types of entries:
    - all_reduced=True: pass through (tensor metrics from replicate_to_host)
    - all_reduced absent/False: reduce via REDUCE_REGISTRY

    On construction, skips to end of existing files (no historical data).
    """

    def __init__(self, experiment_log_dir: str):
        self._log_dir = experiment_log_dir
        self._file_offsets: dict[str, int] = {}
        self._pending: dict[int, list[dict]] = defaultdict(list)
        # Skip historical data
        for fp in glob.glob(os.path.join(self._log_dir, "*.jsonl")):
            self._file_offsets[fp] = os.path.getsize(fp)

    def collect_and_aggregate(self, step: int) -> dict[str, float]:
        """Read new JSONL lines, aggregate entries for the given step.

        Returns:
            Dict of metric_key → reduced float value.
        """
        self._read_new_lines()
        entries = self._pending.pop(step, [])
        # Purge entries for steps older than current — they will never be
        # requested since steps are monotonically increasing.
        stale = [s for s in self._pending if s < step]
        for s in stale:
            del self._pending[s]
        if not entries:
            return {}

        result: dict[str, float] = {}
        by_key: dict[str, list[dict]] = defaultdict(list)

        for entry in entries:
            if entry.get("all_reduced"):
                result[entry["key"]] = entry["value"]
            else:
                by_key[entry["key"]].append(entry)

        for key, key_entries in by_key.items():
            reduce_name = key_entries[0]["reduce"]
            if not all(e["reduce"] == reduce_name for e in key_entries):
                raise ValueError(f"Key '{key}' has mixed reduce types")
            cls = REDUCE_REGISTRY.get(reduce_name)
            if cls is None:
                raise ValueError(f"Unknown reduce type '{reduce_name}' for key '{key}'")
            result[key] = cls.get_reduced_value_from_states(key_entries)

        return result

    def _read_new_lines(self) -> None:
        """Read new lines from all JSONL files since last offset."""
        for fp in glob.glob(os.path.join(self._log_dir, "*.jsonl")):
            offset = self._file_offsets.get(fp, 0)
            try:
                with open(fp) as f:
                    f.seek(offset)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        step = entry.get("step")
                        if step is not None:
                            self._pending[step].append(entry)
                    self._file_offsets[fp] = f.tell()
            except FileNotFoundError:
                continue
