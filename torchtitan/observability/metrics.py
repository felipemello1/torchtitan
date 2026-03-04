# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Non-tensor experiment metrics: record_metric → JSONL → DefaultAggregator → backends.

Data flow:
    record_metric("reward", MeanMetric(sum=r))     fire-and-forget to JSONL
        → ExperimentJSONFormatter                   adds step/rank/source/caller
        → ExperimentLoggingHandler                  writes to per-rank JSONL
    DefaultAggregator.collect_and_aggregate(step)   reads new JSONL lines
        → REDUCE_REGISTRY                           reduces by key
        → dict[str, float]                          ready for SummaryWriter

record_metric is fire-and-forget. The formatter adds step (from ContextVar,
set via ``set_step()``) and rank/source (from formatter instance attributes,
set once in ``init_observability()``). ContextVars provide isolation between
concurrent asyncio tasks on Monarch actors — see common.py module docstring.
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
# Non-tensor metric types (fire-and-forget).
# For tensor metrics inside torch.compile, use TMetricValue (tensor_metrics.py).
# ---------------------------------------------------------------------------


class MetricValue(ABC):
    """Base for non-tensor metric types. Each subclass knows its own state + reduction.

    Subclasses define ``get_state()`` → dict for JSONL, and
    ``get_reduced_value_from_states(states)`` → float for aggregation.

    JSONL output example (from MeanMetric)::

        {"key": "reward", "reduce": "MeanMetric", "sum": 7.2, "weight": 10,
         "step": 42, "rank": 0, "source": "reward", ...}
    """

    reduce_name: str

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return serializable state dict for JSONL logging. Must include 'reduce' key."""
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
    cls.reduce_name: cls
    for cls in (MeanMetric, MaxMetric, MinMetric, SumMetric)
}


# Experiment logger — records are formatted as JSONL by ExperimentJSONFormatter
# and written to per-rank files by ExperimentLoggingHandler. Both are attached
# in init_observability() (structured_logging.py).
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

    Rank/source from ``self`` (set once in ``init_observability``).
    Step from ContextVar (set per-step via ``set_step()``).
    ``source`` identifies the component (e.g., "trainer", "reward").

    Handles two entry types:

    record_metric entry → one JSON line::

        {"key": "reward", "reduce": "MeanMetric", "sum": 7.2, "weight": 10,
         "step": 42, "rank": 0, "source": "reward", "caller": "reward_actor.py:42", ...}

    log_reduced_metrics entry → one JSON line per key::

        {"key": "loss", "value": 2.34, "all_reduced": true, "step": 42, ...}
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
    """Reads experiment JSONL files and reduces metrics by step.

    Usage::

        aggregator = DefaultAggregator(experiment_log_dir="output/experiment_logs")
        # ... training writes JSONL via record_metric / log_reduced_metrics ...
        scalars = aggregator.collect_and_aggregate(step=10)
        # scalars = {"loss": 2.34, "reward": 0.72, ...}
        writer(step=10, values=scalars)

    Handles two entry types:
    - ``all_reduced=True``: pass through (tensor metrics from replicate_to_host)
    - ``all_reduced`` absent: reduce via REDUCE_REGISTRY (e.g., MeanMetric → Σsum/Σweight)

    On construction, skips to end of existing files (no historical data).
    Stale steps (older than the requested step) are purged automatically.
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

        Reads all JSONL files in the log dir since the last call, groups entries
        by step, then reduces entries for the requested step. Entries for older
        steps are discarded (steps are monotonically increasing).

        Returns:
            Dict of metric_key → reduced float value. Empty dict if no entries
            for this step.
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
