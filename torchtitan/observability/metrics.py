# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment metrics: record_metric, MetricValue types, JSONL formatter."""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from torchtitan.observability._constants import _METRIC_ENTRY, EXPERIMENT_LOGGER_NAME
from torchtitan.observability.step_state import _STEP


# ---------------------------------------------------------------------------
# MetricValue types
# ---------------------------------------------------------------------------


class MetricValue(ABC):
    """Base for metric types. Each subclass defines serialization + reduction.

    Example JSONL output (from MeanMetric)::

        {"key": "reward", "reduce": "MeanMetric", "sum": 7.2, "weight": 10,
         "step": 42, "rank": 0, "source": "reward", ...}
    """

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
    """Weighted mean: (Σsum) / (Σweight).

    Example::

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


class NoOpMetric(MetricValue):
    """Passthrough for already-reduced values (e.g., loss after dist_sum, lr).

    Example::

        record_metric("loss/trainer_loss_mean", NoOpMetric(value=0.038))
        record_metric("trainer_schedule/lr", NoOpMetric(value=3e-4))
    """

    reduce_name = "NoOpMetric"

    def __init__(self, value: float) -> None:
        self._value = value

    def get_state(self) -> dict[str, Any]:
        return {"reduce": self.reduce_name, "value": self._value}

    @classmethod
    def get_reduced_value_from_states(cls, states: list[dict[str, Any]]) -> float:
        return states[0]["value"]


# ---------------------------------------------------------------------------
# REDUCE_REGISTRY — extensible type registry for aggregation
# ---------------------------------------------------------------------------

REDUCE_REGISTRY: dict[str, type[MetricValue]] = {
    cls.reduce_name: cls
    for cls in (MeanMetric, MaxMetric, MinMetric, SumMetric, NoOpMetric)
}


_experiment_logger = logging.getLogger(EXPERIMENT_LOGGER_NAME)


# ---------------------------------------------------------------------------
# record_metric — fire-and-forget public API
# ---------------------------------------------------------------------------


def record_metric(key: str, value: MetricValue, _stacklevel: int = 2) -> None:
    """Record a metric to experiment JSONL.

    Formatter adds step (ContextVar), rank/source automatically.

    Example::

        record_metric("trainer_gradient/norm_max", MaxMetric(value=12.3))

    Args:
        key: Metric name (e.g., "loss/trainer_loss_mean").
        value: MetricValue instance (MeanMetric, MaxMetric, NoOpMetric, etc.).
        _stacklevel: For internal callers (e.g., record_span) to report the
            correct caller location. Default 2 (direct caller).
    """
    if _STEP.get() is None:
        raise ValueError("No step in context. Call set_step() before record_metric().")
    state = value.get_state()
    state["key"] = key
    _experiment_logger.info(
        "experiment_metric",
        extra={_METRIC_ENTRY: state},
        stacklevel=_stacklevel,
    )


# ---------------------------------------------------------------------------
# ExperimentJSONFormatter + Handler
# ---------------------------------------------------------------------------


class ExperimentJSONFormatter(logging.Formatter):
    """Formats experiment metric records as flat JSONL.

    Example output::

        {"key": "reward", "reduce": "MeanMetric", "sum": 7.2, "weight": 10,
         "step": 42, "rank": 0, "source": "reward",
         "caller": "torchtitan/trainer.py:542:train_step", "timestamp": 1708200121.7}
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

        state = getattr(record, _METRIC_ENTRY, None)
        if state is not None:
            state = dict(state)
            state["step"] = step
            state["rank"] = self.rank
            state["source"] = self.source
            state[
                "caller"
            ] = f"{os.path.relpath(record.pathname)}:{record.lineno}:{record.funcName}"
            state["timestamp"] = time.time()
            return json.dumps(state)

        return ""


class ExperimentMetricsFilter(logging.Filter):
    """Only pass records with metric data."""

    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, _METRIC_ENTRY)


class ExperimentLoggingHandler(logging.FileHandler):
    """Writes experiment metrics to per-rank JSONL files."""

    def __init__(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        super().__init__(filename=filepath)
        self.addFilter(ExperimentMetricsFilter())
