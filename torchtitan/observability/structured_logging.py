# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Structured JSONL logging for system metrics (phase timing, step context).

Each rank writes its own JSONL file independently — no cross-rank coordination.
The training process writes files; downstream tools (Grafana, DuckDB, Perfetto)
consume them.

Public API:
    record_span(name, event_type)  — context manager for timing phases
    record_event(metrics_dict)     — point-in-time scalar events
    init_observability(...)        — one-time setup (creates file handlers)

Step context (set_step, add_step_tag, clear_step_tags) is in common.py.
"""

import enum
import itertools
import json
import logging
import os
import socket
import threading
import time
from contextlib import ContextDecorator
from timeit import default_timer as timer
from typing import Any

from torchtitan.observability.common import (
    _STEP,
    _STEP_TAGS,
    EXPERIMENT_LOGGER_NAME,
    SYSTEM_LOGGER_NAME,
)
from torchtitan.observability.metrics import (
    ExperimentJSONFormatter,
    ExperimentLoggingHandler,
    MeanMetric,
    record_metric,
)

MAX_MESSAGE_SIZE: int = 1000

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StrEnum(enum.Enum):
    """String-valued enum. str(member) returns the value, not 'Class.name'."""

    def __str__(self) -> str:
        return self.value


class EventType(StrEnum):
    """Phase identifiers for structured JSONL events.

    Each phase has a base event plus _START and _END variants.
    record_span auto-derives _START/_END from the base.
    """
    BINARY_START = "binary_start"
    TORCH_DISTRIBUTED_INIT = "torch_distributed_init"
    TORCH_DISTRIBUTED_INIT_START = "torch_distributed_init_start"
    TORCH_DISTRIBUTED_INIT_END = "torch_distributed_init_end"
    TORCH_DISTRIBUTED_TEARDOWN = "torch_distributed_teardown"
    TORCH_DISTRIBUTED_TEARDOWN_START = "torch_distributed_teardown_start"
    TORCH_DISTRIBUTED_TEARDOWN_END = "torch_distributed_teardown_end"
    MODEL_PARALLELISM_INIT = "model_parallelism_init"
    MODEL_PARALLELISM_INIT_START = "model_parallelism_init_start"
    MODEL_PARALLELISM_INIT_END = "model_parallelism_init_end"
    TOKENIZER_INIT = "tokenizer_init"
    TOKENIZER_INIT_START = "tokenizer_init_start"
    TOKENIZER_INIT_END = "tokenizer_init_end"
    DATA_ITERATOR_INIT = "data_iterator_init"
    DATA_ITERATOR_INIT_START = "data_iterator_init_start"
    DATA_ITERATOR_INIT_END = "data_iterator_init_end"
    RELOAD_DATA_LOADER_STATE = "reload_data_loader_state"
    RELOAD_DATA_LOADER_STATE_START = "reload_data_loader_state_start"
    RELOAD_DATA_LOADER_STATE_END = "reload_data_loader_state_end"
    BUILD_MODEL = "build_model"
    BUILD_MODEL_START = "build_model_start"
    BUILD_MODEL_END = "build_model_end"
    BUILD_LEARNER = "build_learner"
    BUILD_LEARNER_START = "build_learner_start"
    BUILD_LEARNER_END = "build_learner_end"
    OPTIMIZER_INIT = "optimizer_init"
    OPTIMIZER_INIT_START = "optimizer_init_start"
    OPTIMIZER_INIT_END = "optimizer_init_end"
    TRAINING_START = "training_start"
    STEP = "step"
    STEP_START = "step_start"
    STEP_END = "step_end"
    FETCHING_BATCH = "fetching_batch"
    FETCHING_BATCH_START = "fetching_batch_start"
    FETCHING_BATCH_END = "fetching_batch_end"
    FWD_BWD = "fwd_bwd"
    FWD_BWD_START = "fwd_bwd_start"
    FWD_BWD_END = "fwd_bwd_end"
    OPTIM = "optim"
    OPTIM_START = "optim_start"
    OPTIM_END = "optim_end"
    CHECKPOINT = "checkpoint"
    CHECKPOINT_INIT = "checkpoint_init"
    CHECKPOINT_INIT_START = "checkpoint_init_start"
    CHECKPOINT_INIT_END = "checkpoint_init_end"
    CHECKPOINT_START = "checkpoint_start"
    CHECKPOINT_END = "checkpoint_end"
    CHECKPOINT_STAGE = "checkpoint_stage"
    CHECKPOINT_STAGE_START = "checkpoint_stage_start"
    CHECKPOINT_STAGE_END = "checkpoint_stage_end"
    CHECKPOINT_LOAD = "checkpoint_load"
    CHECKPOINT_LOAD_START = "checkpoint_load_start"
    CHECKPOINT_LOAD_END = "checkpoint_load_end"
    EVAL_LAUNCH = "eval_launch"
    EVAL_LAUNCH_START = "eval_launch_start"
    EVAL_LAUNCH_END = "eval_launch_end"
    EVAL = "eval"
    EVAL_START = "eval_start"
    EVAL_END = "eval_end"
    RL_SCORING = "rl_scoring"
    RL_SCORING_START = "rl_scoring_start"
    RL_SCORING_END = "rl_scoring_end"
    SUMMARY_WRITER = "summary_writer"
    SUMMARY_WRITER_START = "summary_writer_start"
    SUMMARY_WRITER_END = "summary_writer_end"
    TORCH_MEMORY_BREAKDOWN = "torch_memory_breakdown"
    TORCH_MEMORY_BREAKDOWN_START = "torch_memory_breakdown_start"
    TORCH_MEMORY_BREAKDOWN_END = "torch_memory_breakdown_end"
    GC_COLLECT = "gc_collect"
    GC_COLLECT_START = "gc_collect_start"
    GC_COLLECT_END = "gc_collect_end"
    METRIC_VALUE = "metric_value"
    STATE_DICT_INIT = "state_dict_init"
    STATE_DICT_INIT_START = "state_dict_init_start"
    STATE_DICT_INIT_END = "state_dict_init_end"
    STATE_DICT_LOAD = "state_dict_load"
    STATE_DICT_LOAD_START = "state_dict_load_start"
    STATE_DICT_LOAD_END = "state_dict_load_end"


class LogType(StrEnum):
    """Distinguishes event records from free-text log records in JSONL."""
    EVENT = "event"
    TEXT = "text"


class ExtraFields(StrEnum):
    """Keys for the `extra` dict passed to logging calls."""
    LOG_TYPE = "log_type"
    LOG_TYPE_NAME = "log_type_name"
    EVENT_NAME = "event_name"
    STEP = "step"
    RELATIVE_STEP = "relative_step"
    CONTEXT = "context"
    VALUE = "value"


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def dict_to_str_list(d: dict[str, str] | None) -> list[str] | None:
    """Convert a dict to a list of "key:value" strings for JSONL normvector field."""
    if d is None:
        return None
    try:
        return [f"{k}:{v}" for k, v in d.items()]
    except Exception:
        return None


def event_extra(
    event_type: EventType,
    event_name: str | None = None,
    step: int | None = None,
    relative_step: int | None = None,
    value: float | int | None = None,
    context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the extra dict for a structured JSONL event record."""
    return {
        str(ExtraFields.LOG_TYPE): str(LogType.EVENT),
        str(ExtraFields.LOG_TYPE_NAME): str(event_type),
        str(ExtraFields.EVENT_NAME): event_name,
        str(ExtraFields.STEP): step,
        str(ExtraFields.RELATIVE_STEP): relative_step,
        str(ExtraFields.VALUE): value,
        str(ExtraFields.CONTEXT): dict_to_str_list(context),
    }



# ---------------------------------------------------------------------------
# JSONL formatting
# ---------------------------------------------------------------------------


def to_structured_json(log_dict: dict[str, Any]) -> str:
    """Convert a log dict to 4-column JSON (int/normal/double/normvector)."""
    int_dict: dict[str, int] = {}
    str_dict: dict[str, str] = {}
    float_dict: dict[str, float] = {}
    vector_str_dict: dict[str, list[str]] = {}

    for k, v in log_dict.items():
        if v is None:
            continue
        if isinstance(v, bool):
            int_dict[k] = int(v)
        elif isinstance(v, int):
            int_dict[k] = v
        elif isinstance(v, str):
            str_dict[k] = v
        elif isinstance(v, float):
            float_dict[k] = v
        elif isinstance(v, (list, tuple)):
            vector_str_dict[k] = [str(e) for e in v]

    structured_dict = {
        "int": int_dict,
        "normal": str_dict,
        "double": float_dict,
        "normvector": vector_str_dict,
    }
    return json.dumps(structured_dict)


# ---------------------------------------------------------------------------
# StructuredJSONFormatter
# ---------------------------------------------------------------------------


class StructuredJSONFormatter(logging.Formatter):
    """Formats system log records as structured JSONL (4-column format).

    Each record becomes a JSON line with int/normal/double/normvector columns.
    Rank and source are constants (set once). Step and step_tags come from
    ContextVars so concurrent async tasks each see their own step.

    Example output (one JSON line):
        {"int": {"rank": 0, "step": 5, "time": 1709500000, "seq_id": 42},
         "normal": {"source": "trainer", "log_type_name": "fwd_bwd_end", ...},
         "double": {"value": 12.5, "delta_ms": 0.3},
         "normvector": {}}

    Usage:
        formatter = StructuredJSONFormatter(rank=0, source="trainer")
        handler.setFormatter(formatter)
    """

    _thread_local = threading.local()

    def __init__(self, rank: int, source: str):
        super().__init__()
        self.rank = rank
        self.source = source
        self._seq_counter = itertools.count()

    def format(self, record: logging.LogRecord) -> str:
        return to_structured_json(self._log_dict(record))

    def _log_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        """Build the flat dict that to_structured_json splits into 4 columns.

        Adds: rank, source, step, step_tags, timing, log_type, event_name,
        value, message, caller info, seq_id.
        """
        log_dict: dict[str, Any] = {}

        log_dict["delta_ms"] = self._refresh_event_delta()
        log_dict["tid"] = threading.get_native_id()
        log_dict["thread_time_ns"] = time.thread_time_ns()

        # Rank/source from self (constants, set once in init_observability)
        log_dict["rank"] = self.rank
        log_dict["source"] = self.source
        log_dict["host_name"] = socket.gethostname()
        log_dict["pid"] = os.getpid()

        # Step/step_tags from ContextVar (mutable, changes every step)
        step = _STEP.get()
        if step is not None:
            log_dict["step"] = step
        step_tags = _STEP_TAGS.get()
        if step_tags:
            log_dict["step_tags"] = list(step_tags)

        log_dict["time"] = int(record.created)
        log_dict["time_ms"] = int(record.created * 1000)

        log_dict["log_type"] = getattr(
            record, str(ExtraFields.LOG_TYPE), str(LogType.TEXT)
        )
        log_dict["log_type_name"] = getattr(
            record, str(ExtraFields.LOG_TYPE_NAME), None
        )

        # Per-record step override (from event_extra)
        record_step = getattr(record, str(ExtraFields.STEP), None)
        if record_step is not None:
            log_dict["step"] = record_step

        relative_step = getattr(record, str(ExtraFields.RELATIVE_STEP), None)
        if relative_step is not None:
            log_dict["relative_step"] = relative_step

        log_dict["event_name"] = getattr(
            record, str(ExtraFields.EVENT_NAME), None
        )

        value = getattr(record, str(ExtraFields.VALUE), None)
        if isinstance(value, (float, int)):
            log_dict["value"] = float(value)

        # Per-record context normvector
        record_context = getattr(record, str(ExtraFields.CONTEXT), None)
        if record_context:
            log_dict["context"] = record_context

        log_dict["log_file"] = record.filename
        log_dict["log_function"] = record.funcName
        log_dict["log_level"] = record.levelname
        log_dict["logger_name"] = record.name
        log_dict["stack_info"] = record.stack_info

        log_dict["seq_id"] = next(self._seq_counter)

        message = record.getMessage()
        if message is not None:
            if len(message) <= MAX_MESSAGE_SIZE:
                log_dict["message"] = message
            else:
                half = MAX_MESSAGE_SIZE // 2
                log_dict["message"] = message[:half] + "..." + message[-half:]

        return log_dict

    def _refresh_event_delta(self) -> float:
        if not hasattr(self._thread_local, "last_event_time"):
            self._thread_local.last_event_time = timer()
        event_delta = (timer() - self._thread_local.last_event_time) * 1000
        self._thread_local.last_event_time = timer()
        return event_delta


# ---------------------------------------------------------------------------
# Filters and handlers
# ---------------------------------------------------------------------------


class EventsOnlyFilter(logging.Filter):
    """Filters logs, only passing events with LOG_TYPE_NAME set."""

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, str(ExtraFields.LOG_TYPE_NAME), None) is not None


class StructuredLoggingHandler(logging.FileHandler):
    """Writes structured JSONL events to per-rank files.

    Creates the output directory if needed. Only passes events
    (records with LOG_TYPE_NAME set) — free-text logs are filtered out.
    """

    def __init__(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        super().__init__(filename=filepath)
        self.addFilter(EventsOnlyFilter())


# ---------------------------------------------------------------------------
# Crash forensics
# ---------------------------------------------------------------------------


class InflightEventTrackingHandler(logging.Handler):
    """Tracks the last structured event for crash forensics.

    If the process crashes, the last event tells you what phase it was in.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_event: str | None = None
        self.last_event_time: float | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event_name = getattr(record, str(ExtraFields.LOG_TYPE_NAME), None)
            if event_name is not None:
                self.last_event = str(event_name)
                self.last_event_time = time.time()
        except Exception:
            return  # Crash forensics handler must never itself crash


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

_system_logger = logging.getLogger(SYSTEM_LOGGER_NAME)


def init_observability(
    source: str, output_dir: str, rank: int | None = None
) -> None:
    """Initialize structured logging. Called once per rank during process setup.

    Creates per-rank JSONL file handlers for both system metrics (phase timing,
    step events) and experiment metrics (record_metric entries).

    Rank and source are baked into the formatter as constants — they never
    change for the lifetime of the process. Step is set per-step via set_step().

    Idempotent: safe to call multiple times (skips if handler already exists).
    Can be called before torch.distributed init — rank defaults to the RANK
    environment variable (set by torchrun/MAST).

    Args:
        source: Component name (e.g., "trainer", "generator").
        output_dir: Root output directory.
        rank: Global rank. If None, reads from RANK env var (default 0).
    """
    if rank is None:
        rank = int(os.environ.get("RANK", 0))

    # --- System handler ---
    sys_logger = logging.getLogger(SYSTEM_LOGGER_NAME)
    if not any(isinstance(h, StructuredLoggingHandler) for h in sys_logger.handlers):
        sys_path = os.path.join(
            output_dir, "system_logs", f"{source}_rank_{rank}_system.jsonl"
        )
        handler = StructuredLoggingHandler(filepath=sys_path)
        handler.setFormatter(StructuredJSONFormatter(rank=rank, source=source))
        sys_logger.addHandler(handler)
        sys_logger.addHandler(InflightEventTrackingHandler())
        sys_logger.propagate = False
        if sys_logger.level == logging.NOTSET or sys_logger.level > logging.INFO:
            sys_logger.setLevel(logging.INFO)

    # --- Experiment handler ---
    exp_logger = logging.getLogger(EXPERIMENT_LOGGER_NAME)
    if not any(isinstance(h, ExperimentLoggingHandler) for h in exp_logger.handlers):
        exp_path = os.path.join(
            output_dir, "experiment_logs",
            f"{source}_rank_{rank}_experiment.jsonl",
        )
        handler = ExperimentLoggingHandler(filepath=exp_path)
        handler.setFormatter(ExperimentJSONFormatter(rank=rank, source=source))
        exp_logger.addHandler(handler)
    exp_logger.propagate = False
    if exp_logger.level == logging.NOTSET or exp_logger.level > logging.INFO:
        exp_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Public API: record_event, record_span
# ---------------------------------------------------------------------------


def record_event(metrics: dict[str, float | int]) -> None:
    """Log point-in-time metrics as structured events.

    Each key-value pair becomes a separate METRIC_VALUE event in JSONL.
    Step is read from ContextVar (set via set_step).
    """
    step = _STEP.get()
    for name, value in metrics.items():
        _system_logger.info(
            f"[step {step if step is not None else 'N/A'}] {name}={value}",
            extra=event_extra(EventType.METRIC_VALUE, event_name=name, value=value, step=step),
            stacklevel=2,
        )


class record_span(ContextDecorator):
    """Context manager/decorator for timing phases.

    Logs START event on enter, END event (with duration) on exit.
    When ``log_to_metrics=True`` (default), also records the duration as an
    experiment metric via ``record_metric``, eliminating the need for manual
    ``time.perf_counter()`` timing.

    Step is read from ContextVar.

    Args:
        description: Human-readable label (e.g., "Forward/Backward").
        event_type: Base EventType (e.g., EventType.FWD_BWD). Must have
            corresponding _START and _END variants.
        log_to_metrics: If True, record ``time/{description}/duration_s`` as
            a MeanMetric to experiment JSONL. Default True.

    Usage:
        with record_span("Forward/Backward", EventType.FWD_BWD):
            output = model(batch)
            loss.backward()
        # duration auto-recorded to both system JSONL and experiment JSONL
    """

    def __init__(self, description: str, event_type: EventType, *, log_to_metrics: bool = True):
        self.description = description
        self.event_type = event_type
        self.log_to_metrics = log_to_metrics
        self.start_time: float | None = None

        # Derive START/END event types. Validate that the base event type
        # is not itself a _START or _END variant — those cannot be used as
        # span types because there's no _start_start or _end_end.
        base_name = str(event_type)
        if base_name.endswith("_start") or base_name.endswith("_end"):
            raise ValueError(
                f"record_span requires a base EventType (e.g., EventType.FWD_BWD), "
                f"not a START/END variant. Got: {event_type}"
            )
        try:
            self.start_event_type = EventType(base_name + "_start")
            self.end_event_type = EventType(base_name + "_end")
        except ValueError:
            raise ValueError(
                f"EventType {event_type} has no _START/_END variants. "
                f"Add {base_name}_start and {base_name}_end to EventType."
            )

    def __enter__(self):
        self.start_time = timer()
        step = _STEP.get()
        _system_logger.info(
            f"[step {step if step is not None else 'N/A'}] {self.description} {self.start_event_type}",
            extra=event_extra(self.start_event_type, step=step),
            stacklevel=2,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = timer()
        step = _STEP.get()
        duration_s = end_time - self.start_time
        delta_ms = duration_s * 1000
        _system_logger.info(
            f"[step {step if step is not None else 'N/A'}] {self.description} {self.end_event_type} took {delta_ms:.2f} ms",
            extra=event_extra(self.end_event_type, value=delta_ms, step=step),
            stacklevel=2,
        )
        if self.log_to_metrics and step is not None:
            record_metric(f"time/{self.description}/duration_s", MeanMetric(sum=duration_s))
        return False  # Don't suppress exceptions
