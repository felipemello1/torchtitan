# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Structured JSONL logging for system metrics (phase timing, step context).

Ported from reference libraries with adaptations. Changes tagged with reason codes:
  META_REMOVAL — removed Meta-internal code
  MONARCH_COMPAT — adapted for Monarch async (ContextVar for step)
  RENAME — renamed internal names to OSS equivalents
  OSS_COMPAT — adapted for OSS (explicit args instead of env vars)
  IMPROVEMENT — minor enhancement over reference
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
from contextvars import ContextVar
from timeit import default_timer as timer
from typing import Any

# ---------------------------------------------------------------------------
# ContextVars for mutable per-task state (MONARCH_COMPAT — decision_003)
# Rank/source are constants stored on the formatter.
# Step/step_tags are mutable, need ContextVar for async safety.
# ---------------------------------------------------------------------------
_STEP: ContextVar[int | None] = ContextVar("_STEP", default=None)
_STEP_TAGS: ContextVar[tuple[str, ...]] = ContextVar("_STEP_TAGS", default=())

# Logger names
SYSTEM_LOGGER_NAME = "torchtitan.observability.system"
EXPERIMENT_LOGGER_NAME = "torchtitan.observability.experiment"

# ---------------------------------------------------------------------------
# StrEnum — ported from reference
# ---------------------------------------------------------------------------


class StrEnum(enum.Enum):
    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# EventType (CHANGED — META_REMOVAL: removed SDC, Triton cache, per-rank
# cache, replicate consistency check entries)
# ---------------------------------------------------------------------------


class EventType(StrEnum):
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


# ---------------------------------------------------------------------------
# LogType, ExtraFields — ported from reference
# ---------------------------------------------------------------------------


class LogType(StrEnum):
    EVENT = "event"
    TEXT = "text"


class ExtraFields(StrEnum):
    LOG_TYPE = "log_type"
    LOG_TYPE_NAME = "log_type_name"
    EVENT_NAME = "event_name"
    STEP = "step"
    RELATIVE_STEP = "relative_step"
    CONTEXT = "context"
    VALUE = "value"


# ---------------------------------------------------------------------------
# dict_to_list_safe — ported from reference
# ---------------------------------------------------------------------------


def dict_to_list_safe(d: dict[str, str] | None) -> list[str] | None:
    if d is None:
        return None
    try:
        return [f"{k}:{v}" for k, v in d.items()]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# event_extra — ported from reference
# ---------------------------------------------------------------------------


def event_extra(
    event_type: EventType,
    event_name: str | None = None,
    step: int | None = None,
    relative_step: int | None = None,
    value: float | int | None = None,
    context: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        str(ExtraFields.LOG_TYPE): str(LogType.EVENT),
        str(ExtraFields.LOG_TYPE_NAME): str(event_type),
        str(ExtraFields.EVENT_NAME): event_name,
        str(ExtraFields.STEP): step,
        str(ExtraFields.RELATIVE_STEP): relative_step,
        str(ExtraFields.VALUE): value,
        str(ExtraFields.CONTEXT): dict_to_list_safe(context),
    }


# ---------------------------------------------------------------------------
# set_step / add_step_tag / clear_step_tags
# (CHANGED — MONARCH_COMPAT: ContextVar instead of global mutable)
# ---------------------------------------------------------------------------


def set_step(step: int) -> None:
    """Set the current training step in the ContextVar.

    Called once per step/endpoint. Rank/source are constants on the formatter,
    so only step needs updating per-task.
    """
    _STEP.set(step)


def add_step_tag(tag: str) -> None:
    """Add a tag to the current step. Tags are immutable tuples (ContextVar safe).

    Uses tuple[str, ...] instead of list to avoid the mutable-list-in-ContextVar
    footgun: ContextVar copies the reference, so a mutable list would leak across
    asyncio tasks.
    """
    current = _STEP_TAGS.get()
    if tag not in current:
        _STEP_TAGS.set(current + (tag,))


def clear_step_tags() -> None:
    """Clear all step tags. Call at step boundaries."""
    _STEP_TAGS.set(())


# ---------------------------------------------------------------------------
# to_structured_json (RENAME + IMPROVEMENT)
# Based on reference handler. Added: None-skipping, bool→int, tuple support.
# ---------------------------------------------------------------------------

MAX_MESSAGE_SIZE: int = 1000


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
# (RENAME + MONARCH_COMPAT + decision_003: rank/source on self, step from ContextVar)
# (CHANGED — MONARCH_COMPAT + decision_003: rank/source on self, step from ContextVar)
# ---------------------------------------------------------------------------


class StructuredJSONFormatter(logging.Formatter):
    """Formats system log records as structured JSONL.

    Rank/source are instance attributes (constants, set once in init_observability).
    Step/step_tags are read from ContextVars (per-task, mutable).
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
        log_dict: dict[str, Any] = {}

        log_dict["delta_ms"] = self._refresh_event_delta()
        log_dict["tid"] = threading.get_native_id()
        log_dict["thread_time_ns"] = time.thread_time_ns()

        # Rank/source from self (constants — decision_003)
        log_dict["rank"] = self.rank
        log_dict["source"] = self.source
        log_dict["host_name"] = socket.gethostname()
        log_dict["pid"] = os.getpid()

        # Step/step_tags from ContextVar (mutable per-task — MONARCH_COMPAT)
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
# EventsOnlyFilter — ported from reference
# ---------------------------------------------------------------------------


class EventsOnlyFilter(logging.Filter):
    """Filters logs, only passing events with LOG_TYPE_NAME set."""

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, str(ExtraFields.LOG_TYPE_NAME), None) is not None


# ---------------------------------------------------------------------------
# StructuredLoggingHandler
# (CHANGED — OSS_COMPAT: filepath from init_observability, not hardcoded /logs)
# ---------------------------------------------------------------------------


class StructuredLoggingHandler(logging.FileHandler):
    """Writes structured JSONL to per-rank files."""

    def __init__(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        super().__init__(filename=filepath)
        self.addFilter(EventsOnlyFilter())


# ---------------------------------------------------------------------------
# InflightEventTrackingHandler (FROM llama4x handler.py:277-288)
# Tracks last structured event for crash forensics.
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
# init_observability
# (CHANGED — OSS_COMPAT + MONARCH_COMPAT: single init, explicit args,
#  rank/source on formatter. PR1 scope: system handler only.
#  PR4 extends this to add experiment handler.)
# ---------------------------------------------------------------------------

_system_logger = logging.getLogger(SYSTEM_LOGGER_NAME)


def init_observability(
    source: str, output_dir: str, rank: int | None = None
) -> None:
    """Initialize system + experiment structured logging. Called ONCE in setup().

    Rank/source are baked into formatters (constants). Step is set per-step
    via set_step(). Idempotent — skips if handlers already exist.

    Can be called before torch.distributed is initialized — rank defaults to
    the RANK environment variable (always set by torchrun/MAST). This matches
    sixlib's pattern where RankContext reads from env vars, not dist.get_rank().

    Args:
        source: Component name (e.g., "trainer", "generator").
        output_dir: Root output directory.
        rank: Global rank. If None, reads from RANK env var (default 0).
    """
    if rank is None:
        rank = int(os.environ.get("RANK", 0))

    # --- System handler (PR1) ---
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

    # --- Experiment handler (PR4) ---
    # Lazy import to avoid circular dependency (metrics.py imports from this module)
    from torchtitan.observability.metrics import (
        ExperimentJSONFormatter,
        ExperimentLoggingHandler,
    )

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
# record_event (RENAME + MONARCH_COMPAT)
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


# ---------------------------------------------------------------------------
# record_span (RENAMED from structured_logger — RENAME + MONARCH_COMPAT)
# ContextDecorator: works as both context manager and decorator.
# Auto-derives _START/_END event types from base.
# ---------------------------------------------------------------------------


class record_span(ContextDecorator):
    """Context manager/decorator for timing phases.

    Logs START event on enter, END event (with duration) on exit.
    Step is read from ContextVar.

    Usage:
        with record_span("Forward/Backward", EventType.FWD_BWD):
            output = model(batch)
            loss.backward()
    """

    def __init__(self, description: str, event_type: EventType):
        self.description = description
        self.event_type = event_type
        self.start_time: float | None = None

        # Derive START/END event types
        base_name = str(event_type)
        self.start_event_type = EventType(base_name + "_start")
        self.end_event_type = EventType(base_name + "_end")

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
        delta_ms = (end_time - self.start_time) * 1000
        _system_logger.info(
            f"[step {step if step is not None else 'N/A'}] {self.description} {self.end_event_type} took {delta_ms:.2f} ms",
            extra=event_extra(self.end_event_type, value=delta_ms, step=step),
            stacklevel=2,
        )
        return False  # Don't suppress exceptions
