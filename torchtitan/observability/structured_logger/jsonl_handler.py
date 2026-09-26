# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Default JSONL backend: formatter, file handler, and factory.

``TraceJsonlFormatter`` is also the base class for backend-specific
formatters (e.g. the Scuba formatter under ``fb/``).
"""

import datetime as dt
import functools
import itertools
import json
import logging
import os
import random
import socket
import string
from typing import Any

from torchtitan.observability.structured_logger.structured_logging import (
    ExtraFields,
    LoggingThreadState,
    LogType,
    TraceEventsOnlyFilter,
)

console_logger: logging.Logger = logging.getLogger(__name__)

MAX_MESSAGE_SIZE: int = 1000


class TraceJsonlFormatter(logging.Formatter):
    """Format trace records as one JSON line per record.

    Per-process fields (rank, source, hostname, local_rank) are captured
    in ``__init__``. The thread id and per-step fields (step, relative_step,
    step_tags) come from ``record.logging_thread_state``, captured by the thread
    that logged: records are formatted on the structured logger's writer thread.

    Subclass to enrich records with backend-specific fields.

    Example output (wrapped for readability)::

        {"rank": 0, "source": "training", "step": 5,
         "log_type_name": "fwd_bwd_end", "value": 12.5, "task_name": "Task-1",
         "step_tags": ["gc"],
         "time_us": 1709500000123456, "caller": "trainer.py:796:train_step",
         "seq_id": 42}
    """

    def __init__(self, rank: int, source: str):
        super().__init__()
        self.rank = rank
        self.source = source
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._host_name = socket.gethostname()
        self._seq_counter = itertools.count()
        # Last record time per logging thread, for delta_ms. Per instance, so
        # formatters of different handlers don't see each other's records.
        self._last_created_by_tid: dict[int, float] = {}

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(self._log_dict(record))

    def _log_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        """Build the flat dict emitted as one JSONL line."""
        log_dict: dict[str, Any] = {}

        # Records built directly on the logger (not through log_trace_*) read it now.
        thread_state = (
            getattr(record, "logging_thread_state", None)
            or LoggingThreadState.current()
        )
        log_dict["delta_ms"] = self._event_delta_ms(thread_state.tid, record.created)
        log_dict["tid"] = thread_state.tid

        # Rank/source from self (constants, set once in init_structured_logger)
        log_dict["rank"] = self.rank
        log_dict["global_rank"] = self.rank
        log_dict["local_rank"] = self._local_rank
        log_dict["source"] = self.source
        log_dict["host_name"] = self._host_name
        log_dict["pid"] = os.getpid()

        # Step/step_tags/relative_step as the logging thread saw them
        if thread_state.step is not None:
            log_dict["step"] = thread_state.step
        if thread_state.relative_step is not None:
            log_dict["relative_step"] = thread_state.relative_step
        if thread_state.step_tags:
            log_dict["step_tags"] = list(thread_state.step_tags)

        log_dict["time"] = int(record.created)
        log_dict["time_ms"] = int(record.created * 1000)
        log_dict["time_us"] = int(record.created * 1_000_000)

        log_dict["log_type"] = getattr(
            record, str(ExtraFields.LOG_TYPE), str(LogType.TEXT)
        )
        log_type_name = getattr(record, str(ExtraFields.LOG_TYPE_NAME), None)
        log_dict["log_type_name"] = log_type_name

        # Per-record step/relative_step override (from event_extra)
        record_step = getattr(record, str(ExtraFields.STEP), None)
        if record_step is not None:
            log_dict["step"] = record_step
        record_relative_step = getattr(record, str(ExtraFields.RELATIVE_STEP), None)
        if record_relative_step is not None:
            log_dict["relative_step"] = record_relative_step

        log_dict["event_name"] = getattr(record, str(ExtraFields.EVENT_NAME), None)

        value = getattr(record, str(ExtraFields.VALUE), None)
        if isinstance(value, (float, int)):
            log_dict["value"] = float(value)

        context = getattr(record, str(ExtraFields.CONTEXT), None)
        if context is not None:
            log_dict["context"] = context

        # task_name pairs start/end records
        task_name = getattr(record, str(ExtraFields.TASK_NAME), None)
        if task_name is not None:
            log_dict["task_name"] = task_name

        # Caller field for source traceability (file:line:function)
        log_dict[
            "caller"
        ] = f"{_relpath(record.pathname)}:{record.lineno}:{record.funcName}"
        log_dict["log_file"] = record.filename
        log_dict["log_function"] = record.funcName
        log_dict["log_level"] = record.levelname
        log_dict["logger_name"] = record.name
        if record.stack_info:
            log_dict["stack_info"] = record.stack_info

        log_dict["seq_id"] = next(self._seq_counter)

        # Truncate long free-text messages. Event records
        # (log_trace_span/scalar/instant) pass msg="", so this branch really
        # only catches plain logger.info() calls routed to the structured
        # logger -- usually a user mistake.
        message = record.getMessage()
        if message is not None:
            if len(message) <= MAX_MESSAGE_SIZE:
                log_dict["message"] = message
            else:
                half = MAX_MESSAGE_SIZE // 2
                log_dict["message"] = message[:half] + "..." + message[-half:]

        return log_dict

    def _event_delta_ms(self, tid: int, created: float) -> float:
        """Milliseconds since the previous record of the same logging thread (0 for its first)."""
        last = self._last_created_by_tid.get(tid, created)
        self._last_created_by_tid[tid] = created
        return (created - last) * 1000


class TraceJsonlHandler(logging.FileHandler):
    """Per-rank JSONL file handler that writes a batch of lines per ``flush()``.

    The structured logger's writer thread flushes every few dozen records. One
    write per batch, not per record, keeps that thread from waiting for the GIL
    after every record while the process is busy.

    File path::

        {output_dir}/structured_logs/{source}.global_rank_{rank}.{timestamp}-{random}.jsonl
    """

    def __init__(self, rank: int, source: str, output_dir: str):
        timestamp_str = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        random_str = "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(6)
        )
        filename = f"{source}.global_rank_{rank}.{timestamp_str}-{random_str}.jsonl"
        filepath = os.path.join(output_dir, "structured_logs", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        super().__init__(filename=filepath)
        self.setFormatter(TraceJsonlFormatter(rank=rank, source=source))
        self.addFilter(TraceEventsOnlyFilter())
        self._pending_lines: list[str] = []
        # A forked child must not write the lines its parent had not written yet.
        os.register_at_fork(after_in_child=self._pending_lines.clear)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._pending_lines.append(self.format(record) + self.terminator)
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        self.acquire()
        try:
            if self._pending_lines:
                if self.stream is None:
                    self.stream = self._open()
                lines = "".join(self._pending_lines)
                self._pending_lines.clear()
                self.stream.write(lines)
            super().flush()
        finally:
            self.release()


@functools.cache
def _relpath(path: str) -> str:
    # os.path.relpath calls os.getcwd() on every record; callers repeat a few paths.
    return os.path.relpath(path)


def register_jsonl_handler(
    *,
    structured_logger: logging.Logger,
    rank: int,
    source: str,
    output_dir: str,
    **kw: Any,
) -> None:
    """Default factory: attach a ``TraceJsonlHandler`` to the structured logger."""
    handler = TraceJsonlHandler(rank=rank, source=source, output_dir=output_dir)
    structured_logger.addHandler(handler)
    console_logger.info("Structured logging -> JSONL: %s", handler.baseFilename)
