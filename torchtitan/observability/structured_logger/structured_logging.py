# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Structured-logging API: init_structured_logger, log_trace_span, log_trace_instant, log_trace_scalar.

Emits structured JSONL events for phase timing, scalars, and diagnostics.
Handler factories (JSONL, custom, etc.) are loaded dynamically via
the ``TITAN_STRUCT_LOGGER_HANDLERS`` env var.

``log_trace_*`` never writes: it queues a small tuple, and one writer thread per
process passes the records to the handlers (see ``_RecordQueue``).
"""

import asyncio
import atexit
import collections
import enum
import functools
import importlib
import inspect
import logging
import marshal
import os
import threading
import time
import traceback
from collections.abc import Callable
from timeit import default_timer as timer
from typing import Any, cast, NamedTuple, TypeVar

import torch

from torchtitan.observability.structured_logger.step_state import (
    get_relative_step,
    get_step,
    get_step_tags,
)

F = TypeVar("F", bound=Callable[..., Any])

# Used by this module for regular logging
console_logger: logging.Logger = logging.getLogger(__name__)

# Dedicated logger for structured events. Uses a name distinct from
# ``__name__`` so we can set ``propagate = False`` here without also silencing
# the console logger above.
_structured_logger: logging.Logger = logging.getLogger("torchtitan.structured_logger")
_structured_logger.propagate = False

# Used to check if handler has been already initialized. If so, re-initializing
# is a no-op
_is_initialized: bool = False
_structured_logger_subprocess_init_fn: Callable[[], None] | None = None

# Set by ``init_structured_logger(enable=False)`` to make all trace calls no-ops.
_disabled: bool = False

# ~250 bytes per queued record: 25 MB when full. At 2k records/s, a stalled sink
# has ~50 s before the first record is dropped.
_MAX_QUEUED_RECORDS = 100_000
# How stale the files get while running, and what a hard kill loses.
_WRITE_INTERVAL_S = 0.1
# The writer flushes the handlers (one write per file) every this many records.
# Each write releases the GIL, so a busy logging thread waits at most this many
# records of formatting for it.
_FLUSH_EVERY_RECORDS = 32
# How long exit waits for a busy sink before giving up on the queued records.
_CLOSE_TIMEOUT_S = 5.0
# Drop and error warnings go to the console at most this often.
_WARNING_INTERVAL_S = 60.0

# Created by ``init_structured_logger``; None before init and when disabled.
_record_queue: "_RecordQueue | None" = None

_DEFAULT_HANDLER_FACTORY = (
    "torchtitan.observability.structured_logger.jsonl_handler.register_jsonl_handler"
)


def _structured_logger_disabled() -> bool:
    """Whether structured logging is disabled.

    Driven by the ``enable`` flag passed to :func:`init_structured_logger`
    (sourced from ``DebugConfig.enable_structured_logging``).
    """
    return _disabled


class LogType(enum.StrEnum):
    """Record kind in the JSONL stream.

    - ``EVENT``: paired span record (``*_start`` / ``*_end`` from ``log_trace_span``).
    - ``INSTANT``: point-in-time record (``log_trace_instant``, ``log_trace_scalar``).
    - ``TEXT``: free-text log record (filtered out by ``TraceEventsOnlyFilter``).
    """

    EVENT = "event"
    INSTANT = "instant"
    TEXT = "text"


class ExtraFields(enum.StrEnum):
    """Keys for the ``extra`` dict passed to logging calls."""

    LOG_TYPE = "log_type"
    LOG_TYPE_NAME = "log_type_name"
    EVENT_NAME = "event_name"
    STEP = "step"
    CONTEXT = "context"
    VALUE = "value"
    RELATIVE_STEP = "relative_step"
    TASK_NAME = "task_name"


class LoggingThreadState(NamedTuple):
    """What the logging thread saw when it logged: its id and its step state.

    Records are formatted on the writer thread, which can't read the logging
    thread's ContextVars, so this is captured when ``log_trace_*`` is called and
    stored on the record as ``logging_thread_state``.
    """

    tid: int
    step: int | None
    relative_step: int | None
    step_tags: tuple[str, ...]

    @classmethod
    def current(cls) -> "LoggingThreadState":
        return cls(
            threading.get_native_id(), get_step(), get_relative_step(), get_step_tags()
        )


class _StructuredRecordForwarder(logging.Handler):
    """Forward structured records from the root logger to trace handlers.

    ``init_structured_logger`` attaches this handler to ``logging.root``,
    Python's process-wide root logger, and attaches the trace handlers to the
    dedicated structured logger. The checkpoint backend uses a separate named
    logger::

        init_structured_logger(source="trainer", output_dir="./outputs")
        structured_logger = logging.getLogger("torchtitan.structured_logger")
        checkpoint_logger = logging.getLogger("torch_checkpointing")
        checkpoint_logger.setLevel(logging.INFO)

        checkpoint_logger.info(
            "checkpoint metric",
            extra=event_extra(
                "log_metric",
                event_name="train.checkpoint_write.latency_ms",
                value=12.5,
            ),
        )

    ``checkpoint_logger.propagate`` defaults to ``True``, so normal Python
    logging propagation sends the checkpoint record to ``logging.root``. This
    handler then bridges it to ``structured_logger``::

        checkpoint_logger.info(...)
            -> logging.root                    normal Python propagation
            -> _StructuredRecordForwarder      handler installed on root
            -> _record_queue.put(record)       explicit bridge
            -> structured_logger.handle(...)   on the writer thread
            -> registered trace handlers

    Native TorchTitan structured events take the shorter path::

        log_trace_*()
            -> _record_queue.put(tuple)
            -> structured_logger.handle(...)   on the writer thread
            -> registered trace handlers

    They are not logged twice because ``structured_logger.propagate`` is
    ``False``. Native records stop after its handlers and never reach
    ``logging.root`` or this forwarder. The bridge is installed on the root so
    future integrated libraries do not need separate handler installation;
    any logger with propagation enabled can opt in by emitting a record with
    ``log_type_name``.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == _structured_logger.name:
            return
        if getattr(record, str(ExtraFields.LOG_TYPE_NAME), None) is None:
            return
        queue = _record_queue
        if queue is None:
            _structured_logger.handle(record)
            return
        # Formatted on the writer thread: capture what only this thread can read.
        record.logging_thread_state = LoggingThreadState.current()
        queue.put(record)


def _ensure_root_forwarder() -> None:
    root_logger = logging.getLogger()
    if not any(
        isinstance(handler, _StructuredRecordForwarder)
        for handler in root_logger.handlers
    ):
        root_logger.addHandler(_StructuredRecordForwarder())


def get_structured_logger_subprocess_init_fn() -> Callable[[], None] | None:
    if _disabled or not _is_initialized or not _structured_logger.handlers:
        return None
    if _structured_logger_subprocess_init_fn is None:
        raise RuntimeError("Structured logger subprocess initializer is missing")
    return _structured_logger_subprocess_init_fn


def event_extra(
    event_type: str,
    event_name: str | None = None,
    step: int | None = None,
    relative_step: int | None = None,
    value: float | int | None = None,
    task_name: str | None = None,
    log_type: LogType = LogType.EVENT,
) -> dict[str, Any]:
    """Build the extra dict for a structured JSONL event record."""
    return {
        str(ExtraFields.LOG_TYPE): str(log_type),
        str(ExtraFields.LOG_TYPE_NAME): str(event_type),
        str(ExtraFields.EVENT_NAME): event_name,
        str(ExtraFields.STEP): step,
        str(ExtraFields.RELATIVE_STEP): relative_step,
        str(ExtraFields.VALUE): value,
        str(ExtraFields.TASK_NAME): task_name,
    }


class TraceEventsOnlyFilter(logging.Filter):
    """Defensive filter: drop any record on the structured logger that did not
    come through the ``log_trace_*`` API.

    How records get a ``log_type_name`` attribute:

    1. ``log_trace_span`` / ``log_trace_instant`` / ``log_trace_scalar`` all
       queue a record that the writer thread builds with
       ``makeRecord(..., extra=event_extra(...))`` (``_to_log_record``).
    2. ``event_extra`` always sets ``log_type_name`` in the ``extra`` dict.
    3. Python's logging attaches ``extra`` keys as attributes on the
       ``LogRecord``, so ``record.log_type_name`` is populated.

    So any record reaching this filter WITHOUT ``log_type_name`` is a plain
    ``.info("text")`` call made directly on the structured logger — bypassing
    the API. That should not happen in this codebase (we never call
    ``_structured_logger`` outside the log_trace_* helpers). The filter exists
    as a safeguard against future accidents: if someone grabs the logger by
    name (``logging.getLogger("torchtitan.structured_logger")``) and writes
    free text, this filter keeps it out of the JSONL stream so the schema
    stays strict. The first drop emits a one-shot warning to make the trap
    discoverable.
    """

    def __init__(self) -> None:
        super().__init__()
        self._warned = False

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, str(ExtraFields.LOG_TYPE_NAME), None) is not None:
            return True
        if not self._warned:
            self._warned = True
            console_logger.warning(
                "Plain-text record on the structured logger was dropped. "
                "Use log_trace_span / log_trace_scalar / log_trace_instant."
            )
        return False


class _RecordQueue:
    """Bounded queue between ``log_trace_*`` callers and the structured logger's handlers.

    ``put`` appends and returns. A daemon thread wakes every ``_WRITE_INTERVAL_S``
    and passes each queued record to the handlers, so a slow or hung sink (NFS, a
    remote database) delays records, never the caller. When ``_MAX_QUEUED_RECORDS``
    are waiting, new records are dropped and counted; the writer then logs a
    ``structured_logger_dropped`` record with the count.

    Items are ``_enqueue`` payloads (bytes), LogRecords forwarded from other
    loggers, or ``threading.Event``s that ``flush`` waits on.

    Example::

        queue = _RecordQueue()
        queue.put(payload)           # returns at once, whatever the handlers do
        queue.flush(timeout_s=10.0)  # True once everything put so far is written
        queue.close(timeout_s=5.0)   # at exit: write the rest, stop the thread
    """

    def __init__(self) -> None:
        self._items: collections.deque[
            bytes | logging.LogRecord | threading.Event
        ] = collections.deque()
        self._wake = threading.Event()
        self._drop_lock = threading.Lock()
        self._num_dropped = 0
        self._stopping = False
        self._closed = False
        self._last_drop_warning_s = float("-inf")
        self._last_error_warning_s = float("-inf")
        self._thread = threading.Thread(
            target=self._write_loop, name="structured-logger-writer", daemon=True
        )
        self._thread.start()

    def put(self, item: bytes | logging.LogRecord) -> None:
        if self._closed:
            self._put_after_close(item)
            return
        # Soft bound: threads racing past this check can add a few more.
        if len(self._items) >= _MAX_QUEUED_RECORDS:
            self._count_dropped()
            return
        self._items.append(item)
        if self._closed and not self._thread.is_alive():
            # close() finished between the check above and the append.
            self._drain()

    def flush(self, timeout_s: float) -> bool:
        """Block until everything put so far reached the handlers, and they flushed."""
        if self._closed:
            if self._thread.is_alive():
                return False
            self._flush_handlers()
            return True
        if threading.current_thread() is self._thread:
            # Called from a handler: waiting for the writer would wait on ourselves.
            return False
        done = threading.Event()
        # Past the size bound on purpose, so a full queue still flushes.
        self._items.append(done)
        self._wake.set()
        return done.wait(timeout_s)

    def close(self, timeout_s: float) -> None:
        """Write what's queued and stop the writer; later records are written on the caller."""
        if self._stopping or threading.current_thread() is self._thread:
            return
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout_s)
        self._closed = True
        if self._thread.is_alive():
            console_logger.warning(
                "Structured logger: the sink is still busy after %.0f s; "
                "%d queued records were not written.",
                timeout_s,
                len(self._items),
            )
            return
        # Records put while the writer was exiting.
        self._drain()

    def _put_after_close(self, item: bytes | logging.LogRecord) -> None:
        # After close() (interpreter exit), write on the caller. If close() gave up
        # on a stuck sink, the writer still holds the handlers: drop, don't wait.
        if self._thread.is_alive():
            self._count_dropped()
            return
        self._handle(item)
        self._flush_handlers()

    def _write_loop(self) -> None:
        while True:
            self._wake.wait(_WRITE_INTERVAL_S)
            self._wake.clear()
            # Read before draining: a close() during this pass gets one more.
            stopping = self._stopping
            try:
                self._drain()
            except Exception:
                self._warn_error(f"writer pass failed:\n{traceback.format_exc()}")
            if stopping:
                return

    def _drain(self) -> None:
        # Only what's queued now, so a pass ends even while producers outrun it.
        for index in range(len(self._items)):
            try:
                item = self._items.popleft()
            except IndexError:
                # After close(), a late put() can drain concurrently.
                break
            if isinstance(item, threading.Event):
                self._report_drops()
                self._flush_handlers()
                item.set()
                continue
            self._handle(item)
            if index % _FLUSH_EVERY_RECORDS == _FLUSH_EVERY_RECORDS - 1:
                self._flush_handlers()
        self._report_drops()
        self._flush_handlers()

    def _handle(self, item: bytes | logging.LogRecord) -> None:
        try:
            if isinstance(item, logging.LogRecord):
                record = item
            elif _structured_logger.isEnabledFor(logging.INFO):
                # Logger.handle skips the level check that Logger.info does.
                record = _to_log_record(item)
            else:
                return
            _structured_logger.handle(record)
        except Exception:
            self._count_dropped()
            self._warn_error(f"a record failed to write:\n{traceback.format_exc()}")

    def _flush_handlers(self) -> None:
        for handler in _structured_logger.handlers:
            try:
                handler.flush()
            except Exception:
                self._warn_error(
                    f"flushing {handler!r} failed:\n{traceback.format_exc()}"
                )

    def _count_dropped(self) -> None:
        with self._drop_lock:
            self._num_dropped += 1

    def _report_drops(self) -> None:
        with self._drop_lock:
            num_dropped, self._num_dropped = self._num_dropped, 0
        if not num_dropped:
            return
        record = _structured_logger.makeRecord(
            _structured_logger.name,
            logging.INFO,
            __file__,
            0,
            f"structured_logger_dropped: {num_dropped} records",
            (),
            None,
            "_report_drops",
            extra=event_extra(
                "structured_logger_dropped",
                value=num_dropped,
                log_type=LogType.INSTANT,
            ),
        )
        # Logged by the writer itself, outside any step.
        record.logging_thread_state = LoggingThreadState(
            threading.get_native_id(), None, None, ()
        )
        try:
            _structured_logger.handle(record)
        except Exception:
            # Not counted as a drop: that would report again on every pass.
            self._warn_error(
                f"a drop report failed to write:\n{traceback.format_exc()}"
            )
        now = time.monotonic()
        if now - self._last_drop_warning_s >= _WARNING_INTERVAL_S:
            self._last_drop_warning_s = now
            console_logger.warning(
                "Structured logger dropped %d records: %d were already queued "
                "(the sink is slower than the logging rate), or they failed to "
                "write (see earlier warnings).",
                num_dropped,
                _MAX_QUEUED_RECORDS,
            )

    def _warn_error(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_error_warning_s >= _WARNING_INTERVAL_S:
            self._last_error_warning_s = now
            console_logger.warning("Structured logger: %s", message)


def _enqueue(
    msg: str,
    *,
    event_type: str,
    log_type: LogType,
    stacklevel: int,
    step: int | None = None,
    value: float | int | None = None,
    event_name: str | None = None,
    task_name: str | None = None,
) -> None:
    """Queue one record for the writer thread; ``_to_log_record`` unpacks the same fields.

    The fields are marshaled to bytes, which the garbage collector doesn't track:
    a backlog of queued records, e.g. behind a stalled sink, triggers no
    collections, while tuples pushed full collections (~45 ms with torch loaded).
    """
    queue = _record_queue
    if queue is None:
        return
    # +1 for this function, so stacklevel=2 is the caller of log_trace_*, as with Logger.info.
    pathname, lineno, func_name, _ = _structured_logger.findCaller(
        False, stacklevel + 1
    )
    queue.put(
        marshal.dumps(
            (
                time.time_ns(),
                pathname,
                lineno,
                func_name,
                msg,
                str(log_type),
                str(event_type),
                event_name,
                step,
                value,
                task_name,
                threading.get_native_id(),
                get_step(),
                get_relative_step(),
                get_step_tags(),
            )
        )
    )


def _to_log_record(payload: bytes) -> logging.LogRecord:
    """Build the LogRecord for one ``_enqueue`` payload, on the writer thread."""
    (
        created_ns,
        pathname,
        lineno,
        func_name,
        msg,
        log_type,
        event_type,
        event_name,
        step,
        value,
        task_name,
        tid,
        thread_step,
        thread_relative_step,
        thread_step_tags,
    ) = marshal.loads(payload)
    record = _structured_logger.makeRecord(
        _structured_logger.name,
        logging.INFO,
        pathname,
        lineno,
        msg,
        (),
        None,
        func_name,
        extra=event_extra(
            event_type,
            event_name=event_name,
            step=step,
            value=value,
            task_name=task_name,
            log_type=LogType(log_type),
        ),
    )
    # makeRecord stamps the writer's clock; keep the time the caller logged at.
    record.created = created_ns / 1e9
    record.msecs = (created_ns % 1_000_000_000) // 1_000_000 + 0.0
    record.logging_thread_state = LoggingThreadState(
        tid, thread_step, thread_relative_step, thread_step_tags
    )
    return record


def _close_record_queue() -> None:
    if _record_queue is not None:
        _record_queue.close(_CLOSE_TIMEOUT_S)


def _new_record_queue_in_forked_child() -> None:
    global _record_queue
    if _record_queue is not None and not _record_queue._closed:
        # The parent writes what it had queued; the child gets its own writer.
        _record_queue = _RecordQueue()


# Registered after ``logging``'s own atexit hook, so it runs first (LIFO) and the
# handlers are still open while the queue drains.
atexit.register(_close_record_queue)
os.register_at_fork(after_in_child=_new_record_queue_in_forked_child)


def flush_structured_logger(timeout_s: float = 10.0) -> bool:
    """Block until every record logged so far reached the handlers, and they flushed.

    Records are written within ~0.1 s anyway; call this before reading the
    structured-log files from the same process. Returns False if the sink didn't
    catch up within ``timeout_s``.
    """
    queue = _record_queue
    return True if queue is None else queue.flush(timeout_s)


# TODO(observability): rename `rank` -> `mesh_rank`. The value is the rank within
# the actor's proc mesh (`current_rank().rank`), not globally unique across
# multiple RL meshes (e.g. trainer + generator each start at 0).
# TODO(observability): handle duplicate `source` across actor meshes -- today
# (source, rank) collides if two meshes share a name (e.g. two generators).
def init_structured_logger(
    source: str, output_dir: str, rank: int | None = None, enable: bool = True
) -> None:
    """Attach handlers to the structured logger. Call once per process.

    Handler factories come from the ``TITAN_STRUCT_LOGGER_HANDLERS`` env var
    (comma-separated ``module.path.factory_name``). When unset, a default
    JSONL handler is registered; when set, ONLY the listed factories run.

    ``rank`` defaults to ``$RANK`` (set by torchrun), so this can run
    before ``torch.distributed`` init. Repeated calls do not duplicate handlers.

    When ``enable=False``, all subsequent ``log_trace_*`` calls become
    no-ops (no handlers are attached).

    Does not configure console output; call ``init_logger()`` for that.

    Example::

        init_structured_logger(source="trainer", output_dir="./outputs")
        log_trace_instant("structured_logger_started")
    """
    global _is_initialized, _disabled, _structured_logger_subprocess_init_fn, _record_queue

    if not enable:
        _disabled = True
        _structured_logger_subprocess_init_fn = None
        console_logger.info(
            "Structured logging disabled via DebugConfig.enable_structured_logging=False"
        )
        return

    if _is_initialized:
        return

    if rank is None:
        rank = int(os.environ.get("RANK", 0))

    factories_env = os.environ.get("TITAN_STRUCT_LOGGER_HANDLERS", "")
    if factories_env.strip():
        factory_paths = [f.strip() for f in factories_env.split(",") if f.strip()]
    else:
        factory_paths = [_DEFAULT_HANDLER_FACTORY]

    for factory_path in factory_paths:
        module_path, func_name = factory_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        getattr(mod, func_name)(
            structured_logger=_structured_logger,
            rank=rank,
            source=source,
            output_dir=output_dir,
        )

    if (
        _structured_logger.level == logging.NOTSET
        or _structured_logger.level > logging.INFO
    ):
        _structured_logger.setLevel(logging.INFO)

    _structured_logger_subprocess_init_fn = functools.partial(
        init_structured_logger,
        source=source,
        output_dir=output_dir,
        rank=rank,
    )
    _record_queue = _RecordQueue()
    _is_initialized = True

    _ensure_root_forwarder()


def log_trace_scalar(scalars: dict[str, float | int], *, stacklevel: int = 2) -> None:
    """Emit a record per (name, value) pair. Useful when adding more context
    to the trace for debugging, e.g. registering `num_tokens_processed`.

    Step is read from ``set_step()``; non-numeric values are skipped
    with a warning. Bump ``stacklevel`` when wrapping in a helper so
    ``caller`` points at the real call site.

    Args:
        scalars: Mapping of scalar name to numeric value. Non-numeric
            values are skipped with a warning.
        stacklevel: Passed through to ``logger.info`` so the ``caller`` field
            in the emitted record points at the real call site. Increase from
            the default 2 if you wrap this function in a helper.

    Example::

        log_trace_scalar({"train.loss": 2.5, "train.tflops": 45.6})
    """
    if _structured_logger_disabled() or torch.compiler.is_compiling():
        return
    step = get_step()
    bad_keys: list[str] = []
    for name, value in scalars.items():
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            bad_keys.append(name)
            continue
        _enqueue(
            f"[step {step if step is not None else 'N/A'}] {name}={value}",
            event_type="metric_value",
            event_name=name,
            value=value,
            step=step,
            log_type=LogType.INSTANT,
            stacklevel=stacklevel,
        )
    if bad_keys:
        console_logger.warning(
            "log_trace_scalar skipped non-numeric values for keys: %s", bad_keys
        )


def log_trace_instant(event_type: str, *, stacklevel: int = 2) -> None:
    """Emit a zero-duration event or marker (e.g. ``"training_start"``).

    Use ``log_trace_span`` when you want start+end+duration.

    Args:
        event_type: Free-form string. Becomes ``log_type_name`` in the
            emitted record.
        stacklevel: Passed through to ``logger.info`` so the ``caller`` field
            in the emitted record points at the real call site. Increase from
            the default 2 if you wrap this function in a helper.

    Example::

        log_trace_instant("training_start")
    """
    if torch.compiler.is_compiling() or _structured_logger_disabled():
        return
    _enqueue(
        str(event_type),
        event_type=event_type,
        log_type=LogType.INSTANT,
        stacklevel=stacklevel,
    )


class log_trace_span:  # noqa: N801
    """Time a block of work; emits ``_start`` and ``_end`` records.

    Usable as a context manager or decorator. On entry, captures the
    enclosing :class:`asyncio.Task`'s name via
    ``asyncio.current_task().get_name()`` (``None`` outside any task)
    and stamps it on both records. Analysis pairs ``_start`` / ``_end``
    via a LIFO stack on ``(source, task_name)``, so nested spans in one
    task pair correctly. The ``_end`` record's ``value`` is the elapsed
    wall-time in ms.

    On exception, emits an extra ``_error`` record (with exception type
    and message), then the normal ``_end`` -- so every ``_start`` has a
    matching close.

    Example::

        # context manager
        with log_trace_span("fwd_bwd"):
            loss = model(batch)
            loss.backward()

        # decorator of sync or async function
        @log_trace_span("rl_rollout")
        async def rollout(self, prompts):
            return await self.engine.generate(prompts)

    Args:
        event_type: Becomes ``log_type_name`` in the records.
        description: Human-readable label in the log line; doesn't
            affect ``log_type_name`` or filtering.
        stacklevel: Bump when wrapping in a helper so ``caller`` points
            at the real call site.
    """

    def __init__(
        self,
        event_type: str,
        description: str | None = None,
        *,
        stacklevel: int = 2,
    ):
        self.base_name = str(event_type)
        self.description = description
        self.stacklevel = stacklevel
        self.start_time: float = 0.0
        self._task_name: str | None = None
        self.start_type_name = self.base_name + "_start"
        self.end_type_name = self.base_name + "_end"

    def __enter__(self):
        if torch.compiler.is_compiling() or _structured_logger_disabled():
            return self

        # Cache the asyncio task name so __exit__ emits the same one as
        # __enter__; pairing relies on (source, task_name) being stable
        # across a span. None in SPMD / non-asyncio code.
        try:
            task = asyncio.current_task()
            self._task_name = task.get_name() if task else None
        except RuntimeError:
            self._task_name = None

        display_name = self.description or self.base_name
        self.start_time = timer()
        step = get_step()
        _enqueue(
            f"[step {step if step is not None else 'N/A'}] {display_name} {self.start_type_name}",
            event_type=self.start_type_name,
            log_type=LogType.EVENT,
            step=step,
            task_name=self._task_name,
            stacklevel=self.stacklevel,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # On success: emit ``_end``. On exception: emit ``_error`` then
        # ``_end``. The trailing ``_end`` carries elapsed-until-crash
        # and keeps pairing simple -- analysis tools don't have to
        # special-case exceptional spans.
        if torch.compiler.is_compiling() or _structured_logger_disabled():
            return None

        end_time = timer()
        step = get_step()
        duration_s = end_time - self.start_time
        delta_ms = duration_s * 1000

        if exc_type is not None:
            error_type_name = self.base_name + "_error"
            _enqueue(
                f"[step {step if step is not None else 'N/A'}] {error_type_name}: {exc_type.__name__}: {exc_val}",
                event_type=error_type_name,
                log_type=LogType.EVENT,
                step=step,
                task_name=self._task_name,
                stacklevel=self.stacklevel,
            )

        _enqueue(
            f"[step {step if step is not None else 'N/A'}] {self.end_type_name} took {delta_ms:.2f} ms",
            event_type=self.end_type_name,
            log_type=LogType.EVENT,
            value=delta_ms,
            step=step,
            task_name=self._task_name,
            stacklevel=self.stacklevel,
        )
        return None

    def __call__(self, func: F) -> F:
        # Decorator support. Each invocation of the decorated function builds
        # a fresh ``log_trace_span`` so concurrent calls don't clobber each
        # other's ``self.start_time`` / ``self._task_name``.
        #
        # The async path wraps so __exit__ runs after ``await func(...)``
        # completes — a plain sync wrapper would close the context before the
        # coroutine runs and the recorded duration would be ~0ms.
        #
        # TODO: ``caller`` is inaccurate for *async* decorator use — it lands
        # on async_wrapper / asyncio / threading internals (incl. Monarch
        # endpoint dispatch). Use as a context manager when this matters.
        base_name, description, stacklevel = (
            self.base_name,
            self.description,
            self.stacklevel,
        )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with log_trace_span(base_name, description, stacklevel=stacklevel):
                    return await func(*args, **kwargs)

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # stacklevel + 1 skips this wrapper so caller points at the user.
            with log_trace_span(base_name, description, stacklevel=stacklevel + 1):
                return func(*args, **kwargs)

        return cast(F, sync_wrapper)
