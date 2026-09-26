# TorchTitan Observability

Structured logging for distributed training. Emits per-rank JSONL events for phase timing, diagnostics, and post-hoc analysis.

Design principles:
- LLM friendly: Emit structured, per-rank data that can be queried during and after a run.
- Handle both SPMD pretraining (one process group) and RL with multiple independent actors (no shared process group).
- Stay invisible to users -- no metric dictionaries to pass around.
- Never block training.
- Support pluggable backends via handler factories.

## Quickstart

```python
from torchtitan.observability.logging import init_logger
from torchtitan.observability import structured_logger as sl

# console logger (stdout, [titan] prefix)
init_logger()

# Register handlers, i.e. the functions that will take the logs
# and save to a local jsonl, a database, or something else.
sl.init_structured_logger(source="training", output_dir="./outputs")

# Use to register a point-in-time marker — training has started.
sl.log_trace_instant("training_start")

loaded_step = 0
for step in range(loaded_step + 1, num_steps + 1):
    # Stamp every subsequent record with `step` and `relative_step`
    sl.set_step(step, relative_step=step - loaded_step)

    if should_garbage_collect:
        # Appends "gc" to `step_tags` on every record for this step; tags
        # reset at the next set_step() call.
        # Users can filter such steps later, e.g. "ignore if tag X".
        sl.add_step_tag("gc")
        with sl.log_trace_span("gc_collect"):
            run_gc()

    with sl.log_trace_span("fwd_bwd"):
        output = model(batch)
        loss.backward()

    with sl.log_trace_span("Optimizer"):
        optimizer.step()

    # Scalars you may want to register to help debug, for example.
    sl.log_trace_scalar({
        "num_trainable_tokens": num_trainable_tokens,
         "batch_size": bsz
         })
```

Call `init_logger()` and `sl.init_structured_logger()` once per process before any trace calls. Rank and source are baked into the formatter at init; every JSONL entry automatically includes `rank`, `source`, `caller` (file:line:function), `time_us`, `step`, `relative_step`, and (when tags are set) `step_tags`.

## API reference

See docstrings for full args:

- `sl.init_structured_logger(source, output_dir, rank=None, enable=True)` -- wire up handlers; call once per process before any trace call. Pass ``enable=False`` (or set ``--debug.enable_structured_logging=False``) to make all trace calls no-ops.
- `sl.log_trace_span(event_type, description=None, *, stacklevel=2)` -- context manager / decorator; emits `_start` / `_end` / optional `_error` records.
- `sl.log_trace_instant(event_type, *, stacklevel=2)` -- point-in-time marker (no duration).
- `sl.log_trace_scalar(scalars, *, stacklevel=2)` -- emit `metric_value` records from a `{name: number}` dict.
- `sl.set_step(step, *, relative_step=None)` -- stamp subsequent records with a step; clears previous step's tags.
- `sl.add_step_tag(tag)` / `sl.clear_step_tags()` -- annotate the current step (e.g. `"gc"`, `"eval"`). `clear_step_tags` is called at `set_step`.
- `sl.flush_structured_logger(timeout_s=10.0)` -- block until every record logged so far is written; call it before reading the JSONL files from the same process.
- `TITAN_STRUCT_LOGGER_HANDLERS` -- Define handlers at the env level

## Flow of information

End-to-end, what happens when user code calls one of the `sl.` helpers:

```
user code  (the thread that logs)
    │   with sl.log_trace_span("fwd_bwd"):
    │       ...
    │
    │   # On entry, log_trace_span queues a tuple and returns:
    │       _enqueue(
    │           "[step 5] fwd_bwd_start",           ← human-readable message
    │           event_type="fwd_bwd_start",         ← → record.log_type_name
    │           step=5,                             ← → record.step
    │           task_name=None,                     ← → record.task_name
    │       )
    │
    │   # The tuple also captures this thread's id, file:line:function, and
    │   # its step state (sl.set_step() / sl.add_step_tag() write a ContextVar
    │   # and a module global), because the writer thread can't read them.
    ▼
_record_queue  (bounded in memory: when 100k records are waiting, new ones
                are dropped and a structured_logger_dropped record says how many)
    │
    ▼  writer thread, one per process, every 0.1 s
_structured_logger  (logging.Logger, name="torchtitan.structured_logger",
                     propagate=False — records stay out of the root logger)
    │
    ▼
TraceEventsOnlyFilter  (drops records that reached the logger WITHOUT a
                        log_type_name attribute — defensive; shouldn't fire
                        in practice because only the sl.* helpers write here)
    │
    ├── TraceJsonlHandler      ──▶  TraceJsonlFormatter     ──▶  {output_dir}/structured_logs/*.jsonl
    └── TraceMyDBHandler*     ──▶  TraceMyDBFormatter     ──▶  MyDB # extra handler defined by user
```

A slow or hung sink (NFS, a remote database) only delays the writer thread; `log_trace_*` never waits on it. Records reach the files within ~0.1 s, so call `sl.flush_structured_logger()` before reading them from the same process.

## Custom handlers

`TITAN_STRUCT_LOGGER_HANDLERS` is a comma-separated list of fully-qualified Python function paths. When set, ONLY the listed factories run.

```bash
export TITAN_STRUCT_LOGGER_HANDLERS="torchtitan.observability.structured_logger.jsonl_handler.register_jsonl_handler,mypackage.my_backend.register_my_db_handler"
```

A handler factory takes the args `sl.init_structured_logger()` forwards and attaches one handler to `structured_logger`. Example: stream events to a remote database instead of writing to disk, and enrich each record with cluster metadata along the way.

Handlers run on the writer thread, so they may block without slowing training. Two things to know:
- Read the logging thread's id and step state from `record.logging_thread_state`. Calling `get_step()` or `threading.get_native_id()` inside a handler returns the writer thread's values.
- Batch I/O in `flush()`, which the writer calls every few dozen records, rather than doing a syscall per record in `emit()`. On a busy process each syscall makes the writer wait for the GIL (see `TraceJsonlHandler`).

```python
import logging
import os

from torchtitan.observability.structured_logger.jsonl_handler import (
    TraceJsonlFormatter,
)
from torchtitan.observability.structured_logger.structured_logging import (
    TraceEventsOnlyFilter,
)


class MyDBFormatter(TraceJsonlFormatter):
    """Enrich each record with backend-specific fields before serialization."""

    def _log_dict(self, record):
        d = super()._log_dict(record)
        d["cluster"] = os.environ.get("CLUSTER_NAME", "unknown")
        return d


class MyDBHandler(logging.Handler):
    """Send each trace event to a remote DB as it's emitted."""

    def __init__(self, rank: int, source: str, db_url: str):
        super().__init__()
        self.client = MyDBClient(db_url)
        self.setFormatter(MyDBFormatter(rank=rank, source=source))
        self.addFilter(TraceEventsOnlyFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.client.insert_row(self.format(record))
        except Exception:
            self.handleError(record)


def register_my_db_handler(
    *, structured_logger: logging.Logger, rank: int, source: str, **kw
) -> None:
    structured_logger.addHandler(
        MyDBHandler(rank=rank, source=source, db_url="mydb://..."),
    )
```

## Analysis

### Gantt trace from JSONL

```python
from torchtitan.observability.structured_logger.gantt_generator import (
    generate_gantt_trace,
)

generate_gantt_trace("outputs/structured_logs/", "outputs/analysis/gantt.json")
```

Reads every rank's JSONL, pairs `_start` / `_end`, and writes a Chrome Trace JSON. Open in [Perfetto](https://ui.perfetto.dev).

Example dummy ASCII sketch:
```
                       step 1                              step 2
                  ┌──────────────────────────┐       ┌───────────────────────────┐
trainer rank 0    │ ▓▓ fwd_bwd ▓▓  ▓ optim ▓ │       │ ▓ fwd_bwd ▓▓ ▓▓ optim ▓   │
trainer rank 1    │  ▓▓ fwd_bwd ▓ ▓ optim ▓  │       │  ▓▓ fwd_bwd ▓▓ ▓ optim ▓  │
trainer rank 2    │ ▓ fwd_bwd ▓▓ ▓ optim ▓▓  │       │  ▓ fwd_bwd ▓▓▓ ▓ optim ▓▓ │
trainer rank 3    │  ▓▓ fwd_bwd ▓▓▓ ▓ optim ▓│       │ ▓ fwd_bwd ▓▓ ▓ optim ▓    │
                  └──────────────────────────┘       └───────────────────────────┘
controller        ▓▓▓▓▓▓ training_s ▓▓▓▓▓▓▓ ▓ scoring ▓ ▓▓▓▓▓▓ training_s ▓▓▓▓▓▓
rollouter         ▓▓▓                                     ▓▓▓
reward                                       ▓▓▓                              ▓▓▓
                  ├──────────────────────────┼───────────┼──────────────────────┤
                  0s                         3s          4s                     7s
```

Each `log_trace_span` becomes a bar. Every source file becomes a separate process row. Auto-named asyncio tasks are slot-packed into a contiguous `0..K-1` range where K is peak concurrency, so the track count stays readable even for long async runs (e.g. RL actors dispatching many concurrent RPCs).
