# TorchTitan Observability

Structured logging, metrics, and profiling for distributed training. Works for
both SPMD pretraining and RL with multiple actors (no shared process group).

## 1. Overview

Everything goes to structured JSONL first, then a background subprocess
collects and aggregates for dashboards.

There are two types of metrics:
- **Experiment metrics** — training values (loss, throughput, memory) that get
  aggregated across ranks and sent to WandB/TB/console.
- **System metrics** — per-rank timing and scalar snapshots for debugging tools
  (Perfetto, DuckDB, LLM agents).

```
Experiment: record_metric(key, Metric) → logger → experiment.jsonl → logging subprocess → aggregate → WandB/TB/console
System:     record_span / record_event → logger → system.jsonl     → analysis tools (Perfetto, DuckDB, LLM agents)
```

This JSONL-first design gives us two things:

1. **No metric plumbing.** Each process calls `record_metric(key, value)`
   locally. No passing dicts between functions or actors. The logging
   subprocess reads all JSONL files in the background.

2. **Debuggability.** Every rank's metrics are preserved as JSONL on disk.

### Quickstart

```python
from torchtitan.observability import (
    init_observability, set_step, add_step_tag, record_span, record_event, EventType,
)
from torchtitan.observability.metrics import record_metric, MeanMetric, NoOpMetric

init_observability(source="trainer", output_dir="./outputs")

set_step(1)
add_step_tag("gc")  # will appear in system JSONL step_tags

with record_span("trainer_time/forward_backward_s", EventType.FWD_BWD):
    loss = model(batch)
    loss.backward()

record_metric("loss/trainer_loss_mean", NoOpMetric(value=0.038))  # already all_reduced
record_metric("trainer_throughput/tps_mean", MeanMetric(sum=1234.5, weight=262))
record_event({"train.tflops": 45.6})
```

## 2. Setup

```python
from torchtitan.observability import init_observability

init_observability(source="trainer", output_dir="/path/to/output")
```

Call once per process, before any logging calls. Sets up:

```
                    ┌─ Console StreamHandler (stdout, [titan] format)
                    │
Root Logger ────────┤
                    │
                    │         ┌─ EventsOnlyFilter
System Logger ──────┤─────────┤─ StructuredLoggingHandler → system.jsonl  (record_span, record_event)
                    │         └─ StructuredJSONFormatter
                    │         └─ InflightEventTrackingHandler (crash forensics)
                    │
                    │         ┌─ ExperimentMetricsFilter
Experiment Logger ──┤─────────┤─ ExperimentLoggingHandler → experiment.jsonl  (record_metric)
                              └─ ExperimentJSONFormatter
```

Rank and source are baked into the formatters at init time. Every JSONL entry
automatically includes `rank`, `source`, `caller` (file:line:function), and
`timestamp`.

## 3. Step Context

```python
from torchtitan.observability import set_step, add_step_tag, clear_step_tags

set_step(42)              # stamps all subsequent JSONL entries with step=42
add_step_tag("gc")        # annotate: GC ran this step
add_step_tag("eval")      # annotate: validation ran this step
clear_step_tags()         # reset tags (called at the start of each step)
```

Step and tags are stored as `ContextVar`s — isolated between concurrent asyncio
tasks (for Monarch actor endpoints in RL) with no overhead in SPMD. These tags and steps
could be used for custom aggregation, e.g. in RL aggregate scores per policy version.

## 4. Experiment Metrics: record_metric

```python
from torchtitan.observability.metrics import (
    record_metric, MeanMetric, MaxMetric, SumMetric, MinMetric, NoOpMetric,
)

record_metric("trainer_throughput/tps_mean", MeanMetric(sum=1234.5, weight=262))
record_metric("trainer_gradient/norm_max", MaxMetric(value=12.3))
record_metric("trainer_memory/ooms_sum", SumMetric(value=0))
record_metric("loss/trainer_loss_mean", NoOpMetric(value=0.038))  # already all_reduced
record_metric("trainer_schedule/lr", NoOpMetric(value=3e-4))      # same on all ranks
```

Each call serializes to experiment JSONL immediately. Step comes from
`set_step()`. Rank, source, caller, and timestamp are added automatically
by the formatter.

**Limitation:** Does not yet work with tensors or inside `torch.compile`
regions. For tensor metrics, manually call `all_reduce` + `.item()` first,
then `record_metric`. In the future, we will have better support for dtensors.

### Reduce types

The reduce type tells the aggregator how to combine entries across ranks:

| Type | Constructor | Aggregation |
|------|------------|-------------|
| `MeanMetric` | `MeanMetric(sum=x, weight=1)` | `sum(sums) / sum(weights)` |
| `MaxMetric` | `MaxMetric(value=x)` | `max(values)` |
| `MinMetric` | `MinMetric(value=x)` | `min(values)` |
| `SumMetric` | `SumMetric(value=x)` | `sum(values)` |
| `NoOpMetric` | `NoOpMetric(value=x)` | Pass through (value is already reduced or identical across ranks) |

### Experiment JSONL format

```json
{"key": "trainer_throughput/tps_mean", "reduce": "MeanMetric",
 "sum": 1234.5, "weight": 262.0,
 "step": 42, "rank": 0, "source": "trainer",
 "caller": "torchtitan/trainer.py:542:train_step", "timestamp": 1708200121.724}
```

## 5. System Metrics: record_span and record_event

### record_span — timing a code region

```python
from torchtitan.observability import record_span, EventType

add_step_tag("gc")  # will appear in the system JSONL step_tags field

with record_span("trainer_time/forward_backward_s", EventType.FWD_BWD):
    output = model(batch)
    loss.backward()
```

On enter, writes a START event to system JSONL. On exit, writes an END event
with the duration in milliseconds. By default (`log_to_metrics=True`), the
exit also calls `record_metric` with the duration in **seconds**.

To write to system JSONL only (no experiment metric):

```python
with record_span("trainer/validation", EventType.EVAL, log_to_metrics=False):
    validator.validate(model_parts, step)
```

### record_event — point-in-time scalars

```python
from torchtitan.observability import record_event

record_event({"train.step": 42, "train.tflops": 45.6})
```

Writes to system JSONL only. Does NOT flow to experiment JSONL or WandB.

### System JSONL format

Values are grouped into four typed columns (`int`, `normal`, `double`,
`normvector`) for easy ingestion into tools like Grafana or DuckDB:

```json
{"int": {"step": 42, "global_rank": 0, "time_ms": 1708200121724},
 "normal": {"log_type_name": "fwd_bwd_end", "component": "trainer",
            "message": "[step 42] trainer_time/forward_backward_s fwd_bwd_end took 123.45 ms",
            "caller": "torchtitan/trainer.py:730:train_step"},
 "double": {"value": 123.45},
 "normvector": {"step_tags": ["gc"]}}
```

## 6. Aggregation (Logging Subprocess)

Aggregation runs in a **separate subprocess** — training never reads JSONL or
writes to backends. On log steps, the training process signals the subprocess:

```python
# Training process (inside MetricsProcessor.log):
torch.distributed.barrier()          # all ranks finished writing
self._log_queue.put((step, False))   # ~0.1ms, non-blocking
self.reset_counters()                # training continues immediately
```

The subprocess reads experiment JSONL, aggregates, and writes to backends:

```python
# Logging subprocess (aggregation.py:logging_worker):
step, is_validation = queue.get()
entries = read_all_new_jsonl(log_dir, offsets)  # sequential, offset tracking
aggregated = aggregate(entries)                  # group by key, reduce across ranks
logger_backend.log(aggregated, step)             # WandB, TensorBoard
logger.info(format_console_line(step, aggregated))  # [titan] console output
```

`aggregate()` groups entries by key and delegates reduction to the
`REDUCE_REGISTRY` based on each entry's `"reduce"` field.

On checkpoint resume, the subprocess skips historical data by recording file
sizes at startup — only new lines from the current run are processed.

**Timing (local filesystem, NOT NFS, 100 metrics/rank):**

| Scale | Read | Aggregate | Total |
|-------|------|-----------|-------|
| 10 ranks (1K entries) | 9ms | 0.5ms | 10ms |
| 100 ranks (10K entries) | 88ms | 6ms | 94ms |
| 500 ranks (50K entries) | 333ms | 35ms | 368ms |

None of this blocks training. Training pays only the signal cost (~0.1ms).
For shared file systems, we expect 'read' to be more expensive, but it can be mitigated
with multithreading.

## 7. File Layout

```
observability/
    __init__.py             # Public API re-exports
    step_state.py           # ContextVars: _STEP, _STEP_TAGS, set_step, add_step_tag, clear_step_tags
    _constants.py           # Logger names, metric entry markers (import cycle breaker)
    structured_logging.py   # System pipeline: init_observability, record_span, record_event, EventType
    metrics.py              # Experiment pipeline: record_metric, MetricValue types, REDUCE_REGISTRY
    aggregation.py          # aggregate() + logging_worker subprocess + JSONL readers
    logging_boundary.py     # EveryNSteps schedule
    rollout_logger.py       # RL: RolloutLogger, filter_top_bottom
    analysis.py             # Post-training: to_chrome_trace (Gantt chart from system JSONL)
    README.md               # This file
```
