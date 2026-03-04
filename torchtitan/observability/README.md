# Observability

```
record_span()          → system JSONL    (phase timing per rank)
record_tensor_metric() → TensorMetricContext → replicate_to_host() → experiment JSONL
record_metric()        → experiment JSONL (non-tensor scalars)

DefaultAggregator reads experiment JSONL → CompositeSummaryWriter → TB/WandB
```

## Files

| File | What |
|------|------|
| `common.py` | Shared ContextVars (`_STEP`, `_STEP_TAGS`), logger names, `set_step()` |
| `structured_logging.py` | System JSONL: `record_span`, `record_event`, `init_observability` |
| `tensor_metric_context.py` | `TensorMetricContext`, `child_context`, `record_tensor_metric` |
| `tensor_metrics.py` | `MeanTMetric`, `replicate_to_host` — compile-safe tensor metrics |
| `metrics.py` | `record_metric`, `MeanMetric`, `DefaultAggregator` — non-tensor metrics |
| `backends.py` | `CompositeSummaryWriter`, TensorBoard, WandB, InMemory writers |
| `logging_boundary.py` | `EveryNSteps` — schedule gate for expensive operations |
| `profiling.py` | `Profiler` orchestrator, `profile_annotation` |
| `rollout_logger.py` | RL rollout JSONL (separate from experiment metrics) |

## Two metric pipelines

**System metrics** (`record_span`, `record_event`): Always fire, every step.
Per-rank JSONL in `system_logs/`. Used for phase timing, Chrome Trace visualization.

**Experiment metrics** (`record_tensor_metric`, `record_metric`): Fire every step,
but `replicate_to_host()` (expensive all-reduce) is gated by `EveryNSteps`.
Per-rank JSONL in `experiment_logs/`. Aggregated by `DefaultAggregator`, sent to
TensorBoard/WandB via `CompositeSummaryWriter`.
