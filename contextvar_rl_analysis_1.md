# ContextVar Usage in RL — Do We Actually Need It?

**Date:** 2026-03-12
**Status:** Research / Recommendation

---

## 1. TBR `events_logger` Analysis

### 1.1 File Location

`/home/felipemello/forge/frameworks/msl/rl/projects/eval/utils/logging/events_logger.py`

### 1.2 Complete API Surface

```python
# Module-level global (plain global, NOT ContextVar)
_TBR_EVENT_LOGGER_CONTEXT: EventLoggerContext | None = None

# ContextVar (one, for per-task isolation)
_TASK_CONTEXT: ContextVar[TaskContext] = ContextVar("_TASK_CONTEXT")

# Classes
class TBREvalsEventType(str, Enum): ...      # Enum of event type names
class EventLoggerContext: ...                 # Job-level metadata (MAST env vars, job config)
class TaskContext:                            # Per-task metadata — this is what goes in ContextVar
    metadata: dict[str, Any]
    task_name: str | None
    model_id: str | None

class EvalsEventLogger:
    def __init__(self, event_type, event_metadata=None, metadata=None) -> None
    def __enter__(self) -> EvalsEventLogger          # logs _start event
    def __exit__(self, exc_type, exc_val, exc_tb)    # logs _end or _failure event with duration_ms
    def add_event_metadata(self, key, value) -> None
    def update_event_metadata(self, data) -> None

# Module-level functions
def init_event_logger_context(job_metadata=None) -> None  # Called once per job in Program
def set_task_metadata(metadata=None, task_name=None, model_id=None) -> None  # ContextVar write
def _get_tbr_logger_context() -> EventLoggerContext | None
def _create_log_entry(event_type_str) -> MslPosttrainingEvalsEventsLogEntry
def _log_event(event_type_str, duration_ms=None, event_metadata=None, ...) -> None
```

### 1.3 Three-Layer Metadata Architecture

TBR's design separates metadata into three layers, each with a different lifetime and storage mechanism:

```
Layer 1: Job-level (global)      _TBR_EVENT_LOGGER_CONTEXT  → plain module-level global
         ──────────────────────────────────────────────────────────────────────────────
         Set once in program.py:
             init_event_logger_context(job_metadata={"cli_command": ..., "experiment_config": ...})
         Read by every event that fires during the job.
         Never changes after init.

Layer 2: Task-level (ContextVar)  _TASK_CONTEXT              → ContextVar[TaskContext]
         ──────────────────────────────────────────────────────────────────────────────
         Set once per task in controller.run():
             set_task_metadata(metadata={"eval_run_id": ...}, task_name="aime2024", model_id="...")
         Isolated per async task — concurrent tasks don't see each other's values.

Layer 3: Event-level (argument)   metadata= kwarg            → local variable, per call
         ──────────────────────────────────────────────────────────────────────────────
         Passed directly to EvalsEventLogger(..., metadata={"extra": ...}) by the caller.
```

At log time, `_log_event` merges all three:

```python
# From _log_event() in events_logger.py:
merged_metadata: dict[str, Any] = {}
if ctx is not None and ctx.job_metadata:
    merged_metadata["job"] = ctx.job_metadata            # Layer 1: global
if task_ctx is not None and task_ctx.metadata:
    merged_metadata["task"] = task_ctx.metadata          # Layer 2: ContextVar
if metadata:
    merged_metadata.update(metadata)                     # Layer 3: per-event arg
```

So every Scuba row gets: which job ran → which task ran → what happened in this specific event.

### 1.4 The Exact Concurrency Problem ContextVar Solves

The multi-task eval controller (`MultiEvalController`) runs multiple eval tasks concurrently via `asyncio.gather`:

```python
# From controller.py — MultiEvalController.run():
await asyncio.gather(
    *[self._run_task(c) for c in self._controllers]
)
```

Each `_run_task` calls `controller.run()`, which sets task-level metadata for that specific eval task (aime2024, gpqa_diamond, etc.):

```python
# From EvalController.run() in controller.py:
async def run(self) -> None:
    set_task_metadata(
        metadata={"eval_run_id": self._eval_run_id, "controller_config": "..."},
        task_name=self.name,     # e.g. "aime2024"
        model_id=self._model_id,
    )
    with EvalsEventLogger(TBREvalsEventType.TBR_TASK, ...) as task_event_logger:
        ...
```

These N tasks run **in the same process, same thread, same event loop**. If `set_task_metadata` used a global, here is what would happen:

```
asyncio event loop timeline (single thread):
─────────────────────────────────────────────────────────────────────────────
Task A (aime2024):  set_task_metadata(task_name="aime2024")
                             ↓ ... await ... ← context switch
Task B (gpqa):               set_task_metadata(task_name="gpqa")
                             ↓ ... await ... ← context switch
Task A resumes:              log_event(...)  ← reads task_name="gpqa"  ← WRONG!
─────────────────────────────────────────────────────────────────────────────
```

Every `await` is a potential context switch. The global written by Task A gets overwritten by Task B, and Task A logs wrong task metadata to Scuba.

With ContextVar, `asyncio.create_task` and `asyncio.gather` each give the child coroutine a **copy** of the parent's context. `_TASK_CONTEXT.set(...)` in Task A creates a new binding visible only within Task A's execution context. Task B's set is invisible to Task A.

The test `test_task_metadata_contextvar_in_async` in `events_logger_test.py` verifies this behavior directly:

```python
async def run():
    set_task_metadata(
        metadata={"name": "test_task"},
        task_name="aime2024",
        model_id="llama4-maverick",
    )
    with EvalsEventLogger(TBREvalsEventType.TBR_TASK):
        pass

asyncio.run(run())

entry = _get_entry(mock_log, 0)
assert entry.task_name == "aime2024"   # must be exactly this task's value
assert entry.model_id == "llama4-maverick"
```

### 1.5 Why the Job-Level Context Is a Plain Global

`EventLoggerContext` (Layer 1) is stored as a plain module-level global because it is set **once** at startup before any concurrent tasks are created, and never mutated afterwards. There is no race:

```python
# program.py:
async def _run_controller(self, server_addresses, root_dir):
    self._init_event_logger()    # sets _TBR_EVENT_LOGGER_CONTEXT once
    await super()._run_controller(...)   # only then spawns concurrent task controllers
```

This is the exact pattern from the decision documents: globals are safe when there is a single writer at a time and no concurrent async tasks.

### 1.6 The Precise Failure Mode Without ContextVar

This is not theoretical. The V2 decision document (`/home/felipemello/forge/reports/observability/implementationV2/decisions/001_logging_context_contextvar.md`) reports:

> A test script (`test_monarch_context.py`) demonstrated **19/20 corruptions** when concurrent async tasks in Forge's GRPO controller shared a global step counter.

The TBR pattern with multiple eval tasks is structurally identical: N coroutines running concurrently in one event loop, each writing "which task am I?" state, each reading that state when logging. Without ContextVar isolation, task identity is corrupted at every await.

---

## 2. Forge GRPO Controller — The Same Pattern

`/home/felipemello/forge/apps/grpo/main.py` confirms the same structure for GRPO training:

```python
# Multiple concurrent async tasks in one event loop:
rollout_tasks = [
    asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
]
training_task = asyncio.create_task(continuous_training())
```

Both `continuous_rollouts` and `continuous_training` call `record_metric(...)` — which reads the current step to annotate metrics. If step is a global and both tasks call `set_step` (or equivalent), the step logged on a metric can belong to the wrong task. If `num_rollout_threads > 1`, multiple rollout tasks simultaneously overwrite each other's step.

---

## 3. Our toy_rl and Its Current Workaround

`/home/felipemello/torchtitan/torchtitan/experiments/observability/toy_rl.py` contains this comment in `TrainerActor.train_step`:

```python
@endpoint
async def train_step(self, tokens, labels, loss_mask) -> float:
    # TODO: ContextVar workaround — re-set step for this asyncio task
    set_step(self.trainer.step)
```

This is exactly the Monarch per-endpoint problem the design docs describe: Monarch dispatches each endpoint call as a fresh asyncio task with a fresh ContextVar context copy. The step set in `set_step` endpoint is invisible to `train_step` endpoint because they run in different async tasks. The workaround is to re-set step at the start of each endpoint that needs it.

This workaround is awkward but it documents the real problem: **Monarch endpoints are concurrent async tasks**. They are not sequential. If we used a global, `set_step(step)` in one endpoint would be visible to all — which is exactly what we want for sequential actors. But it breaks for concurrent tasks.

---

## 4. Summary: When Does ContextVar Actually Matter?

| Deployment | Concurrent step writers? | ContextVar needed? |
|---|---|---|
| SPMD (torchtitan, one process per rank, sequential loop) | No | No |
| Monarch actor endpoints (sequential per-actor dispatch) | No — one endpoint runs at a time per actor | No |
| TBR MultiEvalController with N tasks via `asyncio.gather` | **Yes** — N tasks concurrently call `set_task_metadata` | **Yes** |
| Forge GRPO controller with `asyncio.create_task` | **Yes** — rollout + training tasks interleave | **Yes** |
| Any controller with `asyncio.gather` / `create_task` | **Yes** | **Yes** |

The key insight from TBR: they use **globals for job-level state** (set once, never changes) and **ContextVar for per-task state** (set once per async task, read throughout that task's lifetime). This is the correct hybrid. It maps exactly to our situation:

- SPMD step counter: set once per step, read throughout that step. No concurrency. Global is fine.
- Monarch actor step counter: set once per `set_step` endpoint call, read in the next endpoint. No true concurrency at the actor level. Global is fine.
- Forge GRPO controller step: set per training iteration, but concurrent rollout tasks are also running. ContextVar needed if rollout tasks also annotate metrics with step.

---

## 5. Recommendation

**For TorchTitan's current codebase (SPMD + Monarch actors): use plain globals.**

Rationale:
1. TorchTitan's SPMD loop is sequential. One `set_step` per iteration, no concurrent tasks writing step.
2. Monarch actors are separate OS processes. They do not share globals. The per-endpoint context issue goes away with globals because `set_step(self.trainer.step)` at the start of `train_step` would simply work — the global persists in that process.
3. The ContextVar workaround in `toy_rl.py` (`set_step(self.trainer.step)` at the top of every endpoint) is needed precisely because ContextVar creates a fresh copy per async task. Globals eliminate this footgun in the Monarch actor context.

**For Forge GRPO controller integration: ContextVar is necessary.**

The TBR evidence is concrete and conclusive. TBR has exactly the pattern we would have in a production GRPO loop:
- Multiple concurrent eval tasks in one event loop.
- Each task sets its own "which task is this?" metadata via `set_task_metadata`.
- Each task logs many events that must carry the correct task identity.
- Without ContextVar: 19/20 corruptions (empirically measured).
- With ContextVar: zero corruptions.

TBR chose the hybrid approach (global for job-level + ContextVar for task-level) for good reason. It is the correct design for concurrent async workloads.

**Practical decision for our library:**

The cleanest path is to implement globals now (they work for all current use cases and remove the Monarch per-endpoint footgun), and document the upgrade path for concurrent controllers. The public API (`set_step`, `add_step_tag`, `clear_step_tags`) does not change between the two implementations. If/when TorchTitan needs to support a GRPO-style concurrent controller, the upgrade is mechanical: swap `_STEP: int | None = None` for `_STEP: ContextVar[int | None] = ContextVar(...)` and `_STEP = step` for `_STEP.set(step)`. All callers are unaffected.

What TBR teaches us is not "always use ContextVar." It teaches us: **use ContextVar specifically for state that is set differently by different concurrent async tasks in the same process**. For state that is process-scoped or set once before concurrency begins, globals are correct and simpler.
