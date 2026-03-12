# ContextVar in Real RL Systems: Do We Actually Need It?

**Date:** 2026-03-12
**Status:** Research / Recommendation

This document answers the question: is ContextVar necessary for TorchTitan's observability
library, or can we safely use plain globals? The answer depends on understanding what real
RL systems (TBR, Forge GRPO, rllm) actually do and why.

---

## Part 1: TBR `events_logger.py` — The Original Precedent

### 1.1 Where the File Lives

The file being referenced is:
```
frameworks/msl/rl/projects/eval/utils/logging/events_logger.py
```

This file is **not available** on this machine (it lives in the internal Meta monorepo, not in
the forge checkout here). However, it was read and transcribed in a prior investigation
(`/home/felipemello/forge/reports/observability/implementationV2/contextvar_investigation.md`),
and the exact code and comments are preserved in that report and in the V2 decision document.
The following analysis is drawn directly from those primary sources.

### 1.2 Complete API Surface of `events_logger.py`

Based on the investigation report, the file exposes:

**Module-level storage (two kinds):**

```python
# Process-global — for job-level metadata that never changes during a run
_TBR_EVENT_LOGGER_CONTEXT: EventLoggerContext | None = None

# ContextVar — for per-task metadata that differs between concurrent async tasks
_TASK_CONTEXT: ContextVar[TaskContext] = ContextVar("_TASK_CONTEXT")
```

**Classes:**

```python
@dataclass
class EventLoggerContext:
    # Job-level metadata: set once at startup
    # mast_job_name, mast_job_version, host_name, global_rank, etc.

@dataclass
class TaskContext:
    """Per-task context for event logging.

    Stored in a ContextVar for isolation between concurrent tasks.
    asyncio.gather wraps coroutines in tasks, each getting a context copy,
    so setting this in one task doesn't affect others.
    """
    # Per-task state: e.g., which eval task, which step
```

The docstring on `TaskContext` is the key: it tells you exactly why ContextVar was chosen.
The author knew the difference between globals and ContextVars and made a deliberate choice.

**Functions (inferred from context):**

- `init_event_logger(...)` — sets `_TBR_EVENT_LOGGER_CONTEXT`, called once at startup
- `set_task_context(ctx: TaskContext)` — calls `_TASK_CONTEXT.set(ctx)`, called at the start of each eval task
- `log_event(event_type, ...)` — reads both `_TBR_EVENT_LOGGER_CONTEXT` (for job metadata) and `_TASK_CONTEXT.get()` (for task metadata), writes to Scuba

### 1.3 Where `events_logger.py` Is Called

TBR's controller runs many concurrent eval tasks:

```python
# TBR controller pattern (simplified from the investigation report)
async def run_evals():
    tasks = [
        asyncio.create_task(eval_one_sample(sample))
        for sample in batch
    ]
    results = await asyncio.gather(*tasks)
```

Each `eval_one_sample` call:
1. Sets its own `TaskContext` via `_TASK_CONTEXT.set(TaskContext(sample_id=..., step=...))`
2. Does async I/O (calls model servers, reward models, etc.)
3. Logs events via `log_event(...)` which reads `_TASK_CONTEXT.get()`

Because these tasks all run in the **same process on the same thread**, and because each
`asyncio.create_task()` call gives the new task a snapshot copy of the parent's ContextVar
context, writes in one task (`.set()`) are invisible to sibling tasks. Each task sees its
own isolated `TaskContext`.

### 1.4 Why ContextVar? The Concrete Problem

The TBR code comment explains it directly:

> "asyncio.gather wraps coroutines in tasks, each getting a context copy, so setting this
> in one task doesn't affect others."

Here is the exact failure mode that ContextVar prevents:

```python
# BROKEN with a plain global:
_TASK_STEP: int | None = None   # plain global

async def eval_task(step: int, sample_id: int):
    global _TASK_STEP
    _TASK_STEP = step            # Task A: sets step=42
    await call_model_server()   # <-- SUSPEND: control goes to Task B
                                # Task B: sets step=99 -- overwrites Task A's value!
    # Task A resumes:
    log_event(step=_TASK_STEP)  # logs step=99, not 42 -- WRONG
```

```python
# CORRECT with ContextVar:
_TASK_STEP: ContextVar[int | None] = ContextVar("_TASK_STEP", default=None)

async def eval_task(step: int, sample_id: int):
    _TASK_STEP.set(step)         # Task A: sets step=42 IN ITS OWN CONTEXT COPY
    await call_model_server()   # <-- SUSPEND: control goes to Task B
                                # Task B: sets step=99 IN ITS OWN CONTEXT COPY
    # Task A resumes:
    log_event(step=_TASK_STEP.get())  # logs step=42 -- CORRECT
```

The `await` is the corruption vector. Python's asyncio event loop switches between tasks
at every `await` point. If two concurrent tasks both write to a plain global, the write
from the task that runs last (before any given read) wins. This is non-deterministic and
produces exactly the bug TBR was protecting against.

### 1.5 The Hybrid Pattern: Why Not Make Everything a ContextVar?

TBR makes a deliberate split:
- Job-level metadata (`_TBR_EVENT_LOGGER_CONTEXT`): plain global, set once, never changes.
- Per-task metadata (`_TASK_CONTEXT`): ContextVar, changes per-task.

The reasoning: ContextVar has a cost in conceptual complexity. Making everything a ContextVar
would mean every call to `log_event()` needs to handle "no context set" correctly for both
the global and per-task parts. More importantly, the job-level metadata is genuinely
process-level — it's the same for every task in the process. Putting it in a ContextVar
would require explicitly `.set()`-ing it in every new task's context, which is redundant and
error-prone.

The hybrid pattern separates what genuinely varies (per-task) from what does not (per-process).

### 1.6 Borescope: A Different ContextVar Use Case

TBR's `borescope.py` also uses ContextVar, but for a completely different purpose:

```python
# tools/borescope.py line ~205
_context_var = ContextVar("context")
```

Borescope is a Perfetto-format wall-clock tracer. Its ContextVar stores `(task_name, Context)`
where `Context` holds Perfetto trace "track IDs". The purpose is: each concurrent async task
needs its own Perfetto track so that its trace events appear on the correct horizontal lane
in the Perfetto UI. Without ContextVar isolation, concurrent tasks would write events to the
same track and the trace would be unreadable.

**Do not conflate borescope's ContextVar with the logging context ContextVar.** They solve
different problems.

---

## Part 2: Forge GRPO — Where the Corruption Actually Happens

### 2.1 The Controller Pattern

`/home/felipemello/forge/apps/grpo/main.py` is the key file. The training loop creates
concurrent async tasks:

```python
# From apps/grpo/main.py lines 365-368
rollout_tasks = [
    asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
]
training_task = asyncio.create_task(continuous_training())
```

`continuous_rollouts` and `continuous_training` run in the **same process, same thread,
same event loop**. They interleave at every `await` point. `continuous_rollouts` generates
episodes, scores them, and records metrics. `continuous_training` pulls batches, trains, and
records metrics.

Neither task currently calls `set_step()` in the Forge GRPO code — the step tracking is
done inside the TitanTrainer actor (a separate OS process) and via `mlogger.flush.call_one(training_step)`.
But if our observability library were integrated into the controller with `set_step()` calls
in both `continuous_rollouts` and `continuous_training`, we would have the TBR problem exactly.

### 2.2 Empirical Evidence: 19/20 Corruptions

A test was written (`/home/felipemello/forge/reports/observability/implementationV2/test_context_isolation.py`)
to measure this empirically. From the V2 decision document:

```
Test 4: Controller concurrent tasks (the Forge pattern)
FAIL: 19 corruptions with global state!
  rollout: set 1000, got 0
  training: set 0, got 1001
  rollout: set 1001, got 1
```

```
Test 5: Controller with ContextVar (pure Python, no Monarch)
PASS: No corruption. Each task sees its own step value.
```

The 19/20 corruption rate is not theoretical — it was measured. With a global, nearly every
step the rollout task and training task see each other's step numbers.

### 2.3 Monarch Actors Are Safe with Globals

For completeness, the same test verified that Monarch actors (which run as separate OS
processes) are fully isolated regardless of globals vs ContextVars:

```
Test 1: Actors in separate proc meshes
ActorA: PID=1296768, set=42, read=42
ActorB: PID=1297390, set=77, read=77
Controller: PID=1296155, step=999
PASS: All three have isolated globals (separate PIDs)

Test 2: Multi-rank actor (procs=2)
Rank 0: PID=1298567, set=50, read=50
Rank 1: PID=1298988, set=1050, read=1050
PASS: Each rank is a separate process with isolated globals
```

The key distinction: separate OS processes → globals are safe. Same process, concurrent
async tasks → globals corrupt.

---

## Part 3: rllm — A Third Concurrent Pattern

### 3.1 The rllm SDK Pattern

`/home/felipemello/forge/frameworks/rllm` is a separate RL framework that provides an SDK
for agentic/LLM rollout workflows. Its session tracking also uses ContextVar, and its use
case illuminates a third concurrent pattern.

From `/home/felipemello/forge/frameworks/rllm/rllm/sdk/session/contextvar.py`:

```python
# Session-specific context variables
_current_session: contextvars.ContextVar["ContextVarSession | None"] = contextvars.ContextVar("current_session", default=None)
_session_name: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_name", default=None)
_metadata: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("metadata", default=None)
_sessions_stack: contextvars.ContextVar[list["ContextVarSession"] | None] = contextvars.ContextVar("sessions_stack", default=None)
```

The rllm SDK tracks which LLM call belongs to which training episode. This is not step
tracking — it is session/trajectory grouping. The use case:

```python
# From frameworks/rllm/examples/sdk/solver_judge/solver_judge_flow_session.py
async def generate_solutions(self, problem: str, n_solutions: int = 2):
    tasks = [asyncio.create_task(self.generate_solution(problem)) for _ in range(n_solutions)]
    return await asyncio.gather(*tasks)

async def generate_solution(self, problem: str) -> StepView:
    with session(agent="solver", groupby_key=str(uuid.uuid4())) as sess:
        # Each concurrent solution generation has its own session context
        response = await self.client.chat.completions.create(...)
    return sess.steps[0]   # access this task's traces, not sibling's traces
```

Here, `n_solutions` concurrent LLM calls run in parallel. Each must track its own call trace
independently. If all of them shared a global "current session" pointer, reading `sess.steps`
after `asyncio.gather` would return a mix of all tasks' LLM calls, not just the one task's calls.

The ContextVar gives each `asyncio.create_task()` its own `_current_session` pointer. When
the inner `with session(...)` context manager runs `.set()`, only the current task's context
copy is updated.

### 3.2 The rllm Test That Demonstrates the Need

From `/home/felipemello/forge/frameworks/rllm/tests/sdk/test_session_metadata_contextvar.py`:

```python
def test_parallel_sessions_contextvar():
    results: dict[str, dict[str, Any]] = {}

    def thread_a():
        with ContextVarSession(thread="A", value=1):
            results["A"] = get_current_cv_metadata()

    def thread_b():
        with ContextVarSession(thread="B", value=2):
            results["B"] = get_current_cv_metadata()

    t1 = threading.Thread(target=thread_a)
    t2 = threading.Thread(target=thread_b)
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert results["A"] == {"thread": "A", "value": 1}
    assert results["B"] == {"thread": "B", "value": 2}
```

This test uses threads (since ContextVar is both thread-safe and async-task-safe), but the
same pattern applies to concurrent async tasks. With a global dict, the two contexts would
bleed into each other.

---

## Part 4: MSL Tools — The Reference That Uses Plain Globals

### 4.1 msl_tools `context.py`

`/home/felipemello/forge/frameworks/amaia/third_party/mast_tools/monitoring/structured_logging/context.py`
(available on this machine) is the production-grade reference for structured logging context.
It uses **plain module-level globals**:

```python
GLOBAL_LOGGER_CONTEXT: t.Optional[LoggerContext] = None
THREAD_LOCAL_CONTEXT = threading.local()

def set_step(
    step: int,
    step_attempt: int = 0,
    relative_step: t.Optional[int] = None,
    step_tags: list[str] = [],
) -> None:
    if GLOBAL_LOGGER_CONTEXT is not None:
        GLOBAL_LOGGER_CONTEXT.step = step
        GLOBAL_LOGGER_CONTEXT.step_attempt = step_attempt
        GLOBAL_LOGGER_CONTEXT.step_start_time = time.time()
        ...
```

msl_tools was designed for SLURM/MAST training jobs: one process per rank, sequential
training loop, no concurrent async tasks writing to step state. Plain globals are the
correct choice for that process model.

### 4.2 The genai_evals `events_logging.py` Also Uses Globals

`/home/felipemello/forge/frameworks/amaia/third_party/genai_evals/evals/utils/monitoring/structured_logging/events_logging.py`
(available on this machine) is a simpler version of the same pattern:

```python
# context.py
GLOBAL_LOGGER_CONTEXT: t.Optional[LoggerContext] = None

def init_context(logger_meta_data: t.Optional[t.Dict[str, t.Any]]) -> None:
    global GLOBAL_LOGGER_CONTEXT
    if not _context_initialized:
        GLOBAL_LOGGER_CONTEXT = LoggerContext(meta_data=logger_meta_data)
```

This is used for eval job logging on MAST — sequential, one process per role.

---

## Part 5: Where This Leaves TorchTitan's Observability Library

### 5.1 The Decision Space

There are three distinct deployment patterns, each with different requirements:

| Pattern | Process model | Concurrent `set_step()` writers? | Globals safe? |
|---------|--------------|----------------------------------|---------------|
| **SPMD training** (TorchTitan's main case) | One OS process per rank, sequential loop | No | Yes |
| **Monarch actors** (trainer, generator, reward) | Each actor is a separate OS process, endpoints are sequential | No | Yes |
| **Forge GRPO controller** | Single process, `asyncio.create_task` for rollout + training tasks | **Yes** | **No** |

The third pattern is the only one that requires ContextVar.

### 5.2 Our Current Code

`/home/felipemello/torchtitan/torchtitan/observability/step_state.py` uses ContextVar:

```python
from contextvars import ContextVar

_STEP: ContextVar[int | None] = ContextVar("_STEP", default=None)
_STEP_TAGS: ContextVar[tuple[str, ...]] = ContextVar("_STEP_TAGS", default=())

def set_step(step: int) -> None:
    _STEP.set(step)
    _STEP_TAGS.set(())
```

### 5.3 The Monarch Per-Endpoint Problem (a ContextVar-Specific Complication)

There is a known problem with ContextVar in Monarch actors that does not exist with globals.
From `toy_rl.py`:

```python
@endpoint
async def train_step(self, tokens, labels, loss_mask) -> float:
    # TODO: ContextVar workaround — re-set step for this asyncio task
    set_step(self.trainer.step)    # <-- extra call required
    ...
```

The comment says it all. Monarch creates a **fresh asyncio task context** for each endpoint
dispatch. This means the `set_step()` call from the `set_step` endpoint (which ran in a
different asyncio task context) is invisible to `train_step`. We have to re-call `set_step()`
at the start of `train_step` to re-initialize the ContextVar in the new context.

With plain globals, this problem disappears entirely: `set_step(42)` writes to a module-level
variable that persists for the lifetime of the OS process. The next endpoint call in the
same process sees `step=42` without any workaround.

---

## Part 6: Recommendation

### Should we switch from ContextVar to globals?

**Yes, for TorchTitan's current codebase.** The evidence:

**The case for globals:**

1. **TorchTitan is SPMD.** The training loop is single-threaded and sequential. The process
   model is one rank per OS process. This is exactly msl_tools' deployment model, and
   msl_tools uses plain globals.

2. **Monarch actors are separate OS processes.** Each actor (trainer, generator, reward) has
   its own Python interpreter. Globals in the trainer process are invisible to the generator
   process. No isolation is needed.

3. **The Monarch per-endpoint problem disappears.** With globals, `set_step(42)` in one
   endpoint call persists to the next endpoint call in the same process. The `set_step()`
   workaround in `train_step` is no longer needed.

4. **Simpler code.** The `# TODO: ContextVar workaround` in `toy_rl.py` is a symptom of
   a mismatch between ContextVar semantics and Monarch's dispatch model. With globals, the
   workaround and the comment both go away.

**The case for keeping ContextVar (and why it's not strong enough for now):**

1. **The Forge GRPO controller uses concurrent async tasks.** If `set_step()` were called
   inside `continuous_rollouts()` and `continuous_training()` concurrently, a plain global
   would corrupt — exactly what the empirical test showed.

   **However:** The Forge GRPO controller lives in `apps/grpo/main.py`, which is **outside
   TorchTitan's repository**. TorchTitan's library does not run inside the controller
   process. It runs inside Monarch actors (separate OS processes). When Forge integrates
   TorchTitan's observability library, it will be in the TitanTrainer actor — not in the
   controller. The controller uses its own metric logging via `mlogger`.

2. **Future-proofing.** If someone uses TorchTitan's library directly in an async controller
   process, globals would corrupt. This is a real concern but it is not a current use case.

**The upgrade path is easy.** If the concurrent controller pattern is ever needed:
- Change `_STEP: int | None = None` to `_STEP: ContextVar[int | None] = ContextVar("_STEP", default=None)`
- Change `_STEP = step` to `_STEP.set(step)`
- Change `get_step()` returning `_STEP` to `get_step()` returning `_STEP.get()`
- Public API (`set_step`, `get_step`, `add_step_tag`, `clear_step_tags`) does not change
- All callers are unaffected

### What real RL systems teach us

| System | Uses ContextVar? | Why? |
|--------|-----------------|------|
| **msl_tools** | No — plain global | SLURM/MAST: one rank per process, sequential loop |
| **TBR `events_logger.py`** | Yes — hybrid (global for job metadata, ContextVar for task metadata) | Concurrent async eval tasks in one process |
| **TBR `borescope.py`** | Yes | Per-task Perfetto trace track IDs (different problem) |
| **rllm SDK** | Yes | Per-concurrent-rollout session tracking |
| **Forge GRPO controller** | Would need it if using `set_step()` | Concurrent rollout + training tasks |
| **Forge GRPO actors** | No — separate OS processes | One process per actor, sequential endpoints |

The pattern is clear: **ContextVar is needed when and only when concurrent async tasks in the
same process each need their own independent copy of a piece of state.**

For TorchTitan's actors (SPMD, Monarch) — that condition does not hold. For a Forge-style
concurrent controller — it does. Since TorchTitan's library runs inside actors, not
controllers, plain globals are correct now.

### Implementation Plan (if migrating)

1. **`step_state.py`**: Replace ContextVar definitions with module-level globals. Add
   `get_step()` and `get_step_tags()` getter functions so callers do not need to import
   the raw `_STEP` module attribute.

2. **`structured_logging.py`**: Replace `_STEP.get()` with `get_step()`, `_STEP_TAGS.get()`
   with `get_step_tags()`. Update imports.

3. **`metrics.py`**: Same — replace `.get()` calls with getter functions.

4. **`toy_rl.py`**: Remove the `# TODO: ContextVar workaround` and the redundant `set_step()`
   call at the start of `train_step`. The step set by the `set_step` endpoint now persists
   to subsequent endpoint calls automatically.

5. **Tests**: Replace `_STEP.set(None)` reset calls with `step_state._STEP = None` or a
   `reset_step_state()` test helper.

6. **Comment in `step_state.py`**: Replace the ContextVar rationale comment with a note
   explaining that globals are safe for SPMD/Monarch actors (separate processes, sequential
   endpoints), and that ContextVar is the upgrade path if concurrent async tasks in the same
   process need to call `set_step()` independently.

---

## Appendix: Key Files Referenced

| File | Location | What it shows |
|------|----------|---------------|
| `events_logger.py` (TBR) | Not available locally; content in contextvar_investigation.md | Hybrid global + ContextVar for concurrent eval tasks |
| `context.py` (msl_tools) | `/home/felipemello/forge/frameworks/amaia/third_party/mast_tools/monitoring/structured_logging/context.py` | Plain globals for sequential SLURM training |
| `context.py` (genai_evals) | `/home/felipemello/forge/frameworks/amaia/third_party/genai_evals/evals/utils/monitoring/context.py` | Plain globals for eval jobs |
| `contextvar.py` (rllm) | `/home/felipemello/forge/frameworks/rllm/rllm/sdk/session/contextvar.py` | ContextVar for concurrent LLM call tracking |
| `main.py` (Forge GRPO) | `/home/felipemello/forge/apps/grpo/main.py` | The concurrent rollout+training task pattern |
| `step_state.py` (ours) | `/home/felipemello/torchtitan/torchtitan/observability/step_state.py` | Current ContextVar implementation |
| `toy_rl.py` (ours) | `/home/felipemello/torchtitan/torchtitan/experiments/observability/toy_rl.py` | Shows the per-endpoint ContextVar workaround |
| `contextvar_investigation.md` | `/home/felipemello/forge/reports/observability/implementationV2/contextvar_investigation.md` | TBR events_logger code transcript |
| `decision_001_contextvar.md` (V2) | `/home/felipemello/forge/reports/observability/implementationV2/decisions/001_logging_context_contextvar.md` | Empirical test results (19/20 corruptions) |
| `contextvar_to_global_migration.md` | `/home/felipemello/torchtitan/contextvar_to_global_migration.md` | Prior analysis with before/after code changes |
