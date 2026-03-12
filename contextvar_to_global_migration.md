# ContextVar to Global Migration Analysis

**Date:** 2026-03-12
**Status:** Research / Recommendation

---

## 1. Summary of Findings

### 1.1 Existing Documentation

The decision to use ContextVar (instead of plain globals) for step/step_tags storage is documented across several reports:

- **V2 Investigation 1** (`/home/felipemello/forge/reports/observability/implementationV2/contextvar_investigation.md`): First-principles analysis of globals vs threading.local vs ContextVar. Recommended **plain globals** (Option 1), reasoning that Forge's process model (one actor per proc mesh) made globals safe. This was the initial recommendation before empirical testing.

- **V2 Investigation 2** (`/home/felipemello/forge/reports/observability/implementationV2/contextvar_investigation2.md`): Deeper dive into Monarch's allocators (ProcessAllocator vs LocalAllocator). Confirmed globals are safe for production actors (separate OS processes). Identified one risk: LocalAllocator (used in tests) puts multiple actors in one process.

- **V2 Decision 001** (`/home/felipemello/forge/reports/observability/implementationV2/decisions/001_logging_context_contextvar.md`): The decision document. **Reversed the V2 Investigation 1 recommendation** based on empirical testing. A test script (`test_monarch_context.py`) demonstrated 19/20 corruptions when concurrent async tasks in Forge's GRPO controller shared a global step counter. Decision: **Use ContextVar**.

- **V3 Decision 001** (`/home/felipemello/forge/reports/observability/implementationV3/design/decision_001_contextvar.md`): Reaffirmed ContextVar for the same reasons.

- **V3 Decision 003** (`/home/felipemello/forge/reports/observability/implementationV3/design/decision_003_narrowed_contextvar.md`): Narrowed ContextVar usage to only `_STEP` and `_STEP_TAGS` (not rank, source, or output_dir, which are process-level constants stored on the formatter instance or as globals).

- **Agent E Verification** (`/home/felipemello/forge/reports/observability/implementationV3/research/agent_e_contextvar.md`): Independent verification. Confirmed the ContextVar decision is correct. Checked Monarch dispatch internals (both queue-dispatch and direct Rust dispatch modes).

- **Footgun Analysis** (`/home/felipemello/forge/reports/observability/implementationV3/research/contextvar_footgun.md`): Documents why `_STEP_TAGS` uses `tuple[str, ...]` (immutable) rather than `list[str]` (mutable) -- avoids shared-reference corruption across async tasks.

- **Per-Endpoint Solutions** (`/home/felipemello/forge/reports/observability/implementationV3/research/contextvar_per_endpoint_solutions.md`): 10 solutions for the "Monarch creates fresh ContextVar context per endpoint" problem. Our implementation uses **Solution 8/9**: handler/formatter stores rank/source as instance attributes (set once), ContextVar only for step (the dynamic part).

### 1.2 What msl_tools Does (the original)

**File:** `frameworks/msl/msl_tools/monitoring/structured_logging/context.py`

msl_tools stores step as a **plain module-level global**:

```python
GLOBAL_LOGGER_CONTEXT: Optional[LoggerContext] = None

def set_step(step, step_attempt=0, relative_step=None, step_tags=[]):
    if GLOBAL_LOGGER_CONTEXT is not None:
        GLOBAL_LOGGER_CONTEXT.step = step
        GLOBAL_LOGGER_CONTEXT.step_attempt = step_attempt
        GLOBAL_LOGGER_CONTEXT.step_start_time = time.time()
        GLOBAL_LOGGER_CONTEXT.step_tags = step_tags
```

The formatter (`ScubaJSONFormatter`) reads `GLOBAL_LOGGER_CONTEXT.step` directly at format time. This works because msl_tools runs in a single-threaded-per-process SLURM/MAST setup with no concurrent async tasks.

### 1.3 What sixlib Does

**File:** `frameworks/msl/sixlib/sixlib/invocation_context.py`

sixlib's `InvocationContext` is a **completely different system** -- it is a per-step metric accumulation side-channel (for `add_summary` inside `torch.compile`), not a logging context for step annotation. It uses `threading.local` for a stack-based context manager. It does NOT store step for log formatters. Not relevant to this migration.

### 1.4 What TBR Does

**File:** `frameworks/msl/rl/projects/eval/utils/logging/events_logger.py`

TBR uses a **hybrid approach**: plain global for job-level metadata that never changes, ContextVar for per-task metadata (because TBR runs concurrent eval tasks via `asyncio.gather`). This is the closest precedent to our use case.

### 1.5 What Our Code Does

Our implementation (Solution 8/9 from the per-endpoint analysis) already separates concerns:

- **Process-level constants** (rank, source): stored on the `StructuredJSONFormatter` and `ExperimentJSONFormatter` instances as `self.rank` and `self.source`. Set once in `init_observability()`. Not in ContextVars.
- **Per-step dynamic state** (step, step_tags): stored in ContextVars `_STEP` and `_STEP_TAGS`. Set via `set_step()` and `add_step_tag()`.

---

## 2. What Needs to Change

### 2.1 Complete File Inventory

Every file that reads or writes `_STEP` or `_STEP_TAGS`:

| File | Reads/Writes | What it does |
|------|-------------|--------------|
| `torchtitan/observability/step_state.py` | Defines + writes + reads | Defines `_STEP` and `_STEP_TAGS` ContextVars; `set_step()`, `add_step_tag()`, `clear_step_tags()` |
| `torchtitan/observability/structured_logging.py` | Reads `_STEP`, `_STEP_TAGS` | `StructuredJSONFormatter._log_dict()` reads step/tags; `record_event()` reads step; `record_span.__enter__/__exit__` reads step |
| `torchtitan/observability/metrics.py` | Reads `_STEP` | `record_metric()` checks step is set; `ExperimentJSONFormatter.format()` reads step |
| `torchtitan/observability/__init__.py` | Re-exports | Exports `set_step`, `add_step_tag`, `clear_step_tags` |
| `torchtitan/experiments/observability/toy_spmd.py` | Calls `set_step`, `add_step_tag` | Training loop |
| `torchtitan/experiments/observability/toy_rl.py` | Calls `set_step` | Actor endpoints and controller |
| `torchtitan/experiments/observability/metrics_processor.py` | Calls `set_step` | MetricsProcessor wrapper |
| `tests/unit_tests/observability/test_structured_logging.py` | Calls + reads | Tests for step_state, formatters, record_span |
| `tests/unit_tests/observability/test_metrics.py` | Calls + reads | Tests for record_metric, ExperimentJSONFormatter |

### 2.2 Code Changes: Before and After

#### File 1: `torchtitan/observability/step_state.py`

**Before (ContextVar):**
```python
from contextvars import ContextVar

_STEP: ContextVar[int | None] = ContextVar("_STEP", default=None)
_STEP_TAGS: ContextVar[tuple[str, ...]] = ContextVar("_STEP_TAGS", default=())

def set_step(step: int) -> None:
    _STEP.set(step)
    _STEP_TAGS.set(())

def add_step_tag(tag: str) -> None:
    current = _STEP_TAGS.get()
    if tag not in current:
        _STEP_TAGS.set(current + (tag,))

def clear_step_tags() -> None:
    _STEP_TAGS.set(())
```

**After (globals):**
```python
_STEP: int | None = None
_STEP_TAGS: tuple[str, ...] = ()

def set_step(step: int) -> None:
    global _STEP, _STEP_TAGS
    _STEP = step
    _STEP_TAGS = ()

def add_step_tag(tag: str) -> None:
    global _STEP_TAGS
    if tag not in _STEP_TAGS:
        _STEP_TAGS = _STEP_TAGS + (tag,)

def clear_step_tags() -> None:
    global _STEP_TAGS
    _STEP_TAGS = ()
```

#### File 2: `torchtitan/observability/structured_logging.py`

**Before:** `_STEP.get()` and `_STEP_TAGS.get()` (6 call sites)

**After:** `_STEP` and `_STEP_TAGS` (direct module attribute reads)

Specific lines:
```python
# Line 279: StructuredJSONFormatter._log_dict()
step = _STEP       # was: step = _STEP.get()

# Line 282:
step_tags = _STEP_TAGS  # was: step_tags = _STEP_TAGS.get()

# Line 501: record_event()
step = _STEP       # was: step = _STEP.get()

# Line 569: record_span.__enter__()
step = _STEP       # was: step = _STEP.get()

# Line 579: record_span.__exit__()
step = _STEP       # was: step = _STEP.get()
```

The import also changes:
```python
# Before:
from torchtitan.observability.step_state import _STEP, _STEP_TAGS

# After (must import the module to see mutations):
from torchtitan.observability import step_state
# Then use: step_state._STEP, step_state._STEP_TAGS
```

**Important:** With globals, you cannot `from module import _STEP` and expect to see mutations. You must import the module and access the attribute (`step_state._STEP`), or use a getter function.

#### File 3: `torchtitan/observability/metrics.py`

**Before:** `_STEP.get()` (2 call sites)

**After:** `step_state._STEP` (or getter function)

```python
# Line 176: record_metric()
if step_state._STEP is None:   # was: if _STEP.get() is None:

# Line 208: ExperimentJSONFormatter.format()
step = step_state._STEP        # was: step = _STEP.get()
```

#### Alternative: Use getter functions (cleaner)

Instead of exposing the raw global, add `get_step()` and `get_step_tags()` to `step_state.py`:

```python
def get_step() -> int | None:
    return _STEP

def get_step_tags() -> tuple[str, ...]:
    return _STEP_TAGS
```

Then all consumers call `get_step()` instead of `_STEP.get()`. The import/mutation problem disappears, and the public API is identical to the ContextVar version. This is the cleanest migration path.

### 2.3 Test Changes

Tests that directly access `_STEP` (e.g., `_STEP.set(None)` for cleanup) would change to `step_state._STEP = None` or `set_step(...)` / a reset function. The `conftest.py` fixture that resets ContextVars between tests would reset globals instead.

---

## 3. What We Lose

### 3.1 Concurrent Controller Async Task Isolation

**This is the only thing we lose.**

The original decision to use ContextVar was driven by a specific, empirically-verified scenario:

**Forge's GRPO controller** (`apps/grpo/main.py`) creates concurrent async tasks:
```python
rollout_tasks = [asyncio.create_task(continuous_rollouts()) for _ in range(N)]
training_task = asyncio.create_task(continuous_training())
```

These tasks run in the **same process, same thread, same event loop**. If both call `set_step()`, they overwrite each other at every `await` point. The empirical test showed **19/20 corruptions** with globals.

With ContextVar, each `asyncio.create_task()` gets a snapshot of the parent's context. `_STEP.set(42)` in one task is invisible to other tasks. Zero corruptions.

### 3.2 When Does This Matter?

| Deployment pattern | Concurrent step writers? | Globals safe? |
|---|---|---|
| **SPMD** (torchtitan's main use case) | No -- one process per rank, sequential training loop | Yes |
| **Monarch actors** (trainer, generator, reward) | No -- each is a separate OS process, endpoints are sequential | Yes |
| **Monarch multi-rank actors** (procs=2+) | No -- separate OS process per rank | Yes |
| **Monarch service replicas** | No -- separate OS process per replica | Yes |
| **Sequential controller** (one task at a time) | No | Yes |
| **Forge GRPO concurrent controller** | **Yes** -- rollout + training tasks interleave | **No** |
| **Any controller with `asyncio.gather` / `create_task`** | **Yes** | **No** |

### 3.3 When Does This NOT Matter?

For TorchTitan's current codebase (SPMD training, toy examples), there are zero concurrent async tasks writing to step. Globals are completely safe.

The concurrent controller pattern exists only in Forge's `apps/grpo/main.py`, which is outside TorchTitan's repository.

---

## 4. Recommendation

### Use pure globals.

**Rationale:**

1. **TorchTitan is an SPMD framework.** Its process model is one-process-per-rank with a sequential training loop. There are no concurrent async tasks writing step state. This matches msl_tools' deployment model exactly, and msl_tools uses plain globals.

2. **The Monarch controller pattern is out of scope.** The concurrent `asyncio.create_task` pattern that triggers corruption lives in Forge's `apps/grpo/main.py`, not in TorchTitan. If/when TorchTitan needs to support that pattern, the migration to ContextVar can be done at that time.

3. **Simpler code.** ContextVar adds conceptual overhead (context inheritance semantics, mutable-reference footguns, the `from module import` gotcha) with zero benefit in the current codebase. The current `step_state.py` is 50 lines; the global version would be equally small and more obvious.

4. **The Monarch per-endpoint problem disappears.** The entire investigation into "Monarch creates fresh ContextVar contexts per endpoint" and the 10 proposed solutions become irrelevant with globals. In Monarch actors (separate OS processes), globals persist across endpoint dispatches naturally. No `init_context()` per endpoint needed.

5. **Easy upgrade path.** If concurrent async step writing becomes a requirement:
   - Replace `_STEP: int | None = None` with `_STEP: ContextVar[int | None] = ContextVar("_STEP", default=None)`
   - Replace `_STEP = step` with `_STEP.set(step)`
   - Replace `get_step()` returning `_STEP` with `get_step()` returning `_STEP.get()`
   - The public API (`set_step`, `get_step`, `add_step_tag`, `clear_step_tags`) does not change.
   - Callers are unaffected.

6. **Consistency with the existing architecture.** The formatter already stores rank and source as instance attributes (globals, effectively). Moving step to the same pattern is consistent. The current code is a hybrid: ContextVar for step, instance attributes for rank/source. Pure globals for step would make the storage model uniform.

### Why NOT the dual approach (ContextVar + global fallback)?

A dual approach adds complexity without solving a real problem. If we need ContextVar isolation, we need it everywhere (not sometimes). If we don't need it, we don't need the ContextVar machinery at all. The current codebase does not need it.

### Implementation Plan

1. Change `step_state.py`: Replace ContextVar definitions with module-level globals. Add `get_step()` and `get_step_tags()` getter functions.
2. Change `structured_logging.py`: Replace `_STEP.get()` with `get_step()`, `_STEP_TAGS.get()` with `get_step_tags()`. Update import.
3. Change `metrics.py`: Same pattern -- use getter functions.
4. Update tests: Replace `_STEP.set(None)` resets with `step_state._STEP = None` or a `reset_step_state()` test helper.
5. Update docstrings/comments: Remove references to ContextVar. Add a comment noting that globals are safe for SPMD and that ContextVar is the upgrade path for concurrent async patterns.
