# Agent Task: ContextVar → Global Migration Analysis

## Objective
We're considering switching from ContextVar to plain module-level globals for step/step_tags in our observability library (`torchtitan/observability/step_state.py`). ContextVar was chosen for async task isolation in the Monarch controller, but it causes problems in Monarch actor endpoints (each endpoint gets a fresh ContextVar context, so `set_step` in one endpoint is invisible to `train_step` in another). Plain globals would solve this since actors are separate OS processes.

## Tasks

### A) Find the doc explaining all ContextVar changes
Search in these locations for a document that explains what code was changed from msl_tools/sixlib to use ContextVar instead of globals:
- `/home/felipemello/forge/reports/observability/` (recursively)
- `/home/felipemello/forge/reports/` (any migration or contextvar docs)
- Look for files mentioning "contextvar", "migration", "msl_tools", "sixlib", "logging_context", "InvocationContext"
- Already known docs (don't re-read these, just note them):
  - `/home/felipemello/forge/reports/observability/implementationV3/design/decision_001_contextvar.md`
  - `/home/felipemello/forge/reports/observability/implementationV3/design/decision_002_per_endpoint.md`
  - `/home/felipemello/forge/reports/observability/implementationV3/design/decision_003_narrowed_contextvar.md`

### B) Compare our code to the original msl_tools/sixlib
Compare the ContextVar usage in our observability code vs the original global-based approach in msl_tools. Specifically:

1. Read our current implementation:
   - `/home/felipemello/torchtitan/torchtitan/observability/step_state.py` — where `_STEP` and `_STEP_TAGS` ContextVars are defined
   - `/home/felipemello/torchtitan/torchtitan/observability/structured_logging.py` — where `_STEP.get()` and `_STEP_TAGS.get()` are read by formatters
   - `/home/felipemello/torchtitan/torchtitan/observability/metrics.py` — where `_STEP.get()` is read by ExperimentJSONFormatter and record_metric

2. Find and read the original msl_tools logging context (the doc mentions it's at `frameworks/msl/msl_tools/monitoring/structured_logging/context.py`). Look for:
   - How step is stored (global? dataclass? threading.local?)
   - How step is read by formatters
   - The `set_step` / `get_step` API

3. Find and read sixlib's InvocationContext (mentioned at `frameworks/msl/sixlib/sixlib/invocation_context.py`). Note how it differs.

4. Find and read TBR's events_logger ContextVar usage (mentioned at `frameworks/msl/rl/projects/eval/utils/logging/events_logger.py`). This is the one that originally used ContextVar for the same reason we did.

### C) Write a migration plan
Write a markdown document at `/home/felipemello/torchtitan/contextvar_to_global_migration.md` with:

1. **Summary of findings** from parts A and B
2. **What needs to change** in our code to switch from ContextVar to globals:
   - List every file that reads or writes `_STEP` or `_STEP_TAGS`
   - For each, show the before (ContextVar) and after (global) code
3. **What we lose** — the concurrent controller async task isolation. Explain clearly when this matters (only Forge GRPO pattern with concurrent rollout+training tasks) and when it doesn't (SPMD, Monarch actors, sequential controller).
4. **Recommendation** — should we use pure globals, or the dual approach (ContextVar + global fallback)?

## Important notes
- This is a RESEARCH task. Do NOT edit any source code files in torchtitan/.
- The msl_tools/sixlib paths may be in `/home/felipemello/` somewhere or may not exist on this machine. Search for them. If not found, note that and work with what's available.
