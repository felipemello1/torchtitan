# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared constants and ContextVars for the observability module.

No dependencies on other observability submodules — this file exists to break
circular imports between structured_logging.py and metrics.py (added in PR4).

ContextVars are used instead of globals because they provide isolation between
concurrent asyncio tasks. In Monarch actor endpoints, each request runs as a
separate asyncio task — global variables would be shared and overwritten
across concurrent requests. ContextVars give each task its own copy.

For SPMD (single-process-per-rank), ContextVars behave identically to globals.
"""

from contextvars import ContextVar

# --- ContextVars for mutable per-task state ---
# Rank and source are constants stored on the formatter (set once in init_observability).
# Step and step_tags change every step, so they use ContextVars for async safety.
_STEP: ContextVar[int | None] = ContextVar("_STEP", default=None)
_STEP_TAGS: ContextVar[tuple[str, ...]] = ContextVar("_STEP_TAGS", default=())

# --- Logger names ---
SYSTEM_LOGGER_NAME = "torchtitan.observability.system"
EXPERIMENT_LOGGER_NAME = "torchtitan.observability.experiment"

# --- Metric entry markers (used by PR4's ExperimentJSONFormatter) ---
_METRIC_ENTRY = "_metric_entry"
_REDUCED_METRICS = "_reduced_metrics"


def set_step(step: int) -> None:
    """Set the current training step.

    Called once per step (SPMD) or once per endpoint call (Monarch actors).
    The step is embedded in every JSONL record by the formatter.
    """
    _STEP.set(step)


def add_step_tag(tag: str) -> None:
    """Add a tag to the current step (e.g., "gc_collect", "profiling", "eval").

    Tags are embedded in JSONL records for filtering. Uses tuple (immutable)
    instead of list to avoid the ContextVar mutable-reference footgun:
    ContextVar.set() copies the reference, so a mutable list would leak
    mutations across asyncio tasks.

    Example:
        add_step_tag("profiling")  # mark this step as profiled
        add_step_tag("gc_collect")  # mark GC happened this step
    """
    current = _STEP_TAGS.get()
    if tag not in current:
        _STEP_TAGS.set(current + (tag,))


def clear_step_tags() -> None:
    """Clear all step tags. Call at the start of each step."""
    _STEP_TAGS.set(())
