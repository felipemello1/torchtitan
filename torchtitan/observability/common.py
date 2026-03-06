# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared constants and ContextVars for the observability module.

When producing JSONL logs, we want to embed information in them through handlers,
such as the current training step. This info normally can be stored in **global variables** for SPMD jobs.

However, in Monarch actor endpoints, each request can run as a separate
asyncio task. This means that global variables would be shared and overwritten
across concurrent requests. To solve it, we use ContextVars, that give each task its own copy.

For SPMD (single-process-per-rank), ContextVars behave identically to globals.
"""

from contextvars import ContextVar

# --- ContextVars for mutable per-task state ---
# Rank and source are constants stored on the formatter (set once in init_observability).
# Step and step_tags change every step, so they use ContextVars for async safety.
_STEP: ContextVar[int | None] = ContextVar("_STEP", default=None)
_STEP_TAGS: ContextVar[tuple[str, ...]] = ContextVar("_STEP_TAGS", default=())

# --- Logger names ---
# Separate loggers so system events (e.g. phase timing) and experiment metrics
# (e.g. loss, reward) go to independent JSONL files with independent formatters.
SYSTEM_LOGGER_NAME = "torchtitan.observability.system"
EXPERIMENT_LOGGER_NAME = "torchtitan.observability.experiment"

# --- Metric entry markers ---
# Keys set on LogRecord.extra to distinguish record_metric entries from
# log_reduced_metrics entries in ExperimentJSONFormatter.
_METRIC_ENTRY = "_metric_entry"
_REDUCED_METRICS = "_reduced_metrics"


def set_step(step: int) -> None:
    """Set the current training step into the context, so it can be embedded in JSONL records."""
    _STEP.set(step)


def add_step_tag(tag: str) -> None:
    """Add a tag to the current step (e.g., "gc_collect", "profiling", "eval").
    Tags are embedded in JSONL records for filtering.

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
