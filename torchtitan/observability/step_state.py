# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-step context: current step number and step tags."""

from contextvars import ContextVar

# ContextVars for async-safe per-task isolation (Monarch actors).
# In SPMD, these behave identically to globals.
_STEP: ContextVar[int | None] = ContextVar("_STEP", default=None)
_STEP_TAGS: ContextVar[tuple[str, ...]] = ContextVar("_STEP_TAGS", default=())


def set_step(step: int) -> None:
    """Set the current training step. All subsequent JSONL records will
    include this step number.

    Example::

        for step in range(1, num_steps + 1):
            set_step(step)
            train_step(...)
    """
    _STEP.set(step)


def add_step_tag(tag: str) -> None:
    """Annotate the current step. Tags appear in system JSONL for filtering.

    Example::

        if gc_happened:
            add_step_tag("gc")
        if is_validation:
            add_step_tag("eval")
        # system JSONL: {"normvector": {"step_tags": ["gc", "eval"]}}
    """
    current = _STEP_TAGS.get()
    if tag not in current:
        _STEP_TAGS.set(current + (tag,))


def clear_step_tags() -> None:
    """Reset step tags. Call at the start of each step."""
    _STEP_TAGS.set(())
