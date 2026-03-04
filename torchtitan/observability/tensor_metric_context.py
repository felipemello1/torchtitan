# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Invocation context for collecting tensor metrics during training steps.

Collects tensor metrics during training steps, supporting hierarchical
naming via child_context and deferred reduction via replicate_to_host.

Usage:
    with TensorMetricContext() as ctx:
        # Inside compiled model forward:
        record_tensor_metric("loss", MeanTMetric(sum=loss_sum, weight=mask_sum))
    # Outside compile:
    scalars = replicate_to_host(ctx.summaries())
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.utils._pytree as pytree
from torchtitan.observability import tree
from torchtitan.observability.tensor_metrics import TMetricValue


@dataclass
class _ContextStack(threading.local):
    """A thread-local stack of invocation contexts.

    The stack is used to store the current invocation context.
    """

    thread_id: int
    stack: list["TensorMetricContext"] = field(default_factory=list)


_global_context_stack = _ContextStack(thread_id=threading.get_ident())


class TensorMetricContext:
    def __init__(self) -> None:
        self._summaries: dict[str, TMetricValue] = {}
        self._global_step: int | None = None

    def __enter__(self) -> "TensorMetricContext":
        _global_context_stack.stack.append(self)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if (
            not _global_context_stack.stack
            or _global_context_stack.stack[-1] is not self
        ):
            raise RuntimeError(
                f"Self is not the top of the context stack: {_global_context_stack}"
            )
        _global_context_stack.stack.pop()

    def add_summary(self, name: str, value: TMetricValue) -> None:
        """Adds a summary to the context.

        The summary will be returned by `summaries()`. If a summary with the same name already
        exists, the values are merged using `TMetricValue.merge_()`.

        Args:
            name: The name of the summary.
            value: The value of the summary. Must be a TMetricValue (e.g., MeanTMetric).

        Raises:
            TypeError: If value is not a TMetricValue.
            ValueError: If value contains Partial DTensors with incompatible reduction types.
        """
        if not isinstance(value, TMetricValue):
            raise TypeError(f"Value must be a TMetricValue, got {type(value)}")

        value.validate_placements(name)

        with torch.no_grad():
            value = tree.map(
                lambda x: x.detach()
                if isinstance(x, torch.Tensor) and x.requires_grad
                else x,
                value,
            )
            if name in self._summaries:
                self._summaries[name].merge_(value)
            else:
                self._summaries[name] = value

    def summaries(self) -> dict[str, TMetricValue]:
        """Returns the summaries added to the context.

        Note: The returned metrics may contain Partial DTensors that have not been reduced.
        Use `replicate_to_host()` to reduce and convert to Python scalars for logging.
        For single-metric access, use `maybe_full_tensor(metric.value()).item()`.

        See Note [On Deferred Reduction of Partial DTensors] in the module docstring for details.

        Returns:
            A shallow copy of the summaries dictionary. Modifying it does not affect the context.
        """
        return self._summaries.copy()

    def set_global_step(self, step: int) -> None:
        """Sets the global training step.

        Args:
            step: The global training step.
        """
        self._global_step = step

    def global_step(self) -> int | None:
        """Returns the global training step.

        If the current context doesn't have global_step set, searches up the
        context stack to find the nearest ancestor with global_step set.

        Returns:
            The global training step, or None if not set in any ancestor context.
        """
        if self._global_step is not None:
            return self._global_step

        # Search up the context stack for global_step (ancestors only)
        if not _global_context_stack.stack:
            return None

        # Find the index of self in the stack
        try:
            self_index = _global_context_stack.stack.index(self)
        except ValueError:
            # self is not in the stack, can't search ancestors
            return None

        # Search ancestors (contexts before self in the stack) in reverse order
        for i in range(self_index - 1, -1, -1):
            context = _global_context_stack.stack[i]
            if context._global_step is not None:
                return context._global_step

        return None

    def update(self, other: "TensorMetricContext", prefix: str | None) -> None:
        """Merge another context's summaries into this one.

        Summaries with the same key are merged via merge_(). If prefix is given,
        all keys from other are prefixed with "prefix/". This is how child_context
        builds hierarchical metric names.

        Args:
            other: The child context to merge from.
            prefix: Prefix to prepend to summary keys (e.g., layer name).
        """
        for name, summary in other._summaries.items():
            prefixed_name = name if prefix is None else prefix + "/" + name
            if prefixed_name in self._summaries:
                self._summaries[prefixed_name].merge_(summary)
            else:
                self._summaries[prefixed_name] = summary

    def __flatten__(self):
        """Flattens into (leaves, ctx) for pytree registration."""
        summary_leaves, summary_spec = pytree.tree_flatten(self._summaries)
        return summary_leaves, summary_spec

    @classmethod
    def __unflatten__(cls, leaves, ctx):
        """Reconstructs from (leaves, ctx)."""
        output = cls()
        output._summaries = pytree.tree_unflatten(leaves, ctx)
        return output


def current_tensor_metric_context() -> TensorMetricContext | None:
    if not _global_context_stack.stack:
        return None
    return _global_context_stack.stack[-1]


# See Note [Lazily generating add_summary keys]
@contextmanager
def child_context(context_name: str) -> None:
    try:
        with TensorMetricContext() as ctx:
            yield
    finally:
        parent_ctx = current_tensor_metric_context()
        if parent_ctx:
            parent_ctx.update(ctx, context_name)


# Register a pytree node for compatibility with AOTAutograd.
pytree.register_pytree_node(
    TensorMetricContext,
    lambda x: x.__flatten__(),
    TensorMetricContext.__unflatten__,
)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def record_tensor_metric(key: str, value: TMetricValue) -> None:
    """Record a tensor metric in the current TensorMetricContext.

    No-op if no context is active (non-logging steps).
    Safe to call inside torch.compile — stores tensor refs without I/O.

    Args:
        key: Metric name (e.g., "loss", "grad_norm").
        value: TMetricValue instance (MeanTMetric, MaxTMetric, etc.).
    """
    ctx = current_tensor_metric_context()
    if ctx is None:
        return
    ctx.add_summary(key, value)
