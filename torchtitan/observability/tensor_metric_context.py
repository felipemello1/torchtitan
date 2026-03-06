# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Invocation context for collecting tensor metrics during training steps.

Provides a context manager that collects tensor-valued metrics emitted inside
``torch.compile`` regions. Metrics are stored as tensor references (no I/O)
and reduced lazily via ``replicate_to_host()`` outside compile.

Hierarchical naming via ``child_context("layer_0")`` builds paths like
``layer_0/loss``. Merging (``merge_()``) accumulates metrics with the same key
across multiple calls (e.g., summing loss across micro-batches).

Note [On Deferred Reduction of Partial DTensors]

    When using Tensor Parallel (TP), model outputs may be Partial DTensors
    whose values are only valid after an all-reduce. Rather than reducing each
    metric individually (N NCCL calls), ``replicate_to_host()`` groups them by
    (mesh, placements, dtype, shape) and performs one collective per group.

    On non-logging steps, the context is simply dropped — no all-reduce happens.
    This gives zero overhead except on the steps where you actually read metrics.

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
    """Collects tensor metrics during a training step.

    Acts as a "collection bag" for metrics emitted inside ``torch.compile``.
    Metrics are stored as live tensor references — no I/O or all-reduce happens
    until you explicitly call ``replicate_to_host(ctx.summaries())``.

    Lifecycle:
        1. ``with TensorMetricContext() as ctx:`` — pushes onto thread-local stack
        2. ``record_tensor_metric(key, value)`` — stores tensor refs in the context
        3. ``child_context("name")`` — creates a nested scope that prefixes keys
        4. Context ``__exit__`` — pops from stack
        5. ``replicate_to_host(ctx.summaries())`` — batched all-reduce + GPU→CPU

    On non-logging steps, simply don't create the context — ``record_tensor_metric``
    becomes a no-op (returns immediately when no context is active).
    """

    def __init__(self) -> None:
        self._summaries: dict[str, TMetricValue] = {}

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

        Duplicate keys are merged via ``merge_()``, so calling this twice with
        the same name accumulates values (e.g., summing loss across micro-batches).

        Example::

            ctx.add_summary("loss", MeanTMetric(sum=loss_sum, weight=token_count))

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


# Keys are prefixed lazily on merge — the child context records bare keys
# (e.g., "loss"), and the prefix (e.g., "layer_0/") is added when update()
# merges the child into the parent. This means the compiled code never sees
# the full path, avoiding Dynamo guard issues with dynamic strings.
@contextmanager
def child_context(context_name: str):
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
