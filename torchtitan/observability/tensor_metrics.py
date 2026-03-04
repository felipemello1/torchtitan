# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor metric types for compile-safe experiment metrics.

MeanTMetric, SumTMetric, MaxTMetric, MinTMetric — accumulate values inside
torch.compile regions. Use replicate_to_host() to batch-reduce and convert
to Python scalars for logging.
"""

import enum
import os
from collections.abc import Callable
from typing import Any, Self

import torch
from torch import Tensor
from collections import defaultdict

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate

from torchtitan.observability import tree

Scalar = Tensor | int | float


def _get_partial_reduce_ops(value: Tensor | DTensor | None) -> list[str]:
    """Returns all reduce_ops from Partial placements if present.

    Args:
        value: A Tensor or DTensor to check.

    Returns:
        A list of reduce_op strings (e.g., "sum", "avg", "max") for all Partial
        placements in the DTensor, or an empty list if none are present.
    """
    if isinstance(value, DTensor):
        return [
            placement.reduce_op
            for placement in value.placements
            if isinstance(placement, Partial)
        ]
    return []


def _check_merge_compatibility(
    self_value: Scalar,
    other_value: Scalar,
    context: str,
) -> None:
    """Check that two values are compatible for merging.

    Enforces consistent types across metric merge operations:
    1. Both must be tensors OR both must be Python scalars
    2. For tensors: dtypes must be compatible. Int can be promoted to float
       (lossless), but merging float into int is rejected (lossy).
    3. For tensors: both must be DTensors OR both must be regular Tensors

    Args:
        self_value: The value from self (the metric being merged into).
        other_value: The value from other (the metric being merged from).
        context: Description for error messages (e.g., "MeanTMetric", "SumTMetric").

    Raises:
        TypeError: If tensor vs scalar types don't match, or if dtypes are incompatible.
        ValueError: If DTensor vs Tensor types don't match.
    """
    self_is_tensor = isinstance(self_value, Tensor)
    other_is_tensor = isinstance(other_value, Tensor)

    # Check 1: tensor vs scalar consistency
    if self_is_tensor != other_is_tensor:
        raise TypeError(
            f"Cannot merge {context} with mixed types: "
            f"self is {'Tensor' if self_is_tensor else 'scalar'}, "
            f"other is {'Tensor' if other_is_tensor else 'scalar'}. "
            f"Use consistent types across all merged values."
        )

    if self_is_tensor:
        # Check 2: dtype compatibility
        # Allow int→float promotion (lossless), reject float→int (lossy)
        self_is_float = self_value.dtype.is_floating_point
        other_is_float = other_value.dtype.is_floating_point

        # Reject if self is int but other is float (would lose precision)
        if not self_is_float and other_is_float:
            raise TypeError(
                f"Cannot merge {context}: cannot merge float ({other_value.dtype}) "
                f"into int ({self_value.dtype}). Initialize with float dtype to "
                f"allow merging float values."
            )

        # Check 3: DTensor vs regular Tensor consistency
        self_is_dtensor = isinstance(self_value, DTensor)
        other_is_dtensor = isinstance(other_value, DTensor)

        if self_is_dtensor != other_is_dtensor:
            raise ValueError(
                f"Cannot merge {context} with mixed tensor types: "
                f"{'DTensor' if self_is_dtensor else 'Tensor'} and "
                f"{'DTensor' if other_is_dtensor else 'Tensor'}. "
                f"Ensure all values are either DTensors or regular Tensors."
            )


def _validate_no_tensors_in_context(context: Any, cls_name: str) -> None:
    """Validates that pytree context doesn't contain tensors.

    Tensors must be in leaves (first return value of __flatten__), not context (second return
    value), so that replicate_to_host() can find and convert them.

    This validation only runs in debug mode (debug_level > 0) to avoid overhead in production.

    Args:
        context: The context value from __flatten__.
        cls_name: The class name for error messages.

    Raises:
        ValueError: If context contains a Tensor.
    """
    # Only validate in debug mode (env var check avoids overhead in production)
    debug_level = int(os.environ.get("TORCHTITAN_DEBUG_LEVEL", "0"))
    if debug_level <= 0:
        return
    for leaf in tree.leaves(context):
        if isinstance(leaf, Tensor):
            raise ValueError(
                f"{cls_name}.__flatten__() returned a Tensor in context. "
                f"Tensors must be in leaves (first element), not context (second element), "
                f"so that replicate_to_host() can convert them to scalars."
            )


class TMetricValue:
    """Base class for metric values that can be merged and reduced.

    Subclasses must implement:
    - `value()`: Returns the scalar value of the metric.
    - `merge_()`: Merges another metric into this one.
    - `__flatten__()`: Returns (leaves, context) for pytree traversal.
    - `__unflatten__()`: Class method to reconstruct from (leaves, context).

    The `__flatten__` and `__unflatten__` methods enable pytree traversal, which is required for
    `replicate_to_host()` to convert DTensors to scalars. Subclasses are automatically registered
    with pytree via `__init_subclass__`.

    When used with `TensorMetricContext`, metrics may contain Partial DTensors whose reduction is
    deferred until `summaries()` is called. Each subclass should implement
    `validate_placements()` to check that DTensor placements are compatible with its merge
    semantics.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        """Automatically register subclasses with pytree."""
        super().__init_subclass__(**kwargs)

        # Wrap __flatten__ to validate that context doesn't contain tensors.
        # Use factory function to capture cls_name by value, not reference.
        def make_validated_flatten(cls_name: str):
            def validated_flatten(self):
                leaves, context = self.__flatten__()
                _validate_no_tensors_in_context(context, cls_name)
                return leaves, context

            return validated_flatten

        tree.register_pytree_node(
            cls, make_validated_flatten(cls.__name__), cls.__unflatten__
        )

    def value(self) -> Scalar:
        """Returns the value of the metric."""
        raise NotImplementedError(type(self))

    def merge_(self, other: Self) -> Self:
        """Merges `other` into `self`."""
        raise NotImplementedError(type(self))

    def __flatten__(self) -> tuple[list, Any]:
        """Returns (leaves, context) for pytree flattening.

        Leaves should be tensors or scalars that need to be traversed by `replicate_to_host()`.
        Context stores any additional non-tensor state needed for reconstruction (e.g., reduction
        type, compute function).

        IMPORTANT: Tensors MUST be in leaves, NOT in context. The context is not traversed by
        `replicate_to_host()`, so any tensors there won't be converted to scalars. This is validated
        automatically when the class is registered.

        Returns:
            A tuple of (leaves, context) where leaves is a list of tensor/scalar
            values and context is any hashable non-tensor value for reconstruction.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement __flatten__ for pytree support. "
            f"This is required for replicate_to_host() to work correctly."
        )

    @classmethod
    def __unflatten__(cls, leaves: list, context: Any) -> Self:
        """Reconstructs an instance from (leaves, context).

        Args:
            leaves: The list of tensor/scalar values from __flatten__.
            context: The context value from __flatten__.

        Returns:
            A new instance of this class.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement __unflatten__ for pytree support. "
            f"This is required for replicate_to_host() to work correctly."
        )

    def validate_placements(self, name: str) -> None:
        """Validates that DTensor placements are compatible with deferred reduction.

        This method is called by `TensorMetricContext.add_summary()` to ensure that deferred reduction
        will produce correct results. Subclasses may override this method to allow specific Partial
        DTensor placements that are compatible with their merge semantics.

        For deferred reduction to be correct, the reduction operation R must commute with the merge
        operation ⊕. That is, for values v_i on each rank i:

            R(v_1 ⊕ v_2 ⊕ ... ⊕ v_n) == R(v_1) ⊕ R(v_2) ⊕ ... ⊕ R(v_n)

        This ensures that the order of operations (reduce-then-merge vs merge-then-reduce) does not
        affect the final result. Each TMetricValue subclass must validate that any Partial DTensor
        uses a reduce_op that satisfies this property for its specific merge_() implementation.

        The default implementation rejects any Partial DTensors, requiring all tensors to be fully
        reduced (regular Tensors or Replicated DTensors). This is the backwards-compatible behavior
        that ensures correctness for all metric types.

        Args:
            name: The summary name, used for error messages.

        Raises:
            ValueError: If a Partial DTensor is found.
        """
        # Use class annotations to find tensor attributes. This is cached at the class level.
        for attr_name in type(self).__annotations__:
            reduce_ops = _get_partial_reduce_ops(getattr(self, attr_name, None))
            if reduce_ops:
                raise ValueError(
                    f"Summary '{name}' contains a Partial DTensor in {type(self).__name__}. "
                    f"Partial DTensors are not supported by default. Either reduce the tensor "
                    f"before adding the summary, or use a TMetricValue subclass that explicitly "
                    f"supports deferred reduction (e.g., MeanTMetric with Partial mean)."
                )


class MeanTMetric(TMetricValue):
    """A weighted scalar represents a weighted summable value.

    Internally stores (sum, weight) where sum = mean * weight. The mean is computed lazily
    in value() as sum / weight, allowing both sum and weight to remain as Partial DTensors
    until reduction is needed.

    Weight should be a scalar and is assumed to be non-negative.
    A weight of zero corresponds to zero mean.

    Values preserve their original types: Python scalars remain scalars, tensors remain tensors.
    No implicit conversion occurs. merge_() enforces type consistency between operands.
    """

    _sum: Scalar
    _weight: Scalar

    def __init__(
        self,
        *,
        sum: Scalar,  # noqa: A002
        weight: Scalar = 1,
    ) -> None:
        """Initialize a MeanTMetric.

        Args:
            sum: The numerator (sum of values).
            weight: The denominator (count). Defaults to 1.

        Example:
            MeanTMetric(sum=loss)              # single value, weight=1
            MeanTMetric(sum=total, weight=n)   # weighted: mean = total / n
        """
        self._sum = sum
        self._weight = weight

    @property
    def sum(self) -> Scalar:
        """Returns the sum (numerator)."""
        return self._sum

    @property
    def weight(self) -> Scalar:
        """Returns the weight (denominator)."""
        return self._weight

    def value(self) -> Scalar:
        """Returns the mean value (sum / weight).

        The division happens here, after any Partial DTensors have been reduced via replicate().
        Uses the double-where trick to avoid div-by-zero and possible NaN grads for tensor case.

        Always returns a float (or float tensor), even for integer sum and weight, since
        the mean of integers is typically a float (e.g., mean of [1, 2] is 1.5).

        When _weight is a Python scalar (after replicate_to_host, or when weight is a constant),
        uses simple Python division instead of torch.where.
        """
        sum_val = self._sum
        weight = self._weight

        # If weight is a Python scalar, use simple conditional division.
        # This handles the post-replicate_to_host() case where values are scalars,
        # and future cases where sum may be a Tensor but weight remains a number.
        if not isinstance(weight, Tensor):
            return sum_val / weight if weight > 0 else 0.0

        # Redistribute weight to replicated to avoid implicit redistribution in torch.clamp
        weight = to_replicated_dtensor(weight)
        return to_replicated_dtensor(sum_val / torch.clamp(weight, min=1e-8))

    def validate_placements(self, name: str) -> None:
        """Validates that DTensor placements are compatible with deferred reduction.

        For MeanTMetric with (sum, weight) representation:
        - Sum can be `Partial("sum")` or `Partial("avg")`. These placements are preserved
            through `merge_()` operations (simple addition), allowing reduction to be deferred
            until `summaries()` is called.
        - Weight can also be `Partial("sum")` or `Partial("avg")`. Weights are added during
            merge_(), and both sum and avg reductions commute with addition.

        Args:
            name: The summary name, used for error messages.

        Raises:
            ValueError: If sum or weight uses an unsupported Partial reduction.
        """
        valid_reduce_ops = {"sum", "avg"}

        # Check sum: only "sum" or "avg" are supported for Partial placements.
        sum_reduce_ops = _get_partial_reduce_ops(self._sum)
        invalid_sum_ops = [op for op in sum_reduce_ops if op not in valid_reduce_ops]
        if invalid_sum_ops:
            raise ValueError(
                f"Summary '{name}' has a MeanTMetric with sum "
                f"containing Partial DTensor with reduce_op(s)={invalid_sum_ops}, "
                f"but only 'sum' or 'avg' are supported. Other reduction types "
                f"do not commute with addition in merge_()."
            )

        # Check weight: only "sum" or "avg" are supported for Partial placements.
        weight_reduce_ops = _get_partial_reduce_ops(self._weight)
        invalid_weight_ops = [
            op for op in weight_reduce_ops if op not in valid_reduce_ops
        ]
        if invalid_weight_ops:
            raise ValueError(
                f"Summary '{name}' has a MeanTMetric with weight "
                f"containing Partial DTensor with reduce_op(s)={invalid_weight_ops}, "
                f"but only 'sum' or 'avg' are supported. Other reduction types "
                f"do not commute with addition in merge_()."
            )

    def merge_(self, other: Self) -> Self:
        """Merges `other` into `self` by adding sums and weights.

        This computes:
            new_sum = self._sum + other._sum
            new_weight = self._weight + other._weight

        When `self` and `other` contain Partial("sum") DTensors, the merge is computed on the local
        (unreduced) values. The Partial("sum") placement is preserved, and the reduction happens
        later when `replicate_to_host()` is called.

        Args:
            other: Another MeanTMetric to merge into this one.

        Returns:
            self, modified in-place.

        Raises:
            TypeError: If other is not a MeanTMetric, or if attempting to merge
                tensors with Python scalars, or if dtypes are incompatible.
            ValueError: If attempting to merge DTensor with regular Tensor.
        """
        if not isinstance(other, MeanTMetric):
            raise TypeError(f"Expected MeanTMetric, got {type(other)}.")

        # Check type compatibility for both sum and weight
        _check_merge_compatibility(self._sum, other._sum, "MeanTMetric.sum")
        _check_merge_compatibility(self._weight, other._weight, "MeanTMetric.weight")

        if isinstance(self._sum, torch.Tensor):
            # Tensor path: move other's tensors to self's device
            # .to() is a no-op if already on the same device.
            other_sum = other._sum.to(self._sum.device)
            if isinstance(other._weight, (Tensor, DTensor)):
                other_weight = other._weight.to(self._weight.device)
            else:
                other_weight = other._weight

            # Simple addition preserves Partial placements naturally.
            self._sum = self._sum + other_sum
            self._weight = self._weight + other_weight
        else:
            # Python scalar path: simple addition
            self._sum = self._sum + other._sum
            self._weight = self._weight + other._weight

        return self

    def __eq__(self, value) -> bool:
        if not isinstance(value, MeanTMetric):
            return False
        # Mixed types (tensor vs scalar) are not equal
        if isinstance(self._sum, Tensor) != isinstance(value._sum, Tensor):
            return False
        # bool() works for both tensor (0-d boolean tensor) and scalar comparison results
        return bool(self._sum == value._sum) and bool(self._weight == value._weight)

    def __repr__(self) -> str:
        return f"MeanTMetric(sum={self._sum}, weight={self._weight})"

    def __flatten__(self) -> tuple[list, None]:
        """Returns (leaves, context) for pytree flattening."""
        return [self._sum, self._weight], None

    @classmethod
    def __unflatten__(cls, leaves: list, context: None) -> "MeanTMetric":
        """Reconstructs from (leaves, context)."""
        return cls(sum=leaves[0], weight=leaves[1])


class _ReducedTensorMetric(TMetricValue):
    """A metric represents a value reduced from a set of inputs.

    The reduction is assumed to be associative and commutative.

    _value is stored as-is (no eager conversion to tensor). This avoids unnecessary
    tensor creation overhead on non-logging steps. Conversion to tensor happens lazily
    in merge_() when needed.
    """

    class Reduction(enum.Enum):
        """Reduction operations."""

        SUM = "sum"
        MAX = "max"
        MIN = "min"

    _value: Scalar  # Python scalar or Tensor/DTensor - no eager conversion
    reduction: Reduction

    def __init__(self, value: Scalar, *, reduction: Reduction) -> None:
        # The check seems to fail with dynamo: P2102129865
        if reduction not in self.Reduction:
            raise ValueError(f"Unknown reduction: {reduction}.")
        # Store value as-is, no eager conversion to tensor
        self._value = value
        self.reduction = reduction

    def value(self) -> Scalar:
        """Returns the metric value.

        Returns a Tensor if _value is a Tensor, enabling .backward().
        Returns a Python scalar if _value is a Python scalar.
        """
        return self._value

    def validate_placements(self, name: str) -> None:
        """Validates that any Partial DTensors use a compatible reduction operation.

        For _ReducedTensorMetric, the Partial DTensor's reduce_op must match the metric's
        reduction type. For example:
        - SumMetric requires Partial("sum")
        - MaxMetric requires Partial("max")
        - MinMetric requires Partial("min")

        Python scalars are always valid (no placements to check).

        Args:
            name: The summary name, used for error messages.

        Raises:
            ValueError: If the Partial DTensor's reduce_op doesn't match the metric's reduction.
        """
        # Python scalars have no placements - always valid
        if not isinstance(self._value, Tensor):
            return

        reduce_ops = _get_partial_reduce_ops(self._value)
        expected_op = self.reduction.value
        invalid_ops = [op for op in reduce_ops if op != expected_op]
        if invalid_ops:
            raise ValueError(
                f"Summary '{name}' has a {type(self).__name__} with Partial DTensor "
                f"using reduce_op(s)={invalid_ops}, but this metric requires "
                f"'{expected_op}' to match its merge_() semantics."
            )

    def merge_(self, other: Scalar | Self) -> Self:
        """Merges `other` into `self` using the reduction operation.

        Handles both Python scalars and Tensors. Raises TypeError if types are mixed.

        Args:
            other: Another _ReducedTensorMetric or a scalar value to merge.

        Returns:
            self, modified in-place.

        Raises:
            ValueError: If reduction types don't match, or if DTensor vs Tensor mismatch.
            TypeError: If attempting to merge tensor with Python scalar, or dtype mismatch.
        """
        if isinstance(other, _ReducedTensorMetric):
            if self.reduction != other.reduction:
                raise ValueError(
                    f"Reduction mismatch: {self.reduction} != {other.reduction}."
                )
            other_value = other._value
        else:
            other_value = other

        # Use shared helper for type/dtype/DTensor compatibility checks
        _check_merge_compatibility(self._value, other_value, type(self).__name__)

        if isinstance(self._value, Tensor):
            # Tensor path: move other's tensor to self's device
            # .to() is a no-op if already on the same device
            other_value = other_value.to(self._value.device)

            if self.reduction == self.Reduction.SUM:
                self._value = self._value + other_value
            elif self.reduction == self.Reduction.MAX:
                self._value = torch.max(self._value, other_value)
            elif self.reduction == self.Reduction.MIN:
                self._value = torch.min(self._value, other_value)
            else:
                raise RuntimeError(f"Unknown reduction: {self.reduction}.")
        else:
            # Python scalar path: use Python operators
            if self.reduction == self.Reduction.SUM:
                self._value = self._value + other_value
            elif self.reduction == self.Reduction.MAX:
                self._value = max(self._value, other_value)
            elif self.reduction == self.Reduction.MIN:
                self._value = min(self._value, other_value)
            else:
                raise RuntimeError(f"Unknown reduction: {self.reduction}.")

        return self

    def __flatten__(self) -> tuple[list, None]:
        """Returns (leaves, context) for pytree flattening."""
        return [self._value], None

    @classmethod
    def __unflatten__(cls, leaves: list, context: None):
        """Reconstructs from (leaves, context)."""
        return cls(leaves[0])


class SumTMetric(_ReducedTensorMetric):
    """A metric that represents a sum of values."""

    def __init__(self, value: Scalar) -> None:
        super().__init__(value, reduction=_ReducedTensorMetric.Reduction.SUM)


class MaxTMetric(_ReducedTensorMetric):
    """A metric that represents the maximum of values."""

    def __init__(self, value: Scalar) -> None:
        super().__init__(value, reduction=_ReducedTensorMetric.Reduction.MAX)


class MinTMetric(_ReducedTensorMetric):
    """A metric that represents the minimum of values."""

    def __init__(self, value: Scalar) -> None:
        super().__init__(value, reduction=_ReducedTensorMetric.Reduction.MIN)




# ---------------------------------------------------------------------------
# DTensor helpers for batched reduction
# ---------------------------------------------------------------------------


def to_replicated_dtensor(x: Tensor | DTensor) -> Tensor | DTensor:
    """Redistributes a DTensor to Replicate placement, keeping it as a DTensor.

    Unlike `replicate()` which converts to local Tensor, this function preserves
    the DTensor type. This is useful when you need a fully replicated DTensor
    for operations that require consistent values across ranks.

    Args:
        x: A Tensor or DTensor. If a DTensor, it will be redistributed to
            Replicate placement on all mesh dimensions.

    Returns:
        If input is a DTensor, returns a DTensor with Replicate placement.
        If input is a regular Tensor, returns it unchanged.
    """
    if isinstance(x, DTensor):
        return x.redistribute(
            x.device_mesh,
            [Replicate() for _ in range(x.device_mesh.ndim)],
        )
    return x



def replicate(x):
    """Replicates a tree of DTensors with batched collectives.

    Batches DTensors by (mesh, placements, dtype, shape) and performs a single
    collective per group. This reduces the number of collective operations from
    N to K, where K is the number of unique groups.

    For trees with many small DTensors (e.g., metrics), this can significantly
    reduce collective overhead by amortizing kernel launch costs and improving
    NCCL efficiency for small messages.

    If leaves are already Tensors, they will be returned unchanged.

    Note: This function does not support DTensors that require gradients.

    Args:
        x: A nested structure of Tensors or DTensors. None of the DTensors
            should require gradients.

    Returns:
        A nested structure with all DTensors converted to local Tensors.

    Raises:
        ValueError: If any DTensor requires gradients.
    """
    # Step 1: Flatten the tree and collect DTensors that need reduction
    flat_values = tree.leaves(x)
    treespec = tree.structure(x)

    # Group indices by (mesh_id, placements, dtype, shape) for batching
    groups: dict[tuple, list[tuple[int, DTensor]]] = defaultdict(list)
    results: list[Tensor | None] = [None] * len(flat_values)

    for i, leaf in enumerate(flat_values):
        if isinstance(leaf, DTensor):
            if leaf.requires_grad:
                raise ValueError(
                    "replicate() does not support DTensors that require grad."
                )
            needs_redistribution = any(
                not isinstance(p, Replicate) for p in leaf.placements
            )
            if needs_redistribution:
                key = (
                    id(leaf.device_mesh),
                    leaf.placements,
                    leaf.dtype,
                    leaf.shape,
                )
                groups[key].append((i, leaf))
            else:
                results[i] = leaf.to_local()
        else:
            results[i] = leaf

    # Step 2: Process each group with batched collectives
    async_results: list[tuple[list[int], Tensor]] = []

    for _key, items in groups.items():
        indices, dtensors = zip(*items)
        mesh = dtensors[0].device_mesh

        if len(dtensors) == 1:
            dt = dtensors[0]
            replicated = dt.redistribute(
                placements=[Replicate()] * mesh.ndim,
                async_op=True,
            )
            local = replicated.to_local()
            async_results.append((list(indices), local))
        else:
            stacked_dt = torch.stack(list(dtensors))
            replicated_stacked = stacked_dt.redistribute(
                placements=[Replicate()] * mesh.ndim,
                async_op=True,
            )
            local_stacked = replicated_stacked.to_local()
            async_results.append((list(indices), local_stacked))

    # Step 3: Wait for all async operations and unstack
    for indices, local in async_results:
        # Handle AsyncCollectiveTensor if present
        if hasattr(local, "wait"):
            local = local.wait()

        if len(indices) == 1:
            results[indices[0]] = local
        else:
            unstacked = torch.unbind(local, dim=0)
            for idx, tensor in zip(indices, unstacked):
                results[idx] = tensor

    # Step 4: Reconstruct the tree
    return treespec.unflatten(results)


def _replicate_list(values: list) -> list:
    """Replicate a list of values. Convenience wrapper around replicate()."""
    return replicate(values)


# ---------------------------------------------------------------------------
# replicate_to_host
# Batched GPU→CPU transfer. Groups DTensors by (mesh, placements, dtype, shape)
# for one collective per group. Returns dict[str, float] (callers always want scalars).
# ---------------------------------------------------------------------------


@torch.no_grad()
def replicate_to_host(
    metrics: dict[str, TMetricValue],
) -> dict[str, float]:
    """Redistribute DTensors and batch copy CUDA tensors to CPU.

    This function efficiently converts metrics to Python scalar values by:
    1. Redistributing any Partial/Shard DTensors to Replicate
    2. Batching the GPU→CPU transfer for all CUDA tensors, grouped by dtype
    3. Calling .value() on reconstructed metrics to get Python floats

    Args:
        metrics: Dictionary of metric names to TMetricValue instances.
            May contain Partial/Shard DTensors that need redistribution.

    Returns:
        A new dictionary with the same keys and Python float values.

    Example:
        >>> summaries = ctx.summaries()  # May contain Partial DTensors
        >>> scalars = replicate_to_host(summaries)
        >>> scalars["loss"]  # Python float
    """
    if not metrics:
        return {}

    # Step 1: Redistribute all DTensors to Replicate via batched collectives.
    # replicate() groups DTensors by (mesh, placements, dtype, shape) and performs
    # a single collective per group, reducing N collectives to K.
    metrics = replicate(metrics)

    # Step 2: Flatten each metric, group CUDA tensors by dtype for batched transfer
    per_metric: dict[str, tuple[tree.TreeSpec, list]] = {}
    cuda_tensors_by_dtype: dict[torch.dtype, list[Tensor]] = {}

    for name, metric in metrics.items():
        flat_leaves = tree.leaves(metric)
        spec = tree.structure(metric)
        leaf_info: list = []

        for val in flat_leaves:

            if not isinstance(val, Tensor):
                # Python number - use directly
                leaf_info.append(val)
            elif val.device.type == "cpu":
                # CPU tensor - convert inline (cheap)
                leaf_info.append(val.item())
            elif val.device.type == "cuda":
                # CUDA tensor - add to dtype group and track index
                dtype = val.dtype
                if dtype not in cuda_tensors_by_dtype:
                    cuda_tensors_by_dtype[dtype] = []
                idx = len(cuda_tensors_by_dtype[dtype])
                cuda_tensors_by_dtype[dtype].append(val)
                leaf_info.append((dtype, idx))
            else:
                leaf_info.append(val.item())

        per_metric[name] = (spec, leaf_info)

    # Step 3: Batch GPU→CPU transfer for each dtype group
    cpu_values_by_dtype: dict[torch.dtype, list] = {}
    for dtype, tensors in cuda_tensors_by_dtype.items():
        cpu_values_by_dtype[dtype] = torch.stack(tensors).to("cpu").tolist()

    # Step 4: Reconstruct metrics with scalar leaves, call .value()
    result: dict[str, float] = {}
    for name, (spec, leaf_info) in per_metric.items():
        scalar_leaves: list = []
        for info in leaf_info:
            if isinstance(info, tuple):
                dtype, idx = info
                scalar_leaves.append(cpu_values_by_dtype[dtype][idx])
            else:
                scalar_leaves.append(info)
        reconstructed = spec.unflatten(scalar_leaves)
        val = reconstructed.value()
        result[name] = float(val)

    return result
