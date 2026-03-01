# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pytree utilities for observability metrics.

Reference: sixlib/tree.py. Changes:
  OSS_COMPAT — Nested[_T] type alias replaced with Any (sixlib-specific type).
  OSS_COMPAT — cxx_pytree import guarded with try/except fallback.
"""

from typing import Any, Callable, TypeVar

import torch
import torch.utils._pytree as pytree

# OSS_COMPAT: try cxx_pytree, fall back to pytree (optree may not be installed)
try:
    import torch.utils._cxx_pytree as cxx_pytree
except ImportError:
    cxx_pytree = pytree  # type: ignore[misc]

_T = TypeVar("_T")
_S = TypeVar("_S")
TreeSpec = pytree.TreeSpec


# VERBATIM from sixlib tree.py:24-25
def path_str(kp: pytree.KeyPath) -> str:
    return pytree.keystr(kp)


# CHANGED (OSS_COMPAT): Nested[_T] → Any (type annotation only, zero runtime impact)
def map(
    fn: Callable[..., _S],
    tree: Any,
    *rest,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    if torch._dynamo.is_compiling():
        return pytree.tree_map(fn, tree, *rest, is_leaf=is_leaf)
    return cxx_pytree.tree_map(fn, tree, *rest, is_leaf=is_leaf)


# CHANGED (OSS_COMPAT): same annotation change
def map_with_path(
    fn,
    tree: Any,
    *rest,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    return pytree.tree_map_with_path(fn, tree, *rest, is_leaf=is_leaf)


# VERBATIM
def structure(tree: Any, is_leaf: Callable[[Any], bool] | None = None):
    return cxx_pytree.tree_structure(tree, is_leaf=is_leaf)


# CHANGED (OSS_COMPAT): same annotation change
def leaves(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> list:
    if torch._dynamo.is_compiling():
        return pytree.tree_leaves(tree, is_leaf=is_leaf)
    return cxx_pytree.tree_leaves(tree, is_leaf=is_leaf)


# VERBATIM from sixlib tree.py:122-151 (dedup guard included)
def register_pytree_node(
    cls: type,
    flatten_fn: Callable[[Any], tuple[list, Any]],
    unflatten_fn: Callable[[list, Any], Any],
    *,
    flatten_with_keys_fn: Callable | None = None,
    serialized_type_name: str | None = None,
) -> None:
    """Register a class as a pytree node.

    Registers with both Python pytree and C++ pytree implementations for
    consistency across tracing and normal execution. Skips if already registered.
    """
    if cls in pytree.SUPPORTED_NODES:
        return
    pytree.register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        flatten_with_keys_fn=flatten_with_keys_fn,
        serialized_type_name=serialized_type_name,
    )
