# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pytree utilities for observability metrics.

Wraps PyTorch's pytree API with compile-safe dispatching: uses cxx_pytree
for eager execution and regular pytree when Dynamo is tracing.
"""

from typing import Any, Callable, TypeVar

import torch
import torch.utils._pytree as pytree

try:
    import torch.utils._cxx_pytree as cxx_pytree
except ImportError:
    cxx_pytree = pytree  # type: ignore[misc]

_T = TypeVar("_T")
_S = TypeVar("_S")
TreeSpec = pytree.TreeSpec


def path_str(kp: pytree.KeyPath) -> str:
    return pytree.keystr(kp)


def map(
    fn: Callable[..., _S],
    tree: Any,
    *rest,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    if torch._dynamo.is_compiling():
        return pytree.tree_map(fn, tree, *rest, is_leaf=is_leaf)
    return cxx_pytree.tree_map(fn, tree, *rest, is_leaf=is_leaf)


def map_with_path(
    fn,
    tree: Any,
    *rest,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    return pytree.tree_map_with_path(fn, tree, *rest, is_leaf=is_leaf)


def structure(tree: Any, is_leaf: Callable[[Any], bool] | None = None):
    return cxx_pytree.tree_structure(tree, is_leaf=is_leaf)


def leaves(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> list:
    if torch._dynamo.is_compiling():
        return pytree.tree_leaves(tree, is_leaf=is_leaf)
    return cxx_pytree.tree_leaves(tree, is_leaf=is_leaf)


def register_pytree_node(
    cls: type,
    flatten_fn: Callable[[Any], tuple[list, Any]],
    unflatten_fn: Callable[[list, Any], Any],
    *,
    flatten_with_keys_fn: Callable | None = None,
    serialized_type_name: str | None = None,
) -> None:
    """Register a class as a pytree node. Skips if already registered."""
    if cls in pytree.SUPPORTED_NODES:
        return
    pytree.register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        flatten_with_keys_fn=flatten_with_keys_fn,
        serialized_type_name=serialized_type_name,
    )
