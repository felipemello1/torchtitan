# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Function-scoped torch.compile registration."""

import functools
from collections.abc import Callable
from typing import Any

import torch
from torch._inductor.compile_fx import compile_fx

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode

# Inputs with more elements than this run their own compiled copy. Inductor tunes a
# dynamic-shape kernel's launch config once, on its first input, and keeps it for all sizes.
# Tuned on a decode-sized input, kernels ran 1.8-4x slower at T=64k (SwiGLU, gated RMSNorm,
# GB300); tuned on a prefill-sized input, they matched the static kernel at T=34.
_LARGE_INPUT_NUMEL = 1 << 22


def apply_local_compile(regions: list[str]) -> None:
    """Bind registered functions to eager or compiled implementations.

    Process-wide: a later call replaces this choice for every model in the process.
    """
    unknown = [name for name in regions if name not in _LOCAL_COMPILE_CALLBACKS]
    if unknown:
        raise ValueError(
            f"Unknown local_compile_regions entries {unknown}; "
            f"registered values are {sorted(_LOCAL_COMPILE_CALLBACKS)}"
        )

    for callbacks in _LOCAL_COMPILE_CALLBACKS.values():
        for bind_local_compile_fn in callbacks:
            bind_local_compile_fn(regions)


_LOCAL_COMPILE_CALLBACKS: dict[str, list[Callable[[list[str]], None]]] = {}


def local_compile(
    name: str,
    *,
    batch_invariant: bool,
    **compile_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function that can be compiled independently.

    Args:
        name: Name used to enable the function in a model's ``local_compile_regions``.
        batch_invariant: Whether the compiled function preserves batch invariance.
        **compile_kwargs: Additional ``torch.compile`` keyword arguments;
            ``fullgraph`` is fixed to ``True`` so each function forms one complete
            compile region.
    """
    if compile_kwargs.pop("fullgraph", True) is not True:
        raise ValueError("local_compile requires fullgraph=True.")

    def decorate(reference: Callable[..., Any]) -> Callable[..., Any]:
        fn = reference

        def bind_local_compile(regions: list[str]) -> None:
            nonlocal fn
            enabled = name in regions
            batch_invariant_mode = is_in_batch_invariant_mode()
            if enabled and batch_invariant_mode and not batch_invariant:
                raise ValueError(
                    f"Local compile region {name!r} does not support "
                    "batch-invariant mode; remove it from local_compile_regions."
                )
            if enabled:
                small, large = (
                    _compile_for_size(reference, name, size, compile_kwargs)
                    for size in ("small", "large")
                )
                fn = functools.partial(_call_by_input_size, small, large)
            else:
                fn = reference

        @functools.wraps(reference)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        _LOCAL_COMPILE_CALLBACKS.setdefault(name, []).append(bind_local_compile)
        return wrapped

    return decorate


def _compile_for_size(
    reference: Callable[..., Any], name: str, size: str, compile_kwargs: dict[str, Any]
) -> Callable[..., Any]:
    """Compile ``reference`` with its own Dynamo entries and its own Inductor cache entries.

    Example:
        >>> small = _compile_for_size(fn, "offset_rmsnorm", "small", {})
    """
    kwargs = dict(compile_kwargs)
    if "backend" not in kwargs:
        options = kwargs.pop("options", None)
        tag = f"local_compile:{name}:{size}"

        def backend(gm: torch.fx.GraphModule, example_inputs: list[Any]) -> Any:
            # The two sizes trace identical graphs, so without the tag the FX graph cache
            # would hand one size the other's kernels and their tuned launch configs.
            with torch.compiler.config.patch(cache_key_tag=tag):
                return compile_fx(gm, example_inputs, config_patches=options)

        kwargs["backend"] = backend
    return torch.compile(reference, fullgraph=True, isolate_recompiles=True, **kwargs)


def _call_by_input_size(
    small: Callable[..., Any], large: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Call ``large`` when a tensor argument has more than ``_LARGE_INPUT_NUMEL`` elements.

    Example:
        >>> _call_by_input_size(small, large, torch.empty(4096, 2048))  # runs ``large``
    """
    if torch.compiler.is_compiling():
        # Called from another compiled region: trace into its graph without a size guard.
        return small(*args, **kwargs)
    for arg in (*args, *kwargs.values()):
        if isinstance(arg, torch.Tensor) and arg.numel() > _LARGE_INPUT_NUMEL:
            return large(*args, **kwargs)
    return small(*args, **kwargs)


__all__ = ["apply_local_compile", "local_compile"]
