# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Function-scoped torch.compile registration and configuration."""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx.traceback

from torchtitan.distributed.utils import is_in_batch_invariant_mode


# Regions defined in code every model imports (models/common, components/loss.py).
# A region defined in one model's files (e.g. Qwen3.5's offset_rmsnorm) goes in that model's Config default.
DEFAULT_LOCAL_COMPILE_REGIONS = (
    "gated_rmsnorm",
    "loss",
    "swiglu",
    "situglu",
    "cos_sin_rope",
)


@dataclass(kw_only=True, slots=True)
class LocalCompileConfig:
    regions: list[str] = field(
        default_factory=lambda: list(DEFAULT_LOCAL_COMPILE_REGIONS)
    )
    """Named regions to compile independently with ``torch.compile``.

    Gated RMSNorm, loss, SwiGLU, SiTUGLU, and cos/sin RoPE compilation are
    enabled by default.
    FlexAttention manages its own compilation and is not controlled by this list.
    """

    def apply_local_compile(self, *, tag_regions: bool = False) -> None:
        """Bind registered functions to eager or compiled implementations.

        Process-wide: a later call replaces this choice for every model in the process.

        Args:
            tag_regions: Instead of ``torch.compile``, tag each enabled function
                for a compiler that traces the whole step (GraphTrainer): during a
                non-strict trace, its nodes get ``compile_with_inductor`` with the
                region's name and Inductor ``options`` (other ``torch.compile``
                kwargs do not apply), and it runs with
                ``torch.compiler.is_compiling()`` true, as it would under
                ``torch.compile``. Like the plain binding, this is process-wide.

        Example:
            # GraphTrainer, before tracing the step:
            model_config.local_compile.apply_local_compile(tag_regions=True)
        """
        unknown = [
            name for name in self.regions if name not in _LOCAL_COMPILE_CALLBACKS
        ]
        if unknown:
            raise ValueError(
                f"Unknown local_compile.regions entries {unknown}; "
                f"registered values are {sorted(_LOCAL_COMPILE_CALLBACKS)}"
            )

        for callbacks in _LOCAL_COMPILE_CALLBACKS.values():
            for bind_local_compile_fn in callbacks:
                bind_local_compile_fn(self, tag_regions)


_LOCAL_COMPILE_CALLBACKS: dict[
    str, list[Callable[[LocalCompileConfig, bool], None]]
] = {}


def _tag_for_inductor(
    reference: Callable[..., Any],
    name: str,
    inductor_options: dict[str, Any] | None,
) -> Callable[..., Any]:
    """Run ``reference`` eagerly; inside a non-strict trace, tag its nodes for Inductor."""
    # ``inductor_region`` keeps adjacent regions with different options in
    # separate Inductor partitions.
    annotation = {
        "compile_with_inductor": {
            "inductor_region": name,
            "inductor_configs": dict(inductor_options or {}),
        }
    }

    def tagged(*args: Any, **kwargs: Any) -> Any:
        if not torch.compiler._is_non_strict_tracing():
            return reference(*args, **kwargs)
        # Inductor compiles these nodes, so record the region's compile-only path
        # (code that checks torch.compiler.is_compiling()), as torch.compile would.
        # Private torch API; test_tag_regions_tags_traced_nodes_with_compile_path
        # asserts the compile-only branch is traced, so an upstream change fails loudly.
        with (
            torch.fx.traceback.annotate(annotation),
            torch.compiler._compile_session_context(),
        ):
            return reference(*args, **kwargs)

    return tagged


def local_compile(
    name: str,
    *,
    batch_invariant: bool,
    **compile_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function that can be compiled independently.

    Args:
        name: Name used to enable the function in ``LocalCompileConfig.regions``.
        batch_invariant: Whether the compiled function preserves batch invariance.
        **compile_kwargs: Additional ``torch.compile`` keyword arguments;
            ``fullgraph`` is fixed to ``True`` so each function forms one complete
            compile region.
    """
    if compile_kwargs.pop("fullgraph", True) is not True:
        raise ValueError("local_compile requires fullgraph=True.")

    def decorate(reference: Callable[..., Any]) -> Callable[..., Any]:
        fn = reference

        def bind_local_compile(
            local_compile_config: LocalCompileConfig, tag_regions: bool
        ) -> None:
            nonlocal fn
            enabled = name in local_compile_config.regions
            batch_invariant_mode = is_in_batch_invariant_mode()
            if enabled and batch_invariant_mode and not batch_invariant:
                raise ValueError(
                    f"Local compile region {name!r} does not support "
                    "batch-invariant mode; remove it from local_compile.regions."
                )
            if enabled and tag_regions:
                fn = _tag_for_inductor(reference, name, compile_kwargs.get("options"))
            elif enabled:
                fn = torch.compile(reference, fullgraph=True, **compile_kwargs)
            else:
                fn = reference

        @functools.wraps(reference)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        _LOCAL_COMPILE_CALLBACKS.setdefault(name, []).append(bind_local_compile)
        return wrapped

    return decorate


__all__ = ["DEFAULT_LOCAL_COMPILE_REGIONS", "local_compile", "LocalCompileConfig"]
