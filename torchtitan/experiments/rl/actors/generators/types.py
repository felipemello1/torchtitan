# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared types for the vLLM generator backends.

`SamplingConfig` and `VLLMCudagraphConfig` are imported by both backends and by
`config_registry`; `GeneratorBackend` selects which backend the controller
spawns; `GenerateFn` is the Monarch-free callable a `Task` receives so it can
drive generation without importing the actor.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from torchtitan.experiments.rl.types import Completion
from vllm.config import CompilationConfig


@dataclass(kw_only=True, slots=True)
class SamplingConfig:
    """Sampling parameters passed to vLLM's SamplingParams."""

    n: int = 8
    """Number of completions to generate per prompt (vLLM SamplingParams.n)."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 0.95
    """Nucleus sampling threshold."""

    max_tokens: int = 100
    """Maximum number of tokens to generate per completion."""

    stop_token_ids: list[int] = field(default_factory=list)
    """Role-boundary stop tokens from the renderer (e.g. Qwen3 `<|im_end|>`)."""


@dataclass(kw_only=True, slots=True)
class VLLMCudagraphConfig:
    """CUDA graph capture settings for the vLLM inference engine.

    torch.compile is configured separately via ``CompileConfig`` at the
    ``RLTrainer`` level, shared by both trainer and generator.  Only CUDA
    graph capture, which is vLLM-specific, is controlled here.

    When enabled, vLLM captures the forward pass as a single CUDA graph
    ("full" mode).  "piecewise" modes are intentionally excluded: they
    require vLLM's whole-model torch.compile to split the graph around
    non-capturable ops, which conflicts with per-layer compile.
    """

    enable: bool = True
    """Whether to enable CUDA graph capture (vLLM "full" mode)."""

    # TODO: Validate CUDA graph capture with MoE / Expert Parallelism.
    # MoE routing produces dynamic shapes that may conflict with full
    # CUDA graph capture despite being torch.compile-compatible
    # post https://github.com/pytorch/torchtitan/pull/3142

    # TODO: Explore applying CUDA graph capture on the torchtitan trainer
    # side as well (not just the vLLM generator).
    # https://github.com/pytorch/torchtitan/issues/3175

    def get_vllm_compilation_config(
        self, *, max_num_seqs: int
    ) -> CompilationConfig | None:
        """Build a vLLM ``CompilationConfig``, or return ``None`` when
        CUDA graphs are disabled.

        ``max_num_seqs`` determines CUDA graph capture sizes: powers of
        2 from 1 up to ``max_num_seqs``, plus ``max_num_seqs`` itself
        if it isn't already a power of 2.
        """
        if not self.enable:
            return None
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        sizes = [1 << i for i in range(int(math.log2(max_num_seqs)) + 1)]
        if max_num_seqs not in sizes:
            sizes.append(max_num_seqs)
        return CompilationConfig(
            cudagraph_mode="full",
            mode=0,
            cudagraph_capture_sizes=sorted(sizes),
        )


class GeneratorBackend(StrEnum):
    """Which vLLM integration the controller spawns. Both share the same
    `Config` and the same `generate` contract."""

    LLM_ENGINE = "llm_engine"
    """Raw `LLMEngine` in `external_launcher` (SPMD) mode + a hand-driven engine
    loop that coalesces concurrent requests. The default, runnable path."""

    ASYNC_LLM = "async_llm"
    """`AsyncLLM` under a Monarch executor: vLLM owns the continuous-batching
    loop and TP. Fewer lines here, more vLLM/Monarch integration."""


GenerateFn = Callable[..., Awaitable[Completion]]
"""``async (prompt_token_ids: list[int], *, request_id: str,
sampling_config: SamplingConfig | None = None) -> Completion``.

The controller binds this to one routed generator actor + the rank-0 unwrap, so
a `Task` drives generation by `await`ing it without ever importing Monarch."""
