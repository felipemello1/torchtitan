# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sampling configuration shared by RL rollouts and the vLLM actor."""

from dataclasses import dataclass, field


@dataclass(kw_only=True, slots=True)
class SamplingConfig:
    """Sampling parameters passed to vLLM's ``SamplingParams``."""

    n: int = 1
    """Number of completions to generate per prompt (vLLM SamplingParams.n)."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 0.95
    """Nucleus sampling threshold."""

    max_tokens: int = 100
    """Maximum number of tokens to generate per completion."""

    stop_token_ids: list[int] = field(default_factory=list)
    """Token IDs that stop generation at the assistant-turn boundary."""
