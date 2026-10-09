# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routing data types shared by both routing layers in the RL generator.

- ``RoutingCandidate``: a routable target carrying a ``reserved_load``. Each
  routing layer supplies its own candidate type (``_GeneratorHandle`` for
  generator meshes, ``_DPRankHandle`` for DP ranks).
- ``RoutingContext``: per-request metadata a strategy may consult.
- ``KVCacheBudget`` and ``EngineLoad``: a generator's KV cache size and its
  current load, for the inter-generator router's admission policy.

A ``RoutingStrategy`` (see ``strategies.py``) picks one ``RoutingCandidate``
given a ``RoutingContext``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


class RoutingCandidate(ABC):
    """A routable target."""

    @property
    @abstractmethod
    def reserved_load(self) -> int:
        """Caller-defined reserved load on this target."""


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingContext:
    """Routing metadata for one generation request."""

    estimated_cost: int = 1
    """Estimated request cost used by load-aware routing strategies."""

    session_id: str | None = None
    """Stable session key consumed only by sticky routing strategies; other
    strategies ignore it. ``None`` means the request is unpinned and uses fallback
    routing without session affinity."""


@dataclass(frozen=True, kw_only=True, slots=True)
class KVCacheBudget:
    """A generator's KV cache size and the blocks one session holds in it.

    Example (Qwen3.5-35B-A3B on one GB300: 1 attention group, 3 Gated-DeltaNet groups)::

        budget = KVCacheBudget(
            num_blocks=8169, block_size=1152, num_growing_groups=1, fixed_blocks_per_session=6,
            max_model_len=131072,
        )
        budget.session_blocks(5000)  # 1 * ceil(5000 / 1152) + 6 = 11
    """

    num_blocks: int
    """KV cache blocks on the generator, summed over its data-parallel replicas."""

    block_size: int
    """Tokens per block."""

    num_growing_groups: int
    """KV cache groups where a session holds one block per ``block_size`` tokens of context:
    full attention, and Mamba in vLLM's "all" mode."""

    fixed_blocks_per_session: int
    """Blocks a session holds in the other groups whatever its length, e.g. 2 per Mamba group in
    vLLM's "align" mode, or a sliding window."""

    max_model_len: int
    """Longest context vLLM accepts, in tokens."""

    def session_blocks(self, num_tokens: int) -> int:
        """Return the blocks one session of ``num_tokens`` tokens holds."""
        return (
            self.num_growing_groups * math.ceil(num_tokens / self.block_size)
            + self.fixed_blocks_per_session
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class EngineLoad:
    """A generator's vLLM load, for the inter-generator router's admission policy."""

    kv_usage: float
    """Fraction of KV cache blocks in use (vLLM's ``kv_cache_usage_perc``). Finished requests'
    cached blocks do not count."""

    num_running: int
    """Requests vLLM ran at its last step (vLLM's ``num_requests_running``)."""

    num_waiting: int
    """Requests waiting in vLLM's queue at its last step (vLLM's ``num_requests_waiting``)."""

    num_waiting_for_capacity: int
    """The waiting requests that wait for scheduling capacity, KV blocks or batch slots (vLLM's
    ``num_requests_waiting_by_reason{reason="capacity"}``)."""

    num_preemptions: int
    """Preemptions since the engine started (vLLM's ``num_preemptions``)."""
