# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The inference engine interface the generator's engine loop drives."""

from __future__ import annotations

import abc
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from torchtitan.config import Configurable
from torchtitan.rl.generator.types import EngineRequest, FinishedRequest


ModelStateDictLoadMode = Literal["cpu_prefetch", "gpu_fetch"]


class GenerationEngine(Configurable, abc.ABC):
    """One inference engine replica that runs a TorchTitan model, driven by the engine loop.

    The generator builds the engine with ``config.build(...)`` on its engine loop thread,
    and every method runs there: implementations need not be thread-safe. Every rank of a
    replica receives the same calls in the same order, so collectives inside them (e.g. a
    TP engine step) line up across ranks.

    ``build`` passes the engine's own ``config`` and these keyword arguments, which every
    engine accepts:

        model: The generator's model config.
        parallelism: The generator's parallelism; each rank's DP / TP rank derives from
            ``rank`` and this layout.
        model_path: Path to the HF model checkpoint (tokenizer assets and, with a
            ``checkpointer``, initial weights).
        max_num_seqs: Upper bound on concurrently scheduled sequences.
        precision: The trainer's parameter and compute dtypes, which the engine's model
            mirrors: it computes in ``param_dtype`` and receives the trainer's weights
            in the dtypes the trainer sends them.
        gpu_memory_limit: Fraction of GPU memory the engine may use.
        max_num_batched_tokens: Max tokens scheduled per engine step, or None for the
            engine's default.
        num_topk_logprobs: The most top-k logprobs a request may ask for
            (``SamplingConfig.num_topk_logprobs``); 0 for none.
        checkpointer: Optional initial-weight loader; None when weights arrive through
            model state dict loads.
        dist_moe_runtime: Dist-MoE runtime policy, when the model has Dist-MoE experts.
        debug: Debug and determinism settings.
        reset_kv_cache_on_model_state_dict_load: Whether each model state dict load resets
            all KV (and vLLM's encoder cache for image prompts). When False, a request may
            reuse only KV and image embeddings salted with its ``min_policy_version``.
        model_state_dict_load_mode: ``"cpu_prefetch"`` allocates the pinned CPU buffers of
            ``prefetched_model_state_dict``; ``"gpu_fetch"`` does not.
        vision_processor: Preprocesses image prompts' files (`PromptImage.data`). None:
            text only; requests with images are rejected.
        mooncake_master_address: ``host:port`` of the Mooncake Store master the
            controller started when ``config.uses_mooncake_store``, else None.
        rank: Global rank of this process.
        generator_name: Name of the generator, unique within a run and shared by its
            ranks; names engine metrics and per-generator connector resources.
        output_dir: Output directory, for engine metrics.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        @property
        def uses_mooncake_store(self) -> bool:
            """Whether the generators share a Mooncake Store. The controller then
            starts its master and builds the engines with its address."""
            return False

        @property
        def process_env(self) -> dict[str, str]:
            """Environment variables the generator processes start with, on top of the
            job's (e.g. allocator settings that must be set before CUDA starts)."""
            return {}

    @property
    @abc.abstractmethod
    def data_parallel_size(self) -> int:
        """The data-parallel size the engine derived."""

    @property
    @abc.abstractmethod
    def data_parallel_rank(self) -> int:
        """The data-parallel rank the engine derived for this process."""

    @property
    @abc.abstractmethod
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor] | None:
        """The pinned CPU buffers a model state dict is prefetched into, and loaded from by
        a model state dict load. Safe to fill from any thread while no load runs. None with
        ``model_state_dict_load_mode="gpu_fetch"``."""

    @abc.abstractmethod
    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        """Admit ``requests`` in order and return the ``(request_id, error)`` of each request
        the engine rejected. A rejected request must leave no engine state."""

    @abc.abstractmethod
    def has_unfinished_requests(self) -> bool:
        """Whether requests or their remaining engine work need more steps.

        All ranks must call this together before stepping: implementations may
        synchronize local work so every rank makes the same continue/idle decision.
        """

    @abc.abstractmethod
    def step(self) -> list[FinishedRequest]:
        """Run one engine step and return the requests it finished."""

    @abc.abstractmethod
    def load_model_state_dict(
        self, fetch: Callable[[dict[str, torch.Tensor]], Iterable[str]]
    ) -> None:
        """Load a model state dict by having ``fetch`` fill the model's weights in place, and
        reset the KV cache when configured.

        Args:
            fetch: Fills each tensor of the given subset of the model state dict in place
                with the new weights and returns the keys it filled.
        """

    @property
    def holds_session_kv(self) -> bool:
        """Whether the engine holds multi-turn sessions' KV between turns, so callers must
        release sessions and groups when their rollouts end."""
        return False

    def release_sessions(self, session_ids: list[str], group_ids: list[int]) -> None:
        """Release the held KV of ``session_ids`` and of every session of ``group_ids``."""

    def log_stats(self, step: int) -> None:
        """Log the engine's own per-step stats, if any."""

    def kv_cache_budget(self) -> KVCacheBudget:
        """Return the KV cache size and what a session holds in it, for the inter-generator router's
        admission (``InterGeneratorRouter.Config.admission``). Called from the actor's thread, so it
        reads only what the build fixed."""
        raise NotImplementedError(
            f"{type(self).__name__} does not report its KV cache size, which admission needs"
        )

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release engine-local resources; process teardown is the actor framework's."""


@dataclass(frozen=True, kw_only=True, slots=True)
class KVCacheBudget:
    """One generator's KV cache size, and the blocks a session holds in it while idle or running.

    Example (Qwen3.8-27B with MTP-3 on one GB300: 1 attention group of 1,664-token blocks and
    3 Gated-DeltaNet groups, each a vLLM `MambaSpec` in "align" mode):

        budget = KVCacheBudget(
            num_blocks=3411, free_floor_blocks=381, attention_block_sizes=(1664,),
            num_mamba_groups=3, mamba_blocks_per_running_request=4, max_model_len=262144,
            data_parallel_size=1,
        )
        budget.usable_blocks                  # 3411 - 381 = 3030
        budget.idle_session_blocks(5000)      # ceil(5000 / 1664) + 3 = 7
        budget.running_session_blocks(5000)   # 7 + 3 x 4 + 4 = 23
    """

    # TODO: engine-neutral fields (usable blocks, an idle session's fixed blocks, a running session's extra
    # blocks), built by `VLLMSessionKVHolder`, which owns the free-floor rule. That changes every field and
    # its readers, and waits on the upstream choice between the holder and #5919's evictor.
    num_blocks: int
    """GPU KV cache blocks of one data-parallel replica."""

    free_floor_blocks: int
    """Blocks the session KV holder keeps free: vLLM's watermark plus one prefill chunk per KV
    cache group (`VLLMSessionKVHolder.free_floor_blocks`)."""

    attention_block_sizes: tuple[int, ...]
    """Tokens per block of each full-attention KV cache group. A session holds one block per
    `block_size` tokens of its context in each."""

    num_mamba_groups: int
    """Mamba KV cache groups (Gated-DeltaNet layers) in vLLM's "align" mode. An idle session holds one
    state block in each: the state at its prompt's end, which the holder keeps."""

    mamba_blocks_per_running_request: int
    """More blocks per Mamba group while a request runs: `1 + num_speculative_blocks +
    num_prefill_checkpoint_blocks`. vLLM's align-mode maximum is one more, the prompt-end state that
    `idle_session_blocks` already counts."""

    max_model_len: int
    """Longest context the engine accepts, in tokens."""

    data_parallel_size: int
    """Data-parallel replicas of the generator, each with `num_blocks`."""

    @property
    def num_kv_cache_groups(self) -> int:
        return len(self.attention_block_sizes) + self.num_mamba_groups

    @property
    def usable_blocks(self) -> int:
        """Blocks sessions may hold before the session KV holder starts releasing idle ones."""
        return self.num_blocks - self.free_floor_blocks

    def idle_session_blocks(self, num_tokens: int) -> int:
        """Blocks the session KV holder keeps for a session of `num_tokens` tokens between turns.

        Admission passes a turn's prompt + completion. The holder keeps only the prompt's blocks, so this
        reads about ceil(completion / block_size) + 1 blocks high per idle session: a margin.
        """
        attention_blocks = sum(
            math.ceil(num_tokens / block_size)
            for block_size in self.attention_block_sizes
        )
        return attention_blocks + self.num_mamba_groups

    def running_session_blocks(self, num_tokens: int) -> int:
        """Blocks a session of `num_tokens` tokens holds while its request runs, plus the one block per
        KV cache group the holder keeps free for each running request."""
        return (
            self.idle_session_blocks(num_tokens)
            + self.num_mamba_groups * self.mamba_blocks_per_running_request
            + self.num_kv_cache_groups
        )
