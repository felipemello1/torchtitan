# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapt local CPU offload to Titan's TP groups and session release hints."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import vllm.v1.simple_kv_offload.worker as simple_kv_offload_worker
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector as VllmSimpleCPUOffloadConnector,
)
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_block_hash,
    get_group_id,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import copy_blocks

from torchtitan.rl.generator.vllm.session_kv_holder import cached_block_ids

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request
    from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata


class SimpleCPUOffloadConnector(VllmSimpleCPUOffloadConnector):
    """Local CPU caches with a save-completion quorum scoped to one TP group.

    Every scheduler queries its own CPU cache. Coordinated store completions and
    block release keep those caches consistent; no leader lookup is needed.

    The scheduler side also counts how the CPU cache serves prompts, in plain ints and dicts
    of ints that other threads may read (`VllmOtelStatLogger.observe_kv_reuse` exports them):

        gpu_evicted_blocks{"yes"|"no"}       hashed GPU blocks reused, by whether the CPU had a copy
        broken_chain_requests/tokens{cause}  prompts whose hit stopped before blocks on the CPU
        num_stored_blocks                    blocks copied to the CPU
        num_cpu_gdn_state_blocks             CPU blocks of GDN state groups (vLLM `MambaSpec`),
                                             scanned every VLLM_LOG_STATS_INTERVAL seconds
        load_waits_s                         per request with a CPU hit: admission to load done
        cpu_block_metrics                    vLLM's sampled block lifetimes, on the CPU pool

    The worker side, a separate object in the same process, times the copies:

        copy_time_s{"load"|"store"}          seconds the copy streams spent copying blocks
        copied_blocks{"load"|"store"}        blocks copied
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if self.scheduler_manager is not None:
            # external_launcher world_size includes independent DP replicas.
            # TODO(acisseJZhong): upstream a TP-scoped save quorum instead of
            # setting the private _expected_worker_count.
            self.scheduler_manager._expected_worker_count = get_tp_group().world_size

        self.gpu_evicted_blocks = {"yes": 0, "no": 0}
        self.broken_chain_requests = {"hole": 0, "no_state": 0}
        self.broken_chain_tokens = {"hole": 0, "no_state": 0}
        self.num_stored_blocks = 0
        self.num_cpu_gdn_state_blocks = 0
        self._next_cpu_scan_time = 0.0
        self._gpu_block_pool: BlockPool | None = None
        # Prefix tokens the GPU and CPU caches served at admission, by request id.
        self._cached_tokens_at_admission: dict[str, int] = {}
        # When each request with a CPU hit was admitted, until its load finishes.
        self._load_start_times: dict[str, float] = {}
        # The stat logger drains these on the engine thread.
        self.load_waits_s: list[float] = []
        self.cpu_block_metrics: KVCacheMetricsCollector | None = None
        # Worker side: timed copy batches, oldest first, appended by vLLM's copy thread.
        self.copy_time_s = {"load": 0.0, "store": 0.0}
        self.copied_blocks = {"load": 0, "store": 0}
        self._timed_copies: deque[_TimedCopy] = deque()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Have vLLM's worker build a copy backend that times each copy batch."""
        # TODO: delete when the vLLM pin includes vllm-project/vllm#41790, which times these copies
        # in vLLM's own connector stats.
        original_backend = simple_kv_offload_worker.DmaCopyBackend
        simple_kv_offload_worker.DmaCopyBackend = lambda: _TimedDmaCopyBackend(
            self._timed_copies
        )
        try:
            super().register_kv_caches(kv_caches)
        finally:
            simple_kv_offload_worker.DmaCopyBackend = original_backend

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report finished loads, and add the time of each finished copy batch."""
        finished = super().get_finished(finished_req_ids)
        # Batches finish in order on each stream; one still copying holds back later ones.
        while self._timed_copies and self._timed_copies[0].end.query():
            timed_copy = self._timed_copies.popleft()
            direction = "store" if timed_copy.is_store else "load"
            self.copy_time_s[direction] += (
                timed_copy.start.elapsed_time(timed_copy.end) / 1000
            )
            self.copied_blocks[direction] += timed_copy.num_blocks
        return finished

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Count GPU block reuse through the pool's `metrics_collector` hook."""
        super().bind_gpu_block_pool(gpu_block_pool)
        if gpu_block_pool.metrics_collector is not None:
            raise ValueError(
                "SimpleCPUOffloadConnector counts GPU block evictions through the GPU block "
                "pool's metrics_collector, which vLLM's kv_cache_metrics already uses: turn "
                "ObservabilityConfig.kv_cache_metrics off"
            )
        self._gpu_block_pool = gpu_block_pool
        # TODO: drop this collector once vLLM's SimpleCPU metrics (vllm-project/vllm#57251) count GPU
        # blocks reused without a CPU copy.
        gpu_block_pool.metrics_collector = _CPUCopyEvictionCounter(
            cpu_block_pool=self.scheduler_manager.cpu_block_pool,
            counts=self.gpu_evicted_blocks,
        )
        # vLLM's own collector, sampled: a store allocates a CPU block, a hit accesses it, and an
        # overwrite evicts it.
        self.cpu_block_metrics = KVCacheMetricsCollector()
        self.scheduler_manager.cpu_block_pool.metrics_collector = self.cpu_block_metrics

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        """Record the prefix tokens cached at admission and a CPU load's start; count a broken prefix chain."""
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if num_external_tokens > 0:
            self._load_start_times.setdefault(request.request_id, time.monotonic())

        # vLLM calls this again once a CPU load finishes; only the first call admits.
        stats = request.prefill_stats
        if stats is None or request.request_id in self._cached_tokens_at_admission:
            return

        self._cached_tokens_at_admission[request.request_id] = stats.num_cached_tokens
        self._count_broken_chain(request, blocks, stats.num_cached_tokens)

    def pop_cached_tokens(self, request_id: str) -> int | None:
        """Return and forget the prefix tokens the GPU and CPU caches served at admission.

        None if the request was never admitted with prefill stats (e.g. a preempted re-admission).
        """
        return self._cached_tokens_at_admission.pop(request_id, None)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        # vLLM calls this one, not `request_finished`: this connector supports the hybrid allocator.
        self._cached_tokens_at_admission.pop(request.request_id, None)
        self._load_start_times.pop(request.request_id, None)
        return super().request_finished_all_groups(request, block_ids)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Record how long each finished CPU load kept its request waiting."""
        super().update_connector_output(connector_output)
        now = time.monotonic()
        for request_id in connector_output.finished_recving or ():
            start = self._load_start_times.pop(request_id, None)
            if start is not None:
                self.load_waits_s.append(now - start)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> SimpleCPUOffloadMetadata:
        """Count this step's CPU stores; rescan the CPU pool's GDN state blocks every interval."""
        metadata = super().build_connector_meta(scheduler_output)
        self.num_stored_blocks += len(metadata.store_cpu_blocks)

        now = time.monotonic()
        if now >= self._next_cpu_scan_time:
            self._next_cpu_scan_time = now + envs.VLLM_LOG_STATS_INTERVAL
            manager = self.scheduler_manager
            state_groups = {
                group_id
                for group_id, group in enumerate(
                    manager.cpu_kv_cache_config.kv_cache_groups
                )
                if isinstance(group.kv_cache_spec, MambaSpec)
            }
            self.num_cpu_gdn_state_blocks = sum(
                block.block_hash is not None
                and get_group_id(block.block_hash) in state_groups
                for block in manager.cpu_block_pool.blocks
            )
        return metadata

    def _count_broken_chain(
        self, request: Request, blocks: KVCacheBlocks, num_cached_tokens: int
    ) -> None:
        """Count a prompt whose prefix hit stopped before attention blocks the CPU still holds.

        The prompt's attention blocks past the hit: `c` on the CPU, `g` on the GPU in another
        request's block, `x` on neither:

            hit | c c c x   -> "no_state": the next block is on the CPU, so its GDN state was gone
            hit | x c c x   -> "hole": the next block is gone, later ones are on the CPU
            hit | g c c x   -> not counted: the next block is still on the GPU
            hit | x x x x   -> not counted: the session's tail is gone

        Tokens: from the hit's end to the end of the last CPU block (3 blocks in both counted cases).
        This request's own blocks (`blocks`) are not "on the GPU": `allocate_slots` already hashed them.
        """
        manager = self.scheduler_manager
        block_size = manager.fa_block_size
        hashes_per_block = block_size // manager.hash_block_size
        # A hit never covers the prompt's last token.
        num_hittable_blocks = (request.num_prompt_tokens - 1) // block_size
        block_hashes = [
            make_block_hash_with_group_id(
                request.block_hashes[(index + 1) * hashes_per_block - 1],
                manager.fa_gidx,
            )
            for index in range(num_cached_tokens // block_size, num_hittable_blocks)
        ]
        on_cpu = [
            manager.cpu_block_pool.cached_block_hash_to_block.get_one_block(block_hash)
            is not None
            for block_hash in block_hashes
        ]
        if not any(on_cpu):
            return

        if not on_cpu[0]:
            # A GPU copy outside this request: the hit stopped on its GDN state, no block was lost.
            own_block_ids = {block.block_id for block in blocks.blocks[manager.fa_gidx]}
            # TODO: reads the pool's private cache index, as `release_session_cache` does:
            # https://github.com/vllm-project/vllm/issues/51428
            cached = self._gpu_block_pool.cached_block_hash_to_block._cache.get(
                block_hashes[0], {}
            )
            gpu_copies = cached.values() if isinstance(cached, dict) else [cached]
            if any(copy.block_id not in own_block_ids for copy in gpu_copies):
                return

        # TODO: if "hole" tokens are a material share of prompt tokens, take vllm-project/vllm#65237
        # (head-only lazy walk) and #59383 (stored blocks evicted first), by a vLLM bump or a backport.
        cause = "no_state" if on_cpu[0] else "hole"
        num_lost_blocks = len(on_cpu) - on_cpu[::-1].index(True)
        self.broken_chain_requests[cause] += 1
        self.broken_chain_tokens[cause] += num_lost_blocks * block_size

    def get_pending_save_ids(self) -> set[int]:
        """Snapshot queued/in-flight saves, including the next finish-time batch."""
        manager = self.scheduler_manager
        assert manager is not None
        # TODO(acisseJZhong): reads the manager's private save state. Upstream
        # session eviction hints that apply after pending saves would replace it:
        # https://github.com/vllm-project/vllm/issues/51428
        pending = set(manager._store_event_to_blocks) | set(
            manager._abandoned_store_event_to_blocks
        )
        if manager._pending_finished_stores:
            # build_connector_meta folds these tails into the next store event.
            pending.add(manager._store_event_counter)
        return pending

    def release_session_cache(
        self, block_hashes: set[BlockHash], active_hashes: set[BlockHash]
    ) -> None:
        """Make unused CPU blocks the next evicted, preserving their cached data."""
        manager = self.scheduler_manager
        assert manager is not None
        pool = manager.cpu_block_pool
        blocks = {}
        for block_hash in block_hashes:
            for group_id in range(len(manager.group_block_sizes)):
                cached = pool.cached_block_hash_to_block._cache.get(
                    make_block_hash_with_group_id(block_hash, group_id)
                )
                if cached is None:
                    continue
                # Simultaneous saves may cache several physical copies of a hash.
                for block in cached.values() if isinstance(cached, dict) else [cached]:
                    aliases = [
                        block.block_hash,
                        *pool.cached_block_hashes_by_block.get(block.block_id, ()),
                    ]
                    if block.ref_cnt == 0 and not any(
                        get_block_hash(alias) in active_hashes for alias in aliases
                    ):
                        blocks[block.block_id] = block
        # TODO(acisseJZhong): replace direct cache-index and free-queue access
        # with upstream session eviction hints that protect active aliases:
        # https://github.com/vllm-project/vllm/issues/51428
        # Block IDs agree across TP ranks; hashes may use per-process seeds.
        ordered_blocks = [blocks[block_id] for block_id in sorted(blocks)]
        for block in ordered_blocks:
            pool.free_block_queue.remove(block)
        pool.free_block_queue.prepend_n(ordered_blocks)

    def drop_session_cache(self, block_hashes: set[BlockHash]) -> bool:
        """Uncache every GPU and CPU copy of `block_hashes`; return whether all are gone.

        A GPU block still being saved keeps its hash: vLLM registers the CPU copy under it when the
        save completes. The caller calls again after a step:

            GPU blocks 7 (h2), 9 (h3, saving)  ->  7 uncached, CPU copies of h2, h3 dropped: False
            next step, 9's save done           ->  9 uncached, its new CPU copy dropped: True
        """
        manager = self.scheduler_manager
        # TODO: reads the manager's private save state, as `get_pending_save_ids` does
        # (vllm-project/vllm#51428).
        saving = {
            block_id
            for transfer in [
                *manager._store_event_to_blocks.values(),
                *manager._pending_finished_stores,
            ]
            for block_id in transfer.gpu_block_ids
        }
        gpu_pool = self._gpu_block_pool
        gpu_block_ids = cached_block_ids(gpu_pool, block_hashes)
        # Not the allocator reusing blocks: keep these out of `gpu_evicted_blocks`.
        counter, gpu_pool.metrics_collector = gpu_pool.metrics_collector, None
        gpu_pool.evict_blocks(gpu_block_ids - saving)
        gpu_pool.metrics_collector = counter

        cpu_pool = manager.cpu_block_pool
        collector, cpu_pool.metrics_collector = cpu_pool.metrics_collector, None
        cpu_pool.evict_blocks(cached_block_ids(cpu_pool, block_hashes))
        cpu_pool.metrics_collector = collector
        return not gpu_block_ids & saving


@dataclass(frozen=True, slots=True)
class _TimedCopy:
    """One copy batch's timing events on its copy stream."""

    start: torch.Event
    end: torch.Event
    num_blocks: int
    is_store: bool


class _TimedDmaCopyBackend(DmaCopyBackend):
    """vLLM's DMA copy backend, with a timing event before and after each copy batch.

    `get_finished` reads `start.elapsed_time(end)` once `end` is done, without a host sync.
    """

    def __init__(self, timed_copies: deque[_TimedCopy]) -> None:
        super().__init__()
        self._timed_copies = timed_copies

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        params = self._store_params if is_store else self._load_params
        # `_copy_loop` is a staticmethod: it gets the timing deque in the queue item.
        self._queue.put(
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
                self._timed_copies,
            )
        )

    @staticmethod
    def _copy_loop(
        q,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        """vLLM's `DmaCopyBackend._copy_loop`, plus the two timing events."""
        current_platform.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
                timed_copies,
            ) = item
            stream = store_stream if is_store else load_stream
            if wait_event is not None:
                stream.wait_event(wait_event)
            # After the wait for compute, so the time is the copy's own.
            start = torch.Event(enable_timing=True)
            start.record(stream)
            copy_blocks(src_blocks, dst_blocks, params)
            end = torch.Event(enable_timing=True)
            end.record(stream)
            events_list.append((event_idx, end))
            timed_copies.append(
                _TimedCopy(
                    start=start, end=end, num_blocks=len(src_blocks), is_store=is_store
                )
            )


class _CPUCopyEvictionCounter(KVCacheMetricsCollector):
    """Count hashed GPU blocks the allocator reuses, by whether the CPU cache holds a copy.

    vLLM's `BlockPool` calls `on_block_evicted` on each block it hands out, before it drops the
    block's hash. The sampling hooks that feed vLLM's own block-lifetime metrics do nothing here.

    Example:

        counts = {"yes": 0, "no": 0}
        gpu_pool.metrics_collector = _CPUCopyEvictionCounter(cpu_block_pool=cpu_pool, counts=counts)
        gpu_pool.get_new_blocks(2)  # reuses a hashed block with a CPU copy and an unhashed one
        # counts -> {"yes": 1, "no": 0}
    """

    def __init__(self, *, cpu_block_pool: BlockPool, counts: dict[str, int]) -> None:
        super().__init__()
        self._cpu_block_pool = cpu_block_pool
        self._counts = counts

    def on_block_allocated(self, block: KVCacheBlock) -> None:
        pass

    def on_block_accessed(self, block: KVCacheBlock) -> None:
        pass

    def on_block_evicted(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            return

        cpu_copy = self._cpu_block_pool.cached_block_hash_to_block.get_one_block(
            block.block_hash
        )
        self._counts["no" if cpu_copy is None else "yes"] += 1
