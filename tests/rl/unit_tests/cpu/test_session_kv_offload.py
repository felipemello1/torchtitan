# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Session release must preserve shared prefixes and pending transfer sources; the CPU KV
counters must see what the CPU cache holds."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torchtitan.rl.generator.vllm import simple_cpu_offload
from torchtitan.rl.generator.vllm.simple_cpu_offload import (
    PromptCacheSplit,
    SimpleCPUOffloadConnector,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata


@pytest.mark.parametrize("finished_hashes", [(b"a", b"z"), (b"z", b"a")])
def test_cpu_eviction_preserves_active_hashes_and_rank_independent_order(
    finished_hashes,
):
    pool = BlockPool(7, True, 16)
    (
        active_primary,
        active_secondary,
        finished,
        duplicate,
        other,
        pinned,
    ) = pool.get_new_blocks(6)
    first_hash, second_hash = finished_hashes
    for block, hashes in (
        (active_primary, [b"active-primary", b"released-alias"]),
        (active_secondary, [b"released-primary", b"active-alias"]),
        (finished, [first_hash]),
        (duplicate, [first_hash]),
        (other, [second_hash]),
        (pinned, [b"pinned"]),
    ):
        for block_hash in hashes:
            pool._insert_block_hash(
                make_block_hash_with_group_id(block_hash, 0), block, num_tokens=16
            )
    pool.free_blocks([active_primary, active_secondary, finished, duplicate, other])
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    connector.scheduler_manager = SimpleNamespace(
        cpu_block_pool=pool, group_block_sizes=[16]
    )
    connector.release_session_cache(
        {b"released-alias", b"released-primary", first_hash, second_hash, b"pinned"},
        {b"active-primary", b"active-alias"},
    )
    # Hash order can differ across TP ranks. Physical eviction order must not.
    # Demotion preserves cached data; allocation actually reclaims it later.
    assert pool.get_cached_block(first_hash, [0]) == [finished]
    assert pool.get_new_blocks(5) == [
        finished,
        duplicate,
        other,
        active_primary,
        active_secondary,
    ]
    assert pinned.ref_cnt == 1


def test_cpu_save_snapshot_includes_unsubmitted_tail():
    connector = SimpleCPUOffloadConnector.__new__(SimpleCPUOffloadConnector)
    manager = SimpleNamespace(
        _store_event_to_blocks={3: object()},
        _abandoned_store_event_to_blocks={2: object()},
        _pending_finished_stores=[object()],
        _store_event_counter=4,
    )
    connector.scheduler_manager = manager
    pending = connector.get_pending_save_ids()
    assert pending == {2, 3, 4}
    manager._pending_finished_stores.clear()
    manager._store_event_to_blocks[4] = object()
    manager._store_event_counter = 5
    assert connector.get_pending_save_ids() == pending
    manager._abandoned_store_event_to_blocks.clear()
    manager._store_event_to_blocks = {5: object()}
    assert not pending & connector.get_pending_save_ids()


def _connector(cpu_pool: BlockPool, **manager_fields) -> SimpleCPUOffloadConnector:
    """Our connector over a real CPU pool; vLLM's manager is a stub with `manager_fields`."""
    manager = SimpleNamespace(
        cpu_block_pool=cpu_pool,
        bind_gpu_block_pool=lambda pool: None,
        update_state_after_alloc=lambda *args: None,
        request_finished_all_groups=lambda *args: (False, None),
        **manager_fields,
    )

    def init(connector, *_):
        connector.scheduler_manager = manager

    with (
        patch.object(
            simple_cpu_offload.VllmSimpleCPUOffloadConnector, "__init__", new=init
        ),
        patch.object(
            simple_cpu_offload,
            "get_tp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
    ):
        return SimpleCPUOffloadConnector(Mock(), Mock(), Mock())


def _cache(pool: BlockPool, block_hashes: list[bytes], group_id: int = 0) -> None:
    """Put one block per hash into `pool`'s prefix cache, then free them (cached, evictable)."""
    blocks = pool.get_new_blocks(len(block_hashes))
    for block, block_hash in zip(blocks, block_hashes):
        pool._insert_block_hash(
            make_block_hash_with_group_id(block_hash, group_id), block, num_tokens=16
        )
    pool.free_blocks(blocks)


def test_gpu_block_reuse_counts_hashed_blocks_by_cpu_copy():
    cpu_pool = BlockPool(4, True, 16)
    _cache(cpu_pool, [b"copied"])
    gpu_pool = BlockPool(5, True, 16)
    _cache(gpu_pool, [b"copied", b"uncopied"])
    connector = _connector(cpu_pool)
    connector.bind_gpu_block_pool(gpu_pool)

    # All 4 free GPU blocks: 2 hashed, 2 that never held a hash.
    gpu_pool.get_new_blocks(4)

    assert connector.gpu_evicted_blocks == {"yes": 1, "no": 1}


def test_drop_session_cache_waits_for_a_gpu_block_being_saved():
    cpu_pool = BlockPool(4, True, 16)
    _cache(cpu_pool, [b"a", b"kept"])
    gpu_pool = BlockPool(5, True, 16)
    _cache(gpu_pool, [b"a", b"b"])
    (gpu_b,) = gpu_pool.get_cached_block(b"b", [0])
    connector = _connector(
        cpu_pool,
        _store_event_to_blocks={0: SimpleNamespace(gpu_block_ids=[gpu_b.block_id])},
        _pending_finished_stores=[],
    )
    connector.bind_gpu_block_pool(gpu_pool)

    # b's GPU block is being saved, so it keeps its hash until the save registers the CPU copy.
    assert not connector.drop_session_cache({b"a", b"b"})
    assert gpu_pool.get_cached_block(b"a", [0]) is None
    assert gpu_pool.get_cached_block(b"b", [0]) == [gpu_b]
    assert cpu_pool.get_cached_block(b"a", [0]) is None
    assert cpu_pool.get_cached_block(b"kept", [0]) is not None
    # Dropped blocks are not the allocator's reuse.
    assert connector.gpu_evicted_blocks == {"yes": 0, "no": 0}

    # The save completes with b's CPU copy; the next call drops both copies of b.
    connector.scheduler_manager._store_event_to_blocks.clear()
    _cache(cpu_pool, [b"b"])
    assert connector.drop_session_cache({b"a", b"b"})
    assert gpu_pool.get_cached_block(b"b", [0]) is None
    assert cpu_pool.get_cached_block(b"b", [0]) is None


def test_gpu_block_reuse_counter_refuses_vllm_kv_cache_metrics():
    gpu_pool = BlockPool(5, True, 16, metrics_collector=KVCacheMetricsCollector())
    with pytest.raises(ValueError, match="kv_cache_metrics"):
        _connector(BlockPool(4, True, 16)).bind_gpu_block_pool(gpu_pool)


def _prompt(
    request_id: str,
    block_hashes: list[bytes],
    num_cached_tokens: int,
    num_cpu_cached_tokens: int = 0,
):
    """A prompt of 6 full 16-token blocks plus 1 token, as the connector sees it."""
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=16 * 6 + 1,
        num_local_cached_tokens=num_cached_tokens - num_cpu_cached_tokens,
        num_external_cached_tokens=num_cpu_cached_tokens,
    )
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=16 * 6 + 1,
        block_hashes=block_hashes,
        prefill_stats=prefill_stats,
    )


@pytest.mark.parametrize("hash_block_size", [16, 8])
def test_broken_chain_counts_a_hole_and_a_missing_gdn_state(hash_block_size):
    # Attention blocks h0..h5; the hit covers h0, h1 (32 tokens). Hashes at 8 tokens: two per block.
    def hashes(name: str) -> list[bytes]:
        per_block = 16 // hash_block_size
        return [
            f"{name}{index}-{part}".encode()
            for index in range(6)
            for part in range(per_block)
        ]

    def block_hash(name: str, index: int) -> bytes:
        return hashes(name)[(index + 1) * (16 // hash_block_size) - 1]

    cpu_pool = BlockPool(12, True, 16)
    gpu_pool = BlockPool(4, True, 16)
    # "hole": h2 is gone, h3 and h4 are on the CPU. "no_state": h2 and h3 are on the CPU.
    _cache(cpu_pool, [block_hash("hole", 3), block_hash("hole", 4)])
    _cache(cpu_pool, [block_hash("state", 2), block_hash("state", 3)])
    # "gpu": h2 is in another request's GPU block, h3 on the CPU: the hit stopped on h2's GDN state.
    # "own": h2 is only in this request's new GPU block (vLLM hashes it before the connector sees it).
    _cache(cpu_pool, [block_hash("gpu", 3), block_hash("own", 3)])
    _cache(gpu_pool, [block_hash("gpu", 2)])
    (own_block,) = gpu_pool.get_new_blocks(1)
    gpu_pool._insert_block_hash(
        make_block_hash_with_group_id(block_hash("own", 2), 0), own_block, num_tokens=16
    )
    connector = _connector(
        cpu_pool, fa_block_size=16, hash_block_size=hash_block_size, fa_gidx=0
    )
    connector.bind_gpu_block_pool(gpu_pool)

    for name, own_blocks in (
        ("hole", []),
        ("state", []),
        ("tail", []),
        ("gpu", []),
        ("own", [own_block]),
    ):
        request = _prompt(name, hashes(name), 32)
        blocks = SimpleNamespace(blocks=(own_blocks,))
        connector.update_state_after_alloc(request, blocks, 0)
        # vLLM's second call, once a CPU load finishes, counts nothing.
        connector.update_state_after_alloc(request, blocks, 0)

    assert connector.broken_chain_requests == {"hole": 2, "no_state": 1}
    # hole: h2..h4 (3 blocks) and own's h2..h3 (2 blocks); no_state: h2..h3 (2 blocks).
    assert connector.broken_chain_tokens == {"hole": 80, "no_state": 32}


def test_prompt_cache_split_lasts_until_the_engine_pops_it():
    connector = _connector(
        BlockPool(4, True, 16), fa_block_size=16, hash_block_size=16, fa_gidx=0
    )
    connector.bind_gpu_block_pool(BlockPool(4, True, 16))
    # 97 prompt tokens: 16 from the GPU cache, 32 from the CPU cache, 49 computed.
    request = _prompt(
        "r", [f"r{index}".encode() for index in range(6)], 48, num_cpu_cached_tokens=32
    )
    connector.update_state_after_alloc(request, None, 0)
    split = PromptCacheSplit(
        gpu_cached_tokens=16, cpu_cached_tokens=32, computed_tokens=49
    )
    assert connector.prompt_cache_split("r") == split

    # A finished request keeps it, so the engine can log it after vLLM's step.
    connector.request_finished_all_groups(request, ([],))
    assert connector.prompt_cache_split("r") == split

    connector.pop_prompt_cache_split("r")
    assert connector.prompt_cache_split("r") is None


def test_store_count_and_cpu_gdn_state_scan():
    cpu_pool = BlockPool(6, True, 16)
    _cache(cpu_pool, [b"attention-0", b"attention-1"], group_id=0)
    _cache(cpu_pool, [b"state-0"], group_id=1)
    _cache(cpu_pool, [b"window-0"], group_id=2)
    attention = dict(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
    groups = [
        FullAttentionSpec(**attention),
        MambaSpec(block_size=16, shapes=((4, 8),), dtypes=(torch.float16,)),
        SlidingWindowSpec(**attention, sliding_window=32),
    ]
    connector = _connector(
        cpu_pool,
        fa_gidx=0,
        cpu_kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in groups]
        ),
        build_connector_meta=lambda output: SimpleCPUOffloadMetadata(
            store_gpu_blocks=[3, 4], store_cpu_blocks=[5, 6]
        ),
    )

    connector.build_connector_meta(None)
    connector.build_connector_meta(None)

    assert connector.num_stored_blocks == 4
    # Only the MambaSpec group's block: attention and sliding-window blocks are not GDN states.
    assert connector.num_cpu_gdn_state_blocks == 1
