# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""VLLMSessionKVHolder on vLLM's real KV cache manager: 1 attention group + 3 Mamba "align"
groups, 16-token blocks standing in for Qwen3.5's 1,152."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchtitan.rl.generator.vllm.session_kv_holder import (
    cached_block_ids,
    VLLMSessionKVHolder,
)
from torchtitan.rl.generator.vllm.simple_cpu_offload import PromptCacheSplit
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request, RequestStatus

BLOCK = 16


@pytest.fixture
def manager() -> KVCacheManager:
    return _make_manager()


def _make_manager(num_speculative_blocks: int = 0) -> KVCacheManager:
    """`num_speculative_blocks`: extra GDN state blocks per request, one per draft token."""
    init_none_hash(sha256)
    attention = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    gdn = MambaSpec(
        block_size=BLOCK,
        shapes=((4, 8),),
        dtypes=(torch.float16,),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
    )
    groups = [KVCacheGroupSpec(["attn"], attention)] + [
        KVCacheGroupSpec([f"gdn{i}"], gdn) for i in range(3)
    ]
    config = KVCacheConfig(
        num_blocks=80,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        config,
        max_model_len=4096,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        enable_caching=True,
    )


def _request(request_id: str, tokens: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=10_000),
        pooling_params=None,
        cache_salt="v0",
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _schedule(manager, request, num_tokens, blocks=None, hit=0, sample=None) -> None:
    manager.new_step_starts()
    assert manager.allocate_slots(
        request, num_tokens, num_new_computed_tokens=hit, new_computed_blocks=blocks
    )
    request.num_computed_tokens += hit + num_tokens
    if sample is not None:
        request.append_output_token_ids(sample)


def _prefill(manager, request) -> int:
    """Block-aligned chunk, then the tail (like vLLM's Mamba-aligned split); returns the hit."""
    manager.new_step_starts()
    blocks, hit, _ = manager.get_computed_blocks(request)
    num_prompt = request.num_prompt_tokens
    aligned = num_prompt - num_prompt % BLOCK
    first_hit = hit
    if aligned > hit:
        _schedule(manager, request, aligned - hit, blocks, hit)
        blocks, hit = None, 0
    _schedule(
        manager, request, num_prompt - request.num_computed_tokens - hit, blocks, hit, 7
    )
    return first_hit


def _churn(manager, tag: str) -> None:
    """Push every block through the free queue with throwaway requests."""
    for i in range(80):
        request = _request(f"{tag}{i}", [1000 + 97 * i + j for j in range(BLOCK + 1)])
        _prefill(manager, request)
        manager.free(request)


def _scheduler(
    manager, max_num_scheduled_tokens=0, speculative_config=None, **attrs
) -> SimpleNamespace:
    """The vLLM scheduler attributes the holder reads."""
    return SimpleNamespace(
        kv_cache_manager=manager,
        vllm_config=SimpleNamespace(speculative_config=speculative_config),
        use_eagle_block_drop=False,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        **attrs,
    )


def _holder(
    manager,
    requests,
    floor=0,
    running=(),
    waiting=(),
    skipped_waiting=(),
    offload_connector=None,
    speculative_config=None,
) -> VLLMSessionKVHolder:
    scheduler = _scheduler(
        manager,
        speculative_config=speculative_config,
        requests=requests,
        running=running,
        waiting=waiting,
        skipped_waiting=skipped_waiting,
    )
    holder = VLLMSessionKVHolder.Config().build(
        scheduler=scheduler, offload_connector=offload_connector
    )
    holder.free_floor_blocks = floor
    return holder


@pytest.mark.parametrize(
    "num_draft_tokens", [0, 3], ids=["no_drafts", "mtp3_without_block_drop"]
)
def test_held_prefix_survives_churn_and_the_next_turn_hits_it(num_draft_tokens):
    # Drafts with the block drop off, as the engine sets them for the holder: vLLM's scheduler
    # then builds this manager with use_eagle=False, and each request gets k more GDN state blocks.
    manager = _make_manager(num_speculative_blocks=num_draft_tokens)
    requests = {}
    holder = _holder(
        manager, requests, speculative_config=object() if num_draft_tokens else None
    )
    turn1 = _request("t1", list(range(50)))
    requests["t1"] = turn1
    holder.track("t1", session_id="s", group_id=0)
    _prefill(manager, turn1)
    holder.after_step()
    # Decode moves the GDN checkpoint out of the request; other requests churn the free queue.
    for t in range(20):
        _schedule(manager, turn1, 1, sample=100 + t)
    _churn(manager, "a")
    manager.free(turn1)
    del requests["t1"]
    _churn(manager, "b")

    turn2 = _request("t2", list(turn1.all_token_ids) + [5, 6, 7])
    # The continuation resumes at turn 1's last full prompt block (48 of 50 tokens).
    assert _prefill(manager, turn2) == 48


def test_without_holding_the_next_turn_recomputes(manager):
    turn1 = _request("t1", list(range(50)))
    _prefill(manager, turn1)
    for t in range(20):
        _schedule(manager, turn1, 1, sample=100 + t)
    _churn(manager, "a")
    manager.free(turn1)
    _churn(manager, "b")
    turn2 = _request("t2", list(turn1.all_token_ids) + [5, 6, 7])
    assert _prefill(manager, turn2) == 0


def test_release_returns_every_held_block(manager):
    requests = {}
    holder = _holder(manager, requests)
    free_before = manager.block_pool.get_num_free_blocks()
    turn1 = _request("t1", list(range(50)))
    requests["t1"] = turn1
    holder.track("t1", session_id="s", group_id=3)
    _prefill(manager, turn1)
    holder.after_step()
    manager.free(turn1)
    assert manager.block_pool.get_num_free_blocks() < free_before

    holder.release(group_ids=[3])
    assert holder.num_sessions == 0
    assert manager.block_pool.get_num_free_blocks() == free_before


def _finished_turn(manager, holder, requests, session: str, tokens: list[int]) -> None:
    """One turn of `session`: admit, prefill (held), finish."""
    request = _request(session, tokens)
    requests[session] = request
    holder.track(session, session_id=session, group_id=0)
    _prefill(manager, request)
    holder.after_step()
    manager.free(request)
    del requests[session]
    holder.after_step()


def test_release_frees_sessions_in_sorted_order(manager):
    """Every TP rank frees a release's sessions in one order; set order depends on the process."""
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "b", list(range(50)))
    _finished_turn(manager, holder, requests, "a", list(range(100, 150)))
    order = []
    release_one = holder._release
    holder._release = lambda session_id: (
        order.append(session_id),
        release_one(session_id),
    )

    holder.release(session_ids=["b", "a"])

    assert order == ["a", "b"]


def test_evict_until_free_releases_the_session_idle_longest_first(manager):
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "old", list(range(50)))
    _finished_turn(manager, holder, requests, "new", list(range(500, 550)))

    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.after_step()
    # Only "old" was released: releasing "new" now frees the rest.
    assert holder.num_sessions == 1 and holder.num_evicted == 1
    holder.release(session_ids=["new"])
    assert holder.num_sessions == 0


def test_gpu_pressure_preserves_cpu_cache_until_session_release(manager):
    connector = Mock()
    connector.get_pending_save_ids.return_value = set()
    connector.release_session_cache.return_value = None
    requests = {}
    holder = _holder(manager, requests, offload_connector=connector)
    _finished_turn(manager, holder, requests, "s", list(range(50)))
    hashes = set(holder._cpu_sessions["s"][1])
    assert hashes
    assert holder.num_sessions == 1

    # GPU pressure drops the hold, but the rollout can still reuse its CPU KV.
    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.after_step()
    assert holder.num_sessions == 0
    connector.release_session_cache.assert_not_called()

    # Group completion still reaches the session after its GPU hold is gone.
    holder.release(group_ids=[0])
    connector.release_session_cache.assert_called_once_with(hashes, set())
    assert not holder.has_pending_work


def test_release_waits_for_decode_and_saves_and_preserves_shared_prefix(manager):
    connector = Mock()
    connector.get_pending_save_ids.return_value = {0}
    connector.release_session_cache.return_value = None
    first = _request("first", list(range(50)))
    requests = {first.request_id: first}
    holder = _holder(manager, requests, offload_connector=connector)
    holder.track(first.request_id, "session-0", 0)
    _prefill(manager, first)
    holder.after_step()
    assert holder.num_sessions == 1
    shared_hashes = set(first.block_hashes)
    holder.release(["session-0"], [])
    assert holder.num_sessions == 0
    connector.release_session_cache.assert_not_called()

    # Final decode hashes appear after release; the request is then removed.
    for token in range(BLOCK):
        _schedule(manager, first, 1, sample=100 + token)
    decode_hashes = set(first.block_hashes) - shared_hashes
    assert decode_hashes
    manager.free(first)
    del requests[first.request_id]
    holder.after_step()
    assert holder.has_pending_work
    connector.release_session_cache.assert_not_called()

    # Another session reuses the prefix while the old save is still in flight.
    second = _request("second", list(range(50)))
    requests[second.request_id] = second
    holder.track(second.request_id, "session-1", 1)
    _prefill(manager, second)
    # A newer unrelated save must not keep this release waiting.
    connector.get_pending_save_ids.return_value = {1}
    holder.after_step()
    assert holder.num_sessions == 1
    connector.release_session_cache.assert_called_once_with(
        decode_hashes, shared_hashes
    )
    assert not holder.has_pending_work

    # Group release reaches idle sessions too. The final owner can release it.
    manager.free(second)
    del requests[second.request_id]
    holder.after_step()
    connector.get_pending_save_ids.return_value = set()
    holder.release([], [1])
    assert holder.num_sessions == 0
    assert connector.release_session_cache.call_args.args == (shared_hashes, set())
    assert not holder.has_pending_work
    holder.release(["session-0", "unknown"], [0, 1])
    assert connector.release_session_cache.call_count == 2


def test_aborted_turn_uncaches_its_own_prefix_and_keeps_the_shared_prompt(manager):
    # GRPO siblings s0 and s1 share a 32-token prompt. The engine aborts s0's turn (a non-finite
    # logprob) after the holder held its prefix.
    requests = {}
    holder = _holder(manager, requests)
    shared = list(range(32))
    first_turns = {
        "s0": _request("s0-t0", shared + list(range(100, 118))),
        "s1": _request("s1-t0", shared + list(range(200, 218))),
    }
    for session_id, request in first_turns.items():
        requests[request.request_id] = request
        holder.track(request.request_id, session_id=session_id, group_id=0)
        _prefill(manager, request)
    holder.after_step()
    for request in first_turns.values():
        manager.free(request)
        del requests[request.request_id]
    holder.after_step(aborted_request_ids={"s0-t0"})
    assert holder.num_sessions == 1

    s0_only_hashes = set(first_turns["s0"].block_hashes) - set(
        first_turns["s1"].block_hashes
    )
    assert s0_only_hashes
    assert not cached_block_ids(manager.block_pool, s0_only_hashes)
    # s1 first: s0's recompute would cache the shared prompt again.
    s1_next = _request("s1-t1", list(first_turns["s1"].all_token_ids) + [5, 6, 7])
    assert _prefill(manager, s1_next) == 48
    s0_next = _request("s0-t1", list(first_turns["s0"].all_token_ids) + [5, 6, 7])
    assert _prefill(manager, s0_next) <= len(shared)


def test_aborted_turn_drop_retries_while_a_cpu_save_is_in_flight(manager):
    connector = Mock()
    connector.get_pending_save_ids.return_value = set()
    drop_calls = []

    def drop_session_cache(block_hashes):
        drop_calls.append(set(block_hashes))
        # The first call finds one GPU block still being saved to the CPU.
        return len(drop_calls) > 1

    connector.drop_session_cache.side_effect = drop_session_cache
    requests = {}
    holder = _holder(manager, requests, offload_connector=connector)
    turn = _request("t0", list(range(50)))
    requests["t0"] = turn
    holder.track("t0", session_id="s", group_id=0)
    _prefill(manager, turn)
    holder.after_step()
    manager.free(turn)
    del requests["t0"]

    holder.after_step(aborted_request_ids={"t0"})
    assert holder.has_pending_work
    holder.after_step()
    assert not holder.has_pending_work
    assert drop_calls == [set(turn.block_hashes)] * 2


def test_evict_until_free_skips_a_session_with_a_running_request(manager):
    requests = {}
    busy = _request("busy", list(range(50)))
    holder = _holder(manager, requests, running=[busy])
    requests["busy"] = busy
    holder.track("busy", session_id="busy", group_id=0)
    _prefill(manager, busy)
    holder.after_step()

    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.after_step()
    assert holder.num_sessions == 1 and holder.num_evicted == 0


def test_evict_until_free_releases_a_session_whose_next_request_is_waiting(manager):
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "s", list(range(50)))
    # The next turn is in vLLM but not scheduled yet: only the hold keeps its blocks.
    requests["s/turn2"] = _request("s/turn2", list(range(60)))
    holder.track("s/turn2", session_id="s", group_id=0)

    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.after_step()
    assert holder.num_sessions == 0 and holder.num_evicted == 1


def test_release_before_prefill_takes_no_hold(manager):
    requests = {}
    holder = _holder(manager, requests)
    request = _request("t1", list(range(50)))
    requests["t1"] = request
    holder.track("t1", session_id="s", group_id=7)
    holder.release(session_ids=["s"], group_ids=[7])
    _prefill(manager, request)
    holder.after_step()
    assert holder.num_sessions == 0


def test_next_turn_moves_the_hold_and_release_frees_everything(manager):
    requests = {}
    holder = _holder(manager, requests)
    free_before = manager.block_pool.get_num_free_blocks()
    turn1 = _request("t1", list(range(50)))
    requests["t1"] = turn1
    holder.track("t1", session_id="s", group_id=0)
    _prefill(manager, turn1)
    holder.after_step()
    for t in range(20):
        _schedule(manager, turn1, 1, sample=100 + t)
    manager.free(turn1)
    del requests["t1"]
    holder.after_step()

    turn2 = _request("t2", list(turn1.all_token_ids) + [5, 6, 7])
    requests["t2"] = turn2
    holder.track("t2", session_id="s", group_id=0)
    _prefill(manager, turn2)
    holder.after_step()
    manager.free(turn2)
    del requests["t2"]
    holder.after_step()
    assert holder.num_sessions == 1

    holder.release(session_ids=["s"])
    assert manager.block_pool.get_num_free_blocks() == free_before


def test_free_floor_is_the_watermark_plus_one_prefill_chunk(manager):
    manager.watermark_blocks = 8  # vLLM's watermark=0.1 of the 80-block pool
    scheduler = _scheduler(manager, max_num_scheduled_tokens=100)
    holder = VLLMSessionKVHolder.Config().build(scheduler=scheduler)
    # watermark 8 + 4 KV cache groups x cdiv(100, 16) = 7 chunk blocks
    assert holder.free_floor_blocks == 8 + 4 * 7


def test_holder_rejects_partial_block_hits_and_the_eagle_block_drop(manager):
    manager.coordinator.enable_partial_hash_hits = True
    with pytest.raises(ValueError, match="does not support partial-block prefix hits"):
        VLLMSessionKVHolder.Config().build(scheduler=_scheduler(manager))
    manager.coordinator.enable_partial_hash_hits = False
    scheduler = _scheduler(manager, speculative_config=object())
    scheduler.use_eagle_block_drop = True
    with pytest.raises(ValueError, match="EAGLE block drop off"):
        VLLMSessionKVHolder.Config().build(scheduler=scheduler)


@pytest.mark.parametrize("queue", ["waiting", "skipped_waiting"])
def test_waiting_for_kv_load_keeps_gpu_holds_unless_memory_is_low(manager, queue):
    requests, waiting = {}, []
    holder = _holder(manager, requests, **{queue: waiting})
    _finished_turn(manager, holder, requests, "idle", list(range(50)))
    free_before = manager.block_pool.get_num_free_blocks()
    loading = _request("loading", list(range(200, 260)))
    loading.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    waiting.append(loading)

    holder.after_step()

    assert holder.num_sessions == 1 and holder.num_evicted == 0
    assert manager.block_pool.get_num_free_blocks() == free_before

    # Waiting for a load must not disable genuine memory-pressure eviction.
    holder.free_floor_blocks = free_before + 1
    holder.after_step()
    assert holder.num_sessions == 0 and holder.num_evicted == 1


@pytest.mark.parametrize("queue", ["waiting", "skipped_waiting"])
def test_a_waiting_turn_gets_room_when_held_sessions_fill_the_pool(manager, queue):
    """Nothing runs, free >= floor, and the waiting turn needs more blocks than are free.
    vLLM serves `skipped_waiting` first and leaves a refused request there."""
    requests, waiting = {}, []
    holder = _holder(manager, requests, **{queue: waiting})
    for i in range(12):  # 6 held blocks each: 7 of 80 stay free
        _finished_turn(
            manager, holder, requests, f"s{i}", [1000 * i + j for j in range(50)]
        )
    # Set after the turns: every after_step checks the floor, which would release during setup.
    holder.free_floor_blocks = 4
    assert holder.num_sessions == 12 and manager.block_pool.get_num_free_blocks() == 7
    # 10 attention blocks + GDN state: more than the 7 free.
    head = _request("head", list(range(50_000, 50_160)))
    requests["head"] = head
    # A load in the same queue must not hide another request that needs room.
    loading = _request("loading", list(range(200, 260)))
    loading.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    waiting.extend([loading, head])

    # The engine loop with an empty batch: vLLM's admission check, then the holder.
    for _ in range(20):
        manager.new_step_starts()
        if manager.allocate_slots(
            head, head.num_tokens, full_sequence_must_fit=True, has_scheduled_reqs=False
        ):
            break
        holder.after_step()
    else:
        pytest.fail("held sessions stalled the engine: the waiting turn never fit")


def test_held_blocks_count_a_shared_prompt_once(manager):
    requests = {}
    holder = _holder(manager, requests)
    _finished_turn(manager, holder, requests, "a", list(range(50)))
    num_held_by_a = holder.num_held_blocks
    # GRPO sibling "b" has the same prompt, so it holds the same cached blocks.
    _finished_turn(manager, holder, requests, "b", list(range(50)))
    assert num_held_by_a > 0 and holder.num_held_blocks == num_held_by_a

    holder.release(session_ids=["a"])
    assert holder.num_held_blocks == num_held_by_a
    holder.release(session_ids=["b"])
    assert holder.num_held_blocks == 0


def test_lost_prefix_counts_history_the_next_turn_recomputes(manager):
    cached_at_admission = {}
    connector = Mock()
    connector.get_pending_save_ids.return_value = set()
    connector.prompt_cache_split.side_effect = lambda request_id: (
        None
        if request_id not in cached_at_admission
        else PromptCacheSplit(
            gpu_cached_tokens=cached_at_admission[request_id],
            cpu_cached_tokens=0,
            computed_tokens=0,
        )
    )
    requests = {}
    scheduler = _scheduler(
        manager,
        requests=requests,
        running=(),
        waiting=(),
        skipped_waiting=(),
        block_size=BLOCK,
        hash_block_size=BLOCK,
    )
    holder = VLLMSessionKVHolder.Config().build(
        scheduler=scheduler, offload_connector=connector
    )
    holder.free_floor_blocks = 0

    def finished_turn(request_id, tokens, num_cached_tokens, decode_tokens=0):
        cached_at_admission[request_id] = num_cached_tokens
        request = _request(request_id, tokens)
        requests[request_id] = request
        holder.track(request_id, session_id="s", group_id=0)
        _prefill(manager, request)
        holder.after_step()
        for t in range(decode_tokens):
            _schedule(manager, request, 1, sample=100 + t)
        manager.free(request)
        del requests[request_id]
        holder.after_step()
        return request

    def reusable_and_lost():
        return holder.num_reusable_prefix_tokens, holder.num_lost_prefix_tokens

    # Turn 1 ends at 71 tokens (4 full blocks). A first turn has no history to lose.
    turn1 = finished_turn("t1", list(range(50)), 0, decode_tokens=20)
    assert reusable_and_lost() == (0, 0)

    # Turn 2 (74 tokens) continues it: all 4 blocks reusable, 3 of them hit at admission.
    finished_turn("t2", list(turn1.all_token_ids) + [5, 6, 7], 48)
    assert reusable_and_lost() == (64, 16)

    # A compacted history doesn't continue turn 2: nothing reusable, nothing lost.
    finished_turn("t3", list(range(500, 600)), 0)
    assert reusable_and_lost() == (64, 16)

    # GPU pressure drops the hold first; the group release must still forget the session.
    holder.free_floor_blocks = manager.block_pool.get_num_free_blocks() + 1
    holder.after_step()
    assert holder.num_sessions == 0
    holder.release(group_ids=[0])
    assert not holder._last_turns


def test_stats_count_idle_held_blocks_and_waiting_turns_by_history(manager):
    requests = {}
    waiting = []
    holder = _holder(manager, requests, waiting=waiting)
    _finished_turn(manager, holder, requests, "s", list(range(50)))
    held_blocks = holder.num_held_blocks
    # s's next turn and a new session's first turn wait in vLLM.
    for request_id, session_id in (("s/turn2", "s"), ("new/turn1", "new")):
        requests[request_id] = _request(request_id, list(range(60)))
        waiting.append(requests[request_id])
        holder.track(request_id, session_id=session_id, group_id=0)

    holder._refresh_stats()

    assert holder.num_waiting_requests == {"yes": 1, "no": 1}
    # s has a request in vLLM again: its held blocks are no longer idle.
    assert holder.num_idle_held_blocks == 0
    holder.release(session_ids=["new"])
    del requests["s/turn2"], requests["new/turn1"]
    waiting.clear()
    holder.after_step()
    holder._refresh_stats()
    assert holder.num_idle_held_blocks == held_blocks
