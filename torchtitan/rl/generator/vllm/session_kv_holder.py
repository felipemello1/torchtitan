# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hold each multi-turn session's reusable KV between its turns."""

from __future__ import annotations

import time
from collections import Counter, OrderedDict
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, NamedTuple, TYPE_CHECKING

from vllm import envs
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash, get_block_hash
from vllm.v1.request import Request, RequestStatus

from torchtitan.config import Configurable

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool


class VLLMSessionKVHolder(Configurable):
    """Keeps each multi-turn session's cached prefix on GPU between its turns.

    With a CPU offload connector, session/group release also sends eviction hints
    after outstanding saves finish, protecting hashes used by other local sessions.

    Between turns (a tool call, an env step) a session has no request in vLLM, so its blocks sit in
    vLLM's free queue, where any new request can take them; the next turn then recomputes its
    history. The holder takes a reference on them instead:

        turn 1 prefill done  ->  hold the blocks turn 2 would hit (attention + Gated-DeltaNet state)
        tool call            ->  held: no other request can take them
        turn 2 prefill done  ->  hold the blocks turn 3 would hit, drop turn 1's hold
        rollout ends         ->  release: the blocks go back to vLLM's free queue, still cached
        turn 2 aborted       ->  drop: release the hold, uncache the KV only this session has

    It holds at prefill, not at the turn's end: in vLLM's hybrid "align" mode the prompt-end GDN
    state leaves the request when decode starts.

    When memory runs short, it releases whole idle sessions, idle longest first:
    - after each step, while fewer than ``floor`` blocks are free: the next step needs room, and the
      holder can't act inside a step;
    - after a step where nothing runs and requests wait for blocks, excluding pending KV loads.

    The floor is vLLM's watermark plus the most blocks the next step can add to running requests
    (vLLM schedules them before it admits new ones):

        free_floor_blocks = watermark_blocks
            + per KV cache group: cdiv(max_num_scheduled_tokens, block_size)  # one prefill chunk
        floor = free_floor_blocks
            + num_kv_cache_groups x len(running)  # a block each at a block boundary

    - Held blocks count as used in vLLM's ``kv_cache_usage``.
    - Every mutating method runs on every rank's engine thread in the same order, so the TP ranks'
      schedulers stay identical.

    Example:

        holder = VLLMSessionKVHolder.Config().build(scheduler=scheduler)
        internal_id = engine.add_request(request_id="group=3/rollout=0/turn=2", ...)
        holder.track(internal_id, session_id="group=3/rollout=0", group_id=3)
        engine.step()
        holder.after_step()  # holds new prefixes, then releases idle sessions for the next step
        holder.release(session_ids=["group=3/rollout=0"])  # the rollout ended

    Args:
        scheduler: The vLLM v1 scheduler of this rank's in-process engine.
        offload_connector: Optional connector with ``get_pending_save_ids()``,
            ``release_session_cache(released_hashes, active_hashes)``,
            ``drop_session_cache(block_hashes)`` and
            ``pop_cached_tokens(request_id)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        scheduler: Any,
        offload_connector: Any | None = None,
    ) -> None:
        del config
        # Partial-block prefix hits move the replay point away from the prompt-end block this holds.
        if scheduler.kv_cache_manager.coordinator.enable_partial_hash_hits:
            raise ValueError(
                "session_kv_holder does not support partial-block prefix hits"
            )

        # vLLM's EAGLE/MTP block drop moves the replay point too: the next turn also needs the
        # attention block past the held ones. Once that block is evicted, the hit falls to a block
        # with no GDN state.
        # TODO: delete with the engine's `disable_eagle_block_drop` key (`VLLMEngine._build_engine_args`).
        if (
            scheduler.vllm_config.speculative_config is not None
            and scheduler.use_eagle_block_drop
        ):
            raise ValueError(
                "session_kv_holder needs vLLM's EAGLE block drop off "
                "(speculative_config.disable_eagle_block_drop=True)"
            )

        self._scheduler = scheduler
        kv_cache_manager = scheduler.kv_cache_manager
        self._coordinator = kv_cache_manager.coordinator
        self._block_pool = kv_cache_manager.block_pool
        self._num_kv_cache_groups = kv_cache_manager.num_kv_cache_groups
        self.free_floor_blocks = kv_cache_manager.watermark_blocks + sum(
            cdiv(scheduler.max_num_scheduled_tokens, group.kv_cache_spec.block_size)
            for group in kv_cache_manager.kv_cache_config.kv_cache_groups
        )
        # Requests in vLLM: internal request id -> (request, session, group, prefill seen).
        # The request object outlives scheduler cleanup, so CPU release can read its final
        # block hashes, decode blocks included.
        self._live: dict[str, tuple[Request, str, int, bool]] = {}
        # Held blocks per session, idle longest first.
        self._held: OrderedDict[str, list[Any]] = OrderedDict()
        self._session_groups: dict[str, int] = {}
        self._group_sessions: dict[int, set[str]] = {}
        # Plain ints, so other threads (e.g. a logging endpoint) can read them safely.
        self.num_sessions = 0
        self.num_evicted = 0
        # Distinct held blocks: GRPO siblings hold their shared prompt blocks once each.
        self.num_held_blocks = 0
        self._held_block_refs: Counter[int] = Counter()
        # Refreshed every VLLM_LOG_STATS_INTERVAL seconds (`_refresh_stats`): distinct held blocks of
        # sessions with no request in vLLM, and waiting requests by whether their session ran a turn.
        self.num_gpu_blocks = self._block_pool.num_gpu_blocks
        self.num_idle_held_blocks = 0
        self.num_waiting_requests = {"yes": 0, "no": 0}
        self._next_stats_time = 0.0
        # Requests in vLLM whose session ran a turn before (internal ids).
        self._requests_with_history: set[str] = set()
        # Lost-prefix accounting (with an offload connector only): see `_count_lost_prefix`.
        self.num_reusable_prefix_tokens = 0
        self.num_lost_prefix_tokens = 0
        self._last_turns: dict[str, _LastTurn] = {}
        # CPU release hints (with an offload connector only).
        self._offload_connector = offload_connector
        self._cpu_sessions: dict[str, tuple[int, set[BlockHash]]] = {}
        self._released_sessions: set[str] = set()
        self._released_hashes: set[BlockHash] = set()
        self._eviction_hashes: set[BlockHash] = set()
        self._pending_saves: set[int] = set()
        # Hashes of aborted turns whose cached copies are not all dropped yet: see `_drop_cached_copies`.
        self._dropped_hashes: set[BlockHash] = set()

    def track(self, internal_request_id: str, session_id: str, group_id: int) -> None:
        """Hold this request's prefix once its prefill finishes."""
        request = self._scheduler.requests[internal_request_id]
        self._live[internal_request_id] = (request, session_id, group_id, False)
        # A held session, or one with CPU copies, ran a turn; released ones under pressure keep these.
        if session_id in self._held or session_id in self._cpu_sessions:
            self._requests_with_history.add(internal_request_id)
        if self._offload_connector is not None:
            group, _ = self._cpu_sessions.setdefault(session_id, (group_id, set()))
            assert group == group_id

    def after_step(self, aborted_request_ids: Collection[str] = ()) -> None:
        """After each engine step: hold the prefix of requests whose prefill finished, mark sessions
        whose request left, drop aborted turns, and process GPU pressure and CPU release hints.

        `aborted_request_ids`: vLLM's ids of aborted requests whose cached KV must not be read again.
        """
        # A drop left from an earlier step: its GPU blocks' saves to the CPU may be done now.
        if self._dropped_hashes:
            self._drop_cached_copies()

        for internal_id, (request, session_id, group_id, held) in list(
            self._live.items()
        ):
            if internal_id not in self._scheduler.requests:
                self._requests_with_history.discard(internal_id)
                if internal_id in aborted_request_ids:
                    del self._live[internal_id]
                    self._drop_aborted_turn(request, session_id)
                    continue

                if self._offload_connector is not None:
                    self._cpu_sessions[session_id][1].update(request.block_hashes)
                    self._last_turns[session_id] = _LastTurn(
                        group_id=group_id,
                        num_block_hashes=len(request.block_hashes),
                        last_block_hash=(
                            request.block_hashes[-1] if request.block_hashes else None
                        ),
                    )
                del self._live[internal_id]
                if session_id in self._held:
                    self._held.move_to_end(session_id)
            elif not held and request.num_computed_tokens >= request.num_prompt_tokens:
                self._live[internal_id] = (request, session_id, group_id, True)
                if self._offload_connector is not None:
                    self._count_lost_prefix(request, session_id)
                self._hold(request, session_id, group_id)

        if self._offload_connector is not None:
            self._release_cpu_cache()
        # Nothing runs and requests wait: with an empty batch vLLM skips the watermark, so it refused
        # the head for lack of blocks (or the batch just drained: one cheap extra release).
        # A pending KV load already has destination blocks; it waits for data.
        scheduler = self._scheduler
        if not scheduler.running and any(
            request.status != RequestStatus.WAITING_FOR_REMOTE_KVS
            for request in (*scheduler.skipped_waiting, *scheduler.waiting)
        ):
            free_before = self._block_pool.get_num_free_blocks()
            for session_id in list(self._held):
                self._release(session_id)
                self.num_evicted += 1
                # GRPO siblings share prompt blocks, so one release may free none.
                if self._block_pool.get_num_free_blocks() > free_before:
                    break
        self._evict_until_free()

        now = time.monotonic()
        if now >= self._next_stats_time:
            self._next_stats_time = now + envs.VLLM_LOG_STATS_INTERVAL
            self._refresh_stats()

    def _refresh_stats(self) -> None:
        """Count idle sessions' held blocks and waiting requests by history, for the stat logger."""
        live_sessions = {session for _, session, _, _ in self._live.values()}
        self.num_idle_held_blocks = len(
            {
                block.block_id
                for session_id, blocks in self._held.items()
                if session_id not in live_sessions
                for block in blocks
            }
        )
        scheduler = self._scheduler
        waiting = [*scheduler.skipped_waiting, *scheduler.waiting]
        with_history = sum(
            request.request_id in self._requests_with_history for request in waiting
        )
        # In place: the stat logger holds this dict.
        self.num_waiting_requests["yes"] = with_history
        self.num_waiting_requests["no"] = len(waiting) - with_history

    def _evict_until_free(self) -> None:
        """Release sessions with no running request, idle longest first, until enough blocks are free.

        A session whose request is running is skipped: that request still references most of its
        blocks, so releasing it frees little. A session whose next request is still waiting is
        released: its hold is all that keeps those blocks, and keeping it once memory is full stops
        every waiting request from starting.
        """
        running = self._scheduler.running
        # GRPO siblings decode in lockstep, so all running requests can cross a boundary together.
        floor = self.free_floor_blocks + self._num_kv_cache_groups * len(running)
        if self._block_pool.get_num_free_blocks() >= floor:
            return
        running_sessions = {
            self._live[request.request_id][1]
            for request in running
            if request.request_id in self._live
        }
        for session_id in list(self._held):
            if self._block_pool.get_num_free_blocks() >= floor:
                return
            if session_id not in running_sessions:
                self._release(session_id)
                self.num_evicted += 1

    def release(self, session_ids: list[str] = (), group_ids: list[int] = ()) -> None:
        """Release sessions that make no more calls, and every session of finished groups."""
        # TODO: an ended session's blocks go to the keep end of vLLM's free queue, so they outlive live
        # idle history and the lazy CPU walk may copy them. Moving them to the evict end and dropping their
        # hashes (as #5919's evictor does) needs private free-queue edits and a skip for blocks under a lazy
        # copy; gain unmeasured, and it waits on the upstream choice between this holder and that evictor.
        released = set(session_ids)
        for group_id in group_ids:
            released |= self._group_sessions.pop(group_id, set())
        # Sorted, not set order: string hashes differ per process, and every TP rank must free
        # blocks in the same order.
        for session_id in sorted(released):
            self._release(session_id)
        self._last_turns = {
            session_id: last_turn
            for session_id, last_turn in self._last_turns.items()
            if session_id not in released and last_turn.group_id not in group_ids
        }
        # A request still before its prefill must not re-hold a released session.
        dropped_groups = set(group_ids)
        for internal_id, (request, session_id, group_id, held) in list(
            self._live.items()
        ):
            if session_id in released or group_id in dropped_groups:
                self._live[internal_id] = (request, session_id, group_id, True)
        if self._offload_connector is not None:
            self._released_sessions.update(set(session_ids) & self._cpu_sessions.keys())
            self._released_sessions.update(
                session
                for session, (group, _) in self._cpu_sessions.items()
                if group in dropped_groups
            )
            self._release_cpu_cache()

    @property
    def has_pending_work(self) -> bool:
        """Whether CPU release or an aborted turn's drop still needs engine steps to finish requests
        or saves."""
        return bool(
            self._released_sessions
            or self._released_hashes
            or self._eviction_hashes
            or self._dropped_hashes
        )

    def _release_cpu_cache(self) -> None:
        """Send eviction hints after released requests and their save batch finish."""
        if not self.has_pending_work:
            return
        live_sessions = {session for _, session, _, _ in self._live.values()}
        for session in sorted(self._released_sessions - live_sessions):
            _, hashes = self._cpu_sessions.pop(session)
            self._released_hashes.update(hashes)
            self._released_sessions.remove(session)

        connector = self._offload_connector
        if not self._eviction_hashes:
            if not self._released_hashes:
                return
            self._eviction_hashes, self._released_hashes = self._released_hashes, set()
            # Include finish-time saves queued for the next step. Freeze this
            # batch so unrelated later saves cannot postpone its release forever.
            self._pending_saves = connector.get_pending_save_ids()
        if self._pending_saves & connector.get_pending_save_ids():
            return

        active_hashes = set().union(
            *(hashes for _, hashes in self._cpu_sessions.values())
        )
        for request, _, _, _ in self._live.values():
            active_hashes.update(request.block_hashes)
        released = self._eviction_hashes - active_hashes
        self._eviction_hashes.clear()
        self._pending_saves.clear()
        if released:
            connector.release_session_cache(released, active_hashes)

    def _hold(self, request: Any, session_id: str, group_id: int) -> None:
        # The prompt's last token is recomputed for its logits, so a continuation hits at most
        # num_prompt_tokens - 1 tokens; the lookup returns the attention blocks and, per GDN
        # group, the state block that ends on that aligned boundary. vLLM returns the oldest
        # cached copy of each block, so the hold may take a duplicate of a block this request's
        # table also owns (about one extra block per session while it decodes).
        hit_blocks, _, _ = self._coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_prompt_tokens - 1
        )
        blocks = [b for kv_group in hit_blocks for b in kv_group if not b.is_null]
        # Take the new hold before dropping the old one, so shared prefix blocks stay held.
        self._block_pool.touch(blocks)
        self._release(session_id)
        self._held[session_id] = blocks
        self._held_block_refs.update(block.block_id for block in blocks)
        self.num_held_blocks = len(self._held_block_refs)
        self._session_groups[session_id] = group_id
        self._group_sessions.setdefault(group_id, set()).add(session_id)
        self.num_sessions += 1

    def _release(self, session_id: str) -> None:
        blocks = self._held.pop(session_id, None)
        group_id = self._session_groups.pop(session_id, None)
        if group_id is not None:
            self._group_sessions.get(group_id, set()).discard(session_id)
        if blocks is not None:
            self.num_sessions -= 1
            for block in blocks:
                self._held_block_refs[block.block_id] -= 1
                if self._held_block_refs[block.block_id] == 0:
                    del self._held_block_refs[block.block_id]
            self.num_held_blocks = len(self._held_block_refs)
            self._block_pool.free_blocks(reversed(blocks))

    def _drop_aborted_turn(self, request: Request, session_id: str) -> None:
        """Release the session's hold and uncache the KV of a turn aborted for a non-finite logprob,
        so the session's next turn recomputes that KV instead of reading it again.

        Hashes another session still uses stay cached, e.g. a GRPO group's shared prompt:

            s0 aborts; its sibling s1 shares the prompt blocks p0, p1
            s0's hashes:  p0 p1 a2 a3 a4  ->  every GPU and CPU copy of a2, a3, a4 uncached
            next turns:   s0 hits at most p0 p1 and recomputes the rest; s1 keeps all its hits
        """
        # TODO: delete once the first non-finite logprob's cause is fixed (TODO in
        # `VLLMEngine._abort_running_requests_with_non_finite_logprobs`; not found yet: a write into
        # a cached block, GDN state bookkeeping with MTP drafts, or a kernel). Without this drop,
        # each later turn of the session reads the same bad KV and aborts at its first token.
        other_sessions_hashes: set[BlockHash] = set()
        for other_request, other_session_id, _, _ in self._live.values():
            if other_session_id != session_id:
                other_sessions_hashes.update(other_request.block_hashes)
        for other_session_id, blocks in self._held.items():
            if other_session_id != session_id:
                other_sessions_hashes.update(
                    get_block_hash(block.block_hash)
                    for block in blocks
                    if block.block_hash is not None
                )
        for other_session_id, (_, hashes) in self._cpu_sessions.items():
            if other_session_id != session_id:
                other_sessions_hashes |= hashes

        self._dropped_hashes.update(
            block_hash
            for block_hash in request.block_hashes
            if block_hash not in other_sessions_hashes
        )
        # Uncache first, so `_release` returns the held blocks to the front of the free queue.
        self._drop_cached_copies()
        self._release(session_id)
        # The next turn recomputes this prefix on purpose, so lost-prefix accounting skips it.
        self._last_turns.pop(session_id, None)

    def _drop_cached_copies(self) -> None:
        """Uncache every GPU and CPU copy of `_dropped_hashes`.

        With an offload connector, a GPU block still being saved to the CPU keeps its hash until the
        save completes; the hashes stay in `_dropped_hashes`, and `after_step` calls this again.
        The retry finds copies by hash, so it also uncaches a copy the session's next turn computed
        in between: one more recompute, never a read of the dropped KV.
        """
        # TODO: blocks vLLM freed while still hashed keep their place near the keep end of the GPU
        # and CPU free queues. Moving them to the evict end, as `release_session_cache` does, would
        # need a second private free-queue access, for at most one turn of blocks per abort.
        if self._offload_connector is None:
            self._block_pool.evict_blocks(
                cached_block_ids(self._block_pool, self._dropped_hashes)
            )
            self._dropped_hashes.clear()
        elif self._offload_connector.drop_session_cache(self._dropped_hashes):
            self._dropped_hashes.clear()

    def _count_lost_prefix(self, request: Request, session_id: str) -> None:
        """Count prefix tokens this turn recomputes although the session's last turn computed them.

        Cached tokens come from the connector's admission record (GPU + CPU hit). Every full block
        of the last turn counts as reusable: `VLLMEngine` keeps a GDN state at every boundary
        (`prefix_cache_retention_interval=None`).

        Example, 1,664-token blocks: the last turn ended at 63,000 tokens (37 full blocks); this
        prompt has 63,464 tokens and continues it; 54,912 tokens (33 blocks) were cached:

            reusable = min(37, (63,464 - 1) // 1,664 = 38) x 1,664 = 61,568
            lost     = 61,568 - 54,912 = 6,656 (4 blocks)

        A prompt that doesn't continue the last turn (a compacted history) counts nothing.
        """
        # TODO: without a KV connector, take the cached tokens from the finished request's
        # num_cached_tokens, which VLLMEngine reads at finish, so lost prefix needs no offload.
        num_cached_tokens = self._offload_connector.pop_cached_tokens(
            request.request_id
        )
        last_turn = self._last_turns.get(session_id)
        if num_cached_tokens is None or last_turn is None:
            return

        num_last_hashes = last_turn.num_block_hashes
        continues_last_turn = (
            num_last_hashes > 0
            and len(request.block_hashes) >= num_last_hashes
            and request.block_hashes[num_last_hashes - 1] == last_turn.last_block_hash
        )
        if not continues_last_turn:
            return

        hash_block_size = self._scheduler.hash_block_size
        block_size = self._scheduler.block_size
        num_reusable_hashes = min(
            num_last_hashes, (request.num_prompt_tokens - 1) // hash_block_size
        )
        reusable = num_reusable_hashes * hash_block_size // block_size * block_size
        self.num_reusable_prefix_tokens += reusable
        self.num_lost_prefix_tokens += max(0, reusable - num_cached_tokens)


class _LastTurn(NamedTuple):
    """What lost-prefix accounting needs from a session's last request that left vLLM."""

    group_id: int
    num_block_hashes: int
    last_block_hash: BlockHash | None


def cached_block_ids(pool: BlockPool, block_hashes: set[BlockHash]) -> set[int]:
    """Return the ids of the blocks `pool` caches under any of `block_hashes`, in any KV cache group.

    Example: the GPU caches h2 in block 7 (attention) and block 31 (GDN state), and h3 twice
    (blocks 9 and 12):

        cached_block_ids(gpu_block_pool, {h2, h3})  # -> {7, 9, 12, 31}
    """
    return {
        block.block_id
        for block in pool.blocks
        if any(
            get_block_hash(block_hash) in block_hashes
            # A block may also be cached under extra hashes.
            for block_hash in (
                block.block_hash,
                *pool.cached_block_hashes_by_block.get(block.block_id, ()),
            )
            if block_hash is not None
        )
    }
