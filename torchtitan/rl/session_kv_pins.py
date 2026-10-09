# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hold each multi-turn session's reusable KV between its turns (TBR-style session blocks)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class SessionKVPins:
    """Keeps a session's reusable prefix out of vLLM's free pool until its next turn or its end.

    vLLM frees a finished turn's blocks to an LRU free queue, where any new request can evict them;
    the session's next turn then recomputes its history. In vLLM's hybrid "align" mode the prompt-end
    Gated-DeltaNet state even leaves the request when decode starts. So right after a request's
    prefill, this holds the blocks vLLM's own prefix lookup says a continuation would hit (attention
    blocks and the matching GDN state blocks), and lets go of the session's previous hold. Held blocks
    count as used in vLLM's ``kv_cache_usage``.

    When free blocks fall below ``free_floor_blocks``, whole sessions are released, least recently
    held first (TBR's ``evict_until_free``). Every method must run on the engine thread of every rank,
    in the same order, so the schedulers of all TP ranks stay identical.

    Example::

        pins = SessionKVPins(scheduler, free_floor_blocks=400)
        internal_id = engine.add_request(request_id="group=3/rollout=0/turn=2", ...)
        pins.track(internal_id, session_id="group=3/rollout=0", group_id=3)
        pins.ensure_free()  # before each engine.step()
        engine.step()
        pins.after_step()  # holds the prefix of requests whose prefill just finished
        pins.release(session_ids=["group=3/rollout=0"])  # the rollout ended

    Args:
        scheduler: The vLLM v1 scheduler of this rank's in-process engine.
        free_floor_blocks: Release whole sessions while fewer blocks than this are free.
    """

    def __init__(self, scheduler: Any, free_floor_blocks: int) -> None:
        self._scheduler = scheduler
        self._coordinator = scheduler.kv_cache_manager.coordinator
        self._block_pool = scheduler.kv_cache_manager.block_pool
        self._free_floor_blocks = free_floor_blocks
        # Requests admitted but not yet past prefill: internal request id -> (session, group).
        self._awaiting_prefill: dict[str, tuple[str, int]] = {}
        # Held blocks per session, least recently held first.
        self._held: OrderedDict[str, list[Any]] = OrderedDict()
        self._session_groups: dict[str, int] = {}
        self._group_sessions: dict[int, set[str]] = {}

    @property
    def num_sessions(self) -> int:
        return len(self._held)

    @property
    def num_blocks(self) -> int:
        return sum(len(blocks) for blocks in self._held.values())

    def track(self, internal_request_id: str, session_id: str, group_id: int) -> None:
        """Hold this request's prefix once its prefill finishes."""
        self._awaiting_prefill[internal_request_id] = (session_id, group_id)

    def after_step(self) -> None:
        """Hold the prefix of every tracked request whose prefill finished."""
        for internal_id, (session_id, group_id) in list(self._awaiting_prefill.items()):
            request = self._scheduler.requests.get(internal_id)
            if request is None:  # finished or aborted before its prefill was seen
                del self._awaiting_prefill[internal_id]
            elif request.num_computed_tokens >= request.num_prompt_tokens:
                del self._awaiting_prefill[internal_id]
                self._hold(request, session_id, group_id)

    def ensure_free(self) -> None:
        """Release whole sessions, least recently held first, until enough blocks are free."""
        while (
            self._held
            and self._block_pool.get_num_free_blocks() < self._free_floor_blocks
        ):
            session_id, blocks = self._held.popitem(last=False)
            self._forget(session_id)
            self._block_pool.free_blocks(reversed(blocks))

    def release(self, session_ids: list[str] = (), group_ids: list[int] = ()) -> None:
        """Release sessions that make no more calls, and every session of finished groups."""
        session_ids = list(session_ids)
        for group_id in group_ids:
            session_ids.extend(self._group_sessions.pop(group_id, ()))
        for session_id in session_ids:
            blocks = self._held.pop(session_id, None)
            self._forget(session_id)
            if blocks:
                self._block_pool.free_blocks(reversed(blocks))

    def _hold(self, request: Any, session_id: str, group_id: int) -> None:
        # The prompt's last token is recomputed for its logits, so a continuation hits at most
        # num_prompt_tokens - 1 tokens; the lookup returns the attention blocks and, per GDN
        # group, the state block that ends on that aligned boundary.
        hit_blocks, _, _ = self._coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_prompt_tokens - 1
        )
        blocks = [b for group in hit_blocks for b in group if not b.is_null]
        # Take the new hold before dropping the old one, so shared prefix blocks stay held.
        self._block_pool.touch(blocks)
        previous = self._held.pop(session_id, None)
        self._held[session_id] = blocks
        self._session_groups[session_id] = group_id
        self._group_sessions.setdefault(group_id, set()).add(session_id)
        if previous:
            self._block_pool.free_blocks(reversed(previous))

    def _forget(self, session_id: str) -> None:
        group_id = self._session_groups.pop(session_id, None)
        if group_id is not None:
            self._group_sessions.get(group_id, set()).discard(session_id)
