# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admission: when the inter-generator router starts a new rollout group, and on which generator."""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.rl.generator.engine import KVCacheBudget

# A group starts while its generator's ledger plus the group's charge stays at or under this fraction of
# the generator's usable blocks. The 5% left over absorbs a turn's growth before the holder releases
# idle sessions; on a replay of a lossless load (Qwen3.8-27B, one GB300) the ledger averaged 93%.
ADMIT_FRACTION = 0.95
# A session not started yet is charged the mean first prompt of the last this-many training sessions,
# so the estimate follows the prompt lengths of the tasks being drawn.
FIRST_PROMPT_WINDOW = 64


class KVRoomAdmission(Configurable):
    """Starts a rollout group on a generator only while that generator's KV cache has room for it,
    counting what the engine holds for each live session (`KVCacheBudget`):

        idle session (between turns)       its context's attention blocks + a state block per Mamba group
        call outstanding, or not started   + the blocks vLLM gives a running request + the holder's floor
        room                               ADMIT_FRACTION x usable blocks (the pool minus the holder's floor)

    `place` takes waiting groups in order, validation first (by arrival), then training (lowest id):

        a generator with no live session                     placed, even over the room
        ledger + the group's charge <= room                  placed: training on the most room,
                                                             validation on the fewest validation sessions
        no generator has room for the group at the head      it and every later group keep waiting

    Example (one generator: 3,030 usable blocks, room 2,878; a 4,000-token session charges 22 blocks
    while it runs):

        policy.add_group(7, num_sessions=16, validation=False)
        policy.place()  # -> {7: 0}: 352 <= 2,878
        policy.start_call("g7/r0/0", session_id="g7/r0", group_id=7, num_prompt_tokens=4000)
        policy.end_call("g7/r0/0", num_tokens=5200)  # the turn returned: prompt + completion
        policy.release_session("g7/r0")  # the rollout ended
        policy.release_group(7)  # the group ended

    Args:
        budgets: Each generator's KV cache size, by generator index.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config, *, budgets: Sequence[KVCacheBudget]):
        del config
        # TODO: with DP > 1, `IntraGeneratorRouter` spreads a placed group's sessions over the replicas
        # without looking at their KV, so one replica can fill while a pooled ledger has room. Keep one
        # ledger per replica, and let the intra router follow it, before turning admission on there.
        for index, budget in enumerate(budgets):
            if budget.data_parallel_size != 1:
                raise ValueError(
                    f"admission keeps one ledger per generator and needs data_parallel_size=1; "
                    f"generator {index} has {budget.data_parallel_size}"
                )
        self._budgets = list(budgets)
        self._groups: dict[int, _Group] = {}
        # Waiting groups, in placement order (`_placement_order`).
        self._waiting: list[int] = []
        self._arrivals = itertools.count()
        self._sessions: dict[str, _Session] = {}
        # The session of each call in flight, so `end_call` needs only the request id.
        self._request_sessions: dict[str, str] = {}
        self._first_prompts: deque[int] = deque(maxlen=FIRST_PROMPT_WINDOW)

    # ======== Groups ========

    def add_group(self, group_id: int, *, num_sessions: int, validation: bool) -> None:
        """Queue a group for `place`. A rollout opens one session; a retried attempt opens another."""
        if group_id in self._groups:
            raise ValueError(f"group {group_id} is already waiting or placed")
        self._groups[group_id] = _Group(
            num_sessions=num_sessions,
            validation=validation,
            arrival=next(self._arrivals),
        )
        self._waiting.append(group_id)
        self._waiting.sort(key=self._placement_order)

    def place(self) -> dict[int, int]:
        """Place waiting groups in order until one has to wait; return each placed group's generator.

        Example (groups 4 and 5 wait; generator 1 has room for one of them, generator 0 for none):

            policy.place()  # -> {4: 1}; group 5 keeps waiting
        """
        ledgers = [self.ledger(generator) for generator in range(len(self._budgets))]
        validation_sessions = [0] * len(self._budgets)
        for group in self._groups.values():
            if group.validation and group.generator is not None:
                validation_sessions[group.generator] += group.num_sessions

        placements: dict[int, int] = {}
        while self._waiting:
            group_id = self._waiting[0]
            group = self._groups[group_id]
            generator = self._choose_generator(group, ledgers, validation_sessions)
            if generator is None:
                break

            self._waiting.pop(0)
            group.generator = generator
            ledgers[generator] += self._not_started_blocks(group)
            if group.validation:
                validation_sessions[generator] += group.num_sessions
            placements[group_id] = generator
        return placements

    def generator_of(self, group_id: int) -> int | None:
        """Return the generator a group is placed on; None while it waits, or if it is unknown."""
        group = self._groups.get(group_id)
        return None if group is None else group.generator

    def release_group(self, group_id: int) -> None:
        """Forget a finished group, waiting or placed, and every session it opened."""
        group = self._groups.pop(group_id, None)
        if group is None:
            return
        if group.generator is None:
            self._waiting.remove(group_id)
        for session_id in group.session_ids:
            self._drop_session(session_id)

    # ======== Calls and sessions ========

    def start_call(
        self, request_id: str, *, session_id: str, group_id: int, num_prompt_tokens: int
    ) -> None:
        """Charge a placed group's session for a call: the prompt sets its size, so a compacted (shorter)
        prompt lowers it."""
        group = self._groups[group_id]
        session = self._sessions.get(session_id)
        if session is None:
            session = _Session(group_id=group_id, generator=group.generator)
            self._sessions[session_id] = session
            group.session_ids.add(session_id)
            group.num_started += 1
            if group.first_prompt_tokens is None:
                group.first_prompt_tokens = num_prompt_tokens
            if not group.validation:
                self._first_prompts.append(num_prompt_tokens)

        session.request_ids.add(request_id)
        session.latest_request_id = request_id
        session.num_tokens = num_prompt_tokens
        self._request_sessions[request_id] = session_id

    def end_call(self, request_id: str, *, num_tokens: int | None) -> None:
        """Stop charging a call as running. `num_tokens` (prompt + completion; None if the call failed)
        sets the session's size if this was its latest call.

        Unknown ids are ignored: a call's release can arrive after its session's (`release_session`).
        """
        session_id = self._request_sessions.pop(request_id, None)
        if session_id is None:
            return
        session = self._sessions[session_id]
        session.request_ids.discard(request_id)
        # An older call's late release must not undo a newer call's size.
        if request_id == session.latest_request_id and num_tokens is not None:
            session.num_tokens = num_tokens

    def release_session(self, session_id: str) -> int | None:
        """Stop charging a session whose rollout ended; return the generator it was on, if known."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        self._groups[session.group_id].session_ids.discard(session_id)
        self._drop_session(session_id)
        return session.generator

    # ======== Ledger ========

    def ledger(self, generator: int) -> int:
        """Return the KV blocks the live sessions placed on `generator` charge it now."""
        budget = self._budgets[generator]
        blocks = 0
        for session in self._sessions.values():
            if session.generator != generator:
                continue
            if session.request_ids:
                blocks += budget.running_session_blocks(session.num_tokens)
            else:
                blocks += budget.idle_session_blocks(session.num_tokens)
        for group in self._groups.values():
            if group.generator == generator:
                blocks += self._not_started_blocks(group)
        return blocks

    def ledger_fractions(self) -> list[float]:
        """Return each generator's ledger as a fraction of its usable blocks."""
        return [
            self.ledger(generator) / budget.usable_blocks
            for generator, budget in enumerate(self._budgets)
        ]

    @property
    def num_waiting_groups(self) -> int:
        return len(self._waiting)

    # ======== Helpers ========

    def _placement_order(self, group_id: int) -> tuple[bool, int]:
        group = self._groups[group_id]
        return (not group.validation, group.arrival if group.validation else group_id)

    def _choose_generator(
        self, group: _Group, ledgers: list[int], validation_sessions: list[int]
    ) -> int | None:
        best, best_key = None, None
        for generator, budget in enumerate(self._budgets):
            charge = group.num_sessions * budget.running_session_blocks(
                self._expected_first_prompt(group)
            )
            room = ADMIT_FRACTION * budget.usable_blocks - ledgers[generator] - charge
            if room < 0 and ledgers[generator] > 0:
                continue
            key = (
                (-validation_sessions[generator], room) if group.validation else (room,)
            )
            if best_key is None or key > best_key:
                best, best_key = generator, key
        return best

    def _not_started_blocks(self, group: _Group) -> int:
        """Charge a placed group's sessions that have not called yet, as running."""
        num_not_started = max(0, group.num_sessions - group.num_started)
        budget = self._budgets[group.generator]
        return num_not_started * budget.running_session_blocks(
            self._expected_first_prompt(group)
        )

    def _expected_first_prompt(self, group: _Group) -> int:
        """Tokens a session of `group` not started yet is charged for:

        a sibling has called            that sibling's prompt (GRPO siblings share their task's prompt)
        else, a training session has    the mean first prompt of the last FIRST_PROMPT_WINDOW
        else                            the max model length: one group per generator until then
        """
        if group.first_prompt_tokens is not None:
            return group.first_prompt_tokens
        if self._first_prompts:
            return sum(self._first_prompts) // len(self._first_prompts)
        return max(budget.max_model_len for budget in self._budgets)

    def _drop_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id)
        for request_id in session.request_ids:
            del self._request_sessions[request_id]


@dataclass(kw_only=True, slots=True)
class _Group:
    """A rollout group, waiting (`generator` is None) or placed."""

    num_sessions: int
    validation: bool
    arrival: int
    generator: int | None = None
    session_ids: set[str] = field(default_factory=set)
    """Live sessions."""

    num_started: int = 0
    """Sessions that made a call, released ones included."""

    first_prompt_tokens: int | None = None
    """The first prompt of the group's first session to call."""


@dataclass(kw_only=True, slots=True)
class _Session:
    """A live session's charge."""

    group_id: int
    generator: int
    num_tokens: int = 0
    """Its prompt while a call runs; prompt + completion after it returns."""

    request_ids: set[str] = field(default_factory=set)
    """Calls in flight."""

    latest_request_id: str | None = None
