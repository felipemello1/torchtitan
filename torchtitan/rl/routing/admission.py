# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admission: when the inter-generator router starts a new rollout group, and on which generator; and
which admitted sessions pause when their contexts outgrow the KV cache."""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.rl.generator.engine import KVCacheBudget

# A group starts while its generator's ledger plus the group's charge stays at or under this fraction of
# the generator's usable blocks. The 5% left over absorbs a turn's growth before the holder releases
# idle sessions.
ADMIT_FRACTION = 0.95
# Above PAUSE_FRACTION of usable blocks the session KV holder starts releasing idle sessions' blocks. Then
# sessions pause until the ledger is at DRAIN_FRACTION; a paused session resumes under ADMIT_FRACTION.
PAUSE_FRACTION = 1.0
DRAIN_FRACTION = 0.90
# A session paused this long swaps in for idle sessions, so a session that never fits still resumes.
PAUSE_AGE_BOUND_S = 1800.0
# While work waits, a session idle this long is reclaimed: it leaves the ledger and holds nothing back. E.g. a
# hung env, or a session nobody released (a rollout that raised, a retried attempt's first session).
IDLE_RECLAIM_S = 600.0
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

    Contexts keep growing after admission. `rebalance` pauses sessions so each ledger stays under the point
    where the holder releases idle sessions (`PAUSE_FRACTION`); a paused session's next call waits in
    `start_call` until it resumes:

        ledger > PAUSE_FRACTION x usable    pause training sessions, then validation ones (`_pause_victims`),
                                            until the ledger is at DRAIN_FRACTION; an idle one now, a running
                                            one when its call returns
        ledger + its charge <= room         resume the longest-paused session (no overtaking)
        paused PAUSE_AGE_BOUND_S            swap it in for idle sessions, in pause order, staying in room
        no unpaused session left            resume the longest-paused session at once
        idle IDLE_RECLAIM_S, work waits     reclaim it: it leaves the ledger and holds back no group; once it
                                            calls again, it waits like a paused session
        a generator with paused sessions    takes no new group (reclaimed sessions don't count)

    Both run each `rebalance`: after a pause, the longest-paused sessions take back the room between
    DRAIN_FRACTION and ADMIT_FRACTION; a session paused and resumed in the same `rebalance` is not counted.

    Example (one generator: 3,030 usable blocks, room 2,878; a 4,000-token session charges 22 blocks
    while it runs):

        policy.add_group(7, num_sessions=16, validation=False)
        policy.place()  # -> {7: 0}: 352 <= 2,878
        policy.start_call("g7/r0/0", session_id="g7/r0", group_id=7, num_prompt_tokens=4000)
        policy.end_call("g7/r0/0", num_tokens=5200)  # the turn returned: prompt + completion
        policy.rebalance(now=time.monotonic())  # -> [] (no paused call resumed)
        policy.release_session("g7/r0", group_id=7)  # the rollout ended
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
        # The same sessions by generator, so a generator's ledger reads only its own.
        self._generator_sessions: list[dict[str, _Session]] = [{} for _ in budgets]
        # The session of each call in flight, so `end_call` needs only the request id.
        self._request_sessions: dict[str, str] = {}
        self._first_prompts: deque[int] = deque(maxlen=FIRST_PROMPT_WINDOW)
        # The `now` of the last `rebalance`: pause times are counted from it.
        self._clock = 0.0
        self.num_pauses = 0
        self.num_resumes = 0
        self.num_swaps = 0
        self.num_reclaims = 0

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
        # Paused sessions are already admitted, so they get the room first. A reclaimed one waits for nothing.
        has_paused = [False] * len(self._budgets)
        for session in self._sessions.values():
            if session.paused_at is not None and not session.reclaimed:
                has_paused[session.generator] = True
        validation_sessions = [0] * len(self._budgets)
        for group in self._groups.values():
            if group.validation and group.generator is not None:
                validation_sessions[group.generator] += group.num_sessions

        placements: dict[int, int] = {}
        while self._waiting:
            group_id = self._waiting[0]
            group = self._groups[group_id]
            generator = self._choose_generator(
                group, ledgers, validation_sessions, has_paused
            )
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
    ) -> bool:
        """Charge a placed group's session for a call: the prompt sets its size, so a compacted (shorter)
        prompt lowers it. Return False if the session is paused: the call waits until `rebalance`
        returns its id, or `cancel_call`. A session makes one call at a time."""
        group = self._groups[group_id]
        session = self._sessions.get(session_id)
        if session is not None and session.parked_request_id is not None:
            raise ValueError(
                f"call {request_id!r} of session {session_id!r} arrived while its call "
                f"{session.parked_request_id!r} waits for the session to resume"
            )
        if session is None:
            session = _Session(group_id=group_id, generator=group.generator)
            self._sessions[session_id] = session
            self._generator_sessions[group.generator][session_id] = session
            group.session_ids.add(session_id)
            group.num_started += 1
            if group.first_prompt_tokens is None:
                group.first_prompt_tokens = num_prompt_tokens
            if not group.validation:
                self._first_prompts.append(num_prompt_tokens)

        session.num_tokens = num_prompt_tokens
        self._request_sessions[request_id] = session_id
        # Set before parking too, so an older call's late release can't undo this size.
        session.latest_request_id = request_id
        if session.paused_at is not None:
            if session.reclaimed:
                # It calls again: from now on it waits for room like any paused session.
                session.reclaimed = False
                session.paused_at = self._clock
            session.parked_request_id = request_id
            return False
        session.request_ids.add(request_id)
        session.idle_since = None
        return True

    def cancel_call(self, request_id: str) -> None:
        """Forget a call whose caller is gone: still waiting for its paused session, or resumed but never
        sent. Unknown ids are ignored."""
        session_id = self._request_sessions.pop(request_id, None)
        if session_id is None:
            return
        session = self._sessions[session_id]
        if session.parked_request_id == request_id:
            session.parked_request_id = None
        session.request_ids.discard(request_id)
        if not session.request_ids and session.idle_since is None:
            session.idle_since = self._clock

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
        if session.request_ids:
            return

        session.idle_since = self._clock
        # Picked while its call ran (`_pause`): it leaves the ledger now.
        if session.pause_at_turn_end:
            session.pause_at_turn_end = False
            self._set_paused(session)

    def release_session(self, session_id: str, *, group_id: int) -> int | None:
        """Stop charging a session whose rollout ended; return the generator it was on, if it called.

        the session called                  its charge goes
        it never called                     one fewer not-started session is charged to its group
        released before, or unknown group   ignored
        """
        session = self._sessions.get(session_id)
        if session is not None:
            group = self._groups[session.group_id]
            group.released_session_ids.add(session_id)
            group.session_ids.discard(session_id)
            self._drop_session(session_id)
            return session.generator

        group = self._groups.get(group_id)
        if group is not None and session_id not in group.released_session_ids:
            # The rollout ended before its first call.
            group.released_session_ids.add(session_id)
            group.num_started += 1
        return None

    # ======== Ledger ========

    def ledger(self, generator: int) -> int:
        """Return the KV blocks the live sessions placed on `generator` charge it now."""
        budget = self._budgets[generator]
        blocks = 0
        for session in self._generator_sessions[generator].values():
            if session.paused_at is not None:
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

    def paused_fractions(self) -> list[float]:
        """Return the idle charge of each generator's paused and reclaimed sessions, as a fraction of its
        usable blocks: what the holder may still keep for them."""
        paused_blocks = [0] * len(self._budgets)
        for session in self._sessions.values():
            if session.paused_at is not None:
                budget = self._budgets[session.generator]
                paused_blocks[session.generator] += budget.idle_session_blocks(
                    session.num_tokens
                )
        return [
            blocks / budget.usable_blocks
            for blocks, budget in zip(paused_blocks, self._budgets, strict=True)
        ]

    def longest_pause_s(self) -> float:
        """Return how long the longest-paused session has waited for room, as of the last `rebalance`."""
        return max(
            (
                self._clock - session.paused_at
                for session in self._sessions.values()
                if session.paused_at is not None and not session.reclaimed
            ),
            default=0.0,
        )

    def generator_lines(self) -> list[str]:
        """One line per generator, for a per-step log: ledger and paused charge over usable blocks, and its
        paused and reclaimed sessions.

            generator 3: ledger 0.95, paused 0.07, 6 paused, 1 reclaimed
        """
        paused_fractions = self.paused_fractions()
        lines = []
        for generator, fraction in enumerate(self.ledger_fractions()):
            sessions = self._generator_sessions[generator].values()
            num_reclaimed = sum(session.reclaimed for session in sessions)
            num_paused = (
                sum(session.paused_at is not None for session in sessions)
                - num_reclaimed
            )
            lines.append(
                f"generator {generator}: ledger {fraction:.2f}, paused {paused_fractions[generator]:.2f}, "
                f"{num_paused} paused, {num_reclaimed} reclaimed"
            )
        return lines

    @property
    def num_paused_sessions(self) -> int:
        """Sessions paused to wait for room (reclaimed ones not included)."""
        return sum(
            session.paused_at is not None and not session.reclaimed
            for session in self._sessions.values()
        )

    @property
    def num_reclaimed_sessions(self) -> int:
        return sum(session.reclaimed for session in self._sessions.values())

    @property
    def num_waiting_groups(self) -> int:
        return len(self._waiting)

    # ======== Pause and resume ========

    def rebalance(self, now: float) -> list[str]:
        """Pause and resume sessions on every generator (see the class docstring); return the waiting calls
        of resumed sessions, which may run now.

        Args:
            now: Seconds on a monotonic clock.
        """
        self._clock = now
        for session in self._sessions.values():
            if not session.request_ids and session.idle_since is None:
                session.idle_since = now

        resumed_calls: list[str] = []
        for generator, generator_sessions in enumerate(self._generator_sessions):
            sessions = list(generator_sessions.values())
            ledger = self._pause(generator, sessions)
            resumed_calls += self._resume(generator, sessions, ledger)
        return resumed_calls

    def _pause(self, generator: int, sessions: list[_Session]) -> int:
        """Pause sessions while the ledger is over PAUSE_FRACTION, and hung ones while work waits; return
        the ledger after."""
        budget = self._budgets[generator]
        ledger = self.ledger(generator)
        # Running sessions picked earlier leave the ledger when their call returns.
        leaving = sum(
            budget.running_session_blocks(session.num_tokens)
            for session in sessions
            if session.pause_at_turn_end
        )
        if ledger - leaving > PAUSE_FRACTION * budget.usable_blocks:
            target = ledger - leaving
            for session in self._pause_victims(sessions):
                if target <= DRAIN_FRACTION * budget.usable_blocks:
                    break
                if session.request_ids:
                    target -= budget.running_session_blocks(session.num_tokens)
                    session.pause_at_turn_end = True
                else:
                    blocks = budget.idle_session_blocks(session.num_tokens)
                    target -= blocks
                    ledger -= blocks
                    self._set_paused(session)

        work_waits = bool(self._waiting) or any(
            session.paused_at is not None and not session.reclaimed
            for session in sessions
        )
        if work_waits:
            for session in self._pause_victims(sessions):
                if (
                    not session.request_ids
                    and self._clock - session.idle_since > IDLE_RECLAIM_S
                ):
                    ledger -= budget.idle_session_blocks(session.num_tokens)
                    session.paused_at = self._clock
                    session.reclaimed = True
                    self.num_reclaims += 1
        return ledger

    def _resume(
        self, generator: int, sessions: list[_Session], ledger: int
    ) -> list[str]:
        """Resume paused sessions, longest paused first, while they fit; return their waiting calls."""
        budget = self._budgets[generator]
        room = ADMIT_FRACTION * budget.usable_blocks
        paused = sorted(
            (session for session in sessions if session.paused_at is not None),
            key=lambda session: session.paused_at,
        )
        any_unpaused = any(session.paused_at is None for session in sessions)
        resumed: set[int] = set()
        resumed_calls = []
        for session in paused:
            # A session with a waiting call resumes running. A reclaimed one stays out until it calls.
            if session.parked_request_id is not None:
                blocks = budget.running_session_blocks(session.num_tokens)
            elif session.reclaimed:
                continue
            else:
                blocks = budget.idle_session_blocks(session.num_tokens)

            if ledger + blocks > room and any_unpaused:
                if self._clock - session.paused_at <= PAUSE_AGE_BOUND_S:
                    break
                swap = self._swap_victims(
                    sessions, budget, needed=ledger + blocks - room, resumed=resumed
                )
                if swap is None:
                    break
                for victim in swap:
                    ledger -= budget.idle_session_blocks(victim.num_tokens)
                    self._set_paused(victim)
                self.num_swaps += 1

            if session.paused_at == self._clock:
                # Paused earlier in this `rebalance`: it never waited.
                self.num_pauses -= 1
            else:
                self.num_resumes += 1
            session.paused_at = None
            ledger += blocks
            any_unpaused = True
            resumed.add(id(session))
            if session.parked_request_id is not None:
                session.request_ids.add(session.parked_request_id)
                session.latest_request_id = session.parked_request_id
                session.idle_since = None
                resumed_calls.append(session.parked_request_id)
                session.parked_request_id = None
        return resumed_calls

    def _pause_victims(self, sessions: list[_Session]) -> list[_Session]:
        """Unpaused sessions in pause order: training before validation, idle before running, then the
        newest group first, then the smallest. A group trains once all its rollouts end, so a paused
        session holds back its whole group: pausing the newest group delays the least finished work.

            groups 3 (older), 5 (newer); idle g3/s0 200k tokens, idle g5/s0 250k, idle g5/s1 30k,
            running g5/s2 10k  ->  g5/s1, g5/s0, g3/s0, g5/s2
        """
        candidates = [
            session
            for session in sessions
            if session.paused_at is None and not session.pause_at_turn_end
        ]
        return sorted(
            candidates,
            key=lambda session: (
                self._groups[session.group_id].validation,
                bool(session.request_ids),
                -self._groups[session.group_id].arrival,
                session.num_tokens,
            ),
        )

    def _swap_victims(
        self,
        sessions: list[_Session],
        budget: KVCacheBudget,
        *,
        needed: float,
        resumed: set[int],
    ) -> list[_Session] | None:
        """Idle training sessions, in `_pause_victims` order, that free `needed` blocks; None if they can't.
        Sessions resumed in this `rebalance` (`resumed`, by `id`) are not swapped back out."""
        victims, freed = [], 0
        for session in self._pause_victims(sessions):
            if freed >= needed:
                break
            if self._groups[session.group_id].validation or session.request_ids:
                break
            if id(session) in resumed:
                continue
            victims.append(session)
            freed += budget.idle_session_blocks(session.num_tokens)
        return victims if freed >= needed else None

    def _set_paused(self, session: _Session) -> None:
        session.paused_at = self._clock
        self.num_pauses += 1

    # ======== Helpers ========

    def _placement_order(self, group_id: int) -> tuple[bool, int]:
        group = self._groups[group_id]
        return (not group.validation, group.arrival if group.validation else group_id)

    def _choose_generator(
        self,
        group: _Group,
        ledgers: list[int],
        validation_sessions: list[int],
        has_paused: list[bool],
    ) -> int | None:
        best, best_key = None, None
        for generator, budget in enumerate(self._budgets):
            if has_paused[generator]:
                continue
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
        del self._generator_sessions[session.generator][session_id]
        for request_id in session.request_ids:
            del self._request_sessions[request_id]
        if session.parked_request_id is not None:
            del self._request_sessions[session.parked_request_id]


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
    """Sessions that made a call (released ones included), plus rollouts released before their first call."""

    released_session_ids: set[str] = field(default_factory=set)
    """Sessions already released, so a repeated release counts once."""

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
    """Its last call: only that call's release sets its size."""

    paused_at: float | None = None
    """When it paused, on the `rebalance` clock; None while unpaused."""

    reclaimed: bool = False
    """Paused for idling IDLE_RECLAIM_S while work waited: out of the ledger, and it holds back no group."""

    pause_at_turn_end: bool = False
    """Picked as a pause victim while its call ran: it pauses when the call returns."""

    parked_request_id: str | None = None
    """A call that waits for this paused session to resume."""

    idle_since: float | None = None
    """When its last call returned; None while a call runs."""
