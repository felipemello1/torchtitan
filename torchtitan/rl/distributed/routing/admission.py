# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admission: when the inter-generator router starts a new rollout group, and on which generator."""

from __future__ import annotations

import bisect
import itertools
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.rl.distributed.routing.types import KVCacheBudget

logger = logging.getLogger(__name__)

# R averages over this window; 5 to 30 minutes changed simulated throughput by at most 3%.
GROWTH_WINDOW_S = 900.0
# Every session counts as a max-length context until this many training sessions have ended:
# the first sessions to end are the shortest, so an R learned from a few of them is too low.
MIN_ENDED_SESSIONS = 50
# A group with no call for this long charges nothing, so a hung env cannot hold its generator's room.
IDLE_TTL_S = 7200.0


class KVGrowthEstimateAdmission(Configurable):
    """Starts a new rollout group only while its generator's KV cache, times `limit`, holds every
    placed group at its expected final size, so multi-turn sessions keep their history cached.

    The router's calls, for group 7; each group runs on one generator::

        policy.admit(7, serving=[0, 1])  # -> 1, or None: keep waiting
        policy.set_session_tokens(7, session_id="group=7/rollout=0", num_tokens=5000)  # call starts
        policy.set_session_tokens(7, session_id="group=7/rollout=0", num_tokens=6200)  # it returns
        policy.release_session(7, session_id="group=7/rollout=0")  # the rollout ends
        policy.release_group(7)  # the group ends

    `admit` tries a serving generator charging nothing, else the one with the most room:

        it charges nothing                                   placed, even a group over the limit
        its charge + the group's <= limit x its KV blocks    placed
        its charge + the group's >  limit x its KV blocks    waits
        validation group (negative id)                       placed, never waits

    What each session of a placed group charges its generator, in KV blocks:

        live, grew g blocks since its first call:  its blocks + max(mean(G - g) over G > g, R - g)
        not started yet:                           the live sessions' mean first call + R
        until 50 training sessions have ended:     a max-length context, started or not
        its group made no call for IDLE_TTL_S:     nothing, until it calls again

        G = total growth of each training session that ended in the last 15 min
        R = one session's growth: blocks added / sessions ended in those 15 min (Little's law)
        R - g covers sessions that outgrow the ended ones while lengths still rise
        no session charges more than a max-length context, whatever R says while it ramps

    Example (limit 1.0, 150 blocks per generator, R = 10, ended sessions grew G = 4, 10 or 16)::

        generator 0, two live 20-block sessions:
            grew 0  (first call 20):   20 + mean(4, 10, 16) = 30
            grew 12 (first call 8):    20 + (16 - 12)       = 24
        new group of 2:   54 + 2 x ((20 + 8) / 2 + 10) = 102 <= 150     placed

    Args:
        budgets: Each generator's KV cache size, by generator index.
        group_size: Samples per training group; without `num_sessions`, `admit` books one
            session per sample, and one for a validation group.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        limit: float = 0.95
        """Placed groups' expected final KV blocks may reach `limit` x each generator's KV cache.
            0.95:       every session fits at its expected final size at once, e.g. after a resume,
                        in the 95% of KV a generator fills (`session_kv_free_floor` is 0.05)
            above it:   more sessions; histories get evicted if they reach their final size together
        KV usage settles near limit / 2 when sessions grow far past their first call."""

    def __init__(
        self,
        config: Config,
        *,
        budgets: list[KVCacheBudget],
        group_size: int,
    ):
        self.config = config
        # The generators share one config, so one budget's per-session cost fits all.
        self._budget = budgets[0]
        self._max_session_blocks = self._budget.session_blocks(
            self._budget.max_model_len
        )
        self._group_size = group_size
        self.limits = [int(config.limit * budget.num_blocks) for budget in budgets]
        self._groups: dict[int, _PlacedGroup] = {}

        # As of the last rebuild: the blocks each generator's placed groups charge, what a session
        # not started yet charges, R (None until 50 training sessions have ended), and the mean
        # first call of the live training sessions (kept while none is live).
        self.charged = [0.0] * len(budgets)
        self._new_session_blocks = float(self._max_session_blocks)
        self.growth_per_session: float | None = None
        self._mean_first_call = 0.0

        # (time, blocks) for each growth of a live training session, their sum, and (time, total
        # growth) of each training session that ended. Validation lengths differ from training's.
        self._growth: deque[tuple[float, int]] = deque()
        self._growth_sum = 0
        self._ended: deque[tuple[float, int]] = deque()
        self._num_ended = 0
        logger.info(
            "KV admission: groups may charge %s KV cache blocks per generator; %s",
            self.limits,
            self._budget,
        )

    def admit(
        self, group_id: int, *, serving: list[int], num_sessions: int | None = None
    ) -> int | None:
        """Place a new group on the serving generator with the most room; None keeps it waiting.

        Args:
            group_id: The group to place; negative for a validation group, which always is.
            serving: Indices of the generators that can take a group now (non-empty).
            num_sessions: Routing sessions the group will open, e.g. two per game in two-player
                self-play; None: one per sample.
        """
        now = time.monotonic()
        validation = group_id < 0
        # TODO: prime-rl also caps new groups at a tenth of the pool per 5 s, spreading a cold
        # start's first prefills over ~50 s. Add it if a run shows that burst costs anything: this
        # gate already charges each group its final size when it is placed.
        self._rebuild(now)
        generator = max(
            serving,
            key=lambda index: (
                self.charged[index] == 0,
                self.limits[index] - self.charged[index],
            ),
        )
        expected_sessions = num_sessions
        if expected_sessions is None:
            expected_sessions = 1 if validation else self._group_size
        new_charge = expected_sessions * self._new_session_blocks
        if (
            not validation
            and self.charged[generator] > 0
            and self.charged[generator] + new_charge > self.limits[generator]
        ):
            return None
        self._groups[group_id] = _PlacedGroup(
            generator=generator, expected_sessions=expected_sessions, last_active_s=now
        )
        return generator

    def set_session_tokens(
        self, group_id: int, *, session_id: str | None, num_tokens: int
    ) -> None:
        """Record a session's context length: its prompt when a call starts, prompt plus
        completion when it returns."""
        group = self._groups[group_id]
        now = time.monotonic()
        blocks = self._budget.session_blocks(num_tokens)
        session = group.sessions.get(session_id)
        if session is None:
            if session_id in group.seen_sessions:
                # A call that returned after its rollout released the session.
                return
            group.sessions[session_id] = _Session(
                first_call_blocks=blocks, blocks=blocks
            )
            group.seen_sessions.add(session_id)
        else:
            grown = blocks - session.blocks
            if grown > 0 and group_id >= 0:
                self._growth.append((now, grown))
                self._growth_sum += grown
            session.blocks = blocks
        group.last_active_s = now

    def release_session(self, group_id: int, *, session_id: str | None) -> None:
        """Stop charging a session whose rollout ended, and learn its total growth."""
        group = self._groups[group_id]
        # A rollout that failed before its first call ends a session that never started.
        group.seen_sessions.add(session_id)
        session = group.sessions.pop(session_id, None)
        if session is not None and group_id >= 0:
            self._ended.append(
                (time.monotonic(), session.blocks - session.first_call_blocks)
            )
            self._num_ended += 1

    def release_group(self, group_id: int) -> None:
        """Forget a finished group; its training sessions still live count as ended."""
        group = self._groups.pop(group_id)
        if group_id >= 0:
            # Sessions whose rollout did not release them, e.g. one that raised, end here.
            now = time.monotonic()
            for session in group.sessions.values():
                self._ended.append((now, session.blocks - session.first_call_blocks))
                self._num_ended += 1

    def summary(self) -> str:
        """Return this policy's state for the router's per-step log line."""
        self._rebuild(time.monotonic())
        fractions = [
            round(charged / limit, 2)
            for charged, limit in zip(self.charged, self.limits)
        ]
        growth = (
            "cold start"
            if self.growth_per_session is None
            else f"growth {self.growth_per_session:.0f} blocks per session"
        )
        return (
            f"{len(self._groups)} groups placed, {growth}, "
            f"charged / limit per generator: {fractions}"
        )

    def _rebuild(self, now: float) -> None:
        """Recompute each generator's charge, and what a session not started yet charges."""
        growth = self._growth_per_session(now)
        remaining = _remaining_growth([total for _, total in self._ended])
        live_training_sessions = [
            session
            for group_id, group in self._groups.items()
            if group_id >= 0
            for session in group.sessions.values()
        ]
        if live_training_sessions:
            self._mean_first_call = sum(
                session.first_call_blocks for session in live_training_sessions
            ) / len(live_training_sessions)
        self._new_session_blocks = float(self._max_session_blocks)
        if growth is not None:
            self._new_session_blocks = min(
                self._mean_first_call + growth, self._new_session_blocks
            )
        self.growth_per_session = growth

        charged = [0.0] * len(self.charged)
        for group in self._groups.values():
            if now - group.last_active_s > IDLE_TTL_S:
                continue
            for session in group.sessions.values():
                charge = self._max_session_blocks
                if growth is not None:
                    grown = session.blocks - session.first_call_blocks
                    expected = max(remaining(grown), growth - grown)
                    charge = min(session.blocks + expected, charge)
                charged[group.generator] += charge
            not_started = max(0, group.expected_sessions - len(group.seen_sessions))
            charged[group.generator] += not_started * self._new_session_blocks
        self.charged = charged

    def _growth_per_session(self, now: float) -> float | None:
        """Return R, the blocks a session adds over its whole life; None until 50 training
        sessions have ended.

        Example: in the last 15 min sessions added 9,000 blocks and 300 sessions ended -> 30.
        """
        cutoff = now - GROWTH_WINDOW_S
        while self._growth and self._growth[0][0] < cutoff:
            self._growth_sum -= self._growth.popleft()[1]
        while self._ended and self._ended[0][0] < cutoff:
            self._ended.popleft()
        if self._num_ended < MIN_ENDED_SESSIONS:
            return None
        # No session ended in the last 15 min: keep the last estimate.
        if not self._ended:
            return self.growth_per_session
        # Little's law: blocks added per second / sessions ended per second = one session's growth.
        return self._growth_sum / len(self._ended)


def _remaining_growth(ended_growth: list[int]) -> Callable[[int], float]:
    """Return a function from a live session's growth so far to its expected remaining growth:
    the mean of G - g over the ended sessions' growths G above g, or 0 if none is above.

    Example:

        remaining = _remaining_growth([4, 10, 16])
        remaining(0)   # -> 10.0, mean(4, 10, 16)
        remaining(12)  # -> 4.0, only 16 is above: 16 - 12
        remaining(16)  # -> 0.0, none is above
    """
    sorted_growth = sorted(ended_growth)
    # suffix[i] = sum(sorted_growth[i:])
    suffix = list(itertools.accumulate(reversed(sorted_growth)))[::-1] + [0]

    def remaining(grown: int) -> float:
        i = bisect.bisect_right(sorted_growth, grown)
        above = len(sorted_growth) - i
        return suffix[i] / above - grown if above else 0.0

    return remaining


@dataclass(kw_only=True, slots=True)
class _PlacedGroup:
    """A placed rollout group's sessions."""

    generator: int

    expected_sessions: int
    """Sessions the group is expected to open."""

    last_active_s: float
    """`time.monotonic()` when it was placed, or when any of its sessions last called."""

    sessions: dict[str | None, _Session] = field(default_factory=dict)
    """Live sessions, by routing session id."""

    seen_sessions: set[str | None] = field(default_factory=set)
    """Sessions that have made a call or ended; the rest of `expected_sessions` have not started."""


@dataclass(kw_only=True, slots=True)
class _Session:
    """A live session's KV blocks."""

    first_call_blocks: int
    """Blocks at its first call."""

    blocks: int
    """Blocks for its context so far."""
