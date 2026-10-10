# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admission policies: when the inter-generator router starts a new rollout group, and where.

- `KVUsageAdmission`: a cap on groups in flight, moved by vLLM's measured KV usage (prime-rl's
  controller).
- `KVEstimateAdmission`: the router's own estimate of each generator's KV blocks (ThunderAgent's
  rule).
- `KVGrowthEstimateAdmission`: that estimate plus the blocks each session is expected to add.

prime-rl line references are to prime-rl@0bcb868e9, `src/prime_rl/orchestrator/`.
"""

from __future__ import annotations

import bisect
import itertools
import logging
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from torchtitan.config import Configurable
from torchtitan.rl.distributed.routing.types import EngineLoad, KVCacheBudget

logger = logging.getLogger(__name__)


class AdmissionPolicy(Configurable, ABC):
    """Decides when the router starts a new rollout group, and on which generator.

    The router places each rollout group on one generator at its first generate call, and every
    later call of the group goes there. A new group waits in a FIFO queue until `admit` returns
    a generator for it; turns of placed groups never wait. The router's calls, for group 7::

        policy.observe({0: load0, 1: load1})  # at start, then every 5 s
        policy.admit(7, num_prompt_tokens=5000, serving=[0, 1], skip_queue=False)  # 1, or None
        policy.set_session_tokens(7, session_id="group=7/rollout=0", num_tokens=5000)  # call starts
        policy.set_session_tokens(7, session_id="group=7/rollout=0", num_tokens=6200)  # it returns
        policy.set_session_tokens(7, session_id="group=7/rollout=0", num_tokens=None)  # session ends
        policy.release(7, num_completion_tokens=1200)  # the group ends

    After each of these calls, the router calls `admit` for the oldest waiting group until it
    returns None. Add a mode by subclassing this and selecting its config, e.g.
    `InterGeneratorRouter.Config(admission=MyAdmission.Config())`.

    Args:
        budgets: Each generator's KV cache size, by generator index.
        group_size: Samples per rollout group, e.g. games per start position.
    """

    def __init__(
        self,
        config: Configurable.Config,
        *,
        budgets: list[KVCacheBudget],
        group_size: int,
    ):
        del config, budgets, group_size

    @abstractmethod
    def admit(
        self,
        group_id: int,
        *,
        num_prompt_tokens: int,
        serving: list[int],
        skip_queue: bool,
    ) -> int | None:
        """Start a group: return the generator to place it on, or None to keep it waiting.

        Args:
            group_id: The group to start.
            num_prompt_tokens: Prompt length of the group's first call.
            serving: Indices of the generators that can take a group now (non-empty).
            skip_queue: The group must start now whatever the load, e.g. a validation group.
        """

    @abstractmethod
    def release(self, group_id: int, *, num_completion_tokens: int) -> None:
        """Forget a finished group that `admit` placed.

        Args:
            group_id: The finished group.
            num_completion_tokens: Tokens the group's calls generated.
        """

    def observe(self, loads: dict[int, EngineLoad]) -> None:
        """Take one poll: the load of each generator that answered, by generator index."""
        del loads

    def set_session_tokens(
        self, group_id: int, *, session_id: str | None, num_tokens: int | None
    ) -> None:
        """Set a placed group's session context length; `None` ends the session."""
        del group_id, session_id, num_tokens

    def summary(self) -> str:
        """Return this policy's state for the router's per-step log line."""
        return ""


# prime-rl's constants (concurrency.py:42-89), and its admission window (dispatcher.py:179).
TURNOVER_GROWTH_MAX = 1.2
"""Cap growth per turnover of the groups in flight, at 0% KV usage."""
BINDING_FRACTION = 0.9
"""The cap grows only while groups in flight reach this fraction of it."""
KV_USAGE_SOFT_CAP = 0.8
"""Above this usage, trim the cap; in-flight groups finish on their own."""
KV_USAGE_HARD_CAP = 0.9
"""Above this usage, prime-rl also cancels the excess groups (not ported, see `_resize_down`)."""
KV_USAGE_TARGET = 0.7
"""A trim sets the cap to in flight x target / usage."""
KV_TRIM_COOLDOWN_POLLS = 6
"""Polls between trims, so each trim shows in usage before the next."""
QUEUE_RATIO = 0.5
"""Queue overload: waiting requests above this fraction of running ones, summed over generators."""
QUEUE_PERSISTENCE_POLLS = 6
"""Polls in a row of queue overload before a cut."""
QUEUE_CUT_FRACTION = 0.9
"""A queue-overload cut sets the cap to this fraction of the groups in flight."""
PREEMPTION_CUT_FRACTION = 0.8
"""A preemption cut sets the cap to this fraction of the groups in flight."""
ESCALATED_CUT_FRACTION = 0.5
"""Cut fraction for an overload within the grace window after a drain."""
ESCALATION_GRACE_POLLS = 6
"""Polls after a drain during which a repeat overload cuts at the escalated fraction."""
GROWTH_GATE_TTL_S = 15.0
"""The cap stops growing once the last poll is this old, so stalled polls cannot grow it blind."""
ADMISSION_WINDOW_S = 5.0
"""Groups in flight grow by at most one burst per window."""

Signal = Literal["clear", "soft", "hard"]
"""Engine pressure, in increasing severity."""
_SEVERITY: dict[Signal, int] = {"clear": 0, "soft": 1, "hard": 2}


class KVUsageAdmission(AdmissionPolicy):
    """Caps the rollout groups in flight and moves the cap with vLLM's measured KV usage:
    prime-rl's admission controller, with one permit per rollout group.

    - **Grow:** each finished group multiplies the cap by `m ** (1 / in_flight)`, so the cap
      grows by `m` per turnover of the groups in flight. `m` falls from 1.2 at 0% KV usage to
      1.0 at 80%. Growth needs a clear last poll under 15 s old (usage <= 80%, no waiting request,
      no preemption, no drain) and a binding cap (in flight >= 90% of it).
    - **Trim:** above 80% usage on any generator, the cap drops to `in_flight * 0.7 / usage`, at
      most once per 6 polls.
    - **Cut:** a preemption cuts the cap to 0.8 x in flight; vLLM's waiting requests above half
      its running ones for 6 polls, with the cap full, cut it to 0.9 x. No cut or trim follows
      until in flight drains to the cap with no overload; an overload within 6 polls after that
      cuts to 0.5 x.
    - **Smoothing:** groups in flight grow by at most `max(1, cap // 10)` per 5 s; each finished
      group gives one back.

    A new group goes to the generator that took the fewest groups since its last poll, the lowest
    usage first: usage shows a new group's prefill only at the next poll.

    Example (2 generators, 100 groups in flight, cap 100; `load(u, p)` is an `EngineLoad` with
    KV usage `u` and `p` preemptions so far)::

        policy.observe({0: load(0.85, 0), 1: load(0.60, 0)})
        # soft trim: cap 100 -> 82 (100 x 0.7 / 0.85); no group starts until in flight < 82
        policy.observe({0: load(0.70, 1), 1: load(0.60, 0)})
        # preemption cut: cap 82 -> 80 (0.8 x 100); no cut or trim until in flight <= 80
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        initial_inflight: int | None = None
        """Groups in flight to start with. None starts from a pessimistic bound: the max-length
        contexts every generator's KV cache holds, one per sample of a group."""

        min_inflight: int = 1
        """Fewest groups in flight the cap falls to. Set it to `max_inflight` for a fixed cap."""

        max_inflight: int | None = 1024
        """Most groups in flight (prime-rl's 1024 counts episodes); None for no ceiling."""

    def __init__(
        self,
        config: Config,
        *,
        budgets: list[KVCacheBudget],
        group_size: int,
    ):
        self.config = config
        initial_inflight = config.initial_inflight
        if initial_inflight is None:
            # One max-length context per sample, as prime-rl counts one per episode
            # (concurrency.py:255-265).
            max_contexts = sum(
                budget.num_blocks / budget.session_blocks(budget.max_model_len)
                for budget in budgets
            )
            initial_inflight = max_contexts / group_size
        # `int(cap)` groups may be in flight; growth accumulates its fractions here.
        self.cap = self._clamp(initial_inflight)
        logger.info("Admission starts at %d groups in flight", int(self.cap))

        # Groups in flight: admitted and not yet released. Validation groups take no permit.
        self.admitted_groups: set[int] = set()
        self.signal: Signal = "clear"
        self.growth_multiplier = 1.0
        # Growth gate from the last poll, read when a group finishes.
        self.can_grow = False
        self.can_grow_until = 0.0
        self.queue_overload_polls = 0
        self.trim_cooldown = 0
        self.escalation_grace = 0
        # After a cut, ignore further cuts until in flight drains below the new cap: the overload
        # seen during the drain is stale.
        self.draining = False
        self.escalated = False
        # The last trim or cut, for the per-step log line.
        self.last_change = "none"

        # The last load each generator answered with, and the groups placed on it since.
        self._loads: dict[int, EngineLoad] = {}
        self._placed_since_poll = [0] * len(budgets)
        self._window_start = time.monotonic()
        self._admissions_in_window = 0

    def admit(
        self,
        group_id: int,
        *,
        num_prompt_tokens: int,
        serving: list[int],
        skip_queue: bool,
    ) -> int | None:
        del num_prompt_tokens
        if not skip_queue:
            now = time.monotonic()
            if now - self._window_start >= ADMISSION_WINDOW_S:
                self._window_start, self._admissions_in_window = now, 0
            limit = int(self.cap)
            # A group permit replaces prime-rl's episode permit, so its minimum burst of one
            # group size (dispatcher.py:182) is 1.
            burst = max(1, limit // 10)
            if (
                len(self.admitted_groups) >= limit
                or self._admissions_in_window >= burst
            ):
                return None
            self.admitted_groups.add(group_id)
            self._admissions_in_window += 1

        def usage(g: int) -> float:
            # A generator that has not answered a poll yet is fresh: no groups, no usage.
            return self._loads[g].kv_usage if g in self._loads else 0.0

        generator = min(serving, key=lambda g: (self._placed_since_poll[g], usage(g)))
        self._placed_since_poll[generator] += 1
        return generator

    def release(self, group_id: int, *, num_completion_tokens: int) -> None:
        """Free the group's permit, and grow the cap while the last poll allows it."""
        if group_id not in self.admitted_groups:
            return
        in_flight = len(self.admitted_groups)
        self.admitted_groups.remove(group_id)
        self._admissions_in_window = max(0, self._admissions_in_window - 1)
        # A group that generated nothing used no KV: growing on a storm of failing groups would
        # flood the generators once they recover.
        if num_completion_tokens == 0:
            return
        if (
            self.can_grow
            and time.monotonic() < self.can_grow_until
            and in_flight >= BINDING_FRACTION * int(self.cap)
        ):
            self.cap = self._clamp(self.cap * self.growth_multiplier ** (1 / in_flight))

    def observe(self, loads: dict[int, EngineLoad]) -> None:
        """Classify the pressure, gate growth, and trim or cut the cap."""
        previous = self._loads
        self._loads = {**previous, **loads}
        for g in loads:
            self._placed_since_poll[g] = 0
        in_flight = len(self.admitted_groups)

        # ======== Pressure signal ========
        signal: Signal = "clear"
        preempted = False
        for g, load in loads.items():
            last = previous.get(g)
            # A generator's first answer has no baseline: its preemption count is not new.
            if last is not None and load.num_preemptions > last.num_preemptions:
                preempted = True
                signal = "hard"
            if load.num_waiting > 0 and last is not None and last.num_waiting > 0:
                signal = max(signal, "soft", key=_SEVERITY.__getitem__)
        max_usage = max(load.kv_usage for load in loads.values())
        num_running = sum(load.num_running for load in loads.values())
        num_queued = sum(load.num_waiting_for_capacity for load in loads.values())
        # Not in prime-rl: count an overload only while the cap is full, so a cut is never sized
        # from a pool that is still filling up.
        if (
            num_running > 0
            and num_queued > QUEUE_RATIO * num_running
            and in_flight >= int(self.cap)
        ):
            self.queue_overload_polls += 1
        else:
            self.queue_overload_polls = 0
        queue_overload = self.queue_overload_polls >= QUEUE_PERSISTENCE_POLLS
        if queue_overload:
            signal = "hard"
        elif max_usage > KV_USAGE_SOFT_CAP:
            signal = max(signal, "soft", key=_SEVERITY.__getitem__)
        self.signal = signal

        # ======== Drain, escalation and growth gate ========
        # End a drain only once the generators settled too: preemptions from the backlog of the
        # cut groups must not cut again (prime-rl saw a cascade 1024 -> 8, concurrency.py:233-238).
        if (
            self.draining
            and in_flight <= int(self.cap)
            and not preempted
            and not queue_overload
        ):
            self.draining = False
            self.escalation_grace = ESCALATION_GRACE_POLLS
        if not self.draining and self.escalated:
            self.escalation_grace -= 1
            if self.escalation_grace <= 0:
                self.escalated = False
        self.trim_cooldown = max(0, self.trim_cooldown - 1)
        self.can_grow = signal == "clear" and num_queued == 0 and not self.draining
        self.growth_multiplier = 1.0 + (TURNOVER_GROWTH_MAX - 1.0) * max(
            0.0, 1.0 - max_usage / KV_USAGE_SOFT_CAP
        )
        self.can_grow_until = time.monotonic() + GROWTH_GATE_TTL_S
        if self.draining:
            return

        # ======== Cut on overload, else trim on usage ========
        if preempted or queue_overload:
            if queue_overload:
                self.queue_overload_polls = 0
                fraction, reason = QUEUE_CUT_FRACTION, "queue overload"
            else:
                fraction, reason = PREEMPTION_CUT_FRACTION, "preemption"
            if self.escalated:
                fraction = ESCALATED_CUT_FRACTION
            self._resize_down(int(self._clamp(in_flight * fraction)), reason=reason)
            self.draining = True
            self.escalated = True
        elif (
            max_usage > KV_USAGE_SOFT_CAP and in_flight > 0 and self.trim_cooldown == 0
        ):
            trim = "hard" if max_usage > KV_USAGE_HARD_CAP else "soft"
            self._resize_down(
                int(self._clamp(in_flight * KV_USAGE_TARGET / max_usage)),
                reason=f"{trim} trim at KV usage {max_usage:.2f}",
            )
            self.trim_cooldown = KV_TRIM_COOLDOWN_POLLS

    def summary(self) -> str:
        return (
            f"cap {int(self.cap)}, {len(self.admitted_groups)} admitted, signal {self.signal}, "
            f"draining {self.draining}, last change: {self.last_change}"
        )

    def _resize_down(self, target: int, *, reason: str) -> None:
        """Lower the cap, never raise it; new groups wait until in flight drains below it."""
        # TODO: prime-rl also cancels the excess in flight at a hard trim and on every cut, whole
        # groups, youngest first by their oldest member's start (dispatcher.py:242-269). Deferred:
        # it throws away generated turns. Known gap without it: a cut freezes trims, cuts and growth
        # until ~10-20% of the groups finish, and for good while groups cannot finish (a KV
        # livelock) or vLLM's queue stays long.
        limit = int(self.cap)
        target = min(target, limit)
        if target < limit:
            self.last_change = f"{reason}, cap {limit} -> {target}"
            logger.info("Admission: %s", self.last_change)
        self.cap = float(target)

    def _clamp(self, cap: float) -> float:
        ceiling = self.config.max_inflight or math.inf
        return min(max(cap, float(self.config.min_inflight)), float(ceiling))


class KVEstimateAdmission(AdmissionPolicy):
    """Starts a group only while the router's own estimate of its generator's KV blocks fits:
    ThunderAgent's rule.

    Each group runs on the generator with the most room. A session counts its current context in
    vLLM blocks (`KVCacheBudget.session_blocks`): its prompt while a call runs, then its prompt
    plus completion, until it ends. A group also reserves its first call's blocks for each expected
    session that has not started yet. Sessions count at their current context, so groups admitted
    together, e.g. at startup, can grow past the limit; `KVGrowthEstimateAdmission` reserves for
    that growth.

    Example (3 sessions expected, each first call costs 4 blocks)::

        admit(7, ...) -> placed on gen1 with 3 x 4 = 12 blocks
        all 3 sessions start and grow to 6 blocks each -> blocks = 18
        session r0 ends -> blocks = 12
        release(7, ...) -> blocks[1] -= 12
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        limit: float = 0.9
        """Fraction of each generator's KV cache blocks that live sessions may hold, so that a
        session's history stays cached between its turns."""

        sessions_per_group: int = 1
        """Sessions a new rollout group is expected to open, e.g. its group size, times 2 for
        self-play. A group reserves its first call's blocks for each one that has not started yet.
        The policy raises it to the largest group it has seen."""

    def __init__(
        self,
        config: Config,
        *,
        budgets: list[KVCacheBudget],
        group_size: int,
    ):
        del group_size
        self.config = config
        # The generators share one config, so one budget's per-session cost fits all.
        self._budget = budgets[0]
        # Per generator: the blocks its groups may hold, the blocks they hold, and their sessions
        # (live or not started yet).
        self.limits = [int(config.limit * budget.num_blocks) for budget in budgets]
        self.blocks = [0] * len(budgets)
        self.sessions = [0] * len(budgets)
        self._groups: dict[int, _EstimatedGroup] = {}
        # Most sessions a placed group has started so far.
        self._largest_group = 0
        logger.info(
            "KV estimate admission: groups may hold %s KV cache blocks per generator; %s",
            self.limits,
            self._budget,
        )

    def admit(
        self,
        group_id: int,
        *,
        num_prompt_tokens: int,
        serving: list[int],
        skip_queue: bool,
    ) -> int | None:
        first_call_blocks = self._budget.session_blocks(num_prompt_tokens)
        # Sessions also reserve blocks for their growth, each new one `growth` (0 in this mode).
        growth = self._growth_per_session()
        charged = [b + self._reserved_growth(g) for g, b in enumerate(self.blocks)]
        generator = max(serving, key=lambda g: self.limits[g] - charged[g])
        # A validation group never waits, but reserves one session so the next one spreads out.
        expected_sessions = 1 if skip_queue else self._new_group_sessions()
        new_charged = charged[generator] + expected_sessions * (
            first_call_blocks + growth
        )
        # An empty generator takes any group, so a group larger than the limit still runs.
        if (
            not skip_queue
            and self.blocks[generator] > 0
            and new_charged > self.limits[generator]
        ):
            return None
        group = _EstimatedGroup(
            generator=generator,
            expected_sessions=expected_sessions,
            first_call_blocks=first_call_blocks,
        )
        self._groups[group_id] = group
        self.blocks[generator] += group.blocks
        self.sessions[generator] += group.sessions
        return generator

    def release(self, group_id: int, *, num_completion_tokens: int) -> None:
        del num_completion_tokens
        group = self._groups.pop(group_id)
        self.blocks[group.generator] -= group.blocks
        self.sessions[group.generator] -= group.sessions

    def set_session_tokens(
        self, group_id: int, *, session_id: str | None, num_tokens: int | None
    ) -> None:
        group = self._groups[group_id]
        old_blocks, old_sessions = group.blocks, group.sessions
        if num_tokens is None:
            group.blocks_by_session.pop(session_id, None)
        else:
            group.started_sessions.add(session_id)
            group.blocks_by_session[session_id] = self._budget.session_blocks(
                num_tokens
            )
            self._largest_group = max(self._largest_group, len(group.started_sessions))
        self.blocks[group.generator] += group.blocks - old_blocks
        self.sessions[group.generator] += group.sessions - old_sessions

    def summary(self) -> str:
        fractions = [
            round((b + self._reserved_growth(g)) / limit, 2)
            for g, (b, limit) in enumerate(zip(self.blocks, self.limits))
        ]
        return f"KV blocks / limit per generator: {fractions}"

    def _growth_per_session(self) -> float:
        return 0.0

    def _reserved_growth(self, generator: int) -> float:
        """Return the blocks `generator`'s sessions reserve for their growth: none in this mode."""
        return 0.0

    def _new_group_sessions(self) -> int:
        return max(self.config.sessions_per_group, self._largest_group)


# R averages over this window; 5 to 30 minutes changed simulated throughput by at most 3%.
GROWTH_WINDOW_S = 900.0
# R is half a max-length context until this many sessions have ended in the window.
MIN_ENDED_SESSIONS = 50
# A new group reserves the mean sessions of this many recently finished groups.
GROUP_SIZE_HISTORY = 256


class KVGrowthEstimateAdmission(KVEstimateAdmission):
    """The router's KV estimate plus the blocks each session is expected to add before it ends,
    so a group counts at its expected final size, not its current size (the "growth ledger").

    A session that grew g blocks since its first call reserves its expected remaining growth, from
    the total growth G of each session that ended in the last 15 minutes:

        max(mean(G - g) over the G above g, R - g),  R = blocks live sessions added / sessions ended

    R, one whole session's growth by Little's law, is also what a seat not started yet reserves.
    A new group reserves the mean sessions of recently finished groups; until one finishes, the
    largest group so far, and at least two sessions per sample (self-play opens one per player).
    Reserves are re-estimated at each poll.

    Example (limit 1.0, 150 blocks per generator; ended sessions grew G = 4, 10, 16, so R = 10)::

        generator 0: two live 20-block sessions; one grew 0 (reserves mean(4, 10, 16) = 10), one
        grew 12 (reserves 16 - 12 = 4) -> charged 40 + 10 + 4 = 54
        a new group of 2 sessions with 5-block first calls -> + 2 x (5 + 10) = 84 <= 150: admitted
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        limit: float = 1.6
        """How far the sessions' expected final sizes may overcommit each generator's KV cache.
        A live session has about as much growth ahead as behind, so KV usage settles near
        limit / 2."""

    def __init__(
        self,
        config: Config,
        *,
        budgets: list[KVCacheBudget],
        group_size: int,
    ):
        super().__init__(config, budgets=budgets, group_size=group_size)
        self._group_size = group_size
        # (time, blocks) for each growth of a live session, their sum, and (time, total growth) of
        # each session that ended.
        self._growth: deque[tuple[float, int]] = deque()
        self._growth_sum = 0
        self._ended: deque[tuple[float, int]] = deque()
        self._sessions_per_group: deque[int] = deque(maxlen=GROUP_SIZE_HISTORY)
        self._initial_growth = (
            self._budget.session_blocks(self._budget.max_model_len) / 2
        )
        # Per generator: the blocks its sessions reserve for growth, as of the last poll.
        self._reserve = [0.0] * len(budgets)

    def release(self, group_id: int, *, num_completion_tokens: int) -> None:
        # Validation groups (negative ids, see the controller) have their own size.
        if group_id >= 0:
            group = self._groups[group_id]
            self._sessions_per_group.append(len(group.started_sessions))
        super().release(group_id, num_completion_tokens=num_completion_tokens)

    def admit(
        self,
        group_id: int,
        *,
        num_prompt_tokens: int,
        serving: list[int],
        skip_queue: bool,
    ) -> int | None:
        generator = super().admit(
            group_id,
            num_prompt_tokens=num_prompt_tokens,
            serving=serving,
            skip_queue=skip_queue,
        )
        if generator is not None:
            # Until the next poll, each of the new group's seats reserves R.
            seats = self._groups[group_id].expected_sessions
            self._reserve[generator] += seats * self._growth_per_session()
        return generator

    def observe(self, loads: dict[int, EngineLoad]) -> None:
        """Re-estimate the growth each generator's sessions reserve."""
        del loads
        growth = self._growth_per_session()
        remaining = None
        if len(self._ended) >= MIN_ENDED_SESSIONS:
            remaining = _remaining_growth([g for _, g in self._ended])
        reserve = [0.0] * len(self._reserve)
        for group in self._groups.values():
            for session_id, blocks in group.blocks_by_session.items():
                grown = blocks - group.first_blocks_by_session[session_id]
                expected = growth if remaining is None else remaining(grown)
                reserve[group.generator] += max(expected, growth - grown)
            not_started = max(0, group.expected_sessions - len(group.started_sessions))
            reserve[group.generator] += not_started * growth
        self._reserve = reserve

    def set_session_tokens(
        self, group_id: int, *, session_id: str | None, num_tokens: int | None
    ) -> None:
        group = self._groups[group_id]
        old_blocks = group.blocks_by_session.get(session_id)
        if old_blocks is not None:
            now = time.monotonic()
            if num_tokens is None:
                first = group.first_blocks_by_session.pop(session_id)
                self._ended.append((now, old_blocks - first))
            else:
                grown = self._budget.session_blocks(num_tokens) - old_blocks
                if grown > 0:
                    self._growth.append((now, grown))
                    self._growth_sum += grown
        elif num_tokens is not None:
            group.first_blocks_by_session[session_id] = self._budget.session_blocks(
                num_tokens
            )
        super().set_session_tokens(
            group_id, session_id=session_id, num_tokens=num_tokens
        )

    def summary(self) -> str:
        growth = self._growth_per_session()
        return f"growth {growth:.0f} blocks per session; {super().summary()}"

    def _growth_per_session(self) -> float:
        """Return R, the blocks a session adds over its whole life.

        Example: in the last 15 min live sessions added 9,000 blocks and 300 sessions ended -> 30.
        """
        cutoff = time.monotonic() - GROWTH_WINDOW_S
        while self._growth and self._growth[0][0] < cutoff:
            self._growth_sum -= self._growth.popleft()[1]
        while self._ended and self._ended[0][0] < cutoff:
            self._ended.popleft()
        if len(self._ended) < MIN_ENDED_SESSIONS:
            return self._initial_growth
        # Little's law: blocks added per second / sessions ended per second = one session's growth.
        return self._growth_sum / len(self._ended)

    def _reserved_growth(self, generator: int) -> float:
        return self._reserve[generator]

    def _new_group_sessions(self) -> int:
        if not self._sessions_per_group:
            return max(2 * self._group_size, self._largest_group)
        return math.ceil(sum(self._sessions_per_group) / len(self._sessions_per_group))


def _remaining_growth(ended_growth: list[int]) -> Callable[[int], float]:
    """Return g -> mean of G - g over the ended sessions' growths G above g (0 if none is).

    Example: _remaining_growth([4, 10, 16])(0) -> 10.0; (12) -> 4.0; (16) -> 0.0
    """
    growth = sorted(ended_growth)
    # suffix[i] = sum(growth[i:])
    suffix = list(itertools.accumulate(reversed(growth)))[::-1] + [0]

    def remaining(grown: int) -> float:
        i = bisect.bisect_right(growth, grown)
        above = len(growth) - i
        return suffix[i] / above - grown if above else 0.0

    return remaining


@dataclass(kw_only=True, slots=True)
class _EstimatedGroup:
    """A rollout group's estimated KV blocks on its generator."""

    generator: int

    expected_sessions: int
    """Sessions the group is expected to open."""

    first_call_blocks: int
    """Blocks reserved for each expected session that has not started yet."""

    blocks_by_session: dict[str | None, int] = field(default_factory=dict)
    """Blocks each live session holds for its context so far."""

    first_blocks_by_session: dict[str | None, int] = field(default_factory=dict)
    """Blocks each live session held at its first call (growth ledger only)."""

    started_sessions: set[str | None] = field(default_factory=set)
    """Sessions that have made a call, ended ones included."""

    @property
    def blocks(self) -> int:
        """Blocks charged to the generator: live sessions plus those not started yet."""
        not_started = max(0, self.expected_sessions - len(self.started_sessions))
        return (
            sum(self.blocks_by_session.values()) + not_started * self.first_call_blocks
        )

    @property
    def sessions(self) -> int:
        """Sessions charged to the generator: live sessions plus those not started yet."""
        not_started = max(0, self.expected_sessions - len(self.started_sessions))
        return len(self.blocks_by_session) + not_started
