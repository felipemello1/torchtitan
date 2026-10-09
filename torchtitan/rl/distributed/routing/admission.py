# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admission policies: when the inter-generator router starts a new rollout group, and where.

- ``UsageAIMDAdmission``: a cap on groups in flight, moved by vLLM's KV usage (prime-rl's controller).
- ``KVEstimateAdmission``: the router's own estimate of each generator's KV blocks (ThunderAgent's rule).
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from torchtitan.config import Configurable
from torchtitan.rl.distributed.routing.types import EngineLoad, KVCacheBudget

logger = logging.getLogger(__name__)


class AdmissionPolicy(Configurable, ABC):
    """Decides when the router starts a new rollout group, and on which generator.

    The router places each rollout group on one generator at its first generate call, and every
    later call of the group goes there. A new group waits in a FIFO queue until ``admit`` returns
    a generator for it; turns of placed groups never wait. The router's calls, for group 7::

        policy.observe(loads)  # at start, then every 5 s while groups are placed or waiting
        policy.admit(7, num_prompt_tokens=5000, serving=[0, 1], skip_queue=False)  # 1, or None
        policy.set_session_tokens(7, "group=7/rollout=0", 5000)  # the session's call starts
        policy.set_session_tokens(7, "group=7/rollout=0", 6200)  # the call returns
        policy.set_session_tokens(7, "group=7/rollout=0", None)  # the session ends
        policy.release(7, num_completion_tokens=1200)  # the group ends

    After each of these calls, the router calls ``admit`` for the oldest waiting group until it
    returns None. Add a mode by subclassing this and selecting its config, e.g.
    ``InterGeneratorRouter.Config(admission=MyAdmission.Config())``.

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
        """Forget a finished group that ``admit`` placed.

        Args:
            group_id: The finished group.
            num_completion_tokens: Tokens the group's calls generated.
        """

    def observe(self, loads: list[EngineLoad]) -> None:
        """Take one poll of every generator's load, by generator index."""
        del loads

    def set_session_tokens(
        self, group_id: int, session_id: str | None, num_tokens: int | None
    ) -> None:
        """Set a placed group's session context length; ``None`` ends the session."""
        del group_id, session_id, num_tokens

    def summary(self) -> str:
        """Return this policy's state for the router's per-step log line."""
        return ""


# prime-rl's constants (prime-rl src/prime_rl/orchestrator/concurrency.py:42-89).
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
"""Queue overload: vLLM's waiting requests above this fraction of its running ones ..."""
QUEUE_PERSISTENCE_POLLS = 6
"""... for this many polls in a row."""
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

Signal = Literal["clear", "soft", "hard"]
"""Engine pressure, in increasing severity."""
_SEVERITY: dict[Signal, int] = {"clear": 0, "soft": 1, "hard": 2}


class UsageAIMDAdmission(AdmissionPolicy):
    """Caps the rollout groups in flight and moves the cap with vLLM's KV usage: prime-rl's
    admission controller, with one permit per rollout group.

    - **Grow:** each finished group multiplies the cap by ``m ** (1 / in_flight)``, so the cap
      grows by ``m`` per turnover of the groups in flight. ``m`` falls from 1.2 at 0% KV usage to
      1.0 at 80%. Growth needs a clear last poll under 15 s old (usage <= 80%, no waiting request,
      no preemption, no drain) and a binding cap (in flight >= 90% of it).
    - **Trim:** above 80% usage on any generator, the cap drops to ``in_flight * 0.7 / usage``, at
      most once per 6 polls.
    - **Cut:** a preemption cuts the cap to 0.8 x in flight; vLLM's waiting requests above half
      its running ones for 6 polls cut it to 0.9 x. No cut or trim follows until in flight drains
      to the cap with no overload; an overload within 6 polls after that cuts to 0.5 x.
    - **Smoothing:** groups in flight grow by at most ``max(1, cap // 10)`` per poll; each
      finished group gives one back.

    A new group goes to the generator that took the fewest groups since the last poll, the
    lowest usage first: usage shows a new group's prefill only at the next poll.

    Example (2 generators, 100 groups in flight, cap 100; ``load(u, p)`` is an ``EngineLoad`` with
    KV usage ``u`` and ``p`` preemptions so far)::

        policy.observe([load(0.85, 0), load(0.60, 0)])
        # soft trim: cap 100 -> 82 (100 x 0.7 / 0.85); no group starts until in flight < 82
        policy.observe([load(0.70, 1), load(0.60, 0)])
        # preemption cut: cap 82 -> 80 (0.8 x 100); no cut or trim until in flight <= 80
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        initial_inflight: int | None = None
        """Groups in flight to start with. None starts from a pessimistic bound: the max-length
        contexts every generator's KV cache holds, one per sample of a group."""

        min_inflight: int = 1
        """Fewest groups in flight the cap falls to. Set it to ``max_inflight`` for a fixed cap."""

        max_inflight: int | None = 1024
        """Most groups in flight; None for no ceiling."""

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
        self.cap = self._clamp(initial_inflight)
        """``int(cap)`` groups may be in flight; growth accumulates its fractions here."""
        logger.info("Admission starts at %d groups in flight", int(self.cap))

        self.admitted_groups: set[int] = set()
        """Groups in flight: admitted and not yet released. Validation groups take no permit."""
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
        self.last_change = "none"
        """The last trim or cut, for the per-step log line."""

        self._loads: list[EngineLoad] = []
        self._placed_since_poll = [0] * len(budgets)
        # prime-rl's 5 s admission window (dispatcher.py:173-182, 271-278); here it is the 5 s poll.
        self._admissions_since_poll = 0

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
            limit = int(self.cap)
            # A group permit replaces prime-rl's episode permit, so its minimum burst of one
            # group size (dispatcher.py:182) is 1.
            burst = max(1, limit // 10)
            if (
                len(self.admitted_groups) >= limit
                or self._admissions_since_poll >= burst
            ):
                return None
            self.admitted_groups.add(group_id)
            self._admissions_since_poll += 1
        generator = min(
            serving,
            key=lambda g: (self._placed_since_poll[g], self._loads[g].kv_usage),
        )
        self._placed_since_poll[generator] += 1
        return generator

    def release(self, group_id: int, *, num_completion_tokens: int) -> None:
        """Free the group's permit, and grow the cap while the last poll allows it."""
        if group_id not in self.admitted_groups:
            return
        in_flight = len(self.admitted_groups)
        self.admitted_groups.remove(group_id)
        self._admissions_since_poll = max(0, self._admissions_since_poll - 1)
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

    def observe(self, loads: list[EngineLoad]) -> None:
        """Classify the pressure, gate growth, and trim or cut the cap."""
        previous = self._loads
        self._loads = loads
        self._placed_since_poll = [0] * len(loads)
        self._admissions_since_poll = 0

        # ======== Pressure signal ========
        signal: Signal = "clear"
        preempted = False
        for i, load in enumerate(loads):
            # The first poll has no baseline: its preemption count is not new.
            if previous and load.num_preemptions > previous[i].num_preemptions:
                preempted = True
                signal = "hard"
            if load.num_waiting > 0 and previous and previous[i].num_waiting > 0:
                signal = max(signal, "soft", key=_SEVERITY.__getitem__)
        max_usage = max(load.kv_usage for load in loads)
        num_running = sum(load.num_running for load in loads)
        num_waiting = sum(load.num_waiting for load in loads)
        if num_running > 0 and num_waiting > QUEUE_RATIO * num_running:
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
        in_flight = len(self.admitted_groups)
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
        self.can_grow = signal == "clear" and num_waiting == 0 and not self.draining
        self.growth_multiplier = 1.0 + (TURNOVER_GROWTH_MAX - 1.0) * max(
            0.0, 1.0 - max_usage / KV_USAGE_SOFT_CAP
        )
        self.can_grow_until = time.monotonic() + GROWTH_GATE_TTL_S
        if self.draining:
            return

        # ======== Cut on overload, else trim on usage ========
        if preempted or queue_overload:
            if self.escalated:
                fraction = ESCALATED_CUT_FRACTION
            elif queue_overload:
                fraction = QUEUE_CUT_FRACTION
            else:
                fraction = PREEMPTION_CUT_FRACTION
            if queue_overload:
                self.queue_overload_polls = 0
            reason = "queue overload" if queue_overload else "preemption"
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
        return f"cap {int(self.cap)}, signal {self.signal}, last change: {self.last_change}"

    def _resize_down(self, target: int, *, reason: str) -> None:
        """Lower the cap, never raise it; new groups wait until in flight drains below it."""
        # TODO: prime-rl also cancels the excess in flight at a hard trim and on every cut, whole
        # groups, youngest first by their oldest member's start (dispatcher.py:242-269). Deferred:
        # it throws away generated turns, and `hold_session_kv`'s free floor already releases idle
        # sessions under pressure. Without it, a 0.8 cut keeps new groups out until ~20% finish.
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
    vLLM blocks (``KVCacheBudget.session_blocks``): its prompt while a call runs, then its prompt
    plus completion, until it ends. A group also reserves its first call's blocks for each expected
    session that has not started yet. Sessions count at their current context, so groups admitted
    together, e.g. at startup, can grow past the limit.

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
        # The generators share one config, so one budget's per-session cost fits all.
        self._budget = budgets[0]
        self.limits = [int(config.limit * budget.num_blocks) for budget in budgets]
        """KV cache blocks that the groups placed on each generator may hold."""
        self.blocks = [0] * len(budgets)
        """Estimated KV cache blocks held by the groups placed on each generator."""
        self._groups: dict[int, _EstimatedGroup] = {}
        self._max_sessions_per_group = config.sessions_per_group
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
        generator = max(serving, key=lambda g: self.limits[g] - self.blocks[g])
        # TODO: reserve what the sessions will grow to, e.g. the mean final blocks of
        # released sessions; a session admitted at first-turn size can grow ~2-7x.
        expected_sessions = 0 if skip_queue else self._max_sessions_per_group
        new_blocks = self.blocks[generator] + expected_sessions * first_call_blocks
        # An empty generator takes any group, so a group larger than the limit still runs.
        if (
            not skip_queue
            and self.blocks[generator] > 0
            and new_blocks > self.limits[generator]
        ):
            return None
        group = _EstimatedGroup(
            generator=generator,
            expected_sessions=expected_sessions,
            first_call_blocks=first_call_blocks,
        )
        self._groups[group_id] = group
        self.blocks[generator] += group.blocks
        return generator

    def release(self, group_id: int, *, num_completion_tokens: int) -> None:
        del num_completion_tokens
        group = self._groups.pop(group_id)
        self.blocks[group.generator] -= group.blocks

    def set_session_tokens(
        self, group_id: int, session_id: str | None, num_tokens: int | None
    ) -> None:
        group = self._groups[group_id]
        old_blocks = group.blocks
        if num_tokens is None:
            group.blocks_by_session.pop(session_id, None)
        else:
            group.started_sessions.add(session_id)
            group.blocks_by_session[session_id] = self._budget.session_blocks(
                num_tokens
            )
            self._max_sessions_per_group = max(
                self._max_sessions_per_group, len(group.started_sessions)
            )
        self.blocks[group.generator] += group.blocks - old_blocks

    def summary(self) -> str:
        fractions = [round(b / limit, 2) for b, limit in zip(self.blocks, self.limits)]
        return f"KV blocks / limit per generator: {fractions}"


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

    started_sessions: set[str | None] = field(default_factory=set)
    """Sessions that have made a call, ended ones included."""

    @property
    def blocks(self) -> int:
        """Blocks charged to the generator: live sessions plus those not started yet."""
        not_started = max(0, self.expected_sessions - len(self.started_sessions))
        return (
            sum(self.blocks_by_session.values()) + not_started * self.first_call_blocks
        )
