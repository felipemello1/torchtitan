# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routing strategies.

A ``RoutingStrategy`` chooses one ``RoutingCandidate`` from a set given a
``RoutingContext``. It depends only on the ``RoutingCandidate`` base class -- a
``reserved_load`` field plus object identity -- so the same strategy classes
serve both routing layers in the RL generator:

- Layer 1: ``InterGeneratorRouter`` (router actor side) routes a call across
  generator *meshes* (replicas). See ``inter_generator_router.py``.
- Layer 2: ``IntraGeneratorRouter`` (in-mesh, rank-0 side) routes a request
  across the *data-parallel ranks* within one generator mesh. See
  ``intra_generator_router.py``.

Each layer supplies its own candidate type (``_GeneratorHandle`` for generators,
``_DPRankHandle`` for DP ranks). The data types live in ``types.py``.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.rl.routing.types import RoutingCandidate, RoutingContext


class RoutingStrategy(Configurable, ABC):
    """Policy object that chooses one candidate for a request.

    Add a new strategy by subclassing this, defining a nested ``Config``, and
    selecting it explicitly in config, e.g.
    ``InterGeneratorRouter.Config(strategy=MyRoutingStrategy.Config())``.
    """

    def __init__(self, config: Configurable.Config):
        # Stateless by default; stateful strategies override __init__.
        del config

    @abstractmethod
    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Choose one candidate from the (non-empty) candidates."""

    def pinned_candidate(self, routing_ctx: RoutingContext) -> RoutingCandidate | None:
        """Return the candidate this request's session is pinned to, if any."""
        del routing_ctx
        return None

    def release_session(self, session_id: str) -> None:
        """Forget a session that makes no more requests."""
        del session_id

    def release_groups(self, group_ids: list[int]) -> None:
        """Forget rollout groups that make no more requests."""
        del group_ids


class RoundRobinRoutingStrategy(RoutingStrategy):
    """Cycle over the candidates in order, ignoring load."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        del config
        self._counter = itertools.count()

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Return the next candidate in round-robin order."""

        del routing_ctx
        return candidates[next(self._counter) % len(candidates)]


class LeastLoadedRoutingStrategy(RoutingStrategy):
    """Pick the least loaded candidate, breaking ties by least recently chosen."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        del config
        self._selection_counter = itertools.count()
        # id(candidate) -> the selection number of its most recent selection.
        # Object identity is all a strategy gets from RoutingCandidate, and both
        # routers build their handles once and own them for their own lifetime,
        # so ids stay valid and the map stays bounded by the candidate count.
        self._last_chosen: dict[int, int] = {}

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Return the least recently chosen of the lowest reserved load candidates.

        Ties are common: a request arriving while the candidates are quiescent
        sees all of them at zero load. Resolving every tie to the same candidate
        is harmless on its own because the next request re-picks, but
        ``StickySessionRoutingStrategy`` keeps a session on whichever candidate
        this returns for the session's lifetime, so a fixed tie-break pins every
        session started from an idle state onto one candidate.

        The tie-break tracks per-candidate recency instead of cycling a counter
        over the tied candidates, because generators drain for model state dict loads and
        leave the candidate set: cycling by position over a list whose length
        keeps changing can starve a candidate that is only sometimes present.
        """

        del routing_ctx
        lowest_load = min(h.reserved_load for h in candidates)
        chosen = min(
            (h for h in candidates if h.reserved_load == lowest_load),
            key=lambda h: self._last_chosen.get(id(h), -1),
        )
        self._last_chosen[id(chosen)] = next(self._selection_counter)
        return chosen


class StickySessionRoutingStrategy(RoutingStrategy):
    """Keep requests from the same session on the same candidate."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        max_sessions: int = 262144
        """Maximum session assignments kept (~150 B each; 262144 is ~35 MiB); past it
        the least-recently-used one is evicted. Rollouts release their session when they
        end, so keep it above the rollouts in flight, or live sessions lose their cached KV."""

        fallback_strategy: RoutingStrategy.Config = field(
            default_factory=LeastLoadedRoutingStrategy.Config
        )
        """Routing strategy used for new sessions that don't join their group's home
        (`group_slack`), and for requests without a session."""

        repin_load_ratio: float | None = 1.5
        """Move a session off its pinned candidate once that candidate's in-flight load is above
        `repin_load_ratio` x the mean load. The fallback picks the new candidate; the move
        re-prefills the session's context. None never moves it.

        With 1.5, pinned candidate first:

            loads [6, 2, 2, 2]  mean 3     6 > 4.5    moves
            loads [4, 3, 3, 2]  mean 3     4 <= 4.5   stays
            loads [1, 0, 0, 0]  mean 0.25  1 > 0.375  moves, even at low load
        """

        group_slack: int | None = None
        """Keep a group's siblings together so they share their prompt's KV. None: off.

        A new session joins its group's home, the generator the group's first session went to,
        while home has at most `group_slack` more in-flight calls than the least loaded.
        Higher keeps more of a group together; lower balances load better. With
        `repin_load_ratio` set, a session can join a home that re-pins it on its next turn.

        With 2, home first:

            new session, loads [3, 2, 2]    3 <= 2 + 2   joins home
            new session, loads [5, 2, 3]    5 >  2 + 2   fallback picks
            re-pin off an overloaded pin                 fallback picks, never home

        TODO: only `InterGeneratorRouter` passes group ids, so this has no effect across a
        generator's DP ranks. That layer is untested and unmeasured (DP ranks step in lockstep).
        """

        def __post_init__(self):
            if self.max_sessions <= 0:
                raise ValueError(
                    f"max_sessions must be positive, got {self.max_sessions}"
                )
            if self.repin_load_ratio is not None and self.repin_load_ratio <= 1:
                raise ValueError(
                    f"repin_load_ratio must be > 1 or None, got {self.repin_load_ratio}"
                )
            if self.group_slack is not None and self.group_slack < 0:
                raise ValueError(
                    f"group_slack must be >= 0 or None, got {self.group_slack}"
                )

    def __init__(self, config: Config):
        self._max_sessions = config.max_sessions
        self._fallback_strategy = config.fallback_strategy.build()
        self._repin_load_ratio = config.repin_load_ratio
        self._group_slack = config.group_slack
        self._sessions: OrderedDict[str, RoutingCandidate] = OrderedDict()
        # group_id -> the candidate its first session went to (only with group_slack).
        self._group_homes: dict[int, RoutingCandidate] = {}

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Return the session's assigned candidate, or assign a new one.

        - no ``session_id``: the fallback chooses; nothing is pinned.
        - new session: its group's home with ``group_slack``, else the fallback chooses; the
          session is pinned there.
        - pinned session: reuse its candidate, unless that candidate is gone from
          ``candidates`` or overloaded (``repin_load_ratio``); then the fallback
          chooses and the session is re-pinned.

        Keeps at most ``max_sessions`` pins; evicts the least-recently-used one.
        """

        # Unpinned request: no affinity, defer entirely to the fallback.
        if routing_ctx.session_id is None:
            return self._fallback_strategy.choose(routing_ctx, candidates)

        # Reuse the pinned candidate while it is still a candidate and not overloaded.
        sticky_candidate = self._sessions.get(routing_ctx.session_id)
        if sticky_candidate is not None:
            mean_load = sum(h.reserved_load for h in candidates) / len(candidates)
            overloaded = (
                self._repin_load_ratio is not None
                and sticky_candidate.reserved_load > self._repin_load_ratio * mean_load
            )
            if any(h is sticky_candidate for h in candidates) and not overloaded:
                # End of the dict means it's the most-recently-used session.
                self._sessions.move_to_end(routing_ctx.session_id)
                return sticky_candidate

        # New session: join its group's home while home is within group_slack of the least loaded.
        # A re-pin (pinned candidate gone or overloaded) skips home, so it can't land back on it.
        home = None
        if self._group_slack is not None and sticky_candidate is None:
            home = self._group_homes.get(routing_ctx.group_id)
        home_has_room = (
            home is not None
            and any(h is home for h in candidates)
            and home.reserved_load
            <= min(h.reserved_load for h in candidates) + self._group_slack
        )
        if home_has_room:
            chosen = home
        else:
            # Consult the fallback only when its pick is used, so its tie-break tracks real placements.
            chosen = self._fallback_strategy.choose(routing_ctx, candidates)
            if self._group_slack is not None and routing_ctx.group_id is not None:
                self._group_homes.setdefault(routing_ctx.group_id, chosen)
        self._sessions[routing_ctx.session_id] = chosen
        # End of the dict means it's the most-recently-used session.
        self._sessions.move_to_end(routing_ctx.session_id)
        # Evict the least-recently-used session past max_sessions. `RolloutWorker` releases
        # each rollout's session when it ends, so its finished sessions rarely get here.
        if len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return chosen

    def pinned_candidate(self, routing_ctx: RoutingContext) -> RoutingCandidate | None:
        """Return the candidate this request's session is pinned to, if any."""
        if routing_ctx.session_id is None:
            return None
        return self._sessions.get(routing_ctx.session_id)

    def release_session(self, session_id: str) -> None:
        """Drop the session's assignment."""
        self._sessions.pop(session_id, None)

    def release_groups(self, group_ids: list[int]) -> None:
        """Drop the groups' homes."""
        for group_id in group_ids:
            self._group_homes.pop(group_id, None)
