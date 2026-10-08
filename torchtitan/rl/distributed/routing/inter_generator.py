# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator routing."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any

from monarch.actor import Actor, concurrent_endpoint, current_size

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.distributed.routing.strategies import (
    RoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.distributed.routing.types import (
    KVCacheBudget,
    RoutingCandidate,
    RoutingContext,
)

logger = logging.getLogger(__name__)


class _GeneratorState(Enum):
    """Lifecycle state controlling routability; ``SYNCING`` is only entered when draining (i.e. hot-swap is off)."""

    SERVING = auto()
    SYNCING = auto()


@dataclass(kw_only=True, slots=True)
class _GeneratorHandle(RoutingCandidate):
    """Router-side metadata for one generator mesh."""

    actor: Any
    """Monarch actor handle for the full generator mesh. Used for fan-out calls
    that every rank must run."""

    rank0_actor: Any
    """Cached rank-0 slice of ``actor``. Used for calls that only rank 0 needs
    to run."""

    reserved_load: int = 0
    """Router-side estimate of in-flight routed generation work."""

    state: _GeneratorState = _GeneratorState.SERVING
    """Current routing lifecycle state for this generator."""

    idle: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when this generator has no reserved routed calls."""

    serving: asyncio.Event = field(default_factory=asyncio.Event)
    """Set while this generator is ``SERVING``."""

    kv_limit: int = 0
    """KV cache blocks that the groups placed here may hold (KV admission only)."""

    kv_blocks: int = 0
    """Estimated KV cache blocks held by the groups placed here (KV admission only)."""


@dataclass(kw_only=True, slots=True)
class _PlacedGroup:
    """A rollout group that KV admission placed on one generator.

    Example (3 sessions expected, each first call costs 4 blocks)::

        group 7's first call -> placed on gen1 with 3 x 4 = 12 blocks
        all 3 sessions start and grow to 6 blocks each -> blocks = 18
        session r0 ends -> blocks = 12
        release_groups([7]) -> gen1.kv_blocks -= 12
    """

    handle: _GeneratorHandle

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


@dataclass(kw_only=True, slots=True)
class _WaitingGroup:
    """A new rollout group waiting for KV room."""

    expected_sessions: int
    first_call_blocks: int
    admitted: asyncio.Future[_PlacedGroup]


class InterGeneratorRouter(Actor, Configurable):
    """Routes generation calls across generator meshes and pulls model's state dict.

    This is layer 1 of the two-layer routing design: it routes each call across
    generator *meshes* (replicas). Within the chosen mesh, ``IntraGeneratorRouter``
    then routes the request across that mesh's data-parallel ranks.

    Singleton:
        Routing decisions read and write mutable states such as ``_serving``,
        ``_GeneratorHandle.state``, and so on. These states are not backed by
        shared storage, so if there are multiple router instances, they cannot
        know each others' routing decisions. Instead of using shared storage,
        we solve the problem by enforcing the singleton pattern:
          * there should be only 1 router mesh in a training job;
          * this mesh should consists of only 1 actor.

        This pattern is simpler to implement, and should be good enough to handle
        the RL job's scale because the router is just a proxy, and the number of
        concurrent requests should be reasonable for a singleton to handle.

    Monarch Actor:
       The router singleton needs to be access from different processes or even
       different hosts. If we instantiate the router as an instance of a normal
       Python class, that instance's cannot be accessed from other processes or
       hosts. To solve this problem, we model the router as Monarch Actor, and
       pass the actor reference around.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        strategy: RoutingStrategy.Config = field(
            default_factory=StickySessionRoutingStrategy.Config
        )
        """Routing strategy, selected by its config type. The default keeps a
        session's requests (one multi-turn rollout) on one generator, so each turn
        reuses that generator's prefix KV; new sessions go to the least-loaded
        generator. Other options: ``LeastLoadedRoutingStrategy.Config()``,
        ``RoundRobinRoutingStrategy.Config()``."""

        hot_swap: bool = True
        """When True, pulls model's state dict concurrently with in-flight
        generation (no draining). When False, each generator is drained before
        its pull.

        Draining only waits for a generator's in-flight ``_route`` call (one
        turn) to finish; between turns of a multi-turn rollout the generator is
        idle, so a weight sync may land mid-rollout and successive turns can run
        under different policy versions. A turn whose session is pinned to a
        draining generator waits for it instead of moving to another one."""

        kv_admission_limit: float | None = None
        """Fraction of each generator's KV cache blocks that live sessions may
        hold, so that a session's history stays cached between its turns.
        ``None`` (default) turns it off; when set, ``strategy`` is unused. Meant
        for multi-turn rollouts: vLLM already admits running requests by size,
        but counts a session's history between turns as free.

        Each rollout group runs on the generator with the most room. A new group
        waits (FIFO) until its first turns fit; turns of placed groups never
        wait; nor do validation groups (negative ids). Sessions count at their
        current context, so groups admitted together, e.g. at startup, can grow
        past the limit."""

        kv_admission_sessions_per_group: int = 1
        """Sessions a new rollout group is expected to open, e.g. its group size,
        times 2 for self-play. A group reserves its first call's blocks for each
        one that has not started yet. The router raises it to the largest group
        it has seen."""

    def __init__(
        self,
        config: Config,
        *,
        generators: Sequence[Any],
    ):
        num_actors = math.prod(current_size().values())
        assert (
            num_actors == 1
        ), f"InterGeneratorRouter must be a singleton, but its mesh holds {num_actors} actors"

        self._config = config
        self._generators = [
            _GeneratorHandle(
                actor=generator,
                rank0_actor=generator.flatten("rank").slice(rank=0),
            )
            for generator in generators
        ]
        if not self._generators:
            raise ValueError("InterGeneratorRouter requires at least one generator")
        for h in self._generators:
            h.idle.set()
            h.serving.set()

        self._strategy = config.strategy.build()
        self._serving = asyncio.Event()
        self._refresh_serving_status()
        # Routing sessions of each rollout group, so `release_groups` drops any a rollout did not release.
        self._group_sessions: dict[int, set[str]] = {}

        # KV admission state, used when `kv_admission_limit` is set. The budget is read
        # from the generators in `start_engine_loop`.
        self._kv_budget: KVCacheBudget | None = None
        self._placed_groups: dict[int, _PlacedGroup] = {}
        # New groups in arrival order; admitted first-in, first-out.
        self._waiting_groups: dict[int, _WaitingGroup] = {}
        self._max_sessions_per_group = config.kv_admission_sessions_per_group

    def _candidates(self) -> list[_GeneratorHandle]:
        """Return generator handles that are currently routable."""

        return [h for h in self._generators if h.state is _GeneratorState.SERVING]

    def _refresh_serving_status(self) -> None:
        """Update whether any generator can serve; only changes while draining (i.e. hot-swap is off)."""

        if self._candidates():
            self._serving.set()
        else:
            self._serving.clear()

    def _set_state(self, h: _GeneratorHandle, state: _GeneratorState) -> None:
        """Move a generator between serving and syncing states."""

        h.state = state
        if state is _GeneratorState.SERVING:
            h.serving.set()
        else:
            h.serving.clear()
        self._refresh_serving_status()
        # A generator back from a drain may have room for waiting groups.
        self._admit_waiting_groups()

    def _reserve(self, h: _GeneratorHandle, cost: int) -> None:
        """Reserve estimated generation work on a handle before dispatch."""

        if cost < 0:
            raise ValueError(f"route estimated_cost must be non-negative, got {cost}")
        if h.reserved_load == 0:
            h.idle.clear()
        h.reserved_load += cost

    def _release(self, h: _GeneratorHandle, cost: int) -> None:
        """Release estimated generation work after a routed call finishes."""

        h.reserved_load -= cost
        assert (
            h.reserved_load >= 0
        ), f"generator reserved_load went negative: {h.reserved_load}"
        if h.reserved_load == 0:
            h.idle.set()

    async def _route(
        self,
        method: str,
        *args,
        routing_ctx: RoutingContext,
        **kwargs,
    ) -> Any:
        """Dispatch one call to a strategy-chosen serving generator's rank 0;
        return its result.
        """
        # A turn pinned to a draining generator waits for it instead of re-pinning
        # elsewhere: the drain is short, and the generator keeps the session's KV.
        pinned = self._strategy.pinned_candidate(routing_ctx)
        if pinned is not None:
            await pinned.serving.wait()
        await self._serving.wait()
        candidates = self._candidates()
        assert candidates, "serving event was set with no serving generators"
        h = self._strategy.choose(routing_ctx, candidates)
        return await self._call(
            h, method, *args, cost=routing_ctx.estimated_cost, **kwargs
        )

    async def _call(
        self, h: _GeneratorHandle, method: str, *args, cost: int, **kwargs
    ) -> Any:
        """Call ``method`` on a generator's rank 0, reserving ``cost`` while it runs."""
        self._reserve(h, cost)
        try:
            return await getattr(h.rank0_actor, method).call_one(*args, **kwargs)
        finally:
            self._release(h, cost)

    def _roomiest_generator(self) -> _GeneratorHandle | None:
        """Return the serving generator with the most KV room."""
        return max(
            self._candidates(), key=lambda h: h.kv_limit - h.kv_blocks, default=None
        )

    def _place_group(
        self,
        group_id: int,
        h: _GeneratorHandle,
        expected_sessions: int,
        first_call_blocks: int,
    ) -> _PlacedGroup:
        placed = _PlacedGroup(
            handle=h,
            expected_sessions=expected_sessions,
            first_call_blocks=first_call_blocks,
        )
        self._placed_groups[group_id] = placed
        h.kv_blocks += placed.blocks
        return placed

    def _admit_waiting_groups(self) -> None:
        """Place waiting groups, oldest first, while the oldest fits on a generator."""
        while self._waiting_groups:
            group_id, waiting = next(iter(self._waiting_groups.items()))
            h = self._roomiest_generator()
            # An empty generator takes any group, so a group larger than the limit still runs.
            blocks = waiting.expected_sessions * waiting.first_call_blocks
            if h is None or (h.kv_blocks > 0 and h.kv_blocks + blocks > h.kv_limit):
                return
            del self._waiting_groups[group_id]
            waiting.admitted.set_result(
                self._place_group(
                    group_id, h, waiting.expected_sessions, waiting.first_call_blocks
                )
            )

    async def _admit_group(self, group_id: int, num_tokens: int) -> _PlacedGroup:
        """Return the group's placement; a new group first waits for KV room.

        Args:
            group_id: Rollout group of the call.
            num_tokens: The call's prompt length.
        """
        placed = self._placed_groups.get(group_id)
        if placed is not None:
            return placed
        assert self._kv_budget is not None, "start_engine_loop reads the KV budget"
        first_call_blocks = self._kv_budget.session_blocks(num_tokens)
        if group_id < 0:
            # Validation groups (negative ids, see the controller) skip the queue: the
            # controller blocks on validation.
            await self._serving.wait()
            # A sibling may have placed the group while this call waited.
            placed = self._placed_groups.get(group_id)
            if placed is not None:
                return placed
            return self._place_group(
                group_id, self._roomiest_generator(), 0, first_call_blocks
            )
        waiting = self._waiting_groups.get(group_id)
        if waiting is None:
            # TODO: reserve what the sessions will grow to, e.g. the mean final blocks of
            # released sessions; a session admitted at first-turn size can grow ~2-7x.
            waiting = _WaitingGroup(
                expected_sessions=self._max_sessions_per_group,
                first_call_blocks=first_call_blocks,
                admitted=asyncio.get_running_loop().create_future(),
            )
            self._waiting_groups[group_id] = waiting
            self._admit_waiting_groups()
        with sl.log_trace_span("router_kv_admission_wait"):
            # Shielded: one cancelled sibling must not cancel the group's admission.
            return await asyncio.shield(waiting.admitted)

    def _set_session_tokens(
        self,
        group_id: int,
        placed: _PlacedGroup,
        session_id: str | None,
        num_tokens: int | None,
    ) -> None:
        """Set a session's context length on its group's generator; ``None`` ends the session."""
        if self._placed_groups.get(group_id) is not placed:
            return  # The group was released while this call ran.
        assert self._kv_budget is not None
        old_blocks = placed.blocks
        if num_tokens is None:
            placed.blocks_by_session.pop(session_id, None)
        else:
            placed.started_sessions.add(session_id)
            placed.blocks_by_session[session_id] = self._kv_budget.session_blocks(
                num_tokens
            )
            self._max_sessions_per_group = max(
                self._max_sessions_per_group, len(placed.started_sessions)
            )
        placed.handle.kv_blocks += placed.blocks - old_blocks
        self._admit_waiting_groups()

    async def _generate_with_kv_admission(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        group_id: int,
        routing_session_id: str | None,
        sampling_config: Any | None,
        metrics_prefix: str,
    ) -> Any:
        """Generate on the group's generator, admitting the group first if it is new.

        A call charges its prompt while it runs (vLLM admits what it generates), then its
        prompt plus completion.
        """
        num_prompt_tokens = len(prompt_token_ids)
        placed = await self._admit_group(group_id, num_prompt_tokens)
        self._set_session_tokens(
            group_id, placed, routing_session_id, num_prompt_tokens
        )
        h = placed.handle
        # Wait out a drain on the group's generator: it is short, and it keeps the group's KV.
        while not h.serving.is_set():
            await h.serving.wait()
        completion = await self._call(
            h,
            "generate",
            prompt_token_ids,
            request_id=request_id,
            group_id=group_id,
            routing_session_id=routing_session_id,
            sampling_config=sampling_config,
            metrics_prefix=metrics_prefix,
            cost=1,
        )
        self._set_session_tokens(
            group_id,
            placed,
            routing_session_id,
            num_prompt_tokens + len(completion.token_ids),
        )
        return completion

    async def _read_kv_budgets(self) -> None:
        """Read every generator's KV cache budget and set its admission limit."""
        budgets = await asyncio.gather(
            *[h.rank0_actor.kv_cache_budget.call_one() for h in self._generators]
        )
        for h, budget in zip(self._generators, budgets, strict=True):
            h.kv_limit = int(self._config.kv_admission_limit * budget.num_blocks)
        # The generators share one config, so one budget's per-session cost fits all.
        self._kv_budget = budgets[0]
        logger.info(
            "KV admission: groups may hold %s KV cache blocks per generator; %s",
            [h.kv_limit for h in self._generators],
            self._kv_budget,
        )

    def _release_session(self, group_id: int, session_id: str) -> None:
        """Drop a session's affinity and, with KV admission, its blocks."""
        self._group_sessions.get(group_id, set()).discard(session_id)
        self._strategy.release_session(session_id)
        placed = self._placed_groups.get(group_id)
        if placed is not None:
            self._set_session_tokens(group_id, placed, session_id, None)

    def _release_groups(self, group_ids: list[int]) -> None:
        """Drop the affinity of every session left in these groups, and free their blocks."""
        for group_id in group_ids:
            for session_id in self._group_sessions.pop(group_id, ()):
                self._strategy.release_session(session_id)
            placed = self._placed_groups.pop(group_id, None)
            if placed is not None:
                placed.handle.kv_blocks -= placed.blocks
            waiting = self._waiting_groups.pop(group_id, None)
            if waiting is not None:
                waiting.admitted.cancel()
        self._admit_waiting_groups()

    async def _generate(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        group_id: int,
        routing_session_id: str | None,
        sampling_config: Any | None,
        metrics_prefix: str,
    ) -> Any:
        """Body of the ``generate`` endpoint."""
        if routing_session_id is not None:
            self._group_sessions.setdefault(group_id, set()).add(routing_session_id)
        if self._config.kv_admission_limit is not None:
            return await self._generate_with_kv_admission(
                prompt_token_ids,
                request_id=request_id,
                group_id=group_id,
                routing_session_id=routing_session_id,
                sampling_config=sampling_config,
                metrics_prefix=metrics_prefix,
            )
        return await self._route(
            "generate",
            prompt_token_ids,
            request_id=request_id,
            group_id=group_id,
            # VLLMGenerator.generate also requires this field for its
            # intra-mesh DP routing.
            routing_session_id=routing_session_id,
            sampling_config=sampling_config,
            metrics_prefix=metrics_prefix,
            # Load is measured as in-flight request count (one unit per call).
            routing_ctx=RoutingContext(
                estimated_cost=1,
                session_id=routing_session_id,
            ),
        )

    async def _fanout(
        self,
        method: str,
        *args,
        return_exceptions: bool = False,
        **kwargs,
    ) -> list[Any | BaseException]:
        """Call ``method`` on every generator concurrently and gather results.

        Args:
            method: Actor endpoint name to call on every generator.
            *args: Positional arguments forwarded to each call.
            return_exceptions: If False (default), the first exception
                propagates immediately; if True, each call's exception is
                returned in the list instead of raised. Either way, a failure
                never cancels the other calls.
            **kwargs: Keyword arguments forwarded to each call.

        Returns:
            One entry per generator, in order: its result, or its exception when
            ``return_exceptions`` is True.
        """
        return await asyncio.gather(
            *[getattr(h.actor, method).call(*args, **kwargs) for h in self._generators],
            return_exceptions=return_exceptions,
        )

    async def _pull_model_state_dict(self, *, policy_version: int) -> None:
        """Pull the given policy version's state dict into every generator.

        Args:
            policy_version: Trainer policy version whose state dict to pull.
        """

        async def _pull_one(h: _GeneratorHandle) -> None:
            if self._config.hot_swap:
                # Hot swap: pull concurrently with in-flight generation, without
                # draining. Whether the pull is genuinely concurrent and safe is
                # up to the generator's implementation.
                await h.rank0_actor.pull_model_state_dict.call_one(policy_version)
            else:
                # Drain: stop routing to this generator and wait for in-flight
                # work to finish before pulling, then re-admit it.
                self._set_state(h, _GeneratorState.SYNCING)
                try:
                    with sl.log_trace_span("router_drain_wait"):
                        await h.idle.wait()
                    await h.rank0_actor.pull_model_state_dict.call_one(policy_version)
                finally:
                    self._set_state(h, _GeneratorState.SERVING)

        # Start the pulls in parallel. Technically we could do rolling sync to
        # maintain availability during weight sync, but that's not a priority
        # for now.
        # TODO(perf): stagger the per-generator fetches when num_generators is large so they don't
        #   all read the trainer's CPU-staged weights at once -- bounds trainer host RAM. Matters for
        #   big models / many generators, not at small scale.
        await asyncio.gather(*[_pull_one(h) for h in self._generators])

    @concurrent_endpoint
    async def generate(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        group_id: int,
        routing_session_id: str | None,
        sampling_config: Any | None,
        metrics_prefix: str,
    ) -> Any:
        """Route one generation call to a generator and return its completion."""
        # The logic lives in a private method so tests can call it without an actor mesh.
        return await self._generate(
            prompt_token_ids,
            request_id=request_id,
            group_id=group_id,
            routing_session_id=routing_session_id,
            sampling_config=sampling_config,
            metrics_prefix=metrics_prefix,
        )

    @concurrent_endpoint
    async def start_engine_loop(self) -> None:
        """Start the engine loop on every rank of every generator."""
        await self._fanout("start_engine_loop")
        if self._config.kv_admission_limit is not None:
            await self._read_kv_budgets()

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        """Set the step counter in this process and in every generator rank, and log KV admission."""
        sl.set_step(step)
        if self._config.kv_admission_limit is not None:
            logger.info(
                "KV admission at step %d: %d groups placed, %d waiting; "
                "KV blocks / limit per generator: %s",
                step,
                len(self._placed_groups),
                len(self._waiting_groups),
                [round(h.kv_blocks / h.kv_limit, 2) for h in self._generators],
            )
        await self._fanout("sync_log_step", step)

    @concurrent_endpoint
    async def release_session(self, group_id: int, routing_session_id: str) -> None:
        """Forget a routing session after its rollout's last generation call."""
        self._release_session(group_id, routing_session_id)

    @concurrent_endpoint
    async def release_groups(self, group_ids: list[int]) -> None:
        """Tell every generator that these rollout groups are finished."""
        self._release_groups(group_ids)
        await self._fanout("release_groups", group_ids)

    @concurrent_endpoint
    async def pull_model_state_dict(self, policy_version: int) -> None:
        """Pull the given policy version's state dict into every generator."""
        # Wrapper the logic in a private method so we can test it independently
        # without the need to spawn the Monarch actor mesh.
        await self._pull_model_state_dict(policy_version=policy_version)

    @concurrent_endpoint
    async def close_generators(self) -> list[Any | BaseException]:
        """Close every generator, returning each one's result or exception."""
        return await self._fanout("close", return_exceptions=True)
