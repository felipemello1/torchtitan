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

from monarch.actor import Actor, ActorError, concurrent_endpoint, current_size

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.distributed.routing.strategies import (
    RoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.distributed.routing.types import (
    GeneratorClosedError,
    RoutingCandidate,
    RoutingContext,
)

logger = logging.getLogger(__name__)


class _GeneratorState(Enum):
    """Lifecycle state controlling routability.

    SERVING -> SYNCING -> SERVING   drained for a weight sync (hot-swap off)
    any     -> CLOSED               raised ``GeneratorClosedError``; final
    """

    SERVING = auto()
    SYNCING = auto()
    CLOSED = auto()


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
    """Set unless this generator is ``SYNCING``."""


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

    Closed generators:
        A generator whose engine loop crashed raises ``GeneratorClosedError`` on
        every later call. The first such call fails, and the router marks the
        generator CLOSED for good:
          * new sessions, and sessions pinned to it, go to the other generators;
          * weight syncs and releases skip it;
          * once every generator is closed, routed calls and ``release_groups``
            raise. The rollout loop survives a failed rollout but not a failed
            ``release_groups``, so the job stops instead of starving.
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

    def __init__(
        self,
        config: Config,
        *,
        generators: Sequence[Any],
        forward_session_releases: bool = False,
    ):
        num_actors = math.prod(current_size().values())
        assert (
            num_actors == 1
        ), f"InterGeneratorRouter must be a singleton, but its mesh holds {num_actors} actors"

        self._config = config
        # Tell generators when a session ends, so they release its held KV (`hold_session_kv`).
        self._forward_session_releases = forward_session_releases
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

    def _candidates(self) -> list[_GeneratorHandle]:
        """Return generator handles that are currently routable."""

        return [h for h in self._generators if h.state is _GeneratorState.SERVING]

    def _all_closed(self) -> bool:
        return all(h.state is _GeneratorState.CLOSED for h in self._generators)

    def _refresh_serving_status(self) -> None:
        """Update whether routes may proceed: some generator serves, or all are closed and routes raise."""

        if self._candidates() or self._all_closed():
            self._serving.set()
        else:
            self._serving.clear()

    def _set_state(self, h: _GeneratorHandle, state: _GeneratorState) -> None:
        """Move a generator between states; ``CLOSED`` is final."""

        if h.state is not _GeneratorState.CLOSED:
            h.state = state
        # A CLOSED generator keeps `serving` set, so turns pinned to it wake and re-pin.
        if h.state is _GeneratorState.SYNCING:
            h.serving.clear()
        else:
            h.serving.set()
        self._refresh_serving_status()

    def _mark_closed(self, h: _GeneratorHandle) -> None:
        """Stop routing to a generator for good after it raised ``GeneratorClosedError``."""

        if h.state is _GeneratorState.CLOSED:
            return
        self._set_state(h, _GeneratorState.CLOSED)
        num_left = sum(
            other.state is not _GeneratorState.CLOSED for other in self._generators
        )
        # TODO: export the number of closed generators as a metric. Capacity drops to (N-1)/N with
        # only this log line to show it, and any engine-loop exception closes a generator, even one
        # rejected prompt.
        logger.error(
            f"Generator {self._generators.index(h)} is closed; its own log has the engine loop "
            f"traceback. Routing around it: {num_left} of {len(self._generators)} generators left."
        )

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
        # Re-check after each wake: `_serving` may be cleared again before this waiter runs.
        while not (candidates := self._candidates()):
            if self._all_closed():
                raise RuntimeError(
                    f"All {len(self._generators)} generators are closed; cannot route {method}"
                )
            await self._serving.wait()
        h = self._strategy.choose(routing_ctx, candidates)
        self._reserve(h, routing_ctx.estimated_cost)
        try:
            return await getattr(h.rank0_actor, method).call_one(*args, **kwargs)
        except ActorError as error:
            if isinstance(error.exception, GeneratorClosedError):
                self._mark_closed(h)
            raise
        finally:
            self._release(h, routing_ctx.estimated_cost)

    async def _release_session(self, group_id: int, session_id: str) -> None:
        """Drop a session's affinity, and tell its generator to release the session's held KV."""
        self._group_sessions.get(group_id, set()).discard(session_id)
        pinned = self._strategy.pinned_candidate(RoutingContext(session_id=session_id))
        self._strategy.release_session(session_id)
        # Without a pin (non-sticky routing), `release_groups` frees the held KV at group end.
        # A closed generator holds no KV, and its release raises on the closed engine-loop queue.
        if (
            self._forward_session_releases
            and pinned is not None
            and pinned.state is not _GeneratorState.CLOSED
        ):
            await pinned.rank0_actor.release_sessions.call_one([session_id])

    async def _release_groups(self, group_ids: list[int]) -> None:
        """Drop the affinity of every session left in these groups, then tell every generator that
        is not closed. Raises once all are closed."""
        for group_id in group_ids:
            for session_id in self._group_sessions.pop(group_id, ()):
                self._strategy.release_session(session_id)
        if self._all_closed():
            raise RuntimeError(f"All {len(self._generators)} generators are closed")
        # A closed generator holds no KV, and its release raises on the closed engine-loop queue.
        await asyncio.gather(
            *[
                h.actor.release_groups.call(group_ids)
                for h in self._generators
                if h.state is not _GeneratorState.CLOSED
            ]
        )

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
        """Pull the given policy version's state dict into every generator that is not closed.

        Args:
            policy_version: Trainer policy version whose state dict to pull.
        """

        async def _pull_one(h: _GeneratorHandle) -> None:
            if h.state is _GeneratorState.CLOSED:
                return
            # Transfer over RDMA while the generator remains available
            await h.actor.prefetch_model_state_dict.call()
            if self._config.hot_swap:
                # Hot swap: pull concurrently with in-flight generation, without
                # draining. Whether the pull is genuinely concurrent and safe is
                # up to the generator's implementation.
                await _pull_or_mark_closed(h)
            else:
                # Drain: stop routing to this generator and wait for in-flight
                # work to finish before pulling, then re-admit it.
                # Draining starts only for the local CPU-to-GPU apply rather
                # than the network transfer
                self._set_state(h, _GeneratorState.SYNCING)
                try:
                    with sl.log_trace_span("router_drain_wait"):
                        await h.idle.wait()
                    await _pull_or_mark_closed(h)
                finally:
                    self._set_state(h, _GeneratorState.SERVING)

        async def _pull_or_mark_closed(h: _GeneratorHandle) -> None:
            # Runs inside the drain's `try`, so h is CLOSED before its `finally` re-admits it.
            try:
                await h.rank0_actor.pull_model_state_dict.call_one(policy_version)
            except ActorError as error:
                if not isinstance(error.exception, GeneratorClosedError):
                    raise
                self._mark_closed(h)

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

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        """Set the step counter in this process and in every generator rank."""
        sl.set_step(step)
        await self._fanout("sync_log_step", step)

    @concurrent_endpoint
    async def release_session(self, group_id: int, routing_session_id: str) -> None:
        """Forget a routing session after its rollout's last generation call, and tell its
        generator to release the session's held KV."""
        await self._release_session(group_id, routing_session_id)

    @concurrent_endpoint
    async def release_groups(self, group_ids: list[int]) -> None:
        """Tell every generator that these rollout groups are finished. Raises once all are closed."""
        # The logic lives in a private method so tests can call it without an actor mesh.
        await self._release_groups(group_ids)

    @concurrent_endpoint
    async def pull_model_state_dict(self, policy_version: int) -> None:
        """Pull the given policy version's state dict into every generator that is not closed."""
        # Wrapper the logic in a private method so we can test it independently
        # without the need to spawn the Monarch actor mesh.
        await self._pull_model_state_dict(policy_version=policy_version)

    @concurrent_endpoint
    async def close_generators(self) -> list[Any | BaseException]:
        """Close every generator, returning each one's result or exception."""
        return await self._fanout("close", return_exceptions=True)
