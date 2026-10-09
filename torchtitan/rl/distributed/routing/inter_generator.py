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
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any

from monarch.actor import Actor, concurrent_endpoint, current_size

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.distributed.routing.admission import AdmissionPolicy
from torchtitan.rl.distributed.routing.strategies import (
    RoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.distributed.routing.types import RoutingCandidate, RoutingContext

logger = logging.getLogger(__name__)

# prime-rl polls its engines every 5 s, each with a 5 s timeout
# (prime-rl@0bcb868e9, src/prime_rl/orchestrator/inference_metrics.py:17-18).
_ENGINE_LOAD_POLL_INTERVAL_S = 5.0
_ENGINE_LOAD_TIMEOUT_S = 5.0


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


@dataclass(kw_only=True, slots=True)
class _PlacedGroup:
    """A rollout group the admission policy placed on one generator."""

    handle: _GeneratorHandle

    num_completion_tokens: int = 0
    """Tokens the group's calls generated so far."""


@dataclass(kw_only=True, slots=True)
class _WaitingGroup:
    """A new rollout group waiting for the admission policy."""

    num_prompt_tokens: int
    """Prompt length of the group's first call."""

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

        admission: AdmissionPolicy.Config | None = None
        """When each new rollout group starts, and on which generator. ``None``
        (default) starts every group at once and routes each call with ``strategy``.
        With a policy, ``strategy`` is unused: each group runs on one generator, and
        a new group waits (FIFO) until the policy admits it; turns of placed groups
        never wait, nor do validation groups (negative ids). Meant for multi-turn
        rollouts, whose sessions keep KV between turns. Options:
        ``KVEstimateAdmission.Config()`` (the router's own KV block estimate),
        ``KVGrowthEstimateAdmission.Config()`` (that estimate plus each session's
        expected growth) and ``KVUsageAdmission.Config()`` (vLLM's measured KV
        usage, prime-rl's controller)."""

    def __init__(
        self,
        config: Config,
        *,
        generators: Sequence[Any],
        enable_cpu_weight_prefetch: bool,
        group_size: int,
        forward_session_releases: bool = False,
    ):
        num_actors = math.prod(current_size().values())
        assert (
            num_actors == 1
        ), f"InterGeneratorRouter must be a singleton, but its mesh holds {num_actors} actors"

        self._config = config
        self._enable_cpu_weight_prefetch = enable_cpu_weight_prefetch
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

        # Admission state, used when `admission` is set. `start_engine_loop` builds the policy
        # from the generators' KV budgets and starts polling their loads.
        self._group_size = group_size
        self._admission: AdmissionPolicy | None = None
        # KV usage at each generator's last answered poll, for the per-step log line.
        self._kv_usage: list[float | None] = [None] * len(self._generators)
        self._poll_task: asyncio.Task | None = None
        self._placed_groups: dict[int, _PlacedGroup] = {}
        # New groups in arrival order; admitted first-in, first-out.
        self._waiting_groups: dict[int, _WaitingGroup] = {}

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

    def _serving_indices(self) -> list[int]:
        """Return the indices of the generators that are currently routable."""
        return [
            i
            for i, h in enumerate(self._generators)
            if h.state is _GeneratorState.SERVING
        ]

    def _admit_waiting_groups(self) -> None:
        """Place waiting groups, oldest first, while the admission policy admits the oldest."""
        serving = self._serving_indices()
        while self._waiting_groups and serving:
            group_id, waiting = next(iter(self._waiting_groups.items()))
            generator = self._admission.admit(
                group_id,
                num_prompt_tokens=waiting.num_prompt_tokens,
                serving=serving,
                skip_queue=False,
            )
            if generator is None:
                return
            del self._waiting_groups[group_id]
            placed = _PlacedGroup(handle=self._generators[generator])
            self._placed_groups[group_id] = placed
            waiting.admitted.set_result(placed)

    async def _admit_group(self, group_id: int, num_tokens: int) -> _PlacedGroup:
        """Return the group's placement; a new group first waits for the admission policy.

        Args:
            group_id: Rollout group of the call.
            num_tokens: The call's prompt length.
        """
        placed = self._placed_groups.get(group_id)
        if placed is not None:
            return placed
        assert self._admission is not None, "start_engine_loop builds the policy"
        if group_id < 0:
            # Validation groups (negative ids, see the controller) skip the queue: the
            # controller blocks on validation.
            await self._serving.wait()
            # A sibling may have placed the group while this call waited.
            placed = self._placed_groups.get(group_id)
            if placed is not None:
                return placed
            generator = self._admission.admit(
                group_id,
                num_prompt_tokens=num_tokens,
                serving=self._serving_indices(),
                skip_queue=True,
            )
            placed = _PlacedGroup(handle=self._generators[generator])
            self._placed_groups[group_id] = placed
            return placed
        waiting = self._waiting_groups.get(group_id)
        if waiting is None:
            waiting = _WaitingGroup(
                num_prompt_tokens=num_tokens,
                admitted=asyncio.get_running_loop().create_future(),
            )
            self._waiting_groups[group_id] = waiting
            self._admit_waiting_groups()
        with sl.log_trace_span("router_admission_wait"):
            # Shielded: one cancelled sibling must not cancel the group's admission.
            return await asyncio.shield(waiting.admitted)

    def _set_session_tokens(
        self,
        group_id: int,
        placed: _PlacedGroup,
        session_id: str | None,
        num_tokens: int | None,
    ) -> None:
        """Tell the admission policy a session's context length; ``None`` ends the session."""
        if self._placed_groups.get(group_id) is not placed:
            return  # The group was released while this call ran.
        self._admission.set_session_tokens(
            group_id, session_id=session_id, num_tokens=num_tokens
        )
        self._admit_waiting_groups()

    async def _generate_with_admission(
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

        The policy sees the session's prompt while the call runs (vLLM admits what it generates),
        then its prompt plus completion.
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
        num_completion_tokens = len(completion.token_ids)
        placed.num_completion_tokens += num_completion_tokens
        self._set_session_tokens(
            group_id,
            placed,
            routing_session_id,
            num_prompt_tokens + num_completion_tokens,
        )
        return completion

    async def _start_admission(self) -> None:
        """Build the admission policy from every generator's KV budget, and read their loads once."""
        budgets = await asyncio.gather(
            *[h.rank0_actor.kv_cache_budget.call_one() for h in self._generators]
        )
        self._admission = self._config.admission.build(
            budgets=budgets, group_size=self._group_size
        )
        await self._read_engine_loads()

    async def _read_engine_loads(self) -> None:
        """Pass the load of each generator that answers within 5 s to the admission policy, then
        admit what it allows. A generator that fails or times out is left out of this poll."""
        results = await asyncio.gather(
            *[
                asyncio.wait_for(
                    h.rank0_actor.engine_load.call_one(), _ENGINE_LOAD_TIMEOUT_S
                )
                for h in self._generators
            ],
            return_exceptions=True,
        )
        loads = {}
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning("Reading generator %d's load failed: %r", i, result)
            else:
                loads[i] = result
                self._kv_usage[i] = result.kv_usage
        if loads:
            self._admission.observe(loads)
        self._admit_waiting_groups()

    async def _poll_engine_loads(self) -> None:
        """Read every generator's load every 5 s."""
        while True:
            await asyncio.sleep(_ENGINE_LOAD_POLL_INTERVAL_S)
            try:
                await self._read_engine_loads()
            except Exception:
                # Keep polling: the policy's trims and cuts wait on the next read.
                logger.exception("Admission poll failed")

    def _release_session(self, group_id: int, session_id: str) -> None:
        """Drop a session's affinity and end it in the admission policy, if any."""
        self._group_sessions.get(group_id, set()).discard(session_id)
        self._strategy.release_session(session_id)
        placed = self._placed_groups.get(group_id)
        if placed is not None:
            self._set_session_tokens(group_id, placed, session_id, None)

    def _release_groups(self, group_ids: list[int]) -> None:
        """Drop the affinity of every session left in these groups, and release them from the
        admission policy."""
        for group_id in group_ids:
            for session_id in self._group_sessions.pop(group_id, ()):
                self._strategy.release_session(session_id)
            placed = self._placed_groups.pop(group_id, None)
            if placed is not None:
                self._admission.release(
                    group_id, num_completion_tokens=placed.num_completion_tokens
                )
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
        if self._config.admission is not None:
            return await self._generate_with_admission(
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
            if self._enable_cpu_weight_prefetch:
                # Transfer over RDMA while the generator remains available.
                await h.actor.prefetch_model_state_dict.call()
            if self._config.hot_swap:
                # Hot swap: pull concurrently with in-flight generation, without
                # draining. Whether the pull is genuinely concurrent and safe is
                # up to the generator's implementation.
                await h.rank0_actor.pull_model_state_dict.call_one(policy_version)
            else:
                # Drain: stop routing to this generator and wait for in-flight
                # work to finish before pulling, then re-admit it.
                # With CPU prefetch enabled, draining starts only for the local
                # CPU-to-GPU apply rather than the network transfer.
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
        if self._config.admission is not None:
            await self._start_admission()
            self._poll_task = asyncio.create_task(self._poll_engine_loads())

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        """Set the step counter in this process and in every generator rank, and log admission."""
        sl.set_step(step)
        # Experiment-only: router memory and in-flight calls, to diagnose a router crash.
        rss_gib = (
            int(open("/proc/self/statm").read().split()[1])
            * os.sysconf("SC_PAGE_SIZE")
            / 2**30
        )
        logger.info(
            "Router at step %d: RSS %.2f GiB, %d calls in flight, %d tracked groups",
            step,
            rss_gib,
            sum(h.reserved_load for h in self._generators),
            len(self._group_sessions),
        )
        if self._admission is not None:
            logger.info(
                "Admission at step %d: %d groups placed, %d waiting, "
                "KV usage per generator %s; %s",
                step,
                len(self._placed_groups),
                len(self._waiting_groups),
                [
                    None if usage is None else round(usage, 2)
                    for usage in self._kv_usage
                ],
                self._admission.summary(),
            )
        await self._fanout("sync_log_step", step)

    @concurrent_endpoint
    async def release_session(self, group_id: int, routing_session_id: str) -> None:
        """Forget a routing session after its rollout's last generation call, and tell its
        generator to release the session's held KV."""
        placed = self._placed_groups.get(group_id)
        self._release_session(group_id, routing_session_id)
        # The session's generator lets go of its held KV (`hold_session_kv`). Without an admission
        # policy the router doesn't know which generator holds it; `release_groups` frees it at group end.
        if self._forward_session_releases and placed is not None:
            await placed.handle.rank0_actor.release_sessions.call_one(
                [routing_session_id]
            )

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
        if self._poll_task is not None:
            self._poll_task.cancel()
        return await self._fanout("close", return_exceptions=True)
