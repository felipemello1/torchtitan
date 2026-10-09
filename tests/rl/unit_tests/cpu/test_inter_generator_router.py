# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from types import SimpleNamespace

import pytest

import torchtitan.rl.distributed.routing.admission as admission_module
import torchtitan.rl.distributed.routing.inter_generator as inter_generator_module
from torchtitan.rl.distributed.routing.admission import (
    KVEstimateAdmission,
    KVUsageAdmission,
)
from torchtitan.rl.distributed.routing.inter_generator import (
    _GeneratorState,
    InterGeneratorRouter,
)
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.distributed.routing.types import (
    EngineLoad,
    KVCacheBudget,
    RoutingContext,
)


class _Endpoint:
    def __init__(self, value=None, *, wait: bool = False, raises: bool = False):
        self.value = value
        self.raises = raises
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not wait:
            self.release.set()

    async def call_one(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.started.set()
        await self.release.wait()
        if self.raises:
            raise RuntimeError("endpoint failed")
        return self.value


class _Actor:
    """A single-rank generator mesh fake."""

    def __init__(
        self,
        name: str,
        *,
        wait_generate: bool = False,
        wait_pull: bool = False,
        raises_pull: bool = False,
    ):
        self.generate = _Endpoint(name, wait=wait_generate)
        self.pull_model_state_dict = _Endpoint(None, wait=wait_pull, raises=raises_pull)

    def flatten(self, *args, **kwargs):
        return self

    def slice(self, **kwargs):
        return self

    def __len__(self):
        return 1


def _router(
    actors,
    *,
    strategy=None,
    hot_swap=False,
    admission=None,
) -> InterGeneratorRouter:
    return InterGeneratorRouter(
        InterGeneratorRouter.Config(
            strategy=strategy or LeastLoadedRoutingStrategy.Config(),
            hot_swap=hot_swap,
            admission=admission,
        ),
        generators=actors,
        group_size=1,
    )


def test_least_loaded_routes_to_lowest_reserved_load():
    async def _run():
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_generate=True),
        ]
        router = _router(actors)

        first = asyncio.create_task(
            router._route("generate", routing_ctx=RoutingContext(estimated_cost=3))
        )
        await actors[0].generate.started.wait()

        # gen0 now has reserved load 3, so the next route prefers gen1.
        second = asyncio.create_task(
            router._route("generate", routing_ctx=RoutingContext(estimated_cost=1))
        )
        await actors[1].generate.started.wait()

        actors[0].generate.release.set()
        actors[1].generate.release.set()

        assert await first == "gen0"
        assert await second == "gen1"
        assert [h.reserved_load for h in router._generators] == [0, 0]
        assert all(h.idle.is_set() for h in router._generators)

    asyncio.run(_run())


def test_route_releases_reserved_load_on_failure():
    async def _run():
        actor = _Actor("gen0")
        actor.generate.raises = True
        router = _router([actor])

        with pytest.raises(RuntimeError, match="endpoint failed"):
            await router._route(
                "generate", routing_ctx=RoutingContext(estimated_cost=5)
            )

        assert router._generators[0].reserved_load == 0
        assert router._generators[0].idle.is_set()

    asyncio.run(_run())


def test_least_loaded_tie_break_spreads_over_a_changing_candidate_set():
    async def _run():
        actors = [_Actor(f"gen{i}") for i in range(4)]
        router = _router(actors)

        # gen1 drains for a weight sync on every other request, so the candidate
        # set alternates between four and three generators. Each route finishes
        # before the next starts, so the survivors are always tied at zero load
        # and only the tie-break decides.
        chosen = []
        for i in range(12):
            draining = i % 2 == 1
            if draining:
                router._set_state(router._generators[1], _GeneratorState.SYNCING)
            chosen.append(await router._route("generate", routing_ctx=RoutingContext()))
            if draining:
                router._set_state(router._generators[1], _GeneratorState.SERVING)

        assert [chosen.count(f"gen{i}") for i in range(4)] == [3, 3, 3, 3]

    asyncio.run(_run())


def test_round_robin_cycles_through_generators():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1"), _Actor("gen2")]
        router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())

        results = [
            await router._route("generate", routing_ctx=RoutingContext())
            for _ in range(4)
        ]
        # Cycles through all three in order, then wraps back to the first.
        assert results == ["gen0", "gen1", "gen2", "gen0"]

    asyncio.run(_run())


def test_round_robin_skips_syncing_generators():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())
        router._set_state(router._generators[0], _GeneratorState.SYNCING)

        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen1"
        assert actors[0].generate.calls == []
        assert len(actors[1].generate.calls) == 1

    asyncio.run(_run())


def test_sticky_session_reuses_generator_for_same_session():
    async def _run():
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_generate=True),
        ]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())

        first = asyncio.create_task(
            router._route(
                "generate",
                routing_ctx=RoutingContext(estimated_cost=3, session_id="s0"),
            )
        )
        await actors[0].generate.started.wait()

        second = asyncio.create_task(
            router._route(
                "generate",
                routing_ctx=RoutingContext(estimated_cost=1, session_id="s0"),
            )
        )
        await asyncio.sleep(0)

        assert len(actors[0].generate.calls) == 2
        assert actors[1].generate.calls == []

        actors[0].generate.release.set()
        assert await first == "gen0"
        assert await second == "gen0"

    asyncio.run(_run())


def test_sticky_session_spreads_new_sessions_started_on_idle_generators():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())

        # Each route completes before the next one starts, so the least-loaded
        # fallback sees both generators idle when it places either session. The
        # pin is permanent, so the two sessions must not land on one generator.
        first = await router._route(
            "generate", routing_ctx=RoutingContext(session_id="s0")
        )
        second = await router._route(
            "generate", routing_ctx=RoutingContext(session_id="s1")
        )
        assert {first, second} == {"gen0", "gen1"}

    asyncio.run(_run())


def test_sticky_session_waits_for_its_syncing_generator():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())

        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )

        # gen0 drains for a weight sync: s0's next turn waits instead of moving to gen1.
        router._set_state(router._generators[0], _GeneratorState.SYNCING)
        turn = asyncio.create_task(
            router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
        )
        await asyncio.sleep(0)
        assert not turn.done()

        # A new session is not blocked by the drain.
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s1"))
            == "gen1"
        )

        router._set_state(router._generators[0], _GeneratorState.SERVING)
        assert await turn == "gen0"
        assert len(actors[0].generate.calls) == 2

    asyncio.run(_run())


def test_sticky_session_can_use_round_robin_for_new_sessions():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(
            actors,
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=RoundRobinRoutingStrategy.Config()
            ),
        )

        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s1"))
            == "gen1"
        )
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s2"))
            == "gen0"
        )

    asyncio.run(_run())


def test_sticky_session_without_session_id_uses_fallback_without_affinity():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(
            actors,
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=RoundRobinRoutingStrategy.Config()
            ),
        )

        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen0"
        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen1"
        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen0"

    asyncio.run(_run())


def test_sticky_session_respects_max_sessions():
    async def _run():
        actors = [_Actor("gen0", wait_generate=True), _Actor("gen1")]
        router = _router(
            actors,
            strategy=StickySessionRoutingStrategy.Config(max_sessions=1),
        )

        first = asyncio.create_task(
            router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
        )
        await actors[0].generate.started.wait()

        # s0 is pinned to gen0 and still in flight, so the least-loaded fallback
        # assigns the new s1 session to gen1. Since max_sessions=1, s1 evicts s0.
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s1"))
            == "gen1"
        )
        # s0 was evicted from the sticky map, so this route is a new-session
        # fallback. gen0 still has reserved_load from the first request, while
        # gen1 is idle, so least-loaded picks gen1.
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen1"
        )

        actors[0].generate.release.set()
        assert await first == "gen0"

    asyncio.run(_run())


async def _generate(router, *, group_id: int, session_id: str, prompt_tokens: int = 1):
    return await router._generate(
        [0] * prompt_tokens,
        request_id=f"{session_id}/turn",
        group_id=group_id,
        routing_session_id=session_id,
        sampling_config=None,
        metrics_prefix="generator",
    )


@pytest.mark.parametrize("release", ["session", "group"])
def test_sticky_session_release_drops_its_assignment(release):
    async def _run():
        actors = [_Actor("gen0", wait_generate=True), _Actor("gen1")]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())
        first = asyncio.create_task(_generate(router, group_id=0, session_id="s0"))
        await actors[0].generate.started.wait()

        if release == "session":
            router._release_session(group_id=0, session_id="s0")
        else:
            router._release_groups([0])
        # Released, s0 is a new session: least-loaded picks idle gen1 over busy gen0.
        assert await _generate(router, group_id=0, session_id="s0") == "gen1"

        actors[0].generate.release.set()
        assert await first == "gen0"

    asyncio.run(_run())


def test_sticky_session_rejects_non_positive_max_sessions():
    with pytest.raises(ValueError, match="max_sessions must be positive"):
        StickySessionRoutingStrategy.Config(max_sessions=0).build()


def test_drain_excludes_syncing_generator_from_routes():
    async def _run():
        actors = [
            _Actor("gen0", wait_pull=True),
            _Actor("gen1"),
        ]
        router = _router(actors)

        pull_task = asyncio.create_task(router._pull_model_state_dict(policy_version=1))
        await actors[0].pull_model_state_dict.started.wait()

        assert router._generators[0].state is _GeneratorState.SYNCING
        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen1"

        actors[0].pull_model_state_dict.release.set()
        await pull_task
        assert [h.state for h in router._generators] == [
            _GeneratorState.SERVING,
            _GeneratorState.SERVING,
        ]
        assert [actor.pull_model_state_dict.calls for actor in actors] == [
            [((1,), {})],
            [((1,), {})],
        ]

    asyncio.run(_run())


def test_drain_pulls_idle_generators_while_busy_generator_drains():
    async def _run():
        # gen0 is busy generating, so its pull must wait for the in-flight route
        # to drain; gen1 is idle and starts pulling right away.
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_pull=True),
        ]
        router = _router(actors)

        route_task = asyncio.create_task(
            router._route("generate", routing_ctx=RoutingContext())
        )
        await actors[0].generate.started.wait()

        pull_task = asyncio.create_task(router._pull_model_state_dict(policy_version=2))
        await asyncio.wait_for(
            actors[1].pull_model_state_dict.started.wait(), timeout=1.0
        )

        # Both are SYNCING, but gen0's pull is still waiting for its in-flight
        # route to drain, so only gen1 (idle) has started pulling.
        assert [h.state for h in router._generators] == [
            _GeneratorState.SYNCING,
            _GeneratorState.SYNCING,
        ]
        assert actors[0].pull_model_state_dict.calls == []

        # gen1 finishes pulling and is routable again while gen0 still drains.
        actors[1].pull_model_state_dict.release.set()
        assert (
            await asyncio.wait_for(
                router._route("generate", routing_ctx=RoutingContext()),
                timeout=1.0,
            )
            == "gen1"
        )

        # Draining gen0's route lets its pull proceed; both end up pulled.
        actors[0].generate.release.set()
        assert await route_task == "gen0"
        await pull_task
        assert [actor.pull_model_state_dict.calls for actor in actors] == [
            [((2,), {})],
            [((2,), {})],
        ]

    asyncio.run(_run())


def test_hot_swap_keeps_generators_serving_during_pull():
    async def _run():
        actor = _Actor("gen0", wait_pull=True)
        router = _router([actor], hot_swap=True)

        pull_task = asyncio.create_task(router._pull_model_state_dict(policy_version=3))
        await actor.pull_model_state_dict.started.wait()

        # Hot swap does not quiesce the generator, so it keeps serving.
        assert router._generators[0].state is _GeneratorState.SERVING
        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen0"

        actor.pull_model_state_dict.release.set()
        await pull_task
        assert actor.pull_model_state_dict.calls == [((3,), {})]

    asyncio.run(_run())


def test_single_generator_blocks_routes_while_draining():
    async def _run():
        actor = _Actor("gen0", wait_pull=True)
        router = _router([actor])

        pull_task = asyncio.create_task(router._pull_model_state_dict(policy_version=1))
        await actor.pull_model_state_dict.started.wait()

        route_task = asyncio.create_task(
            router._route("generate", routing_ctx=RoutingContext())
        )
        await asyncio.sleep(0)
        assert not route_task.done()
        assert actor.generate.calls == []

        actor.pull_model_state_dict.release.set()
        await pull_task

        assert await route_task == "gen0"

    asyncio.run(_run())


def test_drain_restores_serving_on_pull_failure():
    async def _run():
        actor = _Actor("gen0", raises_pull=True)
        router = _router([actor])

        with pytest.raises(RuntimeError, match="endpoint failed"):
            await router._pull_model_state_dict(policy_version=1)

        assert router._generators[0].state is _GeneratorState.SERVING
        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen0"

    asyncio.run(_run())


def test_pull_model_state_dict_pulls_every_generator():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors)

        await router._pull_model_state_dict(policy_version=7)

        assert [actor.pull_model_state_dict.calls for actor in actors] == [
            [((7,), {})],
            [((7,), {})],
        ]

    asyncio.run(_run())


# KV estimate admission. Each generator holds 10 blocks of 10 tokens, and a session holds one block
# per 10 tokens plus one fixed block: a 20-token prompt holds 3 blocks, then 4 with a 10-token reply.
_BUDGET = KVCacheBudget(
    num_blocks=10,
    block_size=10,
    num_growing_groups=1,
    fixed_blocks_per_session=1,
    max_model_len=100,
)


def _load(kv_usage: float = 0.0) -> EngineLoad:
    return EngineLoad(
        kv_usage=kv_usage,
        num_running=1,
        num_waiting=0,
        num_waiting_for_capacity=0,
        num_preemptions=0,
    )


class _KVActor(_Actor):
    """A generator fake whose completions have ``completion_tokens`` tokens."""

    def __init__(self, name: str, *, completion_tokens: int = 10, **kwargs):
        super().__init__(name, **kwargs)
        self.generate.value = SimpleNamespace(
            name=name, token_ids=[0] * completion_tokens
        )
        self.kv_cache_budget = _Endpoint(_BUDGET)
        self.engine_load = _Endpoint(_load())


async def _kv_router(actors, sessions_per_group=1) -> InterGeneratorRouter:
    router = _router(
        actors,
        admission=KVEstimateAdmission.Config(
            limit=1.0, sessions_per_group=sessions_per_group
        ),
    )
    await router._start_admission()
    return router


async def _kv_generate(
    router, *, group_id: int, session_id: str, prompt_tokens: int = 20
):
    completion = await _generate(
        router, group_id=group_id, session_id=session_id, prompt_tokens=prompt_tokens
    )
    return completion.name


def test_kv_admission_places_new_groups_on_the_roomiest_generator():
    async def _run():
        router = await _kv_router([_KVActor("gen0"), _KVActor("gen1")])

        assert await _kv_generate(router, group_id=0, session_id="g0/r0") == "gen0"
        assert await _kv_generate(router, group_id=1, session_id="g1/r0") == "gen1"
        # Sessions of a placed group stay on its generator.
        assert await _kv_generate(router, group_id=0, session_id="g0/r1") == "gen0"
        assert router._admission.blocks == [8, 4]

    asyncio.run(_run())


def test_kv_admission_new_group_waits_until_a_session_ends():
    async def _run():
        router = await _kv_router([_KVActor("gen0")])
        await _kv_generate(router, group_id=0, session_id="g0/r0")
        await _kv_generate(router, group_id=1, session_id="g1/r0")
        assert router._admission.blocks[0] == 8

        # 8 + 3 > 10 blocks: group 2 waits, and its sibling waits for the same admission.
        waiting = asyncio.create_task(
            _kv_generate(router, group_id=2, session_id="g2/r0")
        )
        sibling = asyncio.create_task(
            _kv_generate(router, group_id=2, session_id="g2/r1")
        )
        await asyncio.sleep(0)
        assert not waiting.done() and list(router._waiting_groups) == [2]

        # Group 0's session ends, which frees its 4 blocks.
        router._release_session(group_id=0, session_id="g0/r0")
        assert await waiting == "gen0"
        assert await sibling == "gen0"

    asyncio.run(_run())


def test_kv_admission_turns_of_placed_groups_never_wait():
    async def _run():
        router = await _kv_router([_KVActor("gen0")])
        # Self-play: white's first turn places the group and fills the generator.
        await _kv_generate(router, group_id=0, session_id="white", prompt_tokens=60)
        assert router._admission.blocks[0] == 8
        waiting = asyncio.create_task(
            _kv_generate(router, group_id=1, session_id="g1/r0")
        )
        await asyncio.sleep(0)
        assert not waiting.done()

        # Black's first turn joins its placed group past the limit, so the game can go on.
        assert await _kv_generate(router, group_id=0, session_id="black") == "gen0"
        assert router._admission.blocks[0] == 8 + 4
        assert not waiting.done()

        router._release_groups([0])
        assert await waiting == "gen0"

    asyncio.run(_run())


def test_kv_admission_counts_sessions_by_their_context_length():
    async def _run():
        router = await _kv_router([_KVActor("gen0", completion_tokens=5)])

        # 20-token prompt: 3 blocks while it runs, then 25 tokens = 4 blocks.
        await _kv_generate(router, group_id=0, session_id="g0/r0")
        assert router._admission.blocks[0] == 4
        # The next turn extends the history: 45 + 5 = 50 tokens = 6 blocks.
        await _kv_generate(router, group_id=0, session_id="g0/r0", prompt_tokens=45)
        assert router._admission.blocks[0] == 6

        router._release_session(group_id=0, session_id="g0/r0")
        assert router._admission.blocks[0] == 0

    asyncio.run(_run())


def test_kv_admission_reserves_the_largest_group_seen_so_far():
    async def _run():
        router = await _kv_router([_KVActor("gen0", completion_tokens=0)])
        await _kv_generate(router, group_id=0, session_id="g0/r0")
        await _kv_generate(router, group_id=0, session_id="g0/r1")
        assert router._admission.blocks[0] == 6

        # Group 1 reserves two sessions of 3 blocks: 6 + 6 > 10 waits, though one session fits.
        waiting = asyncio.create_task(
            _kv_generate(router, group_id=1, session_id="g1/r0")
        )
        await asyncio.sleep(0)
        assert not waiting.done()
        router._release_groups([0])
        assert await waiting == "gen0"

    asyncio.run(_run())


def test_kv_admission_reserves_the_expected_sessions_until_they_start():
    async def _run():
        router = await _kv_router([_KVActor("gen0")], sessions_per_group=2)
        # One session at 4 blocks, plus 3 reserved for the sibling that has not started.
        await _kv_generate(router, group_id=0, session_id="g0/r0")
        assert router._admission.blocks[0] == 7
        await _kv_generate(router, group_id=0, session_id="g0/r1")
        assert router._admission.blocks[0] == 8

    asyncio.run(_run())


def test_kv_admission_empty_generator_takes_a_group_larger_than_its_limit():
    async def _run():
        router = await _kv_router([_KVActor("gen0")])
        # A 200-token prompt holds 21 blocks, over the 10-block limit.
        assert (
            await _kv_generate(
                router, group_id=0, session_id="g0/r0", prompt_tokens=200
            )
            == "gen0"
        )

    asyncio.run(_run())


def test_kv_admission_validation_groups_skip_the_queue():
    async def _run():
        router = await _kv_router([_KVActor("gen0")])
        await _kv_generate(router, group_id=0, session_id="g0/r0", prompt_tokens=60)
        waiting = asyncio.create_task(
            _kv_generate(router, group_id=1, session_id="g1/r0")
        )
        await asyncio.sleep(0)

        # Validation group ids are negative; the trainer waits on them.
        assert await _kv_generate(router, group_id=-1, session_id="v0") == "gen0"
        assert not waiting.done()
        router._release_groups([0, -1])
        assert await waiting == "gen0"

    asyncio.run(_run())


def test_kv_admission_group_released_during_a_call_frees_all_its_blocks():
    async def _run():
        actor = _KVActor("gen0", wait_generate=True)
        router = await _kv_router([actor])
        call = asyncio.create_task(_kv_generate(router, group_id=0, session_id="g0/r0"))
        await actor.generate.started.wait()

        router._release_groups([0])
        actor.generate.release.set()
        await call
        assert router._admission.blocks[0] == 0

    asyncio.run(_run())


def test_no_admission_starts_every_group_at_once():
    async def _run():
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_generate=True),
        ]
        router = _router(actors)
        calls = [
            asyncio.create_task(_generate(router, group_id=g, session_id=f"g{g}/r0"))
            for g in range(6)
        ]
        await asyncio.sleep(0)
        assert [len(actor.generate.calls) for actor in actors] == [3, 3]

        for actor in actors:
            actor.generate.release.set()
        await asyncio.gather(*calls)

    asyncio.run(_run())


def test_usage_admission_places_waiting_groups_on_the_lowest_usage_generator(
    monkeypatch,
):
    now = [0.0]
    monkeypatch.setattr(
        admission_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )

    async def _run():
        actors = [_KVActor("gen0"), _KVActor("gen1")]
        actors[0].engine_load.value = _load(0.6)
        actors[1].engine_load.value = _load(0.2)
        router = _router(actors, admission=KVUsageAdmission.Config(initial_inflight=2))
        await router._start_admission()

        # A cap of 2 admits one new group per 5 s.
        assert await _kv_generate(router, group_id=0, session_id="g0/r0") == "gen1"
        waiting = asyncio.create_task(
            _kv_generate(router, group_id=1, session_id="g1/r0")
        )
        await asyncio.sleep(0)
        assert not waiting.done()
        # The next poll, 5 s later, finds a new window.
        now[0] += 5.0
        await router._read_engine_loads()
        assert await waiting == "gen1"

        # The cap is full: group 2 waits, but a later session of group 0 does not.
        waiting = asyncio.create_task(
            _kv_generate(router, group_id=2, session_id="g2/r0")
        )
        await asyncio.sleep(0)
        assert await _kv_generate(router, group_id=0, session_id="g0/r1") == "gen1"
        assert not waiting.done()

        # Group 0 ends. gen1 already took a group since the last poll, so group 2 goes to gen0.
        router._release_groups([0])
        assert await waiting == "gen0"

    asyncio.run(_run())


def test_load_poll_skips_a_generator_that_fails_or_times_out(monkeypatch):
    monkeypatch.setattr(inter_generator_module, "_ENGINE_LOAD_TIMEOUT_S", 0.01)

    async def _run():
        actors = [_KVActor("gen0"), _KVActor("gen1"), _KVActor("gen2")]
        actors[0].engine_load.value = _load(0.6)
        router = _router(actors, admission=KVUsageAdmission.Config())
        await router._start_admission()

        actors[0].engine_load.value = _load(0.7)
        actors[1].engine_load.raises = True
        actors[2].engine_load.release.clear()  # never answers
        observed = []
        monkeypatch.setattr(router._admission, "observe", observed.append)
        await router._read_engine_loads()
        assert [list(loads) for loads in observed] == [[0]]
        assert router._kv_usage == [0.7, 0.0, 0.0]

    asyncio.run(_run())
