# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest

from torchtitan.rl.actors.system import ServiceHandle
from torchtitan.rl.generator.engine import KVCacheBudget
from torchtitan.rl.routing.admission import KVRoomAdmission
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
    StickySessionRoutingStrategy,
)


class _Endpoint:
    def __init__(self, *, wait: bool = False, raises: bool = False, returns=None):
        self.raises = raises
        self.returns = returns
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
        return self.returns

    async def call(self, *args, **kwargs):
        return await self.call_one(*args, **kwargs)


class _Actor:
    """A single-rank generator mesh fake."""

    def __init__(
        self,
        name: str,
        *,
        wait_prefetch: bool = False,
        wait_pull: bool = False,
        raises_pull: bool = False,
    ):
        self.name = name
        self.prefetch_model_state_dict = _Endpoint(wait=wait_prefetch)
        self.load_model_state_dict = _Endpoint(wait=wait_pull, raises=raises_pull)
        self.release_sessions = _Endpoint()
        self.release_groups = _Endpoint()
        self.start_engine_loop = _Endpoint()
        self.close = _Endpoint()
        # Qwen3.8-27B with MTP-3 on one GB300: 3,030 usable blocks; a max-length session charges 177,
        # a 4,000-token one 22.
        self.kv_cache_budget = _Endpoint(
            returns=KVCacheBudget(
                num_blocks=3411,
                free_floor_blocks=381,
                attention_block_sizes=(1664,),
                num_mamba_groups=3,
                mamba_blocks_per_running_request=4,
                max_model_len=262144,
                data_parallel_size=1,
            )
        )


def _router(
    actors, *, strategy=None, hot_swap=False, admission=None
) -> InterGeneratorRouter:
    return InterGeneratorRouter(
        InterGeneratorRouter.Config(
            strategy=strategy or LeastLoadedRoutingStrategy.Config(),
            hot_swap=hot_swap,
            admission=admission,
        ),
        generators=[ServiceHandle(all_ranks=actor, rank0=actor) for actor in actors],
    )


def _reserve(router, request_id: str, *, session_id: str | None = None) -> str:
    """Reserve a generator for one generate call of group 0; return its name."""
    return asyncio.run(
        router.reserve(
            request_id, routing_session_id=session_id, group_id=0, num_prompt_tokens=1
        )
    ).name


def _route(router, *, session_id: str | None = None) -> str:
    """Route one generate call that finishes at once; return its generator's name."""
    name = _reserve(router, "probe", session_id=session_id)
    router.release_reservation("probe", num_tokens=None)
    return name


def test_least_loaded_routes_to_lowest_reserved_load():
    router = _router([_Actor("gen0"), _Actor("gen1")])
    assert _reserve(router, "r0") == "gen0"
    assert _reserve(router, "r1") == "gen1"

    # Releasing r1 frees gen1, the generator it reserved.
    router.release_reservation("r1", num_tokens=None)
    assert [h.reserved_load for h in router._generators] == [1, 0]
    # The tie-break alone would pick gen0 (least recently chosen); load picks idle gen1.
    assert _reserve(router, "r2") == "gen1"

    router.release_reservation("r0", num_tokens=None)
    router.release_reservation("r2", num_tokens=None)
    assert [h.reserved_load for h in router._generators] == [0, 0]


def test_least_loaded_tie_break_spreads_over_a_changing_candidate_set():
    router = _router([_Actor(f"gen{i}") for i in range(4)])

    # gen1 drains for a model state dict load on every other request, so the candidate
    # set alternates between four and three generators. Each call finishes
    # before the next starts, so the survivors are always tied at zero load
    # and only the tie-break decides.
    chosen = []
    for i in range(12):
        router._generators[1].draining = i % 2 == 1
        chosen.append(_route(router))

    assert [chosen.count(f"gen{i}") for i in range(4)] == [3, 3, 3, 3]


def test_round_robin_cycles_through_generators():
    actors = [_Actor("gen0"), _Actor("gen1"), _Actor("gen2")]
    router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())

    # Cycles through all three in order, then wraps back to the first.
    assert [_route(router) for _ in range(4)] == ["gen0", "gen1", "gen2", "gen0"]


def test_round_robin_skips_draining_generators():
    actors = [_Actor("gen0"), _Actor("gen1")]
    router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())
    router._generators[0].draining = True

    assert [_route(router) for _ in range(2)] == ["gen1", "gen1"]


def test_sticky_session_reuses_generator_for_same_session():
    actors = [_Actor("gen0"), _Actor("gen1")]
    router = _router(
        actors, strategy=StickySessionRoutingStrategy.Config(repin_load_ratio=None)
    )

    assert _reserve(router, "r0", session_id="s0") == "gen0"
    # gen0 is busier than idle gen1, yet s0's next call stays on gen0.
    assert _reserve(router, "r1", session_id="s0") == "gen0"


def test_sticky_session_spreads_new_sessions_started_on_idle_generators():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")], strategy=StickySessionRoutingStrategy.Config()
    )

    # Each call finishes before the next one starts, so the least-loaded
    # fallback sees both generators idle when it places either session. The
    # pin is permanent, so the two sessions must not land on one generator.
    first = _route(router, session_id="s0")
    second = _route(router, session_id="s1")
    assert {first, second} == {"gen0", "gen1"}


def test_sticky_session_stays_on_its_draining_generator():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")], strategy=StickySessionRoutingStrategy.Config()
    )

    assert _route(router, session_id="s0") == "gen0"

    # gen0 drains for a model state dict load, with 3 turns held back vs 0 on gen1:
    # overloaded by the default 1.5x ratio, yet s0's next turn still goes to gen0.
    router._generators[0].draining = True
    router._generators[0].reserved_load = 3
    assert _route(router, session_id="s0") == "gen0"

    # s0 is still pinned to gen0 after the drain.
    router._generators[0].draining = False
    router._generators[0].reserved_load = 0
    assert _route(router, session_id="s0") == "gen0"


def test_new_sessions_spread_over_generators_that_all_drain():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")], strategy=StickySessionRoutingStrategy.Config()
    )
    for h in router._generators:
        h.draining = True

    # Every generator drains, so new sessions go to them anyway, spread by load.
    chosen = [_reserve(router, f"r{i}", session_id=f"s{i}") for i in range(4)]
    assert sorted(chosen) == ["gen0", "gen0", "gen1", "gen1"]


def test_sticky_session_repins_off_an_overloaded_generator():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")], strategy=StickySessionRoutingStrategy.Config()
    )

    assert _route(router, session_id="s0") == "gen0"

    # gen0 piles up 3 in-flight calls vs 0 on gen1: 3 > default ratio 1.5 * mean load 1.5.
    router._generators[0].reserved_load = 3
    assert _route(router, session_id="s0") == "gen1"

    # The move is a re-pin: s0 stays on gen1 once gen0 recovers.
    router._generators[0].reserved_load = 0
    assert _route(router, session_id="s0") == "gen1"


def test_sticky_session_can_use_round_robin_for_new_sessions():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")],
        strategy=StickySessionRoutingStrategy.Config(
            fallback_strategy=RoundRobinRoutingStrategy.Config()
        ),
    )

    assert _route(router, session_id="s0") == "gen0"
    assert _route(router, session_id="s1") == "gen1"
    assert _route(router, session_id="s0") == "gen0"
    assert _route(router, session_id="s2") == "gen0"


def test_sticky_session_without_session_id_uses_fallback_without_affinity():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")],
        strategy=StickySessionRoutingStrategy.Config(
            fallback_strategy=RoundRobinRoutingStrategy.Config()
        ),
    )

    assert [_route(router) for _ in range(3)] == ["gen0", "gen1", "gen0"]


def test_sticky_session_respects_max_sessions():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")],
        # No repinning, which would also move s0 off its busy generator.
        strategy=StickySessionRoutingStrategy.Config(
            max_sessions=1, repin_load_ratio=None
        ),
    )

    assert _reserve(router, "r0", session_id="s0") == "gen0"
    # s0 is pinned to gen0 and still in flight, so the least-loaded fallback
    # assigns the new s1 session to gen1. Since max_sessions=1, s1 evicts s0.
    assert _route(router, session_id="s1") == "gen1"
    # Evicted, s0 is a new session: least-loaded picks idle gen1 over busy gen0.
    assert _route(router, session_id="s0") == "gen1"


def test_release_session_unpins_it_and_frees_its_kv_on_its_generator():
    actors = [_Actor("gen0"), _Actor("gen1")]
    # No repinning, which would also move s0 off its busy generator.
    router = _router(
        actors, strategy=StickySessionRoutingStrategy.Config(repin_load_ratio=None)
    )
    assert _reserve(router, "r0", session_id="s0") == "gen0"

    asyncio.run(router.release_session("s0", group_id=0))

    assert actors[0].release_sessions.calls == [((["s0"],), {})]
    assert actors[1].release_sessions.calls == []
    # Unpinned, s0 is a new session: least-loaded picks idle gen1 over busy gen0.
    assert _route(router, session_id="s0") == "gen1"


def test_sticky_session_rejects_non_positive_max_sessions():
    with pytest.raises(ValueError, match="max_sessions must be positive"):
        StickySessionRoutingStrategy.Config(max_sessions=0).build()


def test_sticky_session_rejects_repin_load_ratio_at_most_one():
    with pytest.raises(ValueError, match="repin_load_ratio must be > 1"):
        StickySessionRoutingStrategy.Config(repin_load_ratio=1.0)


def test_drain_marks_generator_draining_for_its_load():
    async def _run():
        actors = [_Actor("gen0", wait_pull=True), _Actor("gen1")]
        router = _router(actors)

        pull_task = asyncio.create_task(router.pull_model_state_dict(1))
        await asyncio.wait_for(
            actors[0].load_model_state_dict.started.wait(), timeout=1.0
        )
        # gen1's pull doesn't wait for gen0's.
        await asyncio.wait_for(
            actors[1].load_model_state_dict.started.wait(), timeout=1.0
        )
        assert [h.draining for h in router._generators] == [True, False]
        # Even when gen1 is busier, an unpinned call skips the draining gen0.
        router._generators[1].reserved_load = 2
        assert await _reserve_call(router, "r0", session_id=None, group_id=0) == "gen1"

        actors[0].load_model_state_dict.release.set()
        await pull_task
        assert [h.draining for h in router._generators] == [False, False]

    asyncio.run(_run())


def test_hot_swap_keeps_generators_serving_during_pull():
    async def _run():
        actor = _Actor("gen0", wait_pull=True)
        router = _router([actor], hot_swap=True)

        pull_task = asyncio.create_task(router.pull_model_state_dict(3))
        await actor.load_model_state_dict.started.wait()

        # Hot swap does not drain the generator, so it keeps serving.
        assert not router._generators[0].draining

        actor.load_model_state_dict.release.set()
        await pull_task
        assert actor.load_model_state_dict.calls == [((3,), {"drain": False})]

    asyncio.run(_run())


def test_prefetch_keeps_generator_serving_before_drain():
    async def _run():
        actor = _Actor("gen0", wait_prefetch=True)
        router = _router([actor])

        pull_task = asyncio.create_task(router.pull_model_state_dict(3))
        await actor.prefetch_model_state_dict.started.wait()

        assert not router._generators[0].draining
        assert actor.load_model_state_dict.calls == []

        actor.prefetch_model_state_dict.release.set()
        await pull_task
        assert actor.prefetch_model_state_dict.calls == [((), {})]

    asyncio.run(_run())


def test_drain_clears_draining_on_pull_failure():
    async def _run():
        actor = _Actor("gen0", raises_pull=True)
        router = _router([actor])

        with pytest.raises(RuntimeError, match="endpoint failed"):
            await router.pull_model_state_dict(1)

        assert not router._generators[0].draining

    asyncio.run(_run())


def test_pull_model_state_dict_pulls_every_generator():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors)

        await router.pull_model_state_dict(7)

        assert all(len(actor.prefetch_model_state_dict.calls) == 1 for actor in actors)
        assert [actor.load_model_state_dict.calls for actor in actors] == [
            [((7,), {"drain": True})],
            [((7,), {"drain": True})],
        ]

    asyncio.run(_run())


class _StreamEndpoint:
    """An endpoint whose ``stream()`` yields one reply per rank, in arrival order."""

    def __init__(self, *replies):
        self.replies = replies

    def stream(self):
        return (reply() for reply in self.replies)


def test_wait_engine_loop_raises_when_one_rank_crashes_while_its_peer_hangs():
    async def _run():
        peer_blocked = asyncio.Event()

        async def crashed():
            raise RuntimeError("engine loop crashed")

        async def blocked():
            # A TP peer of the crashed rank stays in a collective and never replies.
            await peer_blocked.wait()

        crashing, healthy = _Actor("gen0"), _Actor("gen1")
        crashing.wait_engine_loop = _StreamEndpoint(crashed, blocked)
        healthy.wait_engine_loop = _StreamEndpoint(blocked)
        router = _router([crashing, healthy])

        with pytest.raises(RuntimeError, match="engine loop crashed"):
            await asyncio.wait_for(router.wait_engine_loop(), timeout=5)

    asyncio.run(_run())


def test_wait_engine_loop_returns_once_every_rank_exits():
    async def _run():
        async def exited():
            return None

        actors = [_Actor("gen0"), _Actor("gen1")]
        for actor in actors:
            actor.wait_engine_loop = _StreamEndpoint(exited, exited)
        router = _router(actors)

        await asyncio.wait_for(router.wait_engine_loop(), timeout=5)

    asyncio.run(_run())


async def _reserve_call(
    router,
    request_id: str,
    *,
    session_id: str | None,
    group_id: int,
    num_prompt_tokens: int = 4000,
) -> str:
    """Reserve a generator for one call of `group_id`; return its name."""
    generator = await router.reserve(
        request_id,
        routing_session_id=session_id,
        group_id=group_id,
        num_prompt_tokens=num_prompt_tokens,
    )
    return generator.name


def test_admission_holds_a_group_until_room_and_routes_its_calls_to_its_generator():
    actors = [_Actor("gen0"), _Actor("gen1")]
    router = _router(actors, admission=KVRoomAdmission.Config())

    async def main():
        await router.start_engine_loop()
        # No first prompt seen yet: each generator takes one group (16 x 177 = 2,832 of 2,878).
        await router.acquire_group(0, group_size=16, validation=False)
        await router.acquire_group(1, group_size=16, validation=False)
        waiting = asyncio.create_task(
            router.acquire_group(2, group_size=16, validation=False)
        )
        await asyncio.sleep(0)
        assert not waiting.done()

        # Group 1 sits on gen1, though least-loaded routing would pick idle gen0 here.
        assert (
            await _reserve_call(router, "r0", session_id="g0/s0", group_id=0) == "gen0"
        )
        assert (
            await _reserve_call(router, "r1", session_id="g1/s0", group_id=1) == "gen1"
        )
        # The release goes to the generator the group was placed on.
        router.release_reservation("r1", num_tokens=4100)
        await router.release_session("g1/s0", group_id=1)
        assert actors[1].release_sessions.calls == [((["g1/s0"],), {})]

        # The first calls showed 4,000-token prompts, so a group now charges 16 x 22 = 352 and
        # group 2 fits: on gen1, which has more room (15 sessions not started vs 16 on gen0).
        await asyncio.wait_for(waiting, timeout=5)
        assert (
            await _reserve_call(router, "r2", session_id="g2/s0", group_id=2) == "gen1"
        )
        await router.close_generators()

    asyncio.run(main())


def test_release_groups_drops_a_group_still_waiting():
    router = _router([_Actor("gen0")], admission=KVRoomAdmission.Config())

    async def main():
        await router.start_engine_loop()
        await router.acquire_group(0, group_size=16, validation=False)
        waiting = asyncio.create_task(
            router.acquire_group(1, group_size=16, validation=False)
        )
        await asyncio.sleep(0)

        await router.release_groups([1])

        with pytest.raises(asyncio.CancelledError):
            await waiting
        metrics = {
            metric.key: metric.value.value for metric in router.admission_metrics()
        }
        assert metrics["admission/groups_waiting"] == 0
        await router.close_generators()

    asyncio.run(main())


def test_admission_metrics_raise_once_the_tick_task_died(monkeypatch):
    router = _router([_Actor("gen0")], admission=KVRoomAdmission.Config())

    async def main():
        await router.start_engine_loop()
        await router.acquire_group(0, group_size=16, validation=False)
        waiting = asyncio.create_task(
            router.acquire_group(1, group_size=16, validation=False)
        )
        await asyncio.sleep(0)

        def broken_place():
            raise RuntimeError("place failed")

        monkeypatch.setattr(router._admission, "place", broken_place)
        await asyncio.wait_for(asyncio.wait([router._admission_task]), timeout=5)

        # Nothing would wake the waiting group: it fails, and so does every later endpoint call.
        with pytest.raises(RuntimeError, match="admission tick task stopped"):
            await asyncio.wait_for(waiting, timeout=5)
        with pytest.raises(RuntimeError, match="place failed"):
            router.admission_metrics()
        with pytest.raises(RuntimeError, match="place failed"):
            await _reserve_call(router, "r0", session_id="g0/s0", group_id=0)
        await router.close_generators()

    asyncio.run(main())


async def _router_over_its_kv_cache(actor) -> InterGeneratorRouter:
    """One generator over its 3,030 usable blocks: 21 idle sessions of ~240k tokens, 148 blocks each.
    A tick pauses the smallest ones; g0/s19 and g0/s20 stay paused (`KVRoomAdmission.rebalance`)."""
    router = _router([actor], admission=KVRoomAdmission.Config())
    await router.start_engine_loop()
    await router.acquire_group(0, group_size=21, validation=False)
    for i in range(21):
        await _reserve_call(
            router,
            f"r{i}",
            session_id=f"g0/s{i}",
            group_id=0,
            num_prompt_tokens=240_000 - i,
        )
        router.release_reservation(f"r{i}", num_tokens=240_000 - i)
    router._admission_tick()
    return router


def test_a_paused_sessions_call_waits_until_the_session_resumes():
    actor = _Actor("gen0")

    async def main():
        router = await _router_over_its_kv_cache(actor)
        call = asyncio.create_task(
            _reserve_call(router, "next", session_id="g0/s20", group_id=0)
        )
        await asyncio.sleep(0)
        assert not call.done()

        # Two rollouts end: the paused sessions fit again, and the waiting call runs.
        await router.release_session("g0/s0", group_id=0)
        await router.release_session("g0/s1", group_id=0)

        assert await asyncio.wait_for(call, timeout=5) == "gen0"
        await router.close_generators()

    asyncio.run(main())


def test_cancel_reserve_drops_a_waiting_call():
    async def main():
        router = await _router_over_its_kv_cache(_Actor("gen0"))
        call = asyncio.create_task(
            _reserve_call(router, "next", session_id="g0/s20", group_id=0)
        )
        await asyncio.sleep(0)

        router.cancel_reserve("next")
        router.cancel_reserve("unknown")  # ignored
        router.release_reservation("unknown", num_tokens=None)  # ignored

        with pytest.raises(asyncio.CancelledError):
            await call
        await router.release_session("g0/s0", group_id=0)
        await router.release_session("g0/s1", group_id=0)
        # The session resumes idle: no call was left to run, and nothing is reserved.
        assert router._reservations == {}
        assert router._generators[0].reserved_load == 0
        await router.close_generators()

    asyncio.run(main())


def test_cancel_reserve_after_the_resume_leaves_no_reservation():
    async def main():
        router = await _router_over_its_kv_cache(_Actor("gen0"))
        call = asyncio.create_task(
            _reserve_call(
                router,
                "next",
                session_id="g0/s20",
                group_id=0,
                num_prompt_tokens=240_000,
            )
        )
        await asyncio.sleep(0)
        ledger_before = router._admission.ledger(0)

        # The resume and the cancel land before the waiting call wakes up.
        router._admission.release_session("g0/s0", group_id=0)
        router._admission.release_session("g0/s1", group_id=0)
        router._admission_tick()
        router.cancel_reserve("next")

        with pytest.raises(asyncio.CancelledError):
            await call
        assert router._reservations == {}
        # g0/s20 resumed idle, not running: 296 freed, 148 + 148 back (g0/s19 and s20).
        assert router._admission.ledger(0) == ledger_before
        await router.close_generators()

    asyncio.run(main())


def test_releasing_a_paused_session_fails_its_waiting_call():
    async def main():
        router = await _router_over_its_kv_cache(_Actor("gen0"))
        call = asyncio.create_task(
            _reserve_call(router, "next", session_id="g0/s20", group_id=0)
        )
        await asyncio.sleep(0)

        await router.release_groups([0])

        with pytest.raises(RuntimeError, match="released while call 'next' waited"):
            await call
        await router.close_generators()

    asyncio.run(main())


def test_a_call_of_a_group_admission_never_placed_raises():
    async def main():
        router = await _router_over_its_kv_cache(_Actor("gen0"))
        with pytest.raises(ValueError, match="group 5 is not placed"):
            await _reserve_call(router, "late", session_id="g5/s0", group_id=5)
        await router.close_generators()

    asyncio.run(main())


def test_a_cancel_before_a_resume_leaves_the_session_idle():
    async def main():
        router = await _router_over_its_kv_cache(_Actor("gen0"))
        call = asyncio.create_task(
            _reserve_call(
                router,
                "next",
                session_id="g0/s20",
                group_id=0,
                num_prompt_tokens=240_000,
            )
        )
        await asyncio.sleep(0)

        # The caller leaves; then, before the waiting call wakes, a release resumes the session.
        router.cancel_reserve("next")
        router._admission.release_session("g0/s0", group_id=0)
        router._admission.release_session("g0/s1", group_id=0)
        router._admission_tick()

        with pytest.raises(asyncio.CancelledError):
            await call
        # g0/s19 and s20 resumed idle (148 each): 2,812 - 296 + 296.
        assert router._admission.ledger(0) == 2812
        await router.release_groups([0])
        await router.close_generators()

    asyncio.run(main())


def test_a_waiting_call_whose_task_is_cancelled_is_uncharged():
    async def main():
        router = await _router_over_its_kv_cache(_Actor("gen0"))
        call = asyncio.create_task(
            _reserve_call(
                router,
                "next",
                session_id="g0/s20",
                group_id=0,
                num_prompt_tokens=240_000,
            )
        )
        await asyncio.sleep(0)

        call.cancel()  # Monarch cancels the endpoint task, no cancel_reserve
        with pytest.raises(asyncio.CancelledError):
            await call
        await router.release_session("g0/s0", group_id=0)
        await router.release_session("g0/s1", group_id=0)

        # Both paused sessions resume idle: no call is left to run.
        assert router._admission.ledger(0) == 2812
        assert router._reservations == {}
        await router.close_generators()

    asyncio.run(main())
