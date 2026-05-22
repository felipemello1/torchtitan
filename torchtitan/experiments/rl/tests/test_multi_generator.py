# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only coverage for multi-generator controller helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchtitan.config import ParallelismConfig
from torchtitan.experiments.rl import grpo
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.types import Completion


def _completion(token_id: int) -> Completion:
    return Completion(
        policy_version=1,
        token_ids=[token_id],
        token_logprobs=[-0.1],
        finish_reason="stop",
    )


def test_generator_routing_is_sticky_for_rollout_session() -> None:
    first_turn = grpo._generator_index_for_request_id(
        "sum_digits/step=7/group=3:sample=1:turn=0",
        num_generators=4,
    )
    second_turn = grpo._generator_index_for_request_id(
        "sum_digits/step=7/group=3:sample=1:turn=1",
        num_generators=4,
    )

    assert 0 <= first_turn < 4
    assert second_turn == first_turn
    assert (
        grpo._generator_index_for_request_id(
            "any-group:sample=0:turn=0",
            num_generators=1,
        )
        == 0
    )


def test_split_and_merge_mixed_generator_sessions_preserves_request_order() -> None:
    async def scenario():
        session_a, session_b = _request_ids_for_distinct_generators()
        request_ids = [
            f"{session_a}:turn=0",
            f"{session_b}:turn=0",
            f"{session_a}:turn=1",
            f"{session_b}:turn=1",
        ]
        prompts = [[10], [20], [11], [21]]
        token_by_request_id = {
            request_id: idx for idx, request_id in enumerate(request_ids)
        }
        calls: list[tuple[int, list[list[int]], list[str], str]] = []

        trainer = _make_rl_trainer_shell()
        trainer.config = SimpleNamespace(
            derived=SimpleNamespace(max_admitted_generation_prompts=None)
        )
        trainer.generators = [
            _GenerateActor(idx, calls, token_by_request_id) for idx in range(2)
        ]
        trainer.generator = trainer.generators[0]

        async def await_rank_0(actor_call, has_gpus=True):
            return await grpo.RLTrainer._await_call(actor_call)

        trainer._await_rank_0 = await_rank_0
        scheduler = trainer._make_generation_scheduler(metrics_prefix="generator")
        try:
            results = await asyncio.gather(
                *[
                    scheduler.submit(
                        prompt_token_ids=prompt,
                        sampling=SamplingConfig(max_tokens=8),
                        request_id=request_id,
                    )
                    for prompt, request_id in zip(prompts, request_ids, strict=True)
                ]
            )
            metrics = scheduler.pop_metrics()
        finally:
            await scheduler.close()

        return calls, results, request_ids, metrics

    calls, results, request_ids, metrics = asyncio.run(scenario())
    by_generator = grpo._group_request_positions_by_generator(
        request_ids,
        num_generators=2,
    )
    assert set(by_generator) == {0, 1}

    flattened_indices = [
        original_idx
        for positions in by_generator.values()
        for original_idx in positions
    ]
    assert sorted(flattened_indices) == list(range(len(request_ids)))

    for generator_idx, _prompts, call_request_ids, metrics_prefix in calls:
        assert call_request_ids == [
            request_ids[position] for position in by_generator[generator_idx]
        ]
        assert metrics_prefix == f"generator/{generator_idx}"

    assert [completion.token_ids for completion in results] == [[0], [1], [2], [3]]
    queue_depth_max = {
        metric.key: metric.value.value
        for metric in metrics
        if isinstance(metric.value, m.Max) and metric.key.endswith("/queue_depth")
    }
    assert queue_depth_max == {
        "generator/0/queue_depth": 2.0,
        "generator/1/queue_depth": 2.0,
    }


def test_single_generator_scheduler_uses_unchanged_prefix() -> None:
    async def scenario():
        request_ids = ["group-a:sample=0:turn=0", "group-b:sample=0:turn=0"]
        calls: list[tuple[int, list[list[int]], list[str], str]] = []
        trainer = _make_rl_trainer_shell()
        trainer.config = SimpleNamespace(
            derived=SimpleNamespace(max_admitted_generation_prompts=None)
        )
        trainer.generators = [
            _GenerateActor(
                0,
                calls,
                {request_id: idx for idx, request_id in enumerate(request_ids)},
            )
        ]
        trainer.generator = trainer.generators[0]

        async def await_rank_0(actor_call, has_gpus=True):
            return await grpo.RLTrainer._await_call(actor_call)

        trainer._await_rank_0 = await_rank_0
        scheduler = trainer._make_generation_scheduler(metrics_prefix="generator")
        try:
            results = await asyncio.gather(
                *[
                    scheduler.submit(
                        prompt_token_ids=[idx],
                        sampling=SamplingConfig(max_tokens=8),
                        request_id=request_id,
                    )
                    for idx, request_id in enumerate(request_ids)
                ]
            )
            metrics = scheduler.pop_metrics()
        finally:
            await scheduler.close()

        return calls, results, metrics

    calls, results, metrics = asyncio.run(scenario())

    assert calls[0][2:] == (
        ["group-a:sample=0:turn=0", "group-b:sample=0:turn=0"],
        "generator",
    )
    assert [completion.token_ids for completion in results] == [[0], [1]]
    assert not any(metric.key.endswith("/queue_depth") for metric in metrics)


def test_config_validation_accepts_multiple_generator_instances() -> None:
    grpo.RLTrainer.Config(
        num_generator_instances=2,
        generator=_valid_generator_config(),
    )


@pytest.mark.parametrize("num_generator_instances", [0, -1])
def test_config_validation_rejects_non_positive_generator_instances(
    num_generator_instances: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="num_generator_instances must be positive",
    ):
        grpo.RLTrainer.Config(
            num_generator_instances=num_generator_instances,
            generator=_valid_generator_config(),
        )


def test_spawn_role_meshes_spawns_one_mesh_per_generator(monkeypatch) -> None:
    calls: list[tuple[dict[str, int], str | None]] = []
    host = _SpawnHost(calls)
    monkeypatch.setattr(grpo, "this_host", lambda: host)
    trainer = _make_rl_trainer_shell()
    trainer.config = SimpleNamespace(num_generator_instances=2)
    trainer.trainer_world_size = 1
    trainer.generator_world_size = 1

    trainer_mesh, generator_meshes = trainer._spawn_role_meshes(
        host_mesh=None,
        trainer_nodes=None,
        generator_nodes=None,
        gpus_per_node=None,
        total_gpus=3,
    )

    assert trainer_mesh.name == "mesh_0"
    assert [mesh.name for mesh in generator_meshes] == ["generator_0", "generator_1"]
    assert calls == [
        ({"gpus": 1}, None),
        ({"gpus": 1}, "generator_0"),
        ({"gpus": 1}, "generator_1"),
    ]


def test_spawn_role_meshes_rejects_multi_node_multigen() -> None:
    trainer = _make_rl_trainer_shell()
    trainer.config = SimpleNamespace(num_generator_instances=2)

    with pytest.raises(ValueError, match="host_mesh is None"):
        trainer._spawn_role_meshes(
            host_mesh=object(),
            trainer_nodes=1,
            generator_nodes=1,
            gpus_per_node=8,
            total_gpus=3,
        )


class _StubEndpoint:
    def __init__(self, name: str, events: list[str]):
        self._name = name
        self._events = events

    async def call(self, *args):
        suffix = "" if not args else f":{','.join(str(arg) for arg in args)}"
        self._events.append(f"{self._name}{suffix}")


class _StubTrainerActor:
    def __init__(self, events: list[str]):
        self.push_model_state_dict = _StubEndpoint("trainer.push", events)


class _StubGeneratorActor:
    def __init__(self, name: str, events: list[str]):
        self.pull_model_state_dict = _StubEndpoint(f"{name}.pull", events)


class _SyncActor:
    def __init__(self, name: str, events: list[str]):
        self.sync_log_step = _StubEndpoint(name, events)


@dataclass
class _StubScheduler:
    events: list[str]

    async def pause_for_weight_sync(self) -> None:
        self.events.append("scheduler.pause")

    async def resume_after_weight_sync(self) -> None:
        self.events.append("scheduler.resume")


def _make_rl_trainer_shell() -> grpo.RLTrainer:
    trainer = grpo.RLTrainer.__new__(grpo.RLTrainer)
    trainer._multi_node = False
    return trainer


def _valid_generator_config() -> VLLMGenerator.Config:
    return VLLMGenerator.Config(
        parallelism=ParallelismConfig(
            enable_sequence_parallel=False,
            disable_loss_parallel=True,
        ),
    )


def _request_ids_for_distinct_generators() -> tuple[str, str]:
    first = "group-a:sample=0"
    first_idx = grpo._generator_index_for_request_id(f"{first}:turn=0", 2)
    for idx in range(1, 64):
        candidate = f"group-{idx}:sample=0"
        if grpo._generator_index_for_request_id(f"{candidate}:turn=0", 2) != first_idx:
            return first, candidate
    raise AssertionError("could not find request IDs for distinct generators")


class _SpawnedMesh:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class _SpawnHost:
    def __init__(self, calls: list[tuple[dict[str, int], str | None]]):
        self._calls = calls
        self._next_idx = 0

    def spawn_procs(
        self,
        *,
        per_host: dict[str, int],
        bootstrap,
        name: str | None = None,
    ) -> _SpawnedMesh:
        self._calls.append((per_host, name))
        mesh = _SpawnedMesh(name or f"mesh_{self._next_idx}")
        self._next_idx += 1
        return mesh


def test_weight_sync_pulls_every_generator_instance() -> None:
    async def scenario() -> list[str]:
        events: list[str] = []
        trainer = _make_rl_trainer_shell()
        trainer.trainer = _StubTrainerActor(events)
        trainer.generators = [
            _StubGeneratorActor("generator0", events),
            _StubGeneratorActor("generator1", events),
        ]
        trainer.generator = trainer.generators[0]

        await trainer._sync_generator_weights(
            generation_scheduler=_StubScheduler(events),
            policy_version=7,
        )

        return events

    assert asyncio.run(scenario()) == [
        "scheduler.pause",
        "trainer.push",
        "generator0.pull:7",
        "generator1.pull:7",
        "scheduler.resume",
    ]


def test_log_step_sync_visits_every_generator_instance() -> None:
    async def scenario() -> list[str]:
        events: list[str] = []
        trainer = _make_rl_trainer_shell()
        trainer.trainer = _SyncActor("trainer.sync", events)
        trainer.generators = [
            _SyncActor("generator0.sync", events),
            _SyncActor("generator1.sync", events),
        ]

        await trainer._sync_actor_log_step(4)

        return events

    assert asyncio.run(scenario()) == [
        "trainer.sync:4",
        "generator0.sync:4",
        "generator1.sync:4",
    ]


def test_close_closes_every_generator_instance_before_mesh_stop() -> None:
    async def scenario() -> list[str]:
        events: list[str] = []
        trainer = _make_rl_trainer_shell()
        trainer.trainer = _CloseActor("trainer.close", events)
        trainer.generators = [
            _CloseActor("generator0.close", events),
            _CloseActor("generator1.close", events),
        ]
        trainer.generator = trainer.generators[0]
        trainer.metrics_processor = _MetricsProcessor()
        trainer._proc_meshes = [_Mesh("mesh.stop[0]", events)]

        await trainer.close()

        assert trainer._proc_meshes == []
        return events

    assert asyncio.run(scenario()) == [
        "trainer.close",
        "generator0.close",
        "generator1.close",
        "mesh.stop[0]",
    ]


class _CloseActor:
    def __init__(self, name: str, events: list[str]):
        self.close = _StubEndpoint(name, events)


class _MetricsProcessor:
    def close(self) -> None:
        return None


class _Mesh:
    def __init__(self, name: str, events: list[str]):
        self._name = name
        self._events = events

    async def stop(self) -> None:
        self._events.append(self._name)


class _GenerateEndpoint:
    def __init__(
        self,
        generator_idx: int,
        calls: list[tuple[int, list[list[int]], list[str], str]],
        token_by_request_id: dict[str, int],
    ):
        self._generator_idx = generator_idx
        self._calls = calls
        self._token_by_request_id = token_by_request_id

    async def call(
        self,
        prompt_token_ids_batch: list[list[int]],
        *,
        request_ids: list[str],
        sampling_config: SamplingConfig,
        metrics_prefix: str,
    ):
        self._calls.append(
            (
                self._generator_idx,
                prompt_token_ids_batch,
                request_ids,
                metrics_prefix,
            )
        )
        return (
            [
                _completion(self._token_by_request_id[request_id])
                for request_id in request_ids
            ],
            [],
        )


class _GenerateActor:
    def __init__(
        self,
        generator_idx: int,
        calls: list[tuple[int, list[list[int]], list[str], str]],
        token_by_request_id: dict[str, int],
    ):
        self.generate = _GenerateEndpoint(
            generator_idx,
            calls,
            token_by_request_id,
        )
