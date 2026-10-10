# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Rollouter's `Curriculum` hooks."""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchtitan.rl.components.data import IterableRLDataLoader, RLDataset
from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.rollout.curriculum import Curriculum
from torchtitan.rl.rollout.environment import MessageEnv
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rubric import Rubric


class _CountingDataset(RLDataset):
    """Yields 0, 1, 2, ..., so each training sample equals its group id."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        value = self.position
        self.position += 1
        return value

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state_dict) -> None:
        self.position = state_dict["position"]


class _RecordingCurriculum(Curriculum):
    """Records every hook call; `prepare` pairs the sample with its step, `update` bumps `level`."""

    @dataclass(kw_only=True, slots=True)
    class Config(Curriculum.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.prepared: list[tuple[object, int]] = []
        self.summarized: list[tuple[object, int]] = []
        self.updates: list[tuple[int, list[object]]] = []
        self.level = 0

    def prepare(self, sample, *, step):
        self.prepared.append((sample, step))
        return (sample, step)

    def summarize(self, sample, group):
        self.summarized.append((sample, group.group_id))
        return f"summary {group.group_id}"

    def update(self, *, step, summaries):
        self.updates.append((step, summaries))
        self.level += 1

    def state_dict(self):
        return {"level": self.level}

    def load_state_dict(self, state_dict):
        self.level = state_dict["level"]


class _WorkerPool:
    """Stands in for the rollout worker actor mesh; records the sample each group ran on."""

    def __init__(self) -> None:
        self.samples: list[object] = []
        self.failing_group_ids: set[int] = set()
        self.run_group = SimpleNamespace(choose=self._run_group)
        self.sync_log_step = SimpleNamespace(call=self._sync_log_step)

    async def _run_group(self, *, sample, group_id, **kwargs) -> RolloutGroup:
        if group_id in self.failing_group_ids:
            raise RuntimeError(f"group {group_id} failed")
        self.samples.append(sample)
        return RolloutGroup(group_id=group_id, rollouts=[])

    async def _sync_log_step(self, step: int) -> None:
        pass


class _InProcessRollouter(Rollouter):
    """Runs groups itself instead of on the worker pool, like `VerifiersRollouter`."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.samples: list[object] = []

    async def _run_group_rollouts(self, *, sample, group_id, **kwargs) -> RolloutGroup:
        self.samples.append(sample)
        return RolloutGroup(group_id=group_id, rollouts=[])


def _rollouter(config_cls=Rollouter.Config, **config_kwargs) -> Rollouter:
    rollouter = config_cls(
        training_dataloader=IterableRLDataLoader.Config(
            dataset=_CountingDataset.Config()
        ),
        validation_dataset=_CountingDataset.Config(),
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(), message_env=MessageEnv.Config()
        ),
        **config_kwargs,
    ).build()
    rollouter._worker_actors = _WorkerPool()
    return rollouter


async def _roll_out(rollouter: Rollouter, group_id: int, sample: object) -> None:
    await rollouter.run_group_rollouts(
        generate_fn=None,
        sample=sample,
        group_id=group_id,
        group_size=1,
        sampling=None,
    )


def test_prepare_gets_the_synced_step_then_the_restored_step() -> None:
    async def run() -> None:
        rollouter = _rollouter(curriculum=_RecordingCurriculum.Config())
        await rollouter.sync_log_step(3)
        await _roll_out(rollouter, *rollouter.get_training_sample())
        await rollouter.sync_log_step(4)
        await _roll_out(rollouter, *rollouter.get_training_sample())
        assert rollouter._curriculum.prepared == [(0, 3), (1, 4)]
        # The worker rolls out the prepared sample.
        assert rollouter._worker_actors.samples == [(0, 3), (1, 4)]

        # Saved at step 4: before any sync, the restored Rollouter prepares at step 4.
        # Group 0 was never acknowledged, so its original sample is replayed first.
        restored = _rollouter(curriculum=_RecordingCurriculum.Config())
        restored.load_state_dict(rollouter.state_dict())
        await _roll_out(restored, *restored.get_training_sample())
        assert restored._curriculum.prepared == [(0, 4)]

    asyncio.run(run())


def test_validation_groups_skip_the_curriculum() -> None:
    async def run() -> None:
        rollouter = _rollouter(curriculum=_RecordingCurriculum.Config())
        await _roll_out(rollouter, -1, "validation sample")
        assert rollouter._worker_actors.samples == ["validation sample"]
        assert rollouter._curriculum.prepared == []
        assert rollouter._curriculum.summarized == []

    asyncio.run(run())


def test_acknowledge_hands_update_each_summary_once() -> None:
    async def run() -> None:
        rollouter = _rollouter(curriculum=_RecordingCurriculum.Config())
        await rollouter.sync_log_step(5)
        for _ in range(3):
            await _roll_out(rollouter, *rollouter.get_training_sample())
        # Group 3 fails in its worker, so it has no summary.
        rollouter._worker_actors.failing_group_ids.add(3)
        with pytest.raises(RuntimeError):
            await _roll_out(rollouter, *rollouter.get_training_sample())

        # An iterator can be read only once; the Rollouter must still see every id.
        rollouter.acknowledge_training_sample_ids(iter([2, 0, 3]))
        rollouter.acknowledge_training_sample_ids([1])
        assert rollouter._curriculum.updates == [
            (5, ["summary 2", "summary 0"]),
            (5, ["summary 1"]),
        ]
        # The loader saw every id, and no summary is left behind.
        assert rollouter.state_dict()["dataloader"]["pending_samples"] == []
        assert rollouter._summaries == {}

    asyncio.run(run())


def test_state_round_trip() -> None:
    async def run() -> None:
        rollouter = _rollouter(curriculum=_RecordingCurriculum.Config())
        await rollouter.sync_log_step(7)
        for _ in range(2):
            await _roll_out(rollouter, *rollouter.get_training_sample())
        rollouter.acknowledge_training_sample_ids([0])

        state = rollouter.state_dict()
        assert state["dataloader"]["pending_samples"] == [(1, 1)]
        assert state["curriculum"] == {"level": 1}
        assert state["step"] == 7

        restored = _rollouter(curriculum=_RecordingCurriculum.Config())
        restored.load_state_dict(state)
        assert restored.state_dict() == state

    asyncio.run(run())


def test_subclass_overriding__run_group_rollouts_gets_the_hooks() -> None:
    async def run() -> None:
        rollouter = _rollouter(
            _InProcessRollouter.Config, curriculum=_RecordingCurriculum.Config()
        )
        # Like `VerifiersRollouter`, it has no worker pool.
        rollouter._worker_actors = None
        await rollouter.sync_log_step(2)
        await _roll_out(rollouter, *rollouter.get_training_sample())
        rollouter.acknowledge_training_sample_ids([0])

        assert rollouter.samples == [(0, 2)]
        assert rollouter._curriculum.summarized == [((0, 2), 0)]
        assert rollouter._curriculum.updates == [(2, ["summary 0"])]

    asyncio.run(run())
