# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Rollouter's controller-side worker dispatch."""

import asyncio
import pathlib
import sys
from types import SimpleNamespace

import cloudpickle
import pytest
from monarch.actor import ActorError, this_host

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch import (
    provisioning,
    rollout_worker as rollout_worker_module,
)
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.system import MonarchActorSystem

from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker


class _ChooseRunGroupEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def choose(self, **kwargs) -> asyncio.Future[RolloutGroup]:
        self.calls.append(kwargs)
        self.started.set()
        return asyncio.create_task(self._execute(kwargs))

    async def _execute(self, kwargs) -> RolloutGroup:
        await self.release.wait()
        return RolloutGroup(group_id=kwargs["group_id"], rollouts=[])


def test_rollout_worker_actor_builds_configured_worker() -> None:
    async def run() -> None:
        worker = object()
        worker_config = SimpleNamespace(build=lambda: worker)
        actor = object.__new__(RolloutWorkerActor)

        RolloutWorkerActor.__init__(
            actor,
            worker_config=worker_config,
            num_threads=1,
            structured_logger_init_fn=None,
        )

        assert actor._worker is worker

    asyncio.run(run())


def test_rollout_worker_actor_inits_structured_logger_with_its_rank(
    monkeypatch,
) -> None:
    async def run() -> None:
        init_calls = []
        monkeypatch.setattr(
            rollout_worker_module, "current_rank", lambda: SimpleNamespace(rank=2)
        )
        actor = object.__new__(RolloutWorkerActor)

        RolloutWorkerActor.__init__(
            actor,
            worker_config=SimpleNamespace(build=lambda: None),
            num_threads=1,
            structured_logger_init_fn=lambda **kwargs: init_calls.append(kwargs),
        )

        assert init_calls == [{"source": "rl_rollout_worker", "rank": 2}]

    asyncio.run(run())


class _SetupEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def call(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _CancelGroupEndpoint:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.error: Exception | None = None
        self.called, self.release = asyncio.Event(), asyncio.Event()

    async def call(self, group_id: int) -> None:
        self.calls.append(group_id)
        self.called.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error


class _WorkerActorMesh:
    def __init__(self) -> None:
        self.setup_async = _SetupEndpoint()
        self.run_group = _ChooseRunGroupEndpoint()
        self.cancel_group = _CancelGroupEndpoint()
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class _WorkerMesh:
    def __init__(self, actor_mesh: _WorkerActorMesh) -> None:
        self.actor_mesh = actor_mesh
        self.spawn_args = None
        self.stopped = False

    def spawn(self, *args, **kwargs) -> _WorkerActorMesh:
        self.spawn_args = (args, kwargs)
        return self.actor_mesh

    async def stop(self) -> None:
        await self.actor_mesh.stop()
        self.stopped = True


class _ControllerHost:
    def __init__(self, worker_mesh: _WorkerMesh) -> None:
        self.worker_mesh = worker_mesh
        self.spawn_kwargs = None

    def spawn_procs(self, **kwargs) -> _WorkerMesh:
        self.spawn_kwargs = kwargs
        return self.worker_mesh


def _rollouter_without_datasets() -> Rollouter:
    class WorkerConfig:
        _owner = RolloutWorker

    worker_config = WorkerConfig()
    rollouter = object.__new__(Rollouter)
    rollouter._config = SimpleNamespace(
        worker=worker_config,
        worker_pool_size=3,
        num_threads_per_worker=2,
    )
    rollouter._worker_actors = None
    rollouter._actor_system = None
    return rollouter


async def _setup(
    rollouter: Rollouter,
    actor_mesh: _WorkerActorMesh,
    monkeypatch,
) -> _WorkerMesh:
    worker_mesh = _WorkerMesh(actor_mesh)
    host = _ControllerHost(worker_mesh)
    monkeypatch.setattr(provisioning, "this_host", lambda: host)
    await rollouter.setup_async(
        actor_system=MonarchActorSystem(MonarchActorSystem.Config()),
        tokenizer_config="tokenizer_config",
        renderer_config="renderer_config",
        hf_assets_path="hf_assets_path",
    )
    return worker_mesh


def test_setup_spawns_worker_pool_on_controller_host(monkeypatch) -> None:
    async def run() -> None:
        worker_mesh = _WorkerMesh(_WorkerActorMesh())
        host = _ControllerHost(worker_mesh)
        monkeypatch.setattr(provisioning, "this_host", lambda: host)
        structured_logger_init_fn = object()
        monkeypatch.setattr(
            sl,
            "get_structured_logger_subprocess_init_fn",
            lambda: structured_logger_init_fn,
        )
        rollouter = _rollouter_without_datasets()

        await rollouter.setup_async(
            actor_system=MonarchActorSystem(MonarchActorSystem.Config()),
            tokenizer_config="tokenizer_config",
            renderer_config="renderer_config",
            hf_assets_path="hf_assets_path",
        )

        assert host.spawn_kwargs == {"per_host": {"cpus": 3}}
        args, kwargs = worker_mesh.spawn_args
        assert args[0] == "rollout_worker"
        assert args[1] is RolloutWorkerActor
        # The actor gets the worker's own config, not the whole Rollouter config.
        assert kwargs == {
            "worker_config": rollouter._config.worker,
            "num_threads": 2,
            "structured_logger_init_fn": structured_logger_init_fn,
        }
        assert worker_mesh.actor_mesh.setup_async.calls == [
            {
                "tokenizer_config": "tokenizer_config",
                "renderer_config": "renderer_config",
                "hf_assets_path": "hf_assets_path",
            }
        ]
        await rollouter.close()
        assert worker_mesh.actor_mesh.stopped
        assert worker_mesh.stopped

    asyncio.run(run())


def test_rollout_group_dispatch_uses_choose(monkeypatch) -> None:
    async def run() -> None:
        actor_mesh = _WorkerActorMesh()
        rollouter = _rollouter_without_datasets()
        await _setup(rollouter, actor_mesh, monkeypatch)

        dispatch = asyncio.create_task(
            rollouter.run_group_rollouts(
                generate_fn="generate_fn",
                release_session_fn="release_session_fn",
                sample="sample",
                group_id=1,
                group_size=2,
                sampling="sampling",
            )
        )
        await actor_mesh.run_group.started.wait()
        assert actor_mesh.run_group.calls == [
            {
                "generate_fn": "generate_fn",
                "release_session_fn": "release_session_fn",
                "sample": "sample",
                "group_id": 1,
                "group_size": 2,
                "sampling": "sampling",
            }
        ]

        actor_mesh.run_group.release.set()
        assert (await dispatch).group_id == 1
        await rollouter.close()
        assert actor_mesh.stopped

    asyncio.run(run())


@pytest.mark.parametrize("cancel_error", [None, RuntimeError("worker is gone")])
def test_cancelled_dispatch_cancels_the_group_on_its_worker(
    monkeypatch, cancel_error
) -> None:
    """A cancelled caller stops the group on its worker; the cancel propagates even if that fails."""

    async def run() -> None:
        actor_mesh = _WorkerActorMesh()
        actor_mesh.cancel_group.error = cancel_error
        rollouter = _rollouter_without_datasets()
        await _setup(rollouter, actor_mesh, monkeypatch)

        dispatch = asyncio.create_task(
            rollouter.run_group_rollouts(
                generate_fn="generate_fn",
                release_session_fn="release_session_fn",
                sample="sample",
                group_id=5,
                group_size=2,
                sampling="sampling",
            )
        )
        await actor_mesh.run_group.started.wait()
        dispatch.cancel()
        await actor_mesh.cancel_group.called.wait()
        # The cancel waits until the worker stopped the group.
        assert not dispatch.done()
        actor_mesh.cancel_group.release.set()
        with pytest.raises(asyncio.CancelledError):
            await dispatch
        assert actor_mesh.cancel_group.calls == [5]
        await rollouter.close()

    asyncio.run(run())


class _BlockingWorker:
    """A rollout worker whose groups run until cancelled; it marks each cancelled group in a file."""

    def __init__(self, marker_dir: str) -> None:
        self.marker_dir = marker_dir

    async def run_group(self, *, group_id: int, **kwargs) -> RolloutGroup:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # A slow cleanup, e.g. env.close(): cancel_group must wait for it.
            await asyncio.sleep(0.5)
            pathlib.Path(self.marker_dir, f"{group_id}.cancelled").touch()
            raise


class _BlockingWorkerConfig:
    def __init__(self, marker_dir: str) -> None:
        self.marker_dir = marker_dir

    def build(self) -> _BlockingWorker:
        return _BlockingWorker(self.marker_dir)


def test_cancel_group_stops_the_group_on_real_worker_processes(tmp_path) -> None:
    """Sent right after `run_group`, `cancel_group` finds the group (Monarch starts one sender's
    concurrent calls in order) and returns only after it stopped."""

    async def run() -> None:
        mesh = this_host().spawn_procs(per_host={"cpus": 2})
        # The worker processes can't import this test module, so ship the fakes by value.
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
        try:
            workers = mesh.spawn(
                "rollout_worker",
                RolloutWorkerActor,
                worker_config=_BlockingWorkerConfig(str(tmp_path)),
                num_threads=1,
                structured_logger_init_fn=None,
            )
            group = workers.run_group.choose(
                generate_fn=None,
                release_session_fn=None,
                sample=None,
                group_id=7,
                group_size=1,
                sampling=None,
            )
            await workers.cancel_group.call(7)
            assert (tmp_path / "7.cancelled").exists()
            with pytest.raises(ActorError):
                await group
            # The worker keeps serving after a cancel.
            await workers.sync_log_step.call(1)
        finally:
            cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
            await mesh.stop()

    asyncio.run(run())
