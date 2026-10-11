# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from types import SimpleNamespace

import pytest

from torchtitan.rl import train
from torchtitan.rl.components.work_buffer import RolloutGroupWork
from torchtitan.rl.controller import AsyncLoopConfig
from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.observability.rollout_recorder import RolloutSampleRecorder
from torchtitan.rl.rollout import RolloutGroup


class _FakeController:
    instances = []

    def __init__(self, config=None):
        self.config = config
        self.events = []
        self.setup_trainer_mesh = None
        self.setup_generator_meshes = None
        self.instances.append(self)

    async def setup_async(
        self,
        *,
        trainer_mesh=None,
        generator_meshes=None,
    ):
        self.events.append("setup")
        self.setup_trainer_mesh = trainer_mesh
        self.setup_generator_meshes = generator_meshes
        if getattr(self.config, "fail_setup", False):
            raise RuntimeError("setup failed")

    async def run(self):
        self.events.append("train")
        if getattr(self.config, "fail_train", False):
            raise RuntimeError("train failed")
        if getattr(self.config, "cancel_train", False):
            raise asyncio.CancelledError()

    async def close(self):
        self.events.append("close")


class _FakeDebug:
    enable_structured_logging = False


class _FakeTrainerConfig:
    debug = _FakeDebug()
    # main() reads config.trainer.parallelism to size the trainer mesh; the
    # value is irrelevant here because stub_mesh_provisioning stubs
    # _compute_trainer_world_size, so a placeholder is enough to resolve the access.
    parallelism = None


class _FakeConfig:
    """Fake config whose build() returns a _FakeController."""

    dump_folder = "/tmp/test_rl"
    trainer = _FakeTrainerConfig()
    # main() also reads config.generator.parallelism (same stubbing applies).
    generator = SimpleNamespace(
        parallelism=None, cuda_graph=VLLMCudaGraphConfig(mode="NONE")
    )
    num_generators = 1

    @property
    def __class__(self):
        # main() does `assert isinstance(config, Controller.Config)`. Reporting
        # that type lets this lightweight stand-in pass the check (the
        # unittest.mock `spec` idiom) without constructing a real Config.
        return train.Controller.Config

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def build(self):
        return _FakeController(config=self)


class _FakeConfigLoader:
    config = _FakeConfig()

    def load(self):
        return self.config


def test_async_loop_config_derives_window_and_max_offpolicy_steps() -> None:
    default_loop = AsyncLoopConfig()
    assert default_loop.target_offpolicy_steps == 3
    assert default_loop.windowed_fifo_batches is None

    # P=8, S=3: windowed_fifo_batches=1 -> 8 ids, cap 4; 3 -> 24 ids, cap 6; None -> no window, no cap
    one_batch = AsyncLoopConfig(num_prompts_per_train_step=8, windowed_fifo_batches=1)
    assert (one_batch.window_size, one_batch.max_offpolicy_steps) == (8, 4)
    three_batches = AsyncLoopConfig(
        num_prompts_per_train_step=8, windowed_fifo_batches=3
    )
    assert (three_batches.window_size, three_batches.max_offpolicy_steps) == (24, 6)
    uncapped = AsyncLoopConfig(num_prompts_per_train_step=8, windowed_fifo_batches=None)
    assert (uncapped.window_size, uncapped.max_offpolicy_steps) == (None, None)

    # a window larger than the buffer is allowed: P=8, S=1 (16 slots), 5 batches -> 40 ids, cap 6
    wide = AsyncLoopConfig(
        num_prompts_per_train_step=8, target_offpolicy_steps=1, windowed_fifo_batches=5
    )
    assert (wide.window_size, wide.max_offpolicy_steps) == (40, 6)


def test_async_loop_config_rejects_bad_windowed_fifo_batches() -> None:
    with pytest.raises(ValueError, match="windowed_fifo_batches"):
        AsyncLoopConfig(windowed_fifo_batches=0)
    with pytest.raises(TypeError, match="window_fraction"):
        AsyncLoopConfig(window_fraction=0.3)  # type: ignore[call-arg]


def test_async_loop_config_ramps_rollout_workers_after_the_first_batch() -> None:
    # P=2, S=2: 6 workers; the first batch's 2 start at once, the other 4 join over 8 s
    ramped = AsyncLoopConfig(
        num_prompts_per_train_step=2, target_offpolicy_steps=2, start_ramp_s=8.0
    )
    assert ramped.rollout_worker_start_delays_s == [0.0, 0.0, 2.0, 4.0, 6.0, 8.0]
    # S=0: every worker serves the first batch, so none waits
    on_policy = AsyncLoopConfig(num_prompts_per_train_step=2, target_offpolicy_steps=0)
    assert on_policy.rollout_worker_start_delays_s == [0.0, 0.0]
    no_ramp = AsyncLoopConfig(
        num_prompts_per_train_step=2, target_offpolicy_steps=2, start_ramp_s=0.0
    )
    assert no_ramp.rollout_worker_start_delays_s == [0.0] * 6
    with pytest.raises(ValueError, match="start_ramp_s"):
        AsyncLoopConfig(start_ramp_s=-1.0)


def test_rollout_loop_waits_its_start_delay_once_before_its_first_claim(monkeypatch):
    # A buffer that hands out groups 0..2, then None (closed)
    waiting = [
        RolloutGroupWork(group_id=group_id, sample=None) for group_id in range(3)
    ]
    finalized_group_ids: list[int] = []

    async def claim_next() -> RolloutGroupWork | None:
        return waiting.pop(0) if waiting else None

    async def finalize_work(rollout_group: RolloutGroup) -> None:
        finalized_group_ids.append(rollout_group.group_id)

    group_buffer = SimpleNamespace(claim_next=claim_next, finalize_work=finalize_work)
    sleeps: list[tuple[float, int]] = []  # (seconds, groups still waiting)

    async def fake_sleep(seconds: float) -> None:
        sleeps.append((seconds, len(waiting)))

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def run_group_rollouts(*, group_id: int, **kwargs) -> RolloutGroup:
        return RolloutGroup(group_id=group_id, rollouts=[])

    async def release_groups(group_ids: list[int]) -> None:
        pass

    controller = SimpleNamespace(
        config=SimpleNamespace(async_loop=SimpleNamespace(num_samples_per_prompt=1)),
        generator_router=SimpleNamespace(
            release_groups=SimpleNamespace(call_one=release_groups)
        ),
        _rollouter=SimpleNamespace(run_group_rollouts=run_group_rollouts),
        _sampling=SamplingConfig(),
        rollout_recorder=SimpleNamespace(record=lambda **kwargs: None),
    )
    asyncio.run(
        train.Controller._rollout_loop(
            controller, group_buffer=group_buffer, generate_fn=None, start_delay_s=7.5
        )
    )

    # one wait, before the first claim; later groups are claimed without waiting
    assert sleeps == [(7.5, 3)]
    assert finalized_group_ids == [0, 1, 2]


def _make_stub_rl_trainer():
    """Create an Controller with a minimal stub config (no VLLMGenerator validation)."""
    from torchtitan.rl.observability import metrics as m

    class _StubRollouter:
        async def close(self):
            pass

    class _StubConfig:
        async_loop = AsyncLoopConfig()
        metrics = m.MetricsProcessor.Config()
        dump_folder = "/tmp/test_rl"
        rollout_recorder = RolloutSampleRecorder.Config()
        hf_assets_path = "./tests/assets/tokenizer"
        # __init__ builds these too; stub them so construction does no real work.
        tokenizer = SimpleNamespace(
            build=lambda *, tokenizer_path: SimpleNamespace(eos_id=0)
        )
        renderer = SimpleNamespace(
            build=lambda *, tokenizer: SimpleNamespace(get_stop_token_ids=lambda: [])
        )
        # __init__ reads generator.sampling (a dataclass, for replace) + generator.debug.seed.
        generator = SimpleNamespace(
            sampling=SamplingConfig(), debug=SimpleNamespace(seed=None)
        )
        rollouter = SimpleNamespace(build=lambda: _StubRollouter())

        def to_dict(self):
            return {}

        def maybe_log(self) -> None:
            pass

    return train.Controller(_StubConfig())


@pytest.fixture
def stub_mesh_provisioning(monkeypatch):
    """Stub the Monarch mesh provisioning ``main()`` runs before ``setup_async``.

    ``spawn_proc_mesh`` spawns real GPU proc meshes and the
    ``_compute_*_world_size`` helpers read parallelism degrees; neither matters
    to shutdown behavior, so stub both and let the test intercept at the
    ``_FakeController`` boundary.
    """
    monkeypatch.setattr(train, "_compute_trainer_world_size", lambda p: 1)
    monkeypatch.setattr(train, "_compute_generator_world_size", lambda p: 1)

    def _spawn_proc_mesh(*args, num_generators=1, **kwargs):
        return (
            "trainer_mesh",
            [f"generator_mesh_{idx}" for idx in range(num_generators)],
        )

    monkeypatch.setattr(train, "spawn_proc_mesh", _spawn_proc_mesh)


def test_main_shuts_down_after_success(monkeypatch, stub_mesh_provisioning):
    _FakeConfigLoader.config = _FakeConfig()
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigLoader", _FakeConfigLoader)

    asyncio.run(train.main())

    trainer = _FakeController.instances[0]
    assert trainer.events == ["setup", "train", "close"]
    assert trainer.setup_trainer_mesh == "trainer_mesh"
    assert trainer.setup_generator_meshes == ["generator_mesh_0"]


def test_main_passes_configured_num_generators(monkeypatch, stub_mesh_provisioning):
    _FakeConfigLoader.config = _FakeConfig(num_generators=2)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigLoader", _FakeConfigLoader)

    asyncio.run(train.main())

    trainer = _FakeController.instances[0]
    assert trainer.events == ["setup", "train", "close"]
    assert trainer.setup_generator_meshes == [
        "generator_mesh_0",
        "generator_mesh_1",
    ]


def test_main_shuts_down_after_train_failure(monkeypatch, stub_mesh_provisioning):
    _FakeConfigLoader.config = _FakeConfig(fail_train=True)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigLoader", _FakeConfigLoader)

    with pytest.raises(RuntimeError, match="train failed"):
        asyncio.run(train.main())

    assert _FakeController.instances[0].events == ["setup", "train", "close"]


def test_main_shuts_down_after_setup_failure(monkeypatch, stub_mesh_provisioning):
    _FakeConfigLoader.config = _FakeConfig(fail_setup=True)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigLoader", _FakeConfigLoader)

    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(train.main())

    assert _FakeController.instances[0].events == ["setup", "close"]


def test_rl_trainer_shutdown_is_noop_before_meshes_spawn():
    trainer = _make_stub_rl_trainer()

    asyncio.run(trainer.close())

    assert trainer.trainer is None
    assert trainer.generator_router is None
    assert trainer._proc_meshes == []


def test_main_reraises_cancellation_after_shutdown(monkeypatch, stub_mesh_provisioning):
    """Signal-driven cancellation surfaces as ``CancelledError`` from the
    running task. Monarch delivers unhandled actor faults the same way, so
    ``main`` runs ``close`` and re-raises to exit nonzero."""
    _FakeConfigLoader.config = _FakeConfig(cancel_train=True)
    _FakeController.instances = []
    monkeypatch.setattr(train, "ConfigLoader", _FakeConfigLoader)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(train.main())

    assert _FakeController.instances[0].events == ["setup", "train", "close"]


def test_vllm_generator_does_not_touch_cuda_from_finalizer():
    assert "__del__" not in VLLMGenerator.__dict__


class _StubEndpoint:
    def __init__(self, name, events, raises=False):
        self._name = name
        self._events = events
        self._raises = raises

    async def call(self):
        self._events.append(self._name)
        if self._raises:
            raise RuntimeError(f"{self._name} failed")


class _RouterCloseEndpoint:
    def __init__(self, router):
        self._router = router

    async def call_one(self):
        return await self._router._fanout("close", return_exceptions=True)


class _StubRouterHandle:
    """Stands in for the router's actor-mesh handle, whose endpoints take adverbs."""

    def __init__(self, router):
        self.close_generators = _RouterCloseEndpoint(router)


class _StubActor:
    def __init__(self, name, events, raises=False):
        self.close = _StubEndpoint(name, events, raises)

    def flatten(self, *args, **kwargs):
        # The router builds a rank-0 handle via flatten("rank").slice(rank=0);
        # a single-rank stub collapses to itself for both.
        return self

    def slice(self, **kwargs):
        return self

    def __len__(self):
        return 1


class _StubMesh:
    def __init__(self, name, events):
        self._name = name
        self._events = events

    async def stop(self):
        self._events.append(self._name)


def _set_generator_router(rl_trainer, generators):
    rl_trainer.generator_router = _StubRouterHandle(
        InterGeneratorRouter(
            InterGeneratorRouter.Config(),
            generators=generators,
        )
    )


def test_shutdown_calls_actor_close_before_mesh_stop():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _StubActor("trainer.close", events)
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer._proc_meshes = [
        _StubMesh("mesh.stop[0]", events),
        _StubMesh("mesh.stop[1]", events),
    ]

    asyncio.run(rl_trainer.close())

    assert events == [
        "trainer.close",
        "generator.close",
        "mesh.stop[0]",
        "mesh.stop[1]",
    ]
    assert rl_trainer._proc_meshes == []


def test_shutdown_closes_all_generators():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _StubActor("trainer.close", events)
    _set_generator_router(
        rl_trainer,
        [
            _StubActor("generator[0].close", events),
            _StubActor("generator[1].close", events),
        ],
    )
    rl_trainer._proc_meshes = [_StubMesh("mesh.stop[0]", events)]

    asyncio.run(rl_trainer.close())

    assert events == [
        "trainer.close",
        "generator[0].close",
        "generator[1].close",
        "mesh.stop[0]",
    ]


def test_shutdown_continues_after_actor_close_failure():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _StubActor("trainer.close", events, raises=True)
    _set_generator_router(rl_trainer, [_StubActor("generator.close", events)])
    rl_trainer._proc_meshes = [_StubMesh("mesh.stop[0]", events)]

    asyncio.run(rl_trainer.close())

    # trainer.close raised, but every later step still ran.
    assert events == ["trainer.close", "generator.close", "mesh.stop[0]"]


def test_shutdown_continues_after_generator_close_failure():
    events: list[str] = []
    rl_trainer = _make_stub_rl_trainer()
    rl_trainer.trainer = _StubActor("trainer.close", events)
    _set_generator_router(
        rl_trainer,
        [
            _StubActor("generator[0].close", events, raises=True),
            _StubActor("generator[1].close", events),
        ],
    )
    rl_trainer._proc_meshes = [_StubMesh("mesh.stop[0]", events)]

    asyncio.run(rl_trainer.close())

    assert events == [
        "trainer.close",
        "generator[0].close",
        "generator[1].close",
        "mesh.stop[0]",
    ]


class _FakeHostMesh:
    """Records the CUDA_VISIBLE_DEVICES of each proc mesh spawned on it."""

    def __init__(self, num_hosts=1):
        self.num_hosts = num_hosts
        self.visible_devices = []

    def __len__(self):
        return self.num_hosts

    def spawn_procs(self, *, per_host, bootstrap, bootstrap_command):
        self.visible_devices.append(bootstrap_command["CUDA_VISIBLE_DEVICES"])


@pytest.fixture
def bootstrap_cmd_returns_env(monkeypatch):
    """`default_bootstrap_cmd().with_env(env)` returns `env`, so `_FakeHostMesh` can read it."""
    monkeypatch.setattr(
        train,
        "default_bootstrap_cmd",
        lambda: SimpleNamespace(with_env=lambda env: env),
    )


def test_spawn_proc_mesh_gives_roles_sharing_a_host_mesh_disjoint_gpus(
    bootstrap_cmd_returns_env,
):
    """Roles on one host mesh get disjoint GPUs; another host mesh starts at GPU 0."""
    mesh_a, mesh_b = _FakeHostMesh(), _FakeHostMesh()

    train.spawn_proc_mesh(
        trainer_world_size=2,
        per_generator_world_size=1,
        host_meshes=train.HostMeshes(
            trainer=mesh_a,
            generators=[mesh_a, mesh_a, mesh_b, mesh_b, mesh_b, mesh_b],
            gpus_per_node=4,
        ),
        num_generators=6,
    )

    assert mesh_a.visible_devices == ["0,1", "2", "3"]
    assert mesh_b.visible_devices == ["0", "1", "2", "3"]


@pytest.mark.parametrize(
    "num_hosts, layout, match",
    [
        (1, "AAABBB", "need 5 GPUs per host, but gpus_per_node is 4"),
        (2, "AABBBB", r"generator0 world size \(1\) .* host count \(2\)"),
    ],
)
def test_spawn_proc_mesh_checks_gpus_per_host_before_spawning(
    bootstrap_cmd_returns_env, num_hosts, layout, match
):
    """Roles that don't fit their host mesh raise before any proc mesh is spawned."""
    meshes = {"A": _FakeHostMesh(num_hosts), "B": _FakeHostMesh(num_hosts)}

    with pytest.raises(ValueError, match=match):
        train.spawn_proc_mesh(
            trainer_world_size=2,
            per_generator_world_size=1,
            host_meshes=train.HostMeshes(
                trainer=meshes["A"],
                generators=[meshes[name] for name in layout],
                gpus_per_node=4,
            ),
            num_generators=len(layout),
        )

    assert all(mesh.visible_devices == [] for mesh in meshes.values())
