# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for periodic validation: the Validator pass, the controller's gates/trigger/worker,
and the WeightSyncManager's DRAIN pull-skip + catch-up. Fakes only (no GPU / Monarch / vLLM)."""

import asyncio
from dataclasses import dataclass

import pytest

from torchtitan.experiments.rl.controller import Controller, ValidationLoopMode
from torchtitan.experiments.rl.components.weight_sync import WeightSyncManager
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout.types import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.types import RolloutTurnID
from torchtitan.experiments.rl.validator import (
    _single_policy_metrics,
    Validator,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def _rollout_turn(*, min_version: int | None, max_version: int | None) -> RolloutTurn:
    return RolloutTurn(
        rollout_id=RolloutTurnID(group_id=-1, rollout_id=0, turn_id=0),
        prompt_token_ids=[1, 2],
        completion_token_ids=[3, 4],
        completion_logprobs=[0.0, 0.0],
        min_policy_version=min_version,
        max_policy_version=max_version,
    )


def _rollout_group(
    *, group_id: int = -1, min_version: int | None = 7, max_version: int | None = 7
) -> RolloutGroup:
    rollout = Rollout(
        group_id=group_id,
        rollout_id=0,
        turns=[_rollout_turn(min_version=min_version, max_version=max_version)],
        status=RolloutStatus.COMPLETED,
        reward=1.0,
    )
    return RolloutGroup(group_id=group_id, rollouts=[rollout])


class _FakeRollouter:
    """Serves validation samples and returns canned rollout groups (or raises).

    `hold` (when given) blocks each pass until the test releases it, so a test can
    drive trainer steps while validation is verifiably in flight.
    """

    def __init__(
        self,
        *,
        policy_version: int = 7,
        fail_all: bool = False,
        hold: asyncio.Event | None = None,
    ):
        self.policy_version = policy_version
        self.fail_all = fail_all
        self.hold = hold
        self.num_group_rollout_calls = 0

    def get_validation_sample(self) -> object:
        return {"prompt": "q"}

    async def run_group_rollouts(self, *, generate_fn, sample, group_id, group_size, sampling, renderer):
        self.num_group_rollout_calls += 1
        if self.hold is not None:
            await self.hold.wait()
        if self.fail_all:
            raise RuntimeError("generation failed")
        return _rollout_group(
            group_id=group_id,
            min_version=self.policy_version,
            max_version=self.policy_version,
        )


class _FakeRecorder:
    def __init__(self):
        self.recorded: list[tuple[bool, int]] = []

    def record(self, *, is_validation, rollout_groups):
        self.recorded.append((is_validation, len(rollout_groups)))


@dataclass
class _FakeSampling:
    """Minimal stand-in for SamplingConfig: `replace()` needs dataclass semantics."""

    temperature: float = 0.8
    top_p: float = 0.95


def _validator(*, num_samples: int = 2, rollouter: _FakeRollouter | None = None, recorder=None):
    return Validator.Config(num_samples=num_samples).build(
        rollouter=rollouter or _FakeRollouter(),
        renderer=object(),
        rollout_recorder=recorder or _FakeRecorder(),
        sampling=_FakeSampling(),
    )


async def _fake_generate_fn(*args, **kwargs):
    raise AssertionError("generate_fn must not be called directly by these tests")


class _Endpoint:
    def __init__(self, on_call):
        self._on_call = on_call

    async def call(self):
        await self._on_call()


class _FakeTrainer:
    def __init__(self, on_push):
        self.push_model_state_dict = _Endpoint(on_push)


class _FakeRouter:
    def __init__(self, on_pull=None):
        self._on_pull = on_pull
        self.pulled_versions: list[int] = []

    async def pull_model_state_dict(self, *, policy_version):
        self.pulled_versions.append(policy_version)
        if self._on_pull is not None:
            await self._on_pull()


class _FakeBuffer:
    def __init__(self):
        self.releases: list[tuple[int, str]] = []

    async def release_active_groups(self, count, *, reason):
        self.releases.append((count, reason))


async def _noop():
    return None


def _weight_sync(
    *, on_push=_noop, on_pull=None, gate: asyncio.Event, num_groups_per_train_step=4
) -> tuple[WeightSyncManager, _FakeRouter, _FakeBuffer]:
    router = _FakeRouter(on_pull)
    buffer = _FakeBuffer()
    manager = WeightSyncManager(
        trainer=_FakeTrainer(on_push),
        generator_router=router,
        group_buffer=buffer,
        num_groups_per_train_step=num_groups_per_train_step,
        generator_pull_gate=gate,
    )
    return manager, router, buffer


def _open_gate() -> asyncio.Event:
    gate = asyncio.Event()
    gate.set()
    return gate


# ---------------------------------------------------------------------------
# Validator: single-policy metrics
# ---------------------------------------------------------------------------


def test_single_policy_metrics_one_version() -> None:
    metrics = _single_policy_metrics([_rollout_group(min_version=7, max_version=7)])
    assert [metric.key for metric in metrics] == ["validation/policy_version"]
    assert metrics[0].value.value == 7.0


def test_single_policy_metrics_mixed_versions_raise() -> None:
    groups = [
        _rollout_group(min_version=7, max_version=7),
        _rollout_group(min_version=8, max_version=8),
    ]
    with pytest.raises(RuntimeError, match="single-policy validation sampled"):
        _single_policy_metrics(groups)


def test_single_policy_metrics_no_generation_returns_empty() -> None:
    assert _single_policy_metrics([_rollout_group(min_version=None, max_version=None)]) == []


def test_evaluate_scores_and_records() -> None:
    async def run() -> None:
        recorder = _FakeRecorder()
        rollouter = _FakeRollouter(policy_version=3)
        validator = _validator(num_samples=4, rollouter=rollouter, recorder=recorder)
        metrics = await validator.evaluate(step=5, generate_fn=_fake_generate_fn)

        assert rollouter.num_group_rollout_calls == 4
        assert recorder.recorded == [(True, 4)]
        by_key = {metric.key for metric in metrics}
        assert "validation/policy_version" in by_key
        assert "validation/group_failures" in by_key
        assert "timing/validate" in by_key

    asyncio.run(run())


def test_evaluate_counts_failed_groups() -> None:
    async def run() -> None:
        validator = _validator(num_samples=3, rollouter=_FakeRollouter(fail_all=True))
        metrics = await validator.evaluate(step=1, generate_fn=_fake_generate_fn)
        failures = [metric for metric in metrics if metric.key == "validation/group_failures"]
        assert failures[0].value.value == 3.0
        # No generation happened -> no policy_version metric, no raise.
        assert not any(metric.key == "validation/policy_version" for metric in metrics)

    asyncio.run(run())


def test_evaluate_disabled_returns_empty() -> None:
    async def run() -> None:
        validator = _validator(num_samples=0)
        assert await validator.evaluate(step=1, generate_fn=_fake_generate_fn) == []

    asyncio.run(run())


# ---------------------------------------------------------------------------
# WeightSyncManager: pull-skip + catch-up
# ---------------------------------------------------------------------------


def test_closed_gate_skips_pull_but_pushes_and_releases() -> None:
    async def run() -> None:
        gate = _open_gate()
        manager, router, buffer = _weight_sync(gate=gate)

        gate.clear()
        manager.start_async_push_pull(version=5)
        pull_metrics = await manager.wait_prev_pull()

        assert router.pulled_versions == []  # pull skipped
        assert buffer.releases == [(4, "trained")]  # push + release still ran
        assert pull_metrics[0].value.value == 0.0  # skipped pull reports 0, not stale time

    asyncio.run(run())


def test_catch_up_pull_noops_without_skips() -> None:
    async def run() -> None:
        manager, router, _ = _weight_sync(gate=_open_gate())
        manager.start_async_push_pull(version=1)
        await manager.wait_prev_pull()
        assert await manager.catch_up_pull() is None
        assert router.pulled_versions == [1]  # only the normal pull; no redundant catch-up

    asyncio.run(run())


def test_catch_up_pull_pulls_latest_skipped_version_once() -> None:
    async def run() -> None:
        gate = _open_gate()
        manager, router, buffer = _weight_sync(gate=gate)

        gate.clear()
        manager.start_async_push_pull(version=5)
        await manager.wait_prev_pull()
        manager.start_async_push_pull(version=6)
        await manager.wait_prev_pull()

        assert router.pulled_versions == []
        assert buffer.releases == [(4, "trained"), (4, "trained")]

        caught_up = await manager.catch_up_pull()
        assert caught_up == 6
        assert router.pulled_versions == [6]  # latest only; no pull of 5
        # State reset: a second catch-up is a no-op.
        assert await manager.catch_up_pull() is None

    asyncio.run(run())


def test_catch_up_registers_as_pull_task() -> None:
    # The trainer's next wait_prev_pull must serialize behind the catch-up pull.
    async def run() -> None:
        gate = _open_gate()
        pull_started = asyncio.Event()
        pull_release = asyncio.Event()

        async def slow_pull():
            pull_started.set()
            await pull_release.wait()

        manager, router, _ = _weight_sync(gate=gate, on_pull=slow_pull)

        gate.clear()
        manager.start_async_push_pull(version=3)
        await manager.wait_prev_pull()

        catch_up = asyncio.create_task(manager.catch_up_pull())
        await pull_started.wait()

        waiter = asyncio.create_task(manager.wait_prev_pull())
        await asyncio.sleep(0)
        assert not waiter.done()  # blocked behind the in-flight catch-up pull

        pull_release.set()
        assert await catch_up == 3
        await waiter

    asyncio.run(run())


def test_catch_up_failure_keeps_skip_state() -> None:
    async def run() -> None:
        gate = _open_gate()

        async def failing_pull():
            raise RuntimeError("pull failed")

        manager, router, _ = _weight_sync(gate=gate, on_pull=failing_pull)
        gate.clear()
        manager.start_async_push_pull(version=2)
        await manager.wait_prev_pull()

        with pytest.raises(RuntimeError, match="pull failed"):
            await manager.catch_up_pull()
        # Skip state not reset on failure.
        assert manager._num_skipped_pulls == 1

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Controller: cadence, trigger, gates, worker
# ---------------------------------------------------------------------------


def _controller(
    *,
    loop_mode: ValidationLoopMode = ValidationLoopMode.PAUSE_TRAINER,
    num_samples: int = 2,
    interval_steps: int = 2,
    weight_sync: WeightSyncManager | None = None,
    rollouter: _FakeRollouter | None = None,
) -> Controller:
    """A Controller with only the periodic-validation state, no meshes/actors."""
    from torchtitan.experiments.rl.controller import ValidationConfig

    controller = Controller.__new__(Controller)

    class _AsyncLoop:
        validation = ValidationConfig(
            validator=Validator.Config(num_samples=num_samples),
            interval_steps=interval_steps,
            loop_mode=loop_mode,
        )

    class _Config:
        async_loop = _AsyncLoop()

    controller.config = _Config()
    controller._validator = _validator(num_samples=num_samples, rollouter=rollouter)
    # The worker logs completed passes at the newest trainer step (W&B monotonicity).
    controller._trainer_policy_version = 0

    controller._validation_gate_data_input = _open_gate()
    controller._validation_gate_trainer = _open_gate()
    # Share the pull-gate object with the manager (as run() does) so the trigger's
    # clear is visible to the weight-sync chain.
    controller._validation_gate_generator_pull = getattr(
        weight_sync, "_generator_pull_gate", None
    ) or _open_gate()
    controller._data_admission_active = asyncio.Lock()
    controller._trainer_step_active = asyncio.Lock()
    controller._validation_idle = _open_gate()
    controller._validation_trigger = asyncio.Queue(maxsize=1)
    controller._weight_sync = weight_sync

    class _Processor:
        def __init__(self):
            self.logged: list[int] = []
            self.logged_metrics: list[list] = []

        def log(self, step, metrics, *, is_validation):
            assert is_validation
            self.logged.append(step)
            self.logged_metrics.append(list(metrics))

    controller.metrics_processor = _Processor()
    controller._make_generate_fn = lambda metrics_prefix: _fake_generate_fn
    return controller


def test_should_start_periodic_validation_cadence() -> None:
    controller = _controller(interval_steps=5)
    should = controller._should_start_periodic_validation
    assert should(step=5, final_step=100)
    assert not should(step=4, final_step=100)  # not a multiple
    assert not should(step=100, final_step=100)  # final step: post-validation covers it
    assert should(step=40, final_step=100)  # resume arithmetic: absolute steps


def test_should_start_periodic_validation_disabled() -> None:
    assert not _controller(interval_steps=0)._should_start_periodic_validation(
        step=2, final_step=10
    )
    assert not _controller(num_samples=0)._should_start_periodic_validation(
        step=2, final_step=10
    )


def test_trigger_closes_mode_gates_and_enqueues() -> None:
    async def run() -> None:
        for mode, closed_gates in [
            (ValidationLoopMode.PAUSE_TRAINER, ("trainer",)),
            (
                ValidationLoopMode.PAUSE_TRAINER_AND_DATA_INPUT,
                ("trainer", "data_input"),
            ),
            (ValidationLoopMode.DRAIN_TRAINER, ("data_input", "generator_pull")),
        ]:
            manager, _, _ = _weight_sync(gate=_open_gate())
            controller = _controller(loop_mode=mode, weight_sync=manager)
            await controller._trigger_periodic_validation(step=2)

            gates = {
                "trainer": controller._validation_gate_trainer,
                "data_input": controller._validation_gate_data_input,
                "generator_pull": controller._validation_gate_generator_pull,
            }
            for name, gate in gates.items():
                assert gate.is_set() != (name in closed_gates), (mode, name)
            assert not controller._validation_idle.is_set()
            assert controller._validation_trigger.get_nowait() == 2

    asyncio.run(run())


def test_trigger_skips_when_pass_running_without_settling() -> None:
    async def run() -> None:
        settled = []

        class _SentinelWeightSync:
            async def wait_inflight_push_pull(self):
                settled.append(True)

        controller = _controller(weight_sync=_SentinelWeightSync())
        controller._validation_idle.clear()  # a pass is in flight
        await controller._trigger_periodic_validation(step=4)

        assert settled == []  # skip-if-running pays nothing
        assert controller._validation_trigger.empty()

    asyncio.run(run())


def test_trigger_settles_inflight_sync_before_closing_gates() -> None:
    async def run() -> None:
        order: list[str] = []

        class _SentinelWeightSync:
            async def wait_inflight_push_pull(self):
                order.append("settle")

        controller = _controller(weight_sync=_SentinelWeightSync())
        await controller._trigger_periodic_validation(step=2)
        order.append("gates_closed" if not controller._validation_gate_trainer.is_set() else "gates_open")
        assert order == ["settle", "gates_closed"]

    asyncio.run(run())


def test_worker_pass_reopens_gates_on_success() -> None:
    async def run() -> None:
        manager, router, _ = _weight_sync(gate=_open_gate())
        controller = _controller(
            loop_mode=ValidationLoopMode.PAUSE_TRAINER, weight_sync=manager
        )
        # PAUSE: the trainer is parked at the trigger version, so completion == trigger step.
        controller._trainer_policy_version = 2
        worker = asyncio.create_task(controller._periodic_validation_loop())

        await controller._trigger_periodic_validation(step=2)
        await asyncio.wait_for(controller._validation_idle.wait(), timeout=5)

        assert controller.metrics_processor.logged == [2]
        assert controller._validation_gate_trainer.is_set()
        assert controller._validation_gate_data_input.is_set()
        assert controller._validation_gate_generator_pull.is_set()

        worker.cancel()

    asyncio.run(run())


def test_worker_failure_keeps_gates_closed() -> None:
    async def run() -> None:
        manager, _, _ = _weight_sync(gate=_open_gate())
        controller = _controller(
            loop_mode=ValidationLoopMode.PAUSE_TRAINER_AND_DATA_INPUT,
            weight_sync=manager,
            rollouter=_FakeRollouter(fail_all=True),
        )

        # Make the pass itself fail (all groups fail -> no raise; force one via recorder).
        def failing_record(*, is_validation, rollout_groups):
            raise RuntimeError("recorder exploded")

        controller._validator._rollout_recorder.record = failing_record

        worker = asyncio.create_task(controller._periodic_validation_loop())
        await controller._trigger_periodic_validation(step=2)

        with pytest.raises(RuntimeError, match="recorder exploded"):
            await asyncio.wait_for(worker, timeout=5)

        # Gates stay closed; idle stays cleared. run() would observe the dead task.
        assert not controller._validation_gate_trainer.is_set()
        assert not controller._validation_gate_data_input.is_set()
        assert not controller._validation_idle.is_set()

    asyncio.run(run())


def test_drain_worker_runs_catch_up_and_reopens() -> None:
    async def run() -> None:
        gate = _open_gate()
        manager, router, _ = _weight_sync(gate=gate)
        # Hold the pass open until both drain steps have pushed, so their pulls are
        # verifiably skipped (not raced by a fast pass reopening the gate first).
        hold = asyncio.Event()
        controller = _controller(
            loop_mode=ValidationLoopMode.DRAIN_TRAINER,
            weight_sync=manager,
            rollouter=_FakeRollouter(hold=hold),
        )
        worker = asyncio.create_task(controller._periodic_validation_loop())

        await controller._trigger_periodic_validation(step=2)
        # Simulate two DRAIN trainer steps: pushes with the gate closed -> skipped pulls.
        manager.start_async_push_pull(version=3)
        await manager.wait_prev_pull()
        manager.start_async_push_pull(version=4)
        await manager.wait_prev_pull()
        assert router.pulled_versions == []  # both pulls skipped while eval runs

        hold.set()
        await asyncio.wait_for(controller._validation_idle.wait(), timeout=5)

        assert router.pulled_versions == [4]  # one coalesced catch-up pull, latest version
        assert controller._validation_gate_data_input.is_set()
        assert controller._validation_gate_generator_pull.is_set()
        assert controller._validation_gate_trainer.is_set()

        worker.cancel()

    asyncio.run(run())


def test_drain_pass_logs_at_completion_step_not_trigger_step() -> None:
    # W&B monotonicity: the pass triggered at step 2 finishes after the trainer logged
    # step 4 -> the validation record must carry backend step 4 (never < the trainer's
    # current step), while validation/policy_version still reports the frozen policy 2.
    async def run() -> None:
        gate = _open_gate()
        manager, router, _ = _weight_sync(gate=gate)
        hold = asyncio.Event()
        controller = _controller(
            loop_mode=ValidationLoopMode.DRAIN_TRAINER,
            weight_sync=manager,
            rollouter=_FakeRollouter(policy_version=2, hold=hold),
        )
        controller._trainer_policy_version = 2
        worker = asyncio.create_task(controller._periodic_validation_loop())

        await controller._trigger_periodic_validation(step=2)
        # Two DRAIN trainer steps advance the trainer to 4 while the pass runs.
        for drained_version in (3, 4):
            controller._trainer_policy_version = drained_version
            manager.start_async_push_pull(version=drained_version)
            await manager.wait_prev_pull()

        hold.set()
        await asyncio.wait_for(controller._validation_idle.wait(), timeout=5)

        assert controller.metrics_processor.logged == [4]  # completion step, monotonic
        policy_versions = [
            metric.value.value
            for metric in controller.metrics_processor.logged_metrics[0]
            if metric.key == "validation/policy_version"
        ]
        assert policy_versions == [2.0]  # the frozen policy actually evaluated

        worker.cancel()

    asyncio.run(run())


def test_max_num_seqs_concurrency_uses_validator_num_samples() -> None:
    # Regression: the vLLM sizing math reads the nested validator config (the
    # bitwise-parity helper reproduces this exact expression); the old flat
    # field on ValidationConfig is gone.
    from torchtitan.experiments.rl.controller import ValidationConfig

    validation = ValidationConfig(validator=Validator.Config(num_samples=500))
    assert not hasattr(validation, "num_samples")
    num_group_workers, group_size = 4, 8
    rollout_concurrency = max(
        num_group_workers * group_size,
        validation.validator.num_samples,
    )
    assert rollout_concurrency == 500


def test_drain_catch_up_waits_active_trainer_step() -> None:
    async def run() -> None:
        gate = _open_gate()
        manager, router, _ = _weight_sync(gate=gate)
        hold = asyncio.Event()
        controller = _controller(
            loop_mode=ValidationLoopMode.DRAIN_TRAINER,
            weight_sync=manager,
            rollouter=_FakeRollouter(hold=hold),
        )
        worker = asyncio.create_task(controller._periodic_validation_loop())

        await controller._trigger_periodic_validation(step=2)
        # One DRAIN trainer step pushes (pull skipped) while eval is in flight.
        manager.start_async_push_pull(version=3)
        await manager.wait_prev_pull()
        assert router.pulled_versions == []

        # A consumed trainer batch holds the lock; the worker's catch-up must wait it out.
        async with controller._trainer_step_active:
            hold.set()  # the pass finishes, but catch-up blocks on this lock
            await asyncio.sleep(0.05)
            assert router.pulled_versions == []  # catch-up blocked behind the step

        await asyncio.wait_for(controller._validation_idle.wait(), timeout=5)
        assert router.pulled_versions == [3]

        worker.cancel()

    asyncio.run(run())


def test_data_input_admission_barrier_is_exact() -> None:
    async def run() -> None:
        controller = _controller(loop_mode=ValidationLoopMode.PAUSE_TRAINER_AND_DATA_INPUT)

        admitted: list[int] = []
        release_admission = asyncio.Event()

        async def fake_admission():
            # Mimics _data_input_loop's admission block: gate check + add under the lock.
            async with controller._data_admission_active:
                if not controller._validation_gate_data_input.is_set():
                    return
                await release_admission.wait()
                admitted.append(1)

        admission = asyncio.create_task(fake_admission())
        await asyncio.sleep(0)  # admission holds the lock now

        controller._validation_gate_data_input.clear()

        async def barrier():
            async with controller._data_admission_active:
                pass

        barrier_task = asyncio.create_task(barrier())
        await asyncio.sleep(0)
        assert not barrier_task.done()  # barrier waits out the in-flight admission

        release_admission.set()
        await barrier_task
        await admission
        # The in-flight admission completed BEFORE the barrier returned; after the barrier,
        # a fresh admission attempt sees the closed gate and admits nothing.
        assert admitted == [1]
        await fake_admission()
        assert admitted == [1]

    asyncio.run(run())


def test_healthy_finish_worker_failure_does_not_hang() -> None:
    # The run() healthy-finish race: worker fails after the trainer finished -> the
    # asyncio.wait on {worker, idle_waiter} returns via the worker, not a hang.
    async def run() -> None:
        idle = asyncio.Event()  # never set: the pass failed

        async def failing_worker():
            raise RuntimeError("pass failed after trainer finish")

        worker = asyncio.create_task(failing_worker())
        idle_waiter = asyncio.create_task(idle.wait())
        done, _ = await asyncio.wait_for(
            asyncio.wait([worker, idle_waiter], return_when=asyncio.FIRST_COMPLETED),
            timeout=5,
        )
        assert worker in done
        idle_waiter.cancel()
        with pytest.raises(RuntimeError, match="pass failed after trainer finish"):
            await worker

    asyncio.run(run())
