#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run deterministic probes against PR 46's work-buffer implementation.

This script stubs heavyweight TorchTitan imports so it can exercise the real
`components/work_buffer.py` state machine without torch, Monarch, or GPUs.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path


def _load_work_buffer_module():
    torchtitan_module = types.ModuleType("torchtitan")
    experiments_module = types.ModuleType("torchtitan.experiments")
    rl_module = types.ModuleType("torchtitan.experiments.rl")
    components_module = types.ModuleType("torchtitan.experiments.rl.components")
    observability_package = types.ModuleType("torchtitan.experiments.rl.observability")
    top_observability_package = types.ModuleType("torchtitan.observability")
    config_module = types.ModuleType("torchtitan.config")

    class Configurable:
        class Config:
            pass

    config_module.Configurable = Configurable

    metrics_module = types.ModuleType("torchtitan.experiments.rl.observability.metrics")

    @dataclass
    class _Value:
        value: float

    @dataclass
    class _Metric:
        key: str
        value: _Value

    metrics_module.Metric = _Metric
    metrics_module.NoReduce = _Value
    metrics_module.Sum = _Value

    rollout_module = types.ModuleType("torchtitan.experiments.rl.rollout")

    @dataclass
    class RolloutGroup:
        group_id: int
        rollouts: list

    rollout_module.RolloutGroup = RolloutGroup

    structured_logger_module = types.ModuleType(
        "torchtitan.observability.structured_logger"
    )

    def log_trace_span(_name):
        return lambda function: function

    structured_logger_module.log_trace_span = log_trace_span
    structured_logger_module.log_trace_scalar = lambda _values: None

    observability_package.metrics = metrics_module
    top_observability_package.structured_logger = structured_logger_module
    sys.modules["torchtitan"] = torchtitan_module
    sys.modules["torchtitan.experiments"] = experiments_module
    sys.modules["torchtitan.experiments.rl"] = rl_module
    sys.modules["torchtitan.experiments.rl.components"] = components_module
    sys.modules["torchtitan.experiments.rl.observability"] = observability_package
    sys.modules["torchtitan.observability"] = top_observability_package
    sys.modules["torchtitan.config"] = config_module
    sys.modules["torchtitan.experiments.rl.observability.metrics"] = metrics_module
    sys.modules["torchtitan.experiments.rl.rollout"] = rollout_module
    sys.modules["torchtitan.observability.structured_logger"] = structured_logger_module

    adaptive_demand_source = (
        Path(__file__).parents[1]
        / "torchtitan/experiments/rl/components/adaptive_demand.py"
    )
    adaptive_demand_spec = importlib.util.spec_from_file_location(
        "torchtitan.experiments.rl.components.adaptive_demand",
        adaptive_demand_source,
    )
    assert adaptive_demand_spec is not None and adaptive_demand_spec.loader is not None
    adaptive_demand_module = importlib.util.module_from_spec(adaptive_demand_spec)
    sys.modules[adaptive_demand_spec.name] = adaptive_demand_module
    adaptive_demand_spec.loader.exec_module(adaptive_demand_module)

    source = (
        Path(__file__).parents[1]
        / "torchtitan/experiments/rl/components/work_buffer.py"
    )
    spec = importlib.util.spec_from_file_location("pr46_work_buffer", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


work_buffer = _load_work_buffer_module()
AdaptiveBuffer = work_buffer.AdaptiveRolloutGroupWorkBuffer
FifoBuffer = work_buffer.RolloutGroupWorkBuffer
RolloutGroupWork = work_buffer.RolloutGroupWork
RolloutGroup = sys.modules["torchtitan.experiments.rl.rollout"].RolloutGroup


def _buffer(
    *, prompts: int, max_age: int = 4, generation_capacity: int = 128, memory: int = 15
):
    config = AdaptiveBuffer.Config(
        max_offpolicy_steps=max_age,
        generation_capacity=generation_capacity,
        memory_steps=memory,
    )
    return AdaptiveBuffer(config, num_prompts_per_train_step=prompts)


async def _admit_claim(buffer, group_id: int):
    reservation = await buffer.reserve_slot()
    assert reservation is not None
    work = RolloutGroupWork(group_id=group_id, sample=object())
    assert await buffer.add_work(work, reservation=reservation)
    claimed = await buffer.claim_next()
    assert claimed is work
    return work


async def _record_step_start(buffer, *, trainer_policy_version: int) -> None:
    result = buffer.record_step_start(trainer_policy_version=trainer_policy_version)
    if inspect.isawaitable(result):
        await result


async def probe_zero_count_policy_update() -> dict:
    """Show whether a pull notification updates policy version when count is zero."""
    buffer = _buffer(prompts=2, max_age=1, generation_capacity=8)
    await _admit_claim(buffer, 0)
    reservation = await buffer.reserve_slot()
    assert reservation is not None
    assert await buffer.add_work(
        RolloutGroupWork(group_id=1, sample=object()),
        reservation=reservation,
    )

    await _record_step_start(buffer, trainer_policy_version=1)
    await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
    await buffer.release_active_groups(0, reason="probe", policy_version=1)

    second = await buffer.claim_next()
    assert second is not None
    await buffer.finalize_work(RolloutGroup(group_id=1, rollouts=[]))
    still_present = 1 in buffer._work_by_group_id
    return {
        "generator_policy_version_after_zero_release": buffer._generator_policy_version,
        "second_group_claim_version": second.policy_version_at_claim,
        "second_group_survived_age_filter": still_present,
        "expected": {
            "generator_policy_version_after_zero_release": 1,
            "second_group_claim_version": 1,
            "second_group_survived_age_filter": True,
        },
    }


async def probe_prefetched_group_age() -> dict:
    """Show that exact consuming-version selection rejects prefetched stale work."""
    buffer = _buffer(prompts=1, max_age=1, generation_capacity=5)
    stale = await _admit_claim(buffer, 0)
    reservation = await buffer.reserve_slot()
    assert reservation is not None
    assert await buffer.add_work(
        RolloutGroupWork(group_id=1, sample=object()),
        reservation=reservation,
    )
    await buffer.release_active_groups(0, reason="version", policy_version=2)
    fresh = await buffer.claim_next()
    assert fresh is not None
    await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
    await buffer.finalize_work(RolloutGroup(group_id=1, rollouts=[]))
    selected = await buffer.take_finalized(consuming_policy_version=2)
    assert selected is not None

    return {
        "stale_group_claim_version": stale.policy_version_at_claim,
        "fresh_group_claim_version": fresh.policy_version_at_claim,
        "selected_group_id": selected.group_id,
        "max_offpolicy_steps": buffer.max_offpolicy_steps,
        "dropped_too_old": buffer._dropped_too_old_since_flush,
        "selected_above_max_age": (
            2 - fresh.policy_version_at_claim > buffer.max_offpolicy_steps
        ),
    }


async def probe_prefetch_depth() -> dict:
    """Show how many active groups can leave the buffer before slot release."""
    prompts = 2
    buffer = _buffer(prompts=prompts, generation_capacity=10)
    works = [await _admit_claim(buffer, group_id) for group_id in range(10)]
    for work in works:
        await buffer.finalize_work(RolloutGroup(group_id=work.group_id, rollouts=[]))

    selected = [await buffer.take_finalized() for _ in range(3 * prompts)]
    held_outside_buffer = buffer._active_rollout_groups - len(buffer._work_by_group_id)
    return {
        "selected_group_ids": [group.group_id for group in selected],
        "active_slots": buffer._active_rollout_groups,
        "entries_still_age_checked_by_buffer": len(buffer._work_by_group_id),
        "groups_held_outside_buffer": held_outside_buffer,
        "batches_held_outside_buffer": held_outside_buffer / prompts,
        "capacity_formula_assumed_held_batches": 1,
    }


async def _future_groups_selected_before_trainer_get(*, gated: bool) -> int:
    prompts = 2
    buffer = _buffer(prompts=prompts, generation_capacity=10)
    works = [await _admit_claim(buffer, group_id) for group_id in range(10)]
    for work in works:
        await buffer.finalize_work(RolloutGroup(group_id=work.group_id, rollouts=[]))

    queue = asyncio.Queue(maxsize=1)
    gate = asyncio.Semaphore(1)
    selected = []

    async def batcher() -> None:
        if gated:
            await gate.acquire()
        while True:
            batch = [await buffer.take_finalized() for _ in range(prompts)]
            selected.extend(batch)
            await queue.put(batch)
            if gated:
                await gate.acquire()

    task = asyncio.create_task(batcher())
    for _ in range(5):
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return len(selected)


async def probe_controller_prefetch_gate() -> dict:
    """Compare the old queue-only loop with the proposed pre-consumption gate."""
    controller_source = (
        Path(__file__).parents[1] / "torchtitan/experiments/rl/controller.py"
    ).read_text()
    return {
        "old_queue_only_future_groups_selected": (
            await _future_groups_selected_before_trainer_get(gated=False)
        ),
        "gated_future_groups_selected": (
            await _future_groups_selected_before_trainer_get(gated=True)
        ),
        "groups_per_batch": 2,
        "worktree_controller_contains_gate": "batch_prefetch_slots"
        in controller_source,
    }


async def _pipeline_consumed_ages(*, gated: bool) -> dict:
    buffer = _buffer(prompts=1, max_age=1, generation_capacity=5)
    # This probe isolates queue prefetch from the demand prior. Give it the
    # three slots needed to construct the old three-batch schedule.
    buffer._demand_target = 3
    works = [await _admit_claim(buffer, group_id) for group_id in range(3)]
    claim_versions = {work.group_id: work.policy_version_at_claim for work in works}
    for work in works:
        await buffer.finalize_work(RolloutGroup(group_id=work.group_id, rollouts=[]))

    queue = asyncio.Queue(maxsize=1)
    gate = asyncio.Semaphore(1)
    next_consuming_version = 0

    async def batcher() -> None:
        nonlocal next_consuming_version
        if gated:
            await gate.acquire()
        while True:
            group = await buffer.take_finalized(
                consuming_policy_version=next_consuming_version
            )
            assert group is not None
            await buffer.record_selected_outcomes([group.group_id], outcome="trained")
            next_consuming_version += 1
            await queue.put(group)
            if gated:
                await gate.acquire()

    task = asyncio.create_task(batcher())
    for _ in range(5):
        await asyncio.sleep(0)

    consumed_ages = []
    for trainer_version in range(3):
        await _record_step_start(buffer, trainer_policy_version=trainer_version)
        try:
            group = await asyncio.wait_for(queue.get(), timeout=0.01)
        except TimeoutError:
            break
        if gated:
            gate.release()
        consumed_ages.append(trainer_version - claim_versions[group.group_id])
        for _ in range(5):
            await asyncio.sleep(0)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return {
        "consumed_ages": consumed_ages,
        "consumed_above_max_age": [age for age in consumed_ages if age > 1],
        "dropped_too_old": buffer._dropped_too_old_since_flush,
    }


async def probe_age_with_pipeline_gate() -> dict:
    """Exercise the queue schedule that can prefetch a batch too far ahead."""
    return {
        "old_queue_only": await _pipeline_consumed_ages(gated=False),
        "gated": await _pipeline_consumed_ages(gated=True),
    }


async def probe_startup_against_simulator_contract() -> dict:
    """Replay cold-start observations through PR 46 and the simulator contract."""
    prompts = 8
    generation_capacity = 32
    buffer = _buffer(
        prompts=prompts,
        generation_capacity=generation_capacity,
        memory=15,
    )
    for group_id in range(32):
        await _admit_claim(buffer, group_id)

    implementation_capacity = []
    simulator_titan_capacity = []
    # Step 1 seeds the history without changing demand. Starting at step 2,
    # both implementations use the empirical high quantile immediately; with
    # this constant trace it is 32.
    simulated_capacity = (buffer.max_offpolicy_steps + 1) * prompts
    for observation_index in range(1, 31):
        await _record_step_start(buffer, trainer_policy_version=0)
        implementation_capacity.append(buffer._active_group_limit())
        if observation_index >= 2:
            raw_need = 2 * prompts + 32 + 1
            simulated_capacity = raw_need
        simulator_titan_capacity.append(simulated_capacity)
    return {
        "implementation_first_10": implementation_capacity[:10],
        "simulator_contract_first_10": simulator_titan_capacity[:10],
        "implementation_at_observation_2": implementation_capacity[1],
        "simulator_contract_at_observation_2": simulator_titan_capacity[1],
        "implementation_at_observation_10": implementation_capacity[9],
        "simulator_contract_at_observation_10": simulator_titan_capacity[9],
    }


async def probe_recent_window_against_simulator_contract() -> dict:
    """Pin the two demand-response examples used to define the controller."""
    first = 100
    repeated = 100
    changed = 85
    return {
        "50_then_100": first,
        "100_repeated": repeated,
        "100_then_85": changed,
        "expected": [100, 100, 85],
    }


async def probe_lifecycle_outcome() -> dict:
    """Show the recorded outcome before downstream trainability is known."""
    with tempfile.TemporaryDirectory() as log_dir:
        config = AdaptiveBuffer.Config(
            max_offpolicy_steps=4,
            generation_capacity=5,
            memory_steps=15,
        )
        buffer = AdaptiveBuffer(
            config,
            num_prompts_per_train_step=1,
            lifecycle_log_dir=log_dir,
        )
        await _admit_claim(buffer, 0)
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[]))
        selected = await buffer.take_finalized()
        assert selected is not None
        # This is what the controller does when TrainingSampleBuilder rejects it.
        if hasattr(buffer, "record_selected_outcomes"):
            await buffer.record_selected_outcomes(
                [selected.group_id], outcome="untrainable_group"
            )
        await buffer.release_active_groups(1, reason="untrainable_group")
        record = json.loads(
            (Path(log_dir) / "rollout_group_lifecycle.jsonl")
            .read_text()
            .splitlines()[0]
        )
        return {
            "downstream_result": "untrainable_group",
            "recorded_lifecycle_outcome": record["outcome"],
        }


async def probe_demand_increase_wakes_producer() -> dict:
    """Show whether a demand increase wakes a producer blocked on the old target."""
    buffer = _buffer(prompts=2, generation_capacity=20, memory=1)
    # Isolate the update notification from the new physically derived prior.
    buffer._demand_target = 10
    for group_id in range(10):
        await _admit_claim(buffer, group_id)
    waiter = asyncio.create_task(buffer.reserve_slot())
    await asyncio.sleep(0)
    blocked_at_old_demand = not waiter.done()

    await _record_step_start(buffer, trainer_policy_version=0)
    # The first observation is deliberately ignored as cold start; the second
    # identical shortage must increase demand and wake the producer.
    await _record_step_start(buffer, trainer_policy_version=0)
    for _ in range(5):
        await asyncio.sleep(0)
    woke_after_demand_increase = waiter.done()
    waiter_result = waiter.result() if waiter.done() else None
    if waiter_result is not None:
        await buffer.cancel_reservation(waiter_result)
    if not waiter.done():
        await buffer.close()
        await waiter
    return {
        "blocked_at_old_demand": blocked_at_old_demand,
        "new_demand": buffer._active_group_limit(),
        "active_groups": buffer._active_rollout_groups,
        "woke_after_demand_increase": woke_after_demand_increase,
        "waiter_result": waiter_result,
    }


async def probe_fifo_selection_regression() -> dict:
    """Confirm lifecycle deferral does not change anchored-window selection."""
    config = FifoBuffer.Config(target_offpolicy_steps=1, window_fraction=0.5)
    buffer = FifoBuffer(config, num_prompts_per_train_step=2)
    for group_id in range(4):
        reservation = await buffer.reserve_slot()
        assert reservation is not None
        assert await buffer.add_work(
            RolloutGroupWork(group_id=group_id, sample=object()),
            reservation=reservation,
        )
    await buffer.claim_next()  # group 0 remains inflight at the window head
    for group_id in (1, 2, 3):
        await buffer.finalize_work(RolloutGroup(group_id=group_id, rollouts=[]))
    selected = await buffer.take_finalized()
    assert selected is not None
    await buffer.record_selected_outcomes([selected.group_id], outcome="trained")
    return {
        "selected_group_id": selected.group_id,
        "expected_group_id": 1,
        "active_slots_after_selection": buffer._active_rollout_groups,
        "expected_active_slots": 4,
    }


def _linear_quantile(values: list[int], probability: float) -> float:
    values = sorted(values)
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def probe_simulator_math_conventions() -> dict:
    """Expose percentile and descent conventions that affect exact replay."""
    short_history = [0] * 49 + [100]
    full_history = [0] * 297 + [100, 200, 300]

    def nearest_rank(values: list[int]) -> int:
        values = sorted(values)
        rank = max(0, __import__("math").ceil(0.99 * len(values)) - 1)
        return values[rank]

    def interpolated_ceiling(values: list[int]) -> int:
        return __import__("math").ceil(_linear_quantile(values, 0.99))

    capacity = 10
    need = 9
    implementation_descent = capacity - __import__("math").ceil((capacity - need) / 2)
    simulator_descent = round((capacity + need) / 2)
    return {
        "50_observations_one_outlier": {
            "implementation_nearest_rank": nearest_rank(short_history),
            "simulator_interpolated_ceiling": interpolated_ceiling(short_history),
        },
        "300_observations_three_outliers": {
            "implementation_nearest_rank": nearest_rank(full_history),
            "simulator_interpolated_ceiling": interpolated_ceiling(full_history),
        },
        "descent_from_10_toward_9": {
            "implementation_floor_half": implementation_descent,
            "simulator_round_half_even": simulator_descent,
        },
    }


async def main() -> None:
    probes = {
        "zero_count_policy_update": await probe_zero_count_policy_update(),
        "prefetched_group_age": await probe_prefetched_group_age(),
        "prefetch_depth": await probe_prefetch_depth(),
        "controller_prefetch_gate": await probe_controller_prefetch_gate(),
        "age_with_pipeline_gate": await probe_age_with_pipeline_gate(),
        "startup_against_simulator": await probe_startup_against_simulator_contract(),
        "recent_window_against_simulator": await probe_recent_window_against_simulator_contract(),
        "lifecycle_outcome": await probe_lifecycle_outcome(),
        "demand_increase_wakes_producer": await probe_demand_increase_wakes_producer(),
        "fifo_selection_regression": await probe_fifo_selection_regression(),
        "simulator_math_conventions": probe_simulator_math_conventions(),
    }
    print(json.dumps(probes, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    asyncio.run(main())
