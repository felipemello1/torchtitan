# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overlap the trainer->generator weight handoff with the next training step."""

import asyncio
import time

from torchtitan.experiments.rl.components.work_buffer import RolloutGroupWorkBuffer
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.observability import structured_logger as sl


# dummy no-op for step 0, used in WeightSyncManager
async def _noop() -> float:
    return 0.0


class WeightSyncManager:
    """Overlap the trainer->generator weight sync with the next training step.

     Trainer weight push:
        - Called after optimizer.step()
        - Awaited before next optimizer.step (weights changes then)
    Generator weight pull:
        - Called after push completes.
        - Awaited before next push (weights changes then)

    Impact on off-policiness: The buffer guarantees that no sample will be born stale,
    as long as we call `self._group_buffer.release_active_groups` after the pull.

    While `generator_pull_gate` is cleared (a DRAIN_TRAINER validation pass), each cycle
    pushes and releases trained slots but skips its generator pull; `catch_up_pull`
    then pulls only the newest pushed version. Safe because the controller keeps data
    admission closed until the catch-up succeeds, so released capacity cannot refill
    against the frozen generator.

    Example:
        for step in training_steps:
            fwd_bwd(batch)
            push_metrics = await weight_sync.wait_prev_push()    # before optim mutates the weights
            optim_result = await trainer.optim_step.call()
            pull_metrics = await weight_sync.wait_prev_pull()  # before the next push overwrites the key
            weight_sync.start_async_push_pull(version=optim_result.policy_version)
        await weight_sync.wait_inflight_push_pull()  # finish the last step's sync before validation
    """

    def __init__(
        self,
        *,
        trainer,  # PolicyTrainer actor handle
        generator_router: InterGeneratorRouter,
        group_buffer: RolloutGroupWorkBuffer,
        num_groups_per_train_step: int,
        generator_pull_gate: asyncio.Event,
    ) -> None:
        self._trainer = trainer
        self._generator_router = generator_router
        self._group_buffer = group_buffer
        self._num_groups_per_train_step = num_groups_per_train_step
        # set == open (normal); the controller clears it only for DRAIN_TRAINER validation passes.
        self._generator_pull_gate = generator_pull_gate
        self._latest_pushed_version: int | None = None
        self._num_skipped_pulls = 0

        # Step 0 has no `wait_prev_push/pull`, so we start with a noop task. Each task
        # returns its own wall-time so a skipped pull reports 0.0, not a previous pull's duration.
        self._trainer_push_task: asyncio.Task[float] = asyncio.create_task(_noop())
        self._generator_pull_task: asyncio.Task[float] = asyncio.create_task(_noop())

    def start_async_push_pull(self, *, version: int) -> None:
        """Fire push -> pull -> buffer-slot release in the background; returns immediately.

        Args:
            version: policy version the generators hold after the pull completes.
        """
        push_task = asyncio.create_task(self._trainer_push())
        self._trainer_push_task = push_task
        self._generator_pull_task = asyncio.create_task(
            self._generator_pull_and_release_buffer_slots(version, push_task)
        )

    async def wait_prev_push(self) -> list[m.Metric]:
        push_s = await self._trainer_push_task
        return [
            m.Metric(
                "timing/weight_sync/trainer_push_model_state_dict",
                m.NoReduce(push_s),
            )
        ]

    async def wait_prev_pull(self) -> list[m.Metric]:
        pull_s = await self._generator_pull_task
        return [
            m.Metric(
                "timing/weight_sync/generator_pull_model_state_dict",
                m.NoReduce(pull_s),
            )
        ]

    async def wait_inflight_push_pull(self) -> None:
        """Finish the last in-flight push+pull so generators hold the final weights (e.g. before validation)."""
        await self.wait_prev_push()
        await self.wait_prev_pull()

    async def catch_up_pull(self) -> int | None:
        """Pull the newest pushed version after skipped pulls; returns it (None if nothing was skipped).

        Called by the validation worker at the end of a DRAIN_TRAINER pass, under the
        controller's `_trainer_step_active` lock so no consumed trainer step is mid-flight.
        Registers itself as the current pull task, so the trainer's next `wait_prev_pull`
        (awaited before its next push) serializes that push behind this pull -- no torn
        read of the staged TorchStore key. Skipped-pull state resets only on success.
        """
        await self._trainer_push_task
        await self._generator_pull_task
        if self._num_skipped_pulls == 0:
            return None

        version = self._latest_pushed_version
        assert version is not None, "skipped pulls imply at least one push"
        pull_task = asyncio.create_task(self._generator_pull(version))
        self._generator_pull_task = pull_task
        await pull_task
        self._num_skipped_pulls = 0
        return version

    async def _trainer_push(self) -> float:
        with sl.log_trace_span("trainer_push_model_state_dict"):
            start = time.perf_counter()
            await self._trainer.push_model_state_dict.call()
            return time.perf_counter() - start

    async def _generator_pull(self, version: int) -> float:
        with sl.log_trace_span("generator_pull_model_state_dict"):
            start = time.perf_counter()
            await self._generator_router.pull_model_state_dict(policy_version=version)
            return time.perf_counter() - start

    async def _generator_pull_and_release_buffer_slots(
        self, version: int, push_task: asyncio.Task[float]
    ) -> float:
        await push_task
        self._latest_pushed_version = version

        pull_s = 0.0
        if self._generator_pull_gate.is_set():
            pull_s = await self._generator_pull(version)
        else:
            self._num_skipped_pulls += 1
        # TODO(perf): pull_model_state_dict awaits ALL generators before we release any buffer slots,
        #   so a generator that finishes its pull early idles until the slowest one. Investigate
        #   per-generator release (router surfaces each pull's completion -> release that generator's
        #   share / resume it early); needs the born-fresh invariant to hold per-generator, not globally.

        # Born-fresh: admit the next groups only now that the generators are on `version`, so a new
        # rollout starts at the current version (keeps policy_age <= max_offpolicy_steps). During a
        # skipped-pull (DRAIN) cycle the controller keeps data admission closed, so releasing here
        # cannot admit work against the frozen generator.
        await self._group_buffer.release_active_groups(
            self._num_groups_per_train_step, reason="trained"
        )
        return pull_s
