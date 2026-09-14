# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Active work buffer shared between the data-input, rollout, and batcher loops.
NOTE: The buffer holds work slots, and not the finalized RolloutGroups necessarily.

Two buffers share one interface, so the controller runs either one unchanged:

    RolloutGroupWorkBuffer             fixed `(target_offpolicy_steps + 1) * P`; windowed FIFO; never drops
    AdaptiveRolloutGroupWorkBuffer  demand set by the trainer's stalls, optionally capped by `target_offpolicy_steps`;
                                    takes the oldest finalized group; drops groups past `max_offpolicy_steps`
"""

import asyncio
import collections
import enum
import json
import math
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.components.adaptive_demand import StallDrivenDemand
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.observability import structured_logger as sl


class _RolloutGroupWorkState(enum.Enum):
    """Where a RolloutGroupWork is in the WAITING -> INFLIGHT -> FINALIZED lifecycle."""

    WAITING = "waiting"
    INFLIGHT = "inflight"
    FINALIZED = "finalized"


@dataclass(slots=True)
class RolloutGroupWork:
    """One prompt group's work, tracked through _RolloutGroupWorkState.

    The input loop sets `group_id` + `sample`; the buffer owns every other field
    (`init=False`, so the input loop can't set them).
    """

    group_id: int
    sample: object
    """Data input produced by `rollouter.get_training_sample()`;
    passed unchanged to the env in `rollouter.run_group_rollouts`."""
    state: _RolloutGroupWorkState = field(
        default=_RolloutGroupWorkState.WAITING, init=False
    )
    rollout_group: RolloutGroup | None = field(
        default=None, init=False
    )  # set once FINALIZED
    policy_version_at_admission: int = field(default=0, init=False)
    """Generator policy version when the group was admitted."""
    policy_version_at_claim: int | None = field(default=None, init=False)
    """Generator policy version when generation started; `None` while WAITING."""
    admitted_at: float = field(default=0.0, init=False)
    claimed_at: float | None = field(default=None, init=False)
    finalized_at: float | None = field(default=None, init=False)
    consuming_policy_version: int | None = field(default=None, init=False)
    """Exact trainer version of the batch that selected or rejected this group."""


@dataclass(frozen=True, slots=True)
class _PendingLifecycleRecord:
    """Lifecycle metadata retained after the rollout payload leaves the buffer."""

    group_id: int
    admitted_at: float
    claimed_at: float | None
    finalized_at: float | None
    policy_version_at_admission: int
    policy_version_at_claim: int | None
    consuming_policy_version: int | None

    @classmethod
    def from_work(cls, work: RolloutGroupWork) -> "_PendingLifecycleRecord":
        return cls(
            group_id=work.group_id,
            admitted_at=work.admitted_at,
            claimed_at=work.claimed_at,
            finalized_at=work.finalized_at,
            policy_version_at_admission=work.policy_version_at_admission,
            policy_version_at_claim=work.policy_version_at_claim,
            consuming_policy_version=work.consuming_policy_version,
        )


class _RolloutGroupLifecycleLog:
    """Append one JSON line per rollout group at its terminal lifecycle outcome.

    Example line (timestamps are `time.time()` seconds):
        {"group_id": 12, "outcome": "trained", "admitted_at": 1.7e9, "claimed_at": 1.7e9, "finalized_at": 1.7e9,
         "left_at": 1.7e9, "policy_version_at_admission": 3, "policy_version_at_claim": 3,
         "trainer_policy_version_at_exit": 5}
    """

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self._file = open(  # noqa: SIM115 - held open for the buffer lifetime
            os.path.join(log_dir, "rollout_group_lifecycle.jsonl"), "a", buffering=1
        )

    def close(self) -> None:
        """Flush and close the lifecycle JSONL file."""
        self._file.close()

    def record(
        self,
        work: RolloutGroupWork | _PendingLifecycleRecord,
        *,
        outcome: str,
        trainer_policy_version: int,
    ) -> None:
        self._file.write(
            json.dumps(
                {
                    "group_id": work.group_id,
                    "outcome": outcome,
                    "admitted_at": work.admitted_at,
                    "claimed_at": work.claimed_at,
                    "finalized_at": work.finalized_at,
                    "left_at": time.time(),
                    "policy_version_at_admission": work.policy_version_at_admission,
                    "policy_version_at_claim": work.policy_version_at_claim,
                    "trainer_policy_version_at_exit": trainer_policy_version,
                    "consuming_policy_version": work.consuming_policy_version,
                }
            )
            + "\n"
        )


class RolloutGroupWorkBuffer(Configurable):
    """Buffer of `RolloutGroupWork` shared between the data-input, rollout, and batcher loops.

    Each entry is a RolloutGroupWork moving WAITING -> INFLIGHT -> FINALIZED. An active-slot budget caps
    the pipeline at `max_active_rollout_groups = (target_offpolicy_steps + 1) * num_prompts_per_train_step`
    active slots; the batcher takes finalized groups within a fixed look-ahead window anchored at the
    oldest entry. A `window_size` of 1 gives strict FIFO.

    For details on the buffer's callers, check the diagram in the controller.py file.

    NOTE: a work slot is **NOT** released when marked as FINALIZED or taken by the batcher.
    Instead, it is only released on `release_active_groups` calls by the trainer
    or data filtering. This is done this way so that we can guarantee we never have more
    than `max_active_rollout_groups` in the entire pipeline (buffer+queue+training).
    Otherwise, we would produce born-stale examples.

    Entry lifecycle vs active slot:
        entry:        WAITING -> INFLIGHT -> FINALIZED -> removed by take_finalized()
        active slot:  charged by add_work() ............ freed by release_active_groups()

    Example:
        # target_offpolicy_steps=1, num_prompts_per_train_step=2 -> capacity=4
        buffer = RolloutGroupWorkBuffer.Config(target_offpolicy_steps=1).build(num_prompts_per_train_step=2)
        for work in (g0, g1, g2, g3):
            reservation = await buffer.reserve_slot()
            assert reservation is not None
            await buffer.add_work(work, reservation=reservation)  # reaches 4/4 active
        g0 = await buffer.take_finalized()                     # g0 leaves the dict; still 4/4 active
        g1 = await buffer.take_finalized()                     # g1 leaves the dict; still 4/4 active
        slot_task = asyncio.create_task(buffer.reserve_slot())  # waits: take_finalized did not free a slot
        assert not slot_task.done()
        await buffer.release_active_groups(2, reason="trained", policy_version=1)  # trainer pulled -> slots free
        reservation = await slot_task
        assert reservation is not None
        await buffer.cancel_reservation(reservation)

    Args:
        num_prompts_per_train_step: Prompt groups per train step (`P`).
        policy_version: Version the generators hold at start (the resumed step).
        lifecycle_log_dir: Where `rollout_group_lifecycle.jsonl` is appended;
            `None` disables it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        target_offpolicy_steps: int = 3
        """Target steady-state offpolicy steps used to set the active buffer size to
        `(S + 1) * P`. Observed offpolicy steps are not guaranteed to equal this
        target: when rollout generation is the bottleneck, the buffer may not fill
        and observed offpolicy steps will be lower. With strict FIFO, observed
        offpolicy steps cannot exceed this target. With windowed FIFO, it may exceed
        this target, up to `max_offpolicy_steps`. See
        ``torchtitan/experiments/rl/docs/windowed_fifo.md`` for details."""

        window_fraction: float | None = 0.3
        """FIFO look-ahead window expressed as a fraction of the buffer size.

        This allows the batcher to bypass an unfinished rollout group at the head of
        the queue and consume younger groups that are already finished. This may
        increase the offpoliciness of the bypassed group. Defaults to 0.3 following
        Section 6.2.4 of https://arxiv.org/pdf/2605.26494. Set to None for strict FIFO;
        otherwise, must be in `(0, 1]`."""

        def __post_init__(self) -> None:
            if self.target_offpolicy_steps < 0:
                raise ValueError(
                    f"target_offpolicy_steps must be >= 0, got {self.target_offpolicy_steps}"
                )
            if self.window_fraction is not None and not (0 < self.window_fraction <= 1):
                raise ValueError(
                    "window_fraction must be None or in (0, 1], got "
                    f"{self.window_fraction}"
                )

        def max_active_rollout_groups(self, num_prompts_per_train_step: int) -> int:
            """Active buffer size in prompt groups, ``B = (S + 1) * P``."""
            return (self.target_offpolicy_steps + 1) * num_prompts_per_train_step

        def max_concurrent_rollout_groups(self, num_prompts_per_train_step: int) -> int:
            """Fixed FIFO keeps its historical one-worker-per-active-slot behavior."""
            return self.max_active_rollout_groups(num_prompts_per_train_step)

        def window_size(self, num_prompts_per_train_step: int) -> int:
            """Derive the fixed FIFO look-ahead window from the configured fraction.

            Symbols:
                ``P``: prompts per train step (``num_prompts_per_train_step``).
                ``S``: target steady-state offpolicy steps (``target_offpolicy_steps``).
                ``f``: fraction of the active buffer visible to windowed FIFO
                    (``window_fraction``).
                ``B``: active buffer size in prompt groups, ``B = (S + 1) * P``.

            Returns:
                The FIFO look-ahead window size, ``max(1, floor(f * B))``. A value
                of 1 is strict FIFO.
            """
            if self.window_fraction is None:
                return 1
            return max(
                1,
                math.floor(
                    self.window_fraction
                    * self.max_active_rollout_groups(num_prompts_per_train_step)
                ),
            )

        def max_offpolicy_steps(self, num_prompts_per_train_step: int) -> int:
            """Return the worst case consume-time offpolicy bound.

            For active buffer size ``B``, window size ``W``, and prompts per
            train step ``P``, the bound is ``(B + W - 2) // P``.

            See ``torchtitan/experiments/rl/docs/windowed_fifo.md`` for the proof and
            a worked example.
            """
            return (
                self.max_active_rollout_groups(num_prompts_per_train_step)
                + self.window_size(num_prompts_per_train_step)
                - 2
            ) // num_prompts_per_train_step

    def __init__(
        self,
        config: Config,
        *,
        num_prompts_per_train_step: int,
        policy_version: int = 0,
        lifecycle_log_dir: str | None = None,
    ) -> None:
        self._num_prompts_per_train_step = num_prompts_per_train_step
        self._max_active_rollout_groups = config.max_active_rollout_groups(
            num_prompts_per_train_step
        )
        self._max_concurrent_rollout_groups = config.max_concurrent_rollout_groups(
            num_prompts_per_train_step
        )
        self._window_size = config.window_size(num_prompts_per_train_step)
        self._max_offpolicy_steps = config.max_offpolicy_steps(
            num_prompts_per_train_step
        )
        if (
            config.window_fraction is not None
            and config.window_fraction * self._max_active_rollout_groups < 1
        ):
            warnings.warn(
                f"window_fraction={config.window_fraction} is too small for "
                f"active_buffer_size={self._max_active_rollout_groups}; forcing "
                "window_size=1 (strict FIFO)",
                stacklevel=2,
            )
        self._init_shared_state(
            policy_version=policy_version, lifecycle_log_dir=lifecycle_log_dir
        )

    def _init_shared_state(
        self, *, policy_version: int, lifecycle_log_dir: str | None
    ) -> None:
        self._generator_policy_version = policy_version
        self._trainer_policy_version = policy_version
        self._trained_awaiting_release = (
            0  # groups in the batch being trained; slots held until the pull
        )
        self._active_rollout_groups = 0
        # metric: Per-flush peak active slots; reset on `.metrics()` call.
        self._active_rollout_groups_peak_since_flush = 0
        self._next_reservation = 0
        self._admission_reservations: set[int] = set()
        self._work_by_group_id: collections.OrderedDict[
            int, RolloutGroupWork
        ] = collections.OrderedDict()
        # TODO(async-rl): Current we use a condition that alerts ALL rollout workers. There is no need to
        # alert all of them. Consider changing it to an async queue + event.

        # One Condition guards all three waits (slot-free / claimable-WAITING / takeable-FINALIZED):
        # every mutation notify_all()s and waiters re-check their predicate.
        self._condition = asyncio.Condition()
        self._closed = False
        self._lifecycle_log = (
            _RolloutGroupLifecycleLog(lifecycle_log_dir)
            if lifecycle_log_dir is not None
            else None
        )
        self._selected_lifecycle_awaiting_outcome: dict[
            int, _PendingLifecycleRecord
        ] = {}

    @property
    def max_active_rollout_groups(self) -> int:
        """Most rollout groups admitted and not yet released at once."""
        return self._max_active_rollout_groups

    @property
    def max_offpolicy_steps(self) -> int:
        """Hard consume-time bound on `trainer_policy_version - min_policy_version`."""
        return self._max_offpolicy_steps

    @property
    def max_concurrent_rollout_groups(self) -> int:
        """Maximum number of groups admitted to generation at once."""
        return self._max_concurrent_rollout_groups

    def _active_group_limit(self) -> int:
        """Active groups the buffer may admit right now."""
        return self._max_active_rollout_groups

    def _has_active_slot_available(self) -> bool:
        generation_pending = sum(
            work.state
            in (_RolloutGroupWorkState.WAITING, _RolloutGroupWorkState.INFLIGHT)
            for work in self._work_by_group_id.values()
        )
        return (
            self._active_rollout_groups + len(self._admission_reservations)
            < self._active_group_limit()
            and generation_pending + len(self._admission_reservations)
            < self._max_concurrent_rollout_groups
        )

    async def reserve_slot(self) -> int | None:
        """Atomically reserve one active and generation slot.

        Example:
            # None means the buffer was closed, so the data input loop exits.
            group_index = 0
            while (reservation := await buffer.reserve_slot()) is not None:
                await buffer.add_work(
                    RolloutGroupWork(group_id=group_index, sample=sample),
                    reservation=reservation,
                )
                group_index += 1
        """
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or self._has_active_slot_available()
            )
            if self._closed:
                return None
            reservation = self._next_reservation
            self._next_reservation += 1
            self._admission_reservations.add(reservation)
            return reservation

    async def cancel_reservation(self, reservation: int) -> None:
        """Release an unused admission reservation, for example after data-input failure."""
        async with self._condition:
            if reservation not in self._admission_reservations:
                return
            self._admission_reservations.remove(reservation)
            self._condition.notify_all()

    async def add_work(self, work: RolloutGroupWork, *, reservation: int) -> bool:
        """Commit a reserved slot as WAITING; return false if close won the race."""
        async with self._condition:
            if self._closed:
                self._admission_reservations.discard(reservation)
                self._condition.notify_all()
                return False
            if reservation not in self._admission_reservations:
                raise RuntimeError(
                    f"unknown or already consumed admission reservation {reservation}"
                )
            self._admission_reservations.remove(reservation)
            self._active_rollout_groups += 1
            self._active_rollout_groups_peak_since_flush = max(
                self._active_rollout_groups_peak_since_flush,
                self._active_rollout_groups,
            )
            work.policy_version_at_admission = self._generator_policy_version
            work.admitted_at = time.time()
            self._work_by_group_id[work.group_id] = work
            self._condition.notify_all()
            return True

    async def claim_next(self) -> RolloutGroupWork | None:
        """Rollout loop: claim the oldest WAITING group (WAITING -> INFLIGHT). None once closed."""
        async with self._condition:
            while True:
                if self._closed:
                    return None
                for work in self._work_by_group_id.values():
                    if work.state is _RolloutGroupWorkState.WAITING:
                        work.state = _RolloutGroupWorkState.INFLIGHT
                        work.policy_version_at_claim = self._generator_policy_version
                        work.claimed_at = time.time()
                        return work
                await self._condition.wait()

    async def finalize_work(self, rollout_group: RolloutGroup) -> None:
        """Rollout loop: store the produced RolloutGroup on its work entry (INFLIGHT -> FINALIZED) and wake the batcher."""
        async with self._condition:
            work = self._work_by_group_id.get(rollout_group.group_id)
            if work is None:
                # run()'s shutdown called close(); drop the result.
                return
            work.rollout_group = rollout_group
            work.state = _RolloutGroupWorkState.FINALIZED
            work.finalized_at = time.time()
            self._condition.notify_all()

    @sl.log_trace_span("take_finalized")
    async def take_finalized(
        self, *, consuming_policy_version: int | None = None
    ) -> RolloutGroup | None:
        """Batcher loop: return the oldest FINALIZED group inside the anchored windowed FIFO range.

        ``consuming_policy_version`` is accepted for the shared buffer interface;
        fixed FIFO does not perform consume-time age eviction.

        The window covers group ids ``[head, head + window_size - 1]``. Entries outside the
        window stay blocked even if they are finalized, so taking non-head groups does not slide
        the window. A `window_size` of 1 gives strict FIFO.

        This anchored-window policy follows MiniMax's rollout scheduling approach; see
        Section 6.2.4 of https://arxiv.org/pdf/2605.26494.

        Example:
            # window_size=3: g0 is INFLIGHT, g1 is WAITING, and g2/g3 are FINALIZED.
            group = await buffer.take_finalized()
            assert group.group_id == 2  # g2 is inside the anchored window [g0, g2].
            # g3 remains blocked because taking g2 does not move the window past g0.
        """
        async with self._condition:
            while True:
                if self._closed:
                    return None
                if self._work_by_group_id:
                    head_group_id = next(iter(self._work_by_group_id))
                    window_end = head_group_id + self._window_size - 1
                    for group_id, work in self._work_by_group_id.items():
                        if group_id > window_end:
                            break
                        if work.state is not _RolloutGroupWorkState.FINALIZED:
                            continue
                        work.consuming_policy_version = consuming_policy_version
                        return self._remove_work(work, outcome=None).rollout_group
                await self._condition.wait()  # nothing finalized inside the window -> stall

    def _remove_work(
        self, work: RolloutGroupWork, *, outcome: str | None
    ) -> RolloutGroupWork:
        """Remove an entry from the buffer, log its lifecycle, and wake waiters. Caller holds the condition."""
        del self._work_by_group_id[work.group_id]
        if outcome is None and self._lifecycle_log is not None:
            self._selected_lifecycle_awaiting_outcome[
                work.group_id
            ] = _PendingLifecycleRecord.from_work(work)
        elif self._lifecycle_log is not None:
            self._lifecycle_log.record(
                work,
                outcome=outcome,
                trainer_policy_version=self._trainer_policy_version,
            )
        self._condition.notify_all()
        return work

    async def record_selected_outcomes(
        self, group_ids: Sequence[int], *, outcome: str
    ) -> None:
        """Record a terminal outcome for selected rollout groups.

        Args:
            group_ids: Selected rollout group ids.
            outcome: Terminal result such as `"trained"` or `"untrainable_group"`.

        Example:
            await buffer.record_selected_outcomes([4, 7], outcome="trained")
        """
        async with self._condition:
            if outcome == "trained":
                # The trainer holds these slots until its weight pull releases them.
                self._trained_awaiting_release += len(group_ids)
            if self._lifecycle_log is None:
                return
            missing_group_ids = [
                group_id
                for group_id in group_ids
                if group_id not in self._selected_lifecycle_awaiting_outcome
            ]
            assert not missing_group_ids, (
                "selected rollout groups have no pending lifecycle record: "
                f"{missing_group_ids}"
            )
            for group_id in group_ids:
                work = self._selected_lifecycle_awaiting_outcome.pop(group_id)
                self._lifecycle_log.record(
                    work,
                    outcome=outcome,
                    trainer_policy_version=self._trainer_policy_version,
                )

    async def record_step_start(self, *, trainer_policy_version: int) -> None:
        """Trainer loop: called right before it waits for the next training batch.

        Args:
            trainer_policy_version: Version of the weights that will train the batch being waited for.
        """
        async with self._condition:
            self._trainer_policy_version = trainer_policy_version

    async def release_active_groups(
        self, count: int, *, reason: str, policy_version: int | None = None
    ) -> None:
        """Free active slots: the trainer releases trained slots after its weight pull; the batcher
        releases untrainable/filtered slots immediately.

        Args:
            count:  Number of rollout groups leaving the active window.
            reason: Metric suffix such as `"trained"` or `"untrainable_group"`.
            policy_version: Version the generators hold after the pull; set only when releasing trained slots.

        Example:
            # generators pulled v=7 after a step over 8 groups -> free their 8 slots
            await buffer.release_active_groups(8, reason="trained", policy_version=7)
            # batcher dropped one zero-std group -> free its single slot
            await buffer.release_active_groups(1, reason="untrainable_group")
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        async with self._condition:
            if count > self._active_rollout_groups:
                raise RuntimeError(
                    f"release_active_groups({count}) exceeds active count {self._active_rollout_groups}"
                )
            self._active_rollout_groups -= count
            if reason == "trained":
                self._trained_awaiting_release = max(
                    0, self._trained_awaiting_release - count
                )
            if policy_version is not None:
                self._generator_policy_version = policy_version
            sl.log_trace_scalar({f"rollout_buffer/released/{reason}": float(count)})
            self._condition.notify_all()

    async def close(self) -> None:
        """run() shutdown calls this once. Sets `_closed`, drops buffered work, and wakes every waiter.

        After this, all four waiters unblock and exit their loops: reserve_slot() returns None, and
        claim_next()/take_finalized() return None.
        """
        async with self._condition:
            self._closed = True
            self._admission_reservations.clear()
            for work in list(self._work_by_group_id.values()):
                self._remove_work(work, outcome="closed")
            if self._lifecycle_log is not None:
                for work in self._selected_lifecycle_awaiting_outcome.values():
                    self._lifecycle_log.record(
                        work,
                        outcome="closed",
                        trainer_policy_version=self._trainer_policy_version,
                    )
                self._lifecycle_log.close()
                self._lifecycle_log = None
            self._selected_lifecycle_awaiting_outcome.clear()
            self._condition.notify_all()

    def metrics(self) -> list[m.Metric]:
        """Trainer loop: point-in-time buffer gauges for this step; resets the per-flush peak."""
        states = [work.state for work in self._work_by_group_id.values()]
        state_enum = _RolloutGroupWorkState
        out = [
            m.Metric(
                "rollout_buffer/num_groups_waiting",
                m.NoReduce(float(states.count(state_enum.WAITING))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_inflight",
                m.NoReduce(float(states.count(state_enum.INFLIGHT))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_finalized",
                m.NoReduce(float(states.count(state_enum.FINALIZED))),
            ),
            m.Metric(
                "rollout_buffer/active_slots_in_use_peak",
                m.NoReduce(float(self._active_rollout_groups_peak_since_flush)),
            ),
            m.Metric(
                "rollout_buffer/available_active_slots",
                m.NoReduce(
                    float(
                        self._active_group_limit()
                        - self._active_rollout_groups
                        - len(self._admission_reservations)
                    )
                ),
            ),
            m.Metric(
                "rollout_buffer/reserved_admission_slots",
                m.NoReduce(float(len(self._admission_reservations))),
            ),
            m.Metric(
                "rollout_buffer/generation_capacity_groups",
                m.NoReduce(float(self._max_concurrent_rollout_groups)),
            ),
            m.Metric(
                "rollout_buffer/available_generation_permits",
                m.NoReduce(
                    float(
                        self._max_concurrent_rollout_groups
                        - states.count(state_enum.WAITING)
                        - states.count(state_enum.INFLIGHT)
                        - len(self._admission_reservations)
                    )
                ),
            ),
        ]
        # Next interval starts from the current gauge, not 0: slots stay occupied across a flush.
        self._active_rollout_groups_peak_since_flush = self._active_rollout_groups
        return out


class AdaptiveRolloutGroupWorkBuffer(RolloutGroupWorkBuffer):
    """Oldest-ready buffer with exact age eviction and stall-driven demand.

    Same callers and lifecycle as `RolloutGroupWorkBuffer`; three rules differ:

    1. Demand. At every step start the buffer measures the shelf (finished groups not yet trained) and hands
       it to `StallDrivenDemand`: a shelf below one batch raises demand by half a batch, two batches on the
       shelf for `patience_steps` steps lower it by one. Generator capacity ``C`` stays a separate deployment
       limit that admission enforces, so demand cannot enlarge vLLM concurrency; demand growth stops when the
       generators are full or when the age cap is what drops groups.
    2. Selection. `take_finalized` returns the oldest FINALIZED group wherever it sits; a slow group never
       blocks younger finished ones and keeps its slot until it finishes.
    3. Age. A finalized group that would be consumed more than `max_offpolicy_steps` versions after it was
       claimed is dropped (slot released at once, prompt not retried). The mean age follows demand, about
       `demand / P - 1` steps when the pipeline is full; `target_offpolicy_steps`, if set, caps demand at
       `(target_offpolicy_steps + 1) * P` so the mean age is bounded too, at the price of stalls.

    Example:
        buffer = AdaptiveRolloutGroupWorkBuffer.Config(
            generation_capacity=60
        ).build(num_prompts_per_train_step=8)
        await buffer.record_step_start(trainer_policy_version=12)   # shelf 5 < 8 -> demand 40 -> 44
        buffer.metrics()   # rollout_buffer/demand_target_groups 44, rollout_buffer/ready_at_step_start 5

        # g0 claimed at version 8; the batch being assembled trains at version 13 -> age 5 > 4 -> dropped
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[...]))
        await buffer.take_finalized(consuming_policy_version=13)
        # -> slot released with reason "too_old"; selection skips to the next finalized group
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        max_offpolicy_steps: int = 4
        """Bounds the age of EVERY trained group: a finalized group older than this at consumption is dropped
        (prompt not retried). Costs waste, never stalls."""

        target_offpolicy_steps: int | None = None
        """Bounds the MEAN age: caps the groups in the pipeline at `(target_offpolicy_steps + 1) * P`, the
        FIFO buffer's cap, so a trained group waited at most that many steps. Costs stalls when generation
        cannot fill a batch within the cap, never waste. None: demand is bounded by generation capacity only."""

        patience_steps: int = 20
        """Consecutive step starts with two batches on the shelf before demand drops by one group.
        Sets the equilibrium stall rate: one short shelf per `patience_steps * P // 2` comfortable steps."""

        max_drop_share: float = 0.15
        """Share of recent completions dropped as too old above which demand stops growing ("age-limited")."""

        generation_capacity: int | None = None
        """Fixed maximum prompt groups the generator service can hold.

        This deployment property is separate from learned reservoir demand. It
        sizes rollout workers and vLLM admission, and must be measured from the
        generator service rather than inferred from trainer-side demand.
        """

        def __post_init__(self) -> None:
            if self.max_offpolicy_steps < 1:
                raise ValueError(
                    f"max_offpolicy_steps must be >= 1, got {self.max_offpolicy_steps}"
                )
            if self.patience_steps < 1:
                raise ValueError(
                    f"patience_steps must be >= 1, got {self.patience_steps}"
                )
            if not (0 < self.max_drop_share < 1):
                raise ValueError(
                    f"max_drop_share must be in (0, 1), got {self.max_drop_share}"
                )
            if self.generation_capacity is None or self.generation_capacity < 1:
                raise ValueError(
                    "generation_capacity must be explicitly set to a positive "
                    "deployment limit"
                )
            if self.target_offpolicy_steps is not None and self.target_offpolicy_steps < 1:
                raise ValueError(
                    f"target_offpolicy_steps must be >= 1 or None, got {self.target_offpolicy_steps}"
                )

        def max_active_rollout_groups(self, num_prompts_per_train_step: int) -> int:
            """Maximum useful demand: generation capacity plus the age window, or the mean-age cap if set."""
            assert self.generation_capacity is not None
            ceiling = (
                self.generation_capacity
                + (self.max_offpolicy_steps + 1) * num_prompts_per_train_step
            )
            if self.target_offpolicy_steps is not None:
                ceiling = min(
                    ceiling,
                    (self.target_offpolicy_steps + 1) * num_prompts_per_train_step,
                )
            return ceiling

        def max_concurrent_rollout_groups(self, num_prompts_per_train_step: int) -> int:
            del num_prompts_per_train_step
            assert self.generation_capacity is not None
            return self.generation_capacity

    def __init__(
        self,
        config: Config,
        *,
        num_prompts_per_train_step: int,
        policy_version: int = 0,
        lifecycle_log_dir: str | None = None,
    ) -> None:
        self._num_prompts_per_train_step = num_prompts_per_train_step
        self._max_active_rollout_groups = config.max_active_rollout_groups(
            num_prompts_per_train_step
        )
        self._max_concurrent_rollout_groups = config.max_concurrent_rollout_groups(
            num_prompts_per_train_step
        )
        self._max_offpolicy_steps = config.max_offpolicy_steps
        self._demand = StallDrivenDemand(
            num_prompts_per_train_step=num_prompts_per_train_step,
            ceiling=self._max_active_rollout_groups,
            generation_capacity=self._max_concurrent_rollout_groups,
            patience_steps=config.patience_steps,
            max_drop_share=config.max_drop_share,
        )
        # metrics: the shelf and the unavailable count at the last step start; flow since the last step start
        self._ready_at_step_start = 0
        self._unavailable_at_step_start = 0
        self._completed_since_step_start = 0
        self._untrainable_since_step_start = 0
        self._dropped_since_step_start = 0
        self._dropped_too_old_since_flush = 0
        self._init_shared_state(
            policy_version=policy_version, lifecycle_log_dir=lifecycle_log_dir
        )

    def _active_group_limit(self) -> int:
        return self._demand.demand

    def _is_too_old(
        self, work: RolloutGroupWork, *, consuming_policy_version: int
    ) -> bool:
        assert work.policy_version_at_claim is not None
        return (
            consuming_policy_version - work.policy_version_at_claim
            > self._max_offpolicy_steps
        )

    def _drop_too_old(self, work: RolloutGroupWork) -> None:
        """Remove a finalized group past `max_offpolicy_steps` and free its slot at once. Caller holds the condition."""
        self._remove_work(work, outcome="dropped_too_old")
        self._active_rollout_groups -= 1
        self._dropped_too_old_since_flush += 1
        self._dropped_since_step_start += 1
        sl.log_trace_scalar({"rollout_buffer/released/too_old": 1.0})

    async def finalize_work(self, rollout_group: RolloutGroup) -> None:
        await super().finalize_work(rollout_group)
        self._completed_since_step_start += 1

    async def release_active_groups(
        self, count: int, *, reason: str, policy_version: int | None = None
    ) -> None:
        await super().release_active_groups(
            count, reason=reason, policy_version=policy_version
        )
        if reason == "untrainable_group":
            self._untrainable_since_step_start += count

    @sl.log_trace_span("take_finalized")
    async def take_finalized(
        self, *, consuming_policy_version: int | None = None
    ) -> RolloutGroup | None:
        """Batcher loop: return the oldest FINALIZED group, dropping any that became too old while waiting.

        Example:
            # g0 is INFLIGHT (slow), g1 and g2 are FINALIZED -> g1 is returned; g0 keeps its slot
            group = await buffer.take_finalized()
            assert group.group_id == 1
        """
        if consuming_policy_version is None:
            consuming_policy_version = self._trainer_policy_version
        async with self._condition:
            while True:
                if self._closed:
                    return None
                for work in list(self._work_by_group_id.values()):
                    if work.state is not _RolloutGroupWorkState.FINALIZED:
                        continue
                    work.consuming_policy_version = consuming_policy_version
                    if self._is_too_old(
                        work, consuming_policy_version=consuming_policy_version
                    ):
                        self._drop_too_old(work)
                        continue
                    return self._remove_work(work, outcome=None).rollout_group
                await self._condition.wait()  # nothing finalized -> stall

    async def record_step_start(self, *, trainer_policy_version: int) -> None:
        """Trainer loop, right before it waits for a batch: measure the shelf and update the demand.

        The shelf is everything admitted that is neither still generating nor held by the trainer:
        `active - (waiting + inflight) - trained awaiting release`, i.e. finalized, selected, and queued groups.
        """
        async with self._condition:
            self._trainer_policy_version = trainer_policy_version
            states = [work.state for work in self._work_by_group_id.values()]
            unavailable = states.count(_RolloutGroupWorkState.INFLIGHT) + states.count(
                _RolloutGroupWorkState.WAITING
            )
            ready = (
                self._active_rollout_groups
                - unavailable
                - self._trained_awaiting_release
            )
            self._unavailable_at_step_start = unavailable
            self._ready_at_step_start = ready
            previous_demand = self._demand.demand
            completed = self._completed_since_step_start
            self._demand.observe(
                ready=ready,
                inflight=unavailable,
                completed=completed,
                trainable=max(
                    0,
                    completed
                    - self._untrainable_since_step_start
                    - self._dropped_since_step_start,
                ),
                dropped=self._dropped_since_step_start,
            )
            self._completed_since_step_start = 0
            self._untrainable_since_step_start = 0
            self._dropped_since_step_start = 0
            if self._demand.demand > previous_demand:
                self._condition.notify_all()

    def metrics(self) -> list[m.Metric]:
        out = [
            *super().metrics(),
            m.Metric(
                "rollout_buffer/demand_target_groups",
                m.NoReduce(float(self._demand.demand)),
            ),
            m.Metric(
                "rollout_buffer/ready_at_step_start",
                m.NoReduce(float(self._ready_at_step_start)),
            ),
            m.Metric(
                "rollout_buffer/unavailable_at_step_start",
                m.NoReduce(float(self._unavailable_at_step_start)),
            ),
            m.Metric(
                "rollout_buffer/dropped_too_old",
                m.NoReduce(float(self._dropped_too_old_since_flush)),
            ),
            # 0 growing normally, 1 age-limited, 2 generation-bound
            m.Metric(
                "rollout_buffer/demand_growth_refused",
                m.NoReduce(
                    float(
                        {"ok": 0, "age-limited": 1, "generation-bound": 2}[
                            self._demand.state
                        ]
                    )
                ),
            ),
        ]
        self._dropped_too_old_since_flush = 0
        return out
