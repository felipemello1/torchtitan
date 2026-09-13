# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Active work buffer shared between the data-input, rollout, and batcher loops.
NOTE: The buffer holds work slots, and not the finalized RolloutGroups necessarily.

Two buffers share one interface, so the controller runs either one unchanged:

    RolloutGroupWorkBuffer          fixed capacity `(target_offpolicy_steps + 1) * P`; windowed FIFO; never drops
    AdaptiveRolloutGroupWorkBuffer  capacity learned from the run; takes the oldest finalized group;
                                    drops groups past `max_offpolicy_steps`
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


class _RolloutGroupLifecycleLog:
    """Append one JSON line per rollout group when it leaves the buffer (taken, dropped, or closed).

    Example line (timestamps are `time.time()` seconds):
        {"group_id": 12, "outcome": "taken", "admitted_at": 1.7e9, "claimed_at": 1.7e9, "finalized_at": 1.7e9,
         "left_at": 1.7e9, "policy_version_at_admission": 3, "policy_version_at_claim": 3,
         "trainer_policy_version_at_exit": 5}
    """

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self._file = open(
            os.path.join(log_dir, "rollout_group_lifecycle.jsonl"), "a", buffering=1
        )

    def record(
        self, work: RolloutGroupWork, *, outcome: str, trainer_policy_version: int
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
        await buffer.add_work(g0); await buffer.add_work(g1)   # 2/4 active
        await buffer.add_work(g2); await buffer.add_work(g3)   # 4/4 active (cap)
        g0 = await buffer.take_finalized()                     # g0 leaves the dict; still 4/4 active
        g1 = await buffer.take_finalized()                     # g1 leaves the dict; still 4/4 active
        slot_task = asyncio.create_task(buffer.wait_for_slot())  # waits: take_finalized did not free a slot
        assert not slot_task.done()
        await buffer.release_active_groups(2, reason="trained", policy_version=1)  # trainer pulled -> slots free
        assert await slot_task                                   # wait_for_slot now returns
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
            """Active buffer size in prompt groups, ``B = (S + 1) * P``; also the number of rollout workers."""
            return (self.target_offpolicy_steps + 1) * num_prompts_per_train_step

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
        """
        Args:
            num_prompts_per_train_step: Prompt groups per train step (`P`).
            policy_version: Version the generators hold at start (the resumed step).
            lifecycle_log_dir: Where `rollout_group_lifecycle.jsonl` is appended; `None` disables it.
        """
        self._num_prompts_per_train_step = num_prompts_per_train_step
        self._max_active_rollout_groups = config.max_active_rollout_groups(
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
        self._active_rollout_groups = 0
        # metric: Per-flush peak active slots; reset on `.metrics()` call.
        self._active_rollout_groups_peak_since_flush = 0
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
        # TODO(async-rl): warm start — admit a small number of groups at first and grow the effective cap as the
        # batcher consumes, so a cold start doesn't fill the whole off-policy window at policy version 0.

    @property
    def max_active_rollout_groups(self) -> int:
        """Most rollout groups ever active at once; the controller spawns one rollout worker per slot."""
        return self._max_active_rollout_groups

    @property
    def max_offpolicy_steps(self) -> int:
        """Hard consume-time bound on `trainer_policy_version - min_policy_version`."""
        return self._max_offpolicy_steps

    def _active_capacity(self) -> int:
        """Active slots the buffer may hold right now."""
        return self._max_active_rollout_groups

    def _has_active_slot_available(self) -> bool:
        return self._active_rollout_groups < self._active_capacity()

    async def wait_for_slot(self) -> bool:
        """Wait until one more rollout group may enter the active buffer.

        Example:
            # False means the buffer was closed, so the data input loop exits.
            group_index = 0
            while await buffer.wait_for_slot():
                await buffer.add_work(RolloutGroupWork(group_id=group_index, sample=sample))
                group_index += 1
        """
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or self._has_active_slot_available()
            )
            return not self._closed

    async def add_work(self, work: RolloutGroupWork) -> None:
        """Admit one rollout group as WAITING and charge one active slot."""
        async with self._condition:
            if not self._has_active_slot_available():
                raise RuntimeError(
                    f"{type(self).__name__}.add_work called without an active slot"
                )
            self._active_rollout_groups += 1
            self._active_rollout_groups_peak_since_flush = max(
                self._active_rollout_groups_peak_since_flush,
                self._active_rollout_groups,
            )
            work.policy_version_at_admission = self._generator_policy_version
            work.admitted_at = time.time()
            self._work_by_group_id[work.group_id] = work
            self._condition.notify_all()

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
    async def take_finalized(self) -> RolloutGroup | None:
        """Batcher loop: return the oldest FINALIZED group inside the anchored windowed FIFO range.

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
                        return self._remove_work(work, outcome="taken").rollout_group
                await self._condition.wait()  # nothing finalized inside the window -> stall

    def _remove_work(self, work: RolloutGroupWork, *, outcome: str) -> RolloutGroupWork:
        """Remove an entry from the buffer, log its lifecycle, and wake waiters. Caller holds the condition."""
        del self._work_by_group_id[work.group_id]
        if self._lifecycle_log is not None:
            self._lifecycle_log.record(
                work,
                outcome=outcome,
                trainer_policy_version=self._trainer_policy_version,
            )
        self._condition.notify_all()
        return work

    def record_step_start(self, *, trainer_policy_version: int) -> None:
        """Trainer loop: called right before it waits for the next training batch.

        Args:
            trainer_policy_version: Version of the weights that will train the batch being waited for.
        """
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
        if count == 0:
            return
        async with self._condition:
            if count > self._active_rollout_groups:
                raise RuntimeError(
                    f"release_active_groups({count}) exceeds active count {self._active_rollout_groups}"
                )
            self._active_rollout_groups -= count
            if policy_version is not None:
                self._generator_policy_version = policy_version
            sl.log_trace_scalar({f"rollout_buffer/released/{reason}": float(count)})
            self._condition.notify_all()

    async def close(self) -> None:
        """run() shutdown calls this once. Sets `_closed`, drops buffered work, and wakes every waiter.

        After this, all four waiters unblock and exit their loops: wait_for_slot() returns False, and
        claim_next()/take_finalized() return None.
        """
        async with self._condition:
            self._closed = True
            for work in list(self._work_by_group_id.values()):
                self._remove_work(work, outcome="closed")
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
                    float(self._active_capacity() - self._active_rollout_groups)
                ),
            ),
        ]
        # Next interval starts from the current gauge, not 0: slots stay occupied across a flush.
        self._active_rollout_groups_peak_since_flush = self._active_rollout_groups
        return out


class AdaptiveRolloutGroupWorkBuffer(RolloutGroupWorkBuffer):
    """Buffer whose capacity is learned from the run instead of set by a target offpolicy step count.

    Same callers and lifecycle as `RolloutGroupWorkBuffer`; three rules differ:

    1. Capacity. At every step start the trainer reports in, and the buffer counts the active slots that are
       not ready (`waiting + inflight`, the "unavailable" slots). It keeps the last `memory_steps` counts and
       sets the capacity so that on the worst `stall_probability` share of steps one batch is still ready:

           capacity = P (batch being trained) + P (batch that must be ready) + unavailable_bad_day + trend + 1

       where `unavailable_bad_day` is the `1 - stall_probability` percentile of the counts and `trend` is
       half the window's fitted rise of the inflight count, so a slowly lengthening workload is followed
       instead of discovered through stalls. The capacity moves up by at most `P // 2` per step and closes
       half the gap per step on the way down. It never exceeds `capacity_ceiling`.
    2. Selection. `take_finalized` returns the oldest FINALIZED group wherever it sits; a slow group never
       blocks younger finished ones and keeps its slot until it finishes.
    3. Age. A finalized group that would be consumed more than `max_offpolicy_steps` versions after it was
       claimed is dropped (slot released at once, prompt not retried). This is the only guarantee on age;
       the mean age follows the capacity: about `capacity / P - 1` steps when the pipeline is full.

    Example:
        # P=8, max_offpolicy_steps=4: starts at capacity 40; the trainer reports 30 unavailable slots at a
        # step start -> need = 8 + 8 + 30 + 0 + 1 = 47 -> capacity rises to 44 (at most P // 2 = 4 per step)
        buffer = AdaptiveRolloutGroupWorkBuffer.Config().build(num_prompts_per_train_step=8)
        buffer.record_step_start(trainer_policy_version=12)
        buffer.metrics()  # rollout_buffer/capacity 44, rollout_buffer/unavailable_at_step_start 30

        # g0 claimed at version 8; the batch being assembled trains at version 13 -> age 5 > 4 -> dropped
        await buffer.finalize_work(RolloutGroup(group_id=0, rollouts=[...]))
        # -> slot released with reason "too_old"; take_finalized() skips to the next finalized group
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        max_offpolicy_steps: int = 4
        """Oldest a group may be at consumption, in policy versions since generation started. Older
        finalized groups are dropped and their prompt is not retried."""

        stall_probability: float = 0.01
        """Share of step starts allowed to find fewer than one batch ready. Sizes the capacity percentile."""

        memory_steps: int = 300
        """Step starts remembered for the percentile; keep it at least `3 / stall_probability`.
        TODO(async-rl): a knob to compare against shorter or recency-weighted memories under drift."""

        capacity_ceiling: int = 128
        """Most rollout groups ever active (admitted and not yet trained); also the number of rollout workers."""

        def __post_init__(self) -> None:
            if self.max_offpolicy_steps < 1:
                raise ValueError(
                    f"max_offpolicy_steps must be >= 1, got {self.max_offpolicy_steps}"
                )
            if not (0 < self.stall_probability < 1):
                raise ValueError(
                    f"stall_probability must be in (0, 1), got {self.stall_probability}"
                )
            if self.memory_steps < 1:
                raise ValueError(f"memory_steps must be >= 1, got {self.memory_steps}")

        def max_active_rollout_groups(self, num_prompts_per_train_step: int) -> int:
            """The ceiling: the controller spawns one rollout worker per slot up to it."""
            if self.capacity_ceiling < 2 * num_prompts_per_train_step:
                raise ValueError(
                    f"capacity_ceiling={self.capacity_ceiling} must hold two batches: one training and one ready "
                    f"(2 * {num_prompts_per_train_step})"
                )
            return self.capacity_ceiling

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
        self._max_offpolicy_steps = config.max_offpolicy_steps
        self._stall_probability = config.stall_probability
        # Start at five batches: one training plus the four-batch reservoir the simulator study started from.
        self._capacity = min(
            self._max_active_rollout_groups, 5 * num_prompts_per_train_step
        )
        self._unavailable_history: collections.deque[int] = collections.deque(
            maxlen=config.memory_steps
        )
        self._inflight_history: collections.deque[int] = collections.deque(
            maxlen=config.memory_steps
        )
        # metrics: last observation and drops since the last `.metrics()` call
        self._unavailable_at_step_start = 0
        self._dropped_too_old_since_flush = 0
        self._init_shared_state(
            policy_version=policy_version, lifecycle_log_dir=lifecycle_log_dir
        )

    def _active_capacity(self) -> int:
        return self._capacity

    def _consuming_policy_version(self) -> int:
        """Version that will train the batch the batcher is assembling now.

        The trainer reports its version when it starts waiting for a batch; groups taken after that go into
        the following batch, trained one optimizer step later. While the trainer is stalled, groups completing
        the awaited batch train at the reported version, so this overestimates their age by one step.
        """
        return self._trainer_policy_version + 1

    def _is_too_old(self, work: RolloutGroupWork) -> bool:
        assert work.policy_version_at_claim is not None
        return (
            self._consuming_policy_version() - work.policy_version_at_claim
            > self._max_offpolicy_steps
        )

    def _drop_too_old(self, work: RolloutGroupWork) -> None:
        """Remove a finalized group past `max_offpolicy_steps` and free its slot at once. Caller holds the condition."""
        self._remove_work(work, outcome="dropped_too_old")
        self._active_rollout_groups -= 1
        self._dropped_too_old_since_flush += 1
        sl.log_trace_scalar({"rollout_buffer/released/too_old": 1.0})

    async def finalize_work(self, rollout_group: RolloutGroup) -> None:
        """Rollout loop: store the RolloutGroup (INFLIGHT -> FINALIZED), or drop it now if already too old."""
        await super().finalize_work(rollout_group)
        async with self._condition:
            work = self._work_by_group_id.get(rollout_group.group_id)
            if work is not None and self._is_too_old(work):
                self._drop_too_old(work)

    @sl.log_trace_span("take_finalized")
    async def take_finalized(self) -> RolloutGroup | None:
        """Batcher loop: return the oldest FINALIZED group, dropping any that became too old while waiting.

        Example:
            # g0 is INFLIGHT (slow), g1 and g2 are FINALIZED -> g1 is returned; g0 keeps its slot
            group = await buffer.take_finalized()
            assert group.group_id == 1
        """
        async with self._condition:
            while True:
                if self._closed:
                    return None
                for work in list(self._work_by_group_id.values()):
                    if work.state is not _RolloutGroupWorkState.FINALIZED:
                        continue
                    if self._is_too_old(work):
                        self._drop_too_old(work)
                        continue
                    return self._remove_work(work, outcome="taken").rollout_group
                await self._condition.wait()  # nothing finalized -> stall

    def record_step_start(self, *, trainer_policy_version: int) -> None:
        """Trainer loop: observe the unavailable slots and move the capacity toward what the bad days need.

        Example:
            # P=8, capacity 40, history of unavailable counts whose 99th percentile is 30, flat inflight trend
            buffer.record_step_start(trainer_policy_version=12)
            # need = 2 * 8 + 30 + 0 + 1 = 47 > 40 -> capacity 44 (up by at most P // 2 per step)
        """
        super().record_step_start(trainer_policy_version=trainer_policy_version)
        num_prompts = self._num_prompts_per_train_step
        states = [work.state for work in self._work_by_group_id.values()]
        num_inflight = states.count(_RolloutGroupWorkState.INFLIGHT)
        unavailable = num_inflight + states.count(_RolloutGroupWorkState.WAITING)
        self._unavailable_at_step_start = unavailable
        self._unavailable_history.append(unavailable)
        self._inflight_history.append(num_inflight)

        # Bad day: the (1 - stall_probability) rank of the remembered counts, plus half the window's fitted rise.
        history = sorted(self._unavailable_history)
        rank = max(0, math.ceil((1 - self._stall_probability) * len(history)) - 1)
        bad_day = history[rank]
        trend = math.ceil(
            max(0.0, _slope(self._inflight_history) * len(self._inflight_history) / 2)
        )
        need = 2 * num_prompts + bad_day + trend + 1
        need = min(self._max_active_rollout_groups, max(2 * num_prompts, need))

        # Up fast (at most half a batch per step); down by halving the gap so a stale history does not drain the stock.
        if need > self._capacity:
            self._capacity = min(need, self._capacity + max(1, num_prompts // 2))
        elif need < self._capacity:
            self._capacity -= math.ceil((self._capacity - need) / 2)

    def metrics(self) -> list[m.Metric]:
        out = [
            *super().metrics(),
            m.Metric("rollout_buffer/capacity", m.NoReduce(float(self._capacity))),
            m.Metric(
                "rollout_buffer/unavailable_at_step_start",
                m.NoReduce(float(self._unavailable_at_step_start)),
            ),
            m.Metric(
                "rollout_buffer/dropped_too_old",
                m.NoReduce(float(self._dropped_too_old_since_flush)),
            ),
        ]
        self._dropped_too_old_since_flush = 0
        return out


def _slope(values: Sequence[int]) -> float:
    """Least-squares slope of `values` against their index; 0 with fewer than two points.

    Example:
        _slope([10, 12, 14, 16])  # -> 2.0
    """
    n = len(values)
    if n < 2:
        return 0.0
    sum_x = n * (n - 1) / 2
    sum_xx = (n - 1) * n * (2 * n - 1) / 6
    sum_y = sum(values)
    sum_xy = sum(i * v for i, v in enumerate(values))
    return (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
