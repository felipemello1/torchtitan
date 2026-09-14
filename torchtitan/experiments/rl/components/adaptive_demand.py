# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Demand rule of the adaptive rollout buffer: how many prompt groups may be in the pipeline."""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class StallDrivenDemand:
    """Set the demand from the trainer's own experience: a short shelf adds half a batch, comfort gives one back.

    At each step start the buffer reports the shelf (finished groups not yet trained) and the groups in flight.
    A shelf below one batch means the trainer is about to wait, so demand rises by `P // 2` at once. A shelf of at
    least two batches for `patience_steps` consecutive steps lowers demand by one. Demand starts at `5 P` and stays
    within `[2 P, ceiling]`.

    Growth is refused, and `state` names the reason, when more demand cannot help:
        "age-limited"       more than `max_drop_share` of the groups completed in the last `guard_window_steps`
                            were dropped as too old: the age cap binds, extra demand becomes stale work
        "generation-bound"  the generators hold every permit (`inflight >= generation_capacity`) and deliver
                            clearly fewer trainable groups per step than the trainer consumes (below 0.9 P)

    Example:
        demand = StallDrivenDemand(num_prompts_per_train_step=8, ceiling=168, generation_capacity=128)
        demand.observe(ready=5, inflight=30, completed=12, trainable=8, dropped=0)   # -> 44: shelf short, +4
        for _ in range(20):
            demand.observe(ready=16, inflight=30, completed=12, trainable=8, dropped=0)
        demand.demand                                                  # -> 43: twenty comfortable steps, -1
    """

    num_prompts_per_train_step: int
    ceiling: int
    generation_capacity: int
    patience_steps: int = 20
    max_drop_share: float = 0.15
    guard_window_steps: int = 10
    demand: int = field(init=False)
    state: str = field(default="ok", init=False)
    _comfortable_steps: int = field(default=0, init=False)
    _dropped: deque = field(init=False)
    _completed: deque = field(init=False)
    _trainable: deque = field(init=False)

    def __post_init__(self) -> None:
        self.demand = min(self.ceiling, 5 * self.num_prompts_per_train_step)
        self._dropped = deque(maxlen=self.guard_window_steps)
        self._completed = deque(maxlen=self.guard_window_steps)
        self._trainable = deque(maxlen=self.guard_window_steps)

    def observe(
        self, *, ready: int, inflight: int, completed: int, trainable: int, dropped: int
    ) -> int:
        """Update the demand from one step start.

        Args:
            ready: Finished groups not yet trained (finalized, selected, or queued for the trainer).
            inflight: Groups generating (waiting or in flight).
            completed: Groups that finished since the previous step start.
            trainable: Of those, groups with a learning signal (classified so far).
            dropped: Groups dropped as too old since the previous step start.
        """
        P = self.num_prompts_per_train_step
        self._dropped.append(dropped)
        self._completed.append(completed)
        self._trainable.append(trainable)
        # TODO: trigger on the previous step's wait (>= 1 s) instead of the shelf. Identical while pulls pause the
        #   generators (waits are 0 or > 10 s); with a fast sync it lands 2 points fewer drops at the same stall rate
        #   (discussions/86 claude/sim/out/compare_wait_trigger.txt). Needs the wait passed into `record_step_start`.
        if ready < P:
            self._comfortable_steps = 0
            drop_share = sum(self._dropped) / max(1, sum(self._completed))
            if (
                len(self._dropped) == self.guard_window_steps
                and drop_share > self.max_drop_share
            ):
                self.state = "age-limited"
                return self.demand
            trainable_per_step = sum(self._trainable) / max(1, len(self._trainable))
            # bound only when the flow is clearly short: a shelf can still be built from a small surplus
            if inflight >= self.generation_capacity and trainable_per_step < 0.9 * P:
                self.state = "generation-bound"
                return self.demand
            self.state = "ok"
            self.demand = min(self.ceiling, self.demand + max(1, P // 2))
        elif ready >= 2 * P:
            self._comfortable_steps += 1
            if self._comfortable_steps >= self.patience_steps:
                self._comfortable_steps = 0
                self.demand = max(2 * P, self.demand - 1)
        else:
            self._comfortable_steps = 0
        return self.demand
