# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from torchtitan.config import Configurable


class SandboxCreationLimiter(Configurable):
    """Caps how many rollouts start per minute, and so how many sandboxes they create.

    Each Verifiers rollout creates one sandbox when it starts. Without a cap, a run's first
    wave reaches the provider while it is still scaling up, and it answers with HTTP 429.

        creates_per_minute=60, A, B, C arrive at t=0:  A starts at 0 s, B at 1 s, C at 2 s
        C is still waiting after max_wait_sec:         C raises TimeoutError, D takes its slot
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        creates_per_minute: float
        """Rollout starts per minute in this controller. Keep it between:
        - the run's steady-state start rate, rollouts per step / step time (below it, every
          step slows down);
        - how fast the provider adds sandboxes while it scales up (above it, HTTP 429)."""

        max_wait_sec: float = 3600.0
        """A rollout that waits this long for its slot ends as an ERROR rollout without
        starting, which drops its group from training. Cover the first wave: 1,536 rollouts
        at 60/min all start within 26 min."""

        def __post_init__(self) -> None:
            if self.creates_per_minute <= 0:
                raise ValueError(
                    f"creates_per_minute must be positive, got {self.creates_per_minute}"
                )
            if self.max_wait_sec <= 0:
                raise ValueError(
                    f"max_wait_sec must be positive, got {self.max_wait_sec}"
                )

    def __init__(self, config: Config) -> None:
        self.config = config
        self._interval_sec = 60 / config.creates_per_minute
        self._lock = asyncio.Lock()
        # Event-loop time at which the next rollout may start.
        self._next_start = 0.0

    async def wait_for_slot(self) -> float:
        """Wait for this rollout's start slot and return the seconds waited.

        Raises:
            TimeoutError: after `max_wait_sec`. Like a cancelled rollout, it takes no
                slot, so the next one in line starts on schedule.
        """
        loop = asyncio.get_running_loop()
        arrived = loop.time()
        try:
            async with asyncio.timeout(self.config.max_wait_sec):
                # asyncio.Lock wakes waiters in arrival order; the holder sleeps until its slot.
                async with self._lock:
                    await asyncio.sleep(self._next_start - loop.time())
                    self._next_start = loop.time() + self._interval_sec
        except TimeoutError:
            raise TimeoutError(
                f"waited {self.config.max_wait_sec:g}s for a sandbox creation slot at "
                f"{self.config.creates_per_minute:g}/min; raise creates_per_minute or "
                "max_wait_sec"
            ) from None
        return loop.time() - arrived
