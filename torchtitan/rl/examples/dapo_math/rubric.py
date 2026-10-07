# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.rl.examples.dapo_math.data import DapoMathSample
from torchtitan.rl.examples.dapo_math.grader import MathVerifyPool
from torchtitan.rl.rollout import Rollout
from torchtitan.rl.rubric import RewardFn


class RewardMathVerify(RewardFn):
    """Binary reward for a mathematically equivalent final answer."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        timeout_seconds: float = 5.0
        """Wall-clock limit to score one answer; a slower answer scores 0. Normal answers
        take 2-20 ms. 5 s matches Math-Verify's own default."""

        num_workers: int = 4
        """Worker processes (~70 MB each) scoring answers in parallel, per rollout worker."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._pool = MathVerifyPool(
            num_workers=config.num_workers, timeout_seconds=config.timeout_seconds
        )

    async def __call__(self, rollout: Rollout, env_input: DapoMathSample) -> float:
        """Return 1 when Math-Verify equates the response and ground truth."""
        if not rollout.turns:
            return 0.0
        completion_message = rollout.turns[-1].completion_message
        response = (
            (completion_message.get("content") or "") if completion_message else ""
        )
        return await self._pool.score(response, env_input.ground_truth)
