# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal-Bench reward with partial credit for the tests that passed."""

from dataclasses import dataclass

from torchtitan.rl.examples.verifiers.rollouter import VERIFIERS_REWARD_KEY
from torchtitan.rl.examples.verifiers.terminal_bench.taskset import (
    TESTS_PASSED_FRAC_KEY,
)
from torchtitan.rl.rollout.rubric import RewardFn
from torchtitan.rl.rollout.types import Rollout


class RewardFromVerifiersPartialCredit(RewardFn):
    """Verifiers' reward when every test passed, else partial credit for the tests that passed.

    Example, with the default max_partial_credit=0.5:
        all tests pass     -> 1.0
        2 of 3 tests pass  -> 0.5 * 2/3 = 0.33
        no test passes     -> 0.0
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        max_partial_credit: float = 0.5
        """Weight of the passed-test fraction for a rollout that did not solve the task; below 1
        so a solve always scores highest."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._config = config

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        del env_input
        rewards = next(
            (
                turn.env_rewards
                for turn in reversed(rollout.turns)
                if VERIFIERS_REWARD_KEY in turn.env_rewards
            ),
            {},
        )
        if rewards.get(VERIFIERS_REWARD_KEY, 0.0) >= 1.0:
            return 1.0
        return self._config.max_partial_credit * rewards.get(TESTS_PASSED_FRAC_KEY, 0.0)
