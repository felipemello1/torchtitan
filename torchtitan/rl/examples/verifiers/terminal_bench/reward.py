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

# Terminus-2's reply to an agent turn without a valid JSON action.
_PARSE_ERROR_PREFIX = "Previous response had parsing errors"


class RewardFromVerifiersPartialCredit(RewardFn):
    """Verifiers' reward when every test passed, else partial credit for the tests that passed,
    minus a penalty for the share of agent turns the harness could not parse.

    Example, with max_partial_credit=0.5 and malformed_turn_penalty=0.2:
        all tests pass, any malformed turns          -> 1.0
        2 of 3 tests pass, no malformed turn         -> 0.5 * 2/3 = 0.33
        2 of 3 tests pass, 5 of 20 turns malformed   -> 0.33 - 0.2 * 5/20 = 0.28
        no test passes, 5 of 20 turns malformed      -> 0.0 - 0.2 * 5/20 = -0.05
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        max_partial_credit: float = 0.5
        """Weight of the passed-test fraction for a rollout that did not solve the task; below 1
        so a solve always scores highest."""

        malformed_turn_penalty: float = 0.0
        """For a rollout that did not solve the task, subtracted times its share of agent turns
        the harness answered with a parse error. Solves stay at 1.0, so all-pass groups still tie
        and are dropped; a share, not a count, so a shorter failure doesn't score higher."""

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
        num_malformed_turns = sum(
            any(
                message["content"].startswith(_PARSE_ERROR_PREFIX)
                for message in turn.env_messages
            )
            for turn in rollout.turns
        )
        return self._config.max_partial_credit * rewards.get(
            TESTS_PASSED_FRAC_KEY, 0.0
        ) - self._config.malformed_turn_penalty * num_malformed_turns / max(
            1, len(rollout.turns)
        )
