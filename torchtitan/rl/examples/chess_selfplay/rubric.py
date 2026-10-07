# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.rl.examples.chess_selfplay.data import ChessSample
from torchtitan.rl.rollout import Rollout
from torchtitan.rl.rubric import RewardFn


class RewardChessScore(RewardFn):
    """This player's game score: win 1, draw 0.5, loss 0; a game cut at the ply cap is scored by material.

    The env puts the score on the last turn. A rollout that stopped before the game ended (its reply
    hit `max_tokens`, its history outgrew `max_rollout_tokens`) has no score; set the rubric's
    `truncation_reward` / `error_reward` to 0.0, a loss, to match the forfeit the game records.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: ChessSample) -> float:
        return rollout.turns[-1].env_rewards["score"]
