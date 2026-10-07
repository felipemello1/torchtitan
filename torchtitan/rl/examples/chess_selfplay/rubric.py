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
    """This player's reward from `ChessGame.rewards`: a win is 1, a game cut at the ply cap or forfeited
    by the other player scores 0.25 to 0.75 by material, a draw up to 0.5, and being checkmated or
    forfeiting is negative. Draws, checkmates against the player, and forfeits score better the more
    moves the player lasted.

    The env puts the reward on the last turn. For a player that stopped mid-game (its reply hit
    `max_tokens`, its history outgrew `max_rollout_tokens`), `ChessSelfPlayWorker` puts the game's
    forfeit reward there; leave the rubric's `truncation_reward` / `error_reward` unset so this fn scores it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: ChessSample) -> float:
        return rollout.turns[-1].env_rewards["score"]
