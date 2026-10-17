# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.rl.examples.chess_selfplay.data import ChessSample
from torchtitan.rl.rollout import Rollout
from torchtitan.rl.rollout.rubric import RewardFn


class RewardChessScore(RewardFn):
    """This player's `ChessGame.rewards`, read from its last turn.

    The env puts the reward on the last turn; for a player that stopped mid-game, `ChessSelfPlayWorker`
    puts the forfeit reward there. Leave the rubric's `truncation_reward` / `error_reward` unset.
    """

    # TODO: a reward cost per force-closed turn, against thinking loops.

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: ChessSample) -> float:
        return rollout.turns[-1].env_rewards["score"]
