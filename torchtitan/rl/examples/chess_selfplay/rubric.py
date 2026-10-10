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
    """This player's reward from `ChessGame.rewards`: a checkmate is 10, a game cut at the ply cap or forfeited
    by the other player scores 0.25 to 0.75 by material, a draw up to 0.5, and being checkmated or
    forfeiting is negative. Draws, checkmates against the player, and forfeits score better the more
    moves the player lasted.

    The env puts the reward on the last turn. For a player that stopped mid-game (its reply hit
    `max_tokens`, its history outgrew `max_rollout_tokens`), `ChessSelfPlayWorker` puts the game's
    forfeit reward there; leave the rubric's `truncation_reward` / `error_reward` unset so this fn scores it.

    With `forced_close_penalty`, a turn whose thinking the `ThinkingBudget` had to close costs its share:
    reward - penalty * (force-closed turns / turns).

    Example:

        RewardChessScore.Config(forced_close_penalty=0.1)
        # a checkmate with 10 of 30 turns force-closed -> 10.0 - 0.1 * 10 / 30 = 9.967
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        forced_close_penalty: float = 0.0
        """The most a rollout loses when every turn's thinking was force-closed; 0.1 keeps a checkmate (9.9)
        above every non-win (at most 0.75)."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._forced_close_penalty = config.forced_close_penalty

    async def __call__(self, rollout: Rollout, env_input: ChessSample) -> float:
        # the budget masks the tokens it forced out of the loss; no other path does
        num_forced = sum(
            turn.completion_loss_mask is not None and not all(turn.completion_loss_mask)
            for turn in rollout.turns
        )
        forced_share = num_forced / len(rollout.turns)
        return (
            rollout.turns[-1].env_rewards["score"]
            - self._forced_close_penalty * forced_share
        )
