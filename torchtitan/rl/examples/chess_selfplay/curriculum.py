# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from torchtitan.rl.examples.chess_selfplay.data import ChessSample
from torchtitan.rl.rollout.curriculum import Curriculum
from torchtitan.rl.rollout.types import RolloutGroup


class ChessCurriculum(Curriculum):
    """Moves "curriculum" games up a ladder of bots as the policy wins. Validation games keep their bot.

    Example:

        config.rollouter.training_dataloader.dataset.bots = ("curriculum",)
        config.rollouter.curriculum = ChessCurriculum.Config(bots=("sf_random", "sf_eps75", "sf_eps50"))
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Curriculum.Config):
        bots: tuple[str, ...]
        """Bots from `bots.BOTS`, easiest first, for games whose opponent is "curriculum"."""

        promote_win_rate: float = 0.6
        """Moves to the next bot once the policy wins more than this share of a train step's games
        against the current one (by checkmate; a material lead at the ply cap does not count)."""

        min_games: int = 64
        """Games against the current bot a train step needs to test for a promotion. At a true 53% win
        rate, a test passes 60% by luck 12.5% of the time with 64 games, 1.3% with 256."""

    def __init__(self, config: Config) -> None:
        self._bots = config.bots
        self._promote_win_rate = config.promote_win_rate
        self._min_games = config.min_games
        self.level = 0
        """Index of the current bot in `bots`."""

    def prepare(self, sample: ChessSample, *, step: int) -> ChessSample:
        """Resolve a "curriculum" opponent to the current bot; return other samples unchanged.

        Example (bots=("sf_random", "sf_eps75"), level 1):

            prepare(ChessSample(opponent="curriculum", ...), step=0)
            # ChessSample(opponent="sf_eps75", curriculum_level=1, ...)
        """
        if sample.opponent != "curriculum":
            return sample
        return replace(
            sample, opponent=self._bots[self.level], curriculum_level=self.level
        )

    def summarize(
        self, sample: ChessSample, group: RolloutGroup
    ) -> tuple[int, list[float]] | None:
        """For a curriculum group, its level and whether the policy won each game; None for other groups.

        A game the policy never moved in has no rollout, so it doesn't count.

        Example: 3 games at level 2, the second won -> (2, [0.0, 1.0, 0.0])
        """
        if sample.curriculum_level is None:
            return None
        return sample.curriculum_level, [
            rollout.turns[-1].env_rewards["won"] for rollout in group.rollouts
        ]

    def update(
        self, *, step: int, summaries: list[tuple[int, list[float]] | None]
    ) -> None:
        """Move to the next bot if the policy won more than `promote_win_rate` of this step's games
        against the current one.

        Example (promote_win_rate=0.6, min_games=64, level 2):

            100 games at level 2, 63 won (63%)   -> level 3
            50 games at level 2, 45 won          -> level 2: fewer than `min_games`, no test
            games at level 1 (prepared before the last promotion) don't count
        """
        curriculum_summaries = [summary for summary in summaries if summary is not None]
        wins = [
            win
            for level, group_wins in curriculum_summaries
            if level == self.level
            for win in group_wins
        ]
        if (
            wins
            and len(wins) >= self._min_games
            and sum(wins) / len(wins) > self._promote_win_rate
        ):
            self.level = min(self.level + 1, len(self._bots) - 1)

    def state_dict(self) -> dict[str, Any]:
        return {"level": self.level}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # empty when the run was saved without a curriculum
        self.level = state_dict.get("level", 0)
