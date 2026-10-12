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
    """Moves "curriculum" games up a ladder of bots as the policy wins, and can grow training games'
    ply cap with the step. Validation games keep their bot and the worker's `max_plies`.

    Example:

        config.rollouter.training_dataloader.dataset.bots = ("curriculum",)
        config.rollouter.curriculum = ChessCurriculum.Config(bots=("sf_random", "sf_eps75", "sf_eps50"))
        # self-play games from 60 plies at step 0 to 120 at step 200, then 120
        ChessCurriculum.Config(start_max_plies=60, end_max_plies=120, max_plies_ramp_steps=200)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Curriculum.Config):
        bots: tuple[str, ...] = ()
        """Bots from `bots.BOTS`, easiest first, for games whose opponent is "curriculum"."""

        promote_win_rate: float = 0.6
        """Moves to the next bot once the policy wins more than this share of a train step's games
        against the current one (by checkmate; a material lead at the ply cap does not count)."""

        min_games: int = 64
        """Games against the current bot a train step needs to test for a promotion. At a true 53% win
        rate, a test passes 60% by luck 12.5% of the time with 64 games, 1.3% with 256."""

        start_max_plies: int | None = None
        """A training game's ply cap at step 0, growing linearly to `end_max_plies` at step
        `max_plies_ramp_steps`, then held: 60 -> 90 at step 100 -> 120 from step 200. None: games
        keep the worker's `max_plies`."""

        end_max_plies: int = 120
        """The ply cap from step `max_plies_ramp_steps` on, with `start_max_plies` set."""

        max_plies_ramp_steps: int = 200
        """Steps the ply cap takes to grow from `start_max_plies` to `end_max_plies`."""

    def __init__(self, config: Config) -> None:
        self._bots = config.bots
        self._promote_win_rate = config.promote_win_rate
        self._min_games = config.min_games
        self._start_max_plies = config.start_max_plies
        self._end_max_plies = config.end_max_plies
        self._max_plies_ramp_steps = config.max_plies_ramp_steps
        self.level = 0
        """Index of the current bot in `bots`."""

    def prepare(self, sample: ChessSample, *, step: int) -> ChessSample:
        """Set the ply cap for `step` (with `start_max_plies`) and resolve a "curriculum" opponent to
        the current bot.

        Example (bots=("sf_random", "sf_eps75"), level 1; start_max_plies=60, end 120 at step 200):

            prepare(ChessSample(opponent="curriculum", ...), step=100)
            # ChessSample(opponent="sf_eps75", curriculum_level=1, max_plies=90, ...)
        """
        if self._start_max_plies is not None:
            ramp = min(step / self._max_plies_ramp_steps, 1.0)
            max_plies = self._start_max_plies + ramp * (
                self._end_max_plies - self._start_max_plies
            )
            sample = replace(sample, max_plies=round(max_plies))
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
