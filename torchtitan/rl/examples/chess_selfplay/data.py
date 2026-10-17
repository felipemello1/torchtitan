# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

import chess

from torchtitan.rl.examples.chess_selfplay.openings import OPENINGS

from torchtitan.rl.rollout.data import RLDataset


@dataclass(frozen=True, kw_only=True, slots=True)
class ChessSample:
    """A start position, and whether the policy plays it against itself or against a bot.

    Example:

        # the policy plays both colors
        ChessSample(fen=chess.STARTING_FEN, opponent="self", seed=7)
        # the policy plays Black against Stockfish with 75% random moves (see `bots.BOTS`)
        ChessSample(fen=chess.STARTING_FEN, opponent="sf_eps75", policy_color=chess.BLACK, seed=7)
    """

    fen: str
    opponent: str
    """"self", or the name of a bot in `bots.BOTS`."""
    policy_color: chess.Color = chess.WHITE
    """The policy's color against a bot; unused in self-play."""
    seed: int = 0
    """Seeds the legal-move order shown to the model and the bot's random moves."""
    max_plies: int | None = None
    """This game's ply cap; None: the worker's `max_plies`. `ChessCurriculum` sets it from the train step."""


class ChessSelfPlayDataset(RLDataset):
    """Provides self-play training groups: start positions 0 to 16 plies into a random line of
    `openings.OPENINGS`.

    Example:

        dataset = ChessSelfPlayDataset.Config().build()
        next(dataset)  # e.g. ChessSample(fen=<3 plies into a Sicilian>, opponent="self", seed=...)
    """

    # TODO: mix in groups against a bot (e.g. sf_eps25), so blunders that self-play misses get
    # punished; no recipe needs it yet.

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        seed: int = 42

    def __init__(self, config: Config) -> None:
        self._rng = random.Random(config.seed)

    def __iter__(self) -> Iterator[ChessSample]:
        return self

    def __next__(self) -> ChessSample:
        board = _random_opening(self._rng)
        return ChessSample(
            fen=board.fen(), opponent="self", seed=self._rng.getrandbits(32)
        )

    def state_dict(self) -> dict:
        """Snapshot the stream so a run can resume at the same point."""
        return {"rng_state": self._rng.getstate()}

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])


class ChessVsBotDataset(RLDataset):
    """Provides a fixed set of `num_games` games against bots, cycled in order, so every validation
    pass plays the same games. Games cycle through each bot with the policy as White, then as Black.

    Example:

        dataset = ChessVsBotDataset.Config(num_games=4, opponents=("sf_eps90", "sf_eps50")).build()
        [(s.opponent, s.policy_color) for s in (next(dataset) for _ in range(4))]
        # [("sf_eps90", True), ("sf_eps90", False), ("sf_eps50", True), ("sf_eps50", False)]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        seed: int = 99

        num_games: int = 64

        opponents: tuple[str, ...] = (
            "sf_eps90",
            "sf_eps75",
            "sf_eps50",
            "sf_eps25",
            "sf_elo1320",
        )
        """Bot names from `bots.BOTS`."""

    def __init__(self, config: Config) -> None:
        rng = random.Random(config.seed)
        self._samples = [
            ChessSample(
                fen=_random_opening(rng).fen(),
                opponent=config.opponents[(game_idx // 2) % len(config.opponents)],
                policy_color=game_idx % 2 == 0,
                seed=rng.getrandbits(32),
            )
            for game_idx in range(config.num_games)
        ]
        self._position = 0

    def __iter__(self) -> Iterator[ChessSample]:
        return self

    def __next__(self) -> ChessSample:
        sample = self._samples[self._position % len(self._samples)]
        self._position += 1
        return sample

    def state_dict(self) -> dict:
        return {"position": self._position}

    def load_state_dict(self, state_dict: dict) -> None:
        self._position = state_dict["position"]


def _random_opening(rng: random.Random) -> chess.Board:
    """Return the position a random number of plies (0 to the whole line) into a random book line.

    Example:

        _random_opening(rng)
        # e.g. 3 plies into "B22 Sicilian Defense: Heidenfeld Variation": 1. e4 c5 2. c3, Black to move
    """
    _, line = rng.choice(OPENINGS)
    moves = line.split()
    board = chess.Board()
    for san in moves[: rng.randint(0, len(moves))]:
        board.push_san(san)
    return board
