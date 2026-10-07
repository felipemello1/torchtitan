# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import chess

from torchtitan.config import Configurable


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
    split: Literal["train", "validation"] = "train"
    """Keeps validation metrics apart: they log under `val_chess_*` instead of `chess_*`."""


class ChessSelfPlayDataset(Configurable):
    """Provides training groups: start positions (the standard position after 0 to `max_opening_plies`
    random legal moves) for self-play, with a `bot_fraction` of groups played against a bot instead.

    A bot group draws its bot uniformly from `bots` and its policy color at random. Bot groups are
    spread evenly: with `bot_fraction=0.5`, every other group.

    Example:

        dataset = ChessSelfPlayDataset.Config(bots=("sf_eps90", "sf_eps50"), bot_fraction=0.5).build()
        [next(dataset).opponent for _ in range(4)]  # ["self", "sf_eps90", "self", "sf_eps90"]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 42

        max_opening_plies: int = 2
        """Random legal plies played before the game starts, drawn uniformly from [0, max_opening_plies]."""

        bots: tuple[str, ...] = ()
        """Bot names from `bots.BOTS` for bot groups; empty means self-play only."""

        bot_fraction: float = 0.5
        """Fraction of groups played against a bot when `bots` is set."""

    def __init__(self, config: Config) -> None:
        self._max_opening_plies = config.max_opening_plies
        self._bots = config.bots
        self._bot_fraction = config.bot_fraction if config.bots else 0.0
        self._rng = random.Random(config.seed)
        # adds bot_fraction per group; a bot group is due each time it reaches 1
        self._bot_credit = 0.0

    def __iter__(self) -> Iterator[ChessSample]:
        return self

    def __next__(self) -> ChessSample:
        board = _random_opening(self._rng, max_plies=self._max_opening_plies)
        self._bot_credit += self._bot_fraction
        if self._bot_credit >= 1.0:
            self._bot_credit -= 1.0
            return ChessSample(
                fen=board.fen(),
                opponent=self._rng.choice(self._bots),
                policy_color=self._rng.random() < 0.5,
                seed=self._rng.getrandbits(32),
            )
        return ChessSample(
            fen=board.fen(), opponent="self", seed=self._rng.getrandbits(32)
        )

    def state_dict(self) -> dict:
        """Snapshot the stream so a run can resume at the same point."""
        return {"rng_state": self._rng.getstate(), "bot_credit": self._bot_credit}

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._bot_credit = state_dict["bot_credit"]


class ChessVsBotDataset(Configurable):
    """Provides a fixed set of `num_games` games against bots, cycled in order, so every validation
    pass plays the same games. Games cycle through each bot with the policy as White, then as Black.

    Example:

        dataset = ChessVsBotDataset.Config(num_games=4, opponents=("sf_eps90", "sf_eps50")).build()
        [(s.opponent, s.policy_color) for s in (next(dataset) for _ in range(4))]
        # [("sf_eps90", True), ("sf_eps90", False), ("sf_eps50", True), ("sf_eps50", False)]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
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

        max_opening_plies: int = 2
        """Random legal plies played before the game starts, drawn uniformly from [0, max_opening_plies]."""

    def __init__(self, config: Config) -> None:
        rng = random.Random(config.seed)
        self._samples = [
            ChessSample(
                fen=_random_opening(rng, max_plies=config.max_opening_plies).fen(),
                opponent=config.opponents[(game_idx // 2) % len(config.opponents)],
                policy_color=game_idx % 2 == 0,
                seed=rng.getrandbits(32),
                split="validation",
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


def _random_opening(rng: random.Random, *, max_plies: int) -> chess.Board:
    """Return the standard position after 0 to `max_plies` random legal plies."""
    board = chess.Board()
    for _ in range(rng.randint(0, max_plies)):
        board.push(rng.choice(list(board.legal_moves)))
    return board
