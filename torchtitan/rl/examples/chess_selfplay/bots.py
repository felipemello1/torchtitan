# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import os
import random
import shutil
import tempfile
from dataclasses import dataclass

import chess
import chess.engine


@dataclass(frozen=True, kw_only=True, slots=True)
class BotSpec:
    """A Stockfish opponent at a fixed, measured strength."""

    elo: float
    """Rating on a full-rules scale anchored at Stockfish `UCI_Elo` 1320 (100 ms per move)."""
    random_move_prob: float = 0.0
    """Probability of a uniformly random legal move instead of Stockfish's move."""
    depth: int | None = None
    """Search depth of Stockfish's move, for the diluted bots."""
    uci_elo: int | None = None
    """Stockfish's own strength limiter (1320-3190), searched for 100 ms per move."""


# Stockfish diluted with random moves fills the range below Stockfish's 1320 floor. Ratings: 96 games
# per pair under full rules, anchored at UCI_Elo 1320 (research/elo_bots.md, 2026-10-07).
BOTS: dict[str, BotSpec] = {
    "sf_eps90": BotSpec(elo=441, random_move_prob=0.9, depth=5),
    "sf_eps75": BotSpec(elo=571, random_move_prob=0.75, depth=5),
    "sf_eps50": BotSpec(elo=747, random_move_prob=0.5, depth=5),
    "sf_eps25": BotSpec(elo=1048, random_move_prob=0.25, depth=5),
    "sf_elo1320": BotSpec(elo=1320, uci_elo=1320),
    "sf_elo1500": BotSpec(elo=1500, uci_elo=1500),
}


class StockfishBot:
    """Plays one game's bot moves with its own Stockfish process; call `close` when the game ends.

    Engine calls run in a thread, so a 100 ms search does not stall the worker's other games.

    Example:

        bot = StockfishBot(BOTS["sf_eps75"], name="sf_eps75", engine_path="stockfish", seed=7)
        move = await bot.move(chess.Board())  # a random move 75% of the time, else Stockfish's
        bot.close()
    """

    def __init__(
        self, spec: BotSpec, *, name: str, engine_path: str, seed: int
    ) -> None:
        self.name = name
        self.elo = spec.elo
        self._spec = spec
        self._engine_path = engine_path
        self._rng = random.Random(seed)
        self._engine: chess.engine.SimpleEngine | None = None

    async def move(self, board: chess.Board) -> chess.Move:
        if self._rng.random() < self._spec.random_move_prob:
            return self._rng.choice(list(board.legal_moves))
        return await asyncio.to_thread(self._engine_move, board.copy())

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()

    def _engine_move(self, board: chess.Board) -> chess.Move:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._engine_path)
            options = {"Threads": 1, "Hash": 16}
            if self._spec.uci_elo is not None:
                options |= {"UCI_LimitStrength": True, "UCI_Elo": self._spec.uci_elo}
            self._engine.configure(options)
        if self._spec.uci_elo is not None:
            limit = chess.engine.Limit(time=0.1)
        else:
            limit = chess.engine.Limit(depth=self._spec.depth)
        return self._engine.play(board, limit).move


def centipawn_losses(
    *,
    fen: str,
    moves: list[chess.Move],
    scored_color: chess.Color | None,
    engine_path: str,
) -> list[int]:
    """Stockfish's loss in centipawns for each move of `scored_color` (both colors if `None`):
    the eval of the best move minus the eval of the played move, from the mover's side, at depth 8.

    Example:

        centipawn_losses(fen=chess.STARTING_FEN, moves=[chess.Move.from_uci("f2f3")], scored_color=None,
                         engine_path="stockfish")  # [~110]; e4 would be ~0
    """
    board = chess.Board(fen)
    limit = chess.engine.Limit(depth=8)
    losses = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        engine.configure({"Threads": 1, "Hash": 16})
        best = engine.analyse(board, limit)["score"]
        for move in moves:
            mover = board.turn
            board.push(move)
            after = engine.analyse(board, limit)["score"]
            if scored_color is None or mover == scored_color:
                # a mate counts as 2000 centipawns, so one missed mate does not dominate the mean
                best_cp = best.pov(mover).score(mate_score=2000)
                played_cp = after.pov(mover).score(mate_score=2000)
                losses.append(max(0, best_cp - played_cp))
            # the eval after this move is the best eval for the next mover
            best = after
    return losses


def executable_stockfish(path: str | None) -> str | None:
    """Return `path`, or an executable copy of it when the file lacks the execute bit, e.g. a binary
    staged from Manifold, which keeps no file modes, onto a read-only mount.

    Example:

        executable_stockfish("stockfish")                       # "stockfish" (found on PATH)
        executable_stockfish("/mnt/dome_staged/.../stockfish")  # "/tmp/stockfish_x1y2/stockfish", mode 0755
    """
    if path is None or not os.path.isfile(path) or os.access(path, os.X_OK):
        return path
    copy = os.path.join(tempfile.mkdtemp(prefix="stockfish_"), os.path.basename(path))
    shutil.copyfile(path, copy)
    os.chmod(copy, 0o755)
    return copy
