# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import math
import random
import re
from dataclasses import dataclass

import chess
from renderers import Message

from torchtitan.rl.examples.chess_selfplay.bots import StockfishBot
from torchtitan.rl.examples.chess_selfplay.data import ChessSample
from torchtitan.rl.rollout.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)

_COLOR_NAMES = {chess.WHITE: "White", chess.BLACK: "Black"}
_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


class ChessPlayerEnv(MessageEnv):
    """One player's half of a `ChessGame`, as a chat: each user message shows the board and the legal
    moves, and each assistant reply ends with a move in `\\boxed{}`.

    `step` plays the move, then waits for the other player's reply. The rollout ends when the game
    does, and the last step's `env_rewards["score"]` is this player's score (win 1, draw 0.5, loss 0).

    Example (self-play; White's view):

        init:                    "You are playing chess as White ... Legal moves: c3 Nf3 ... e4 ..."
        step("... \\boxed{e4}")   -> waits for Black's move -> "Black played c5. <board> Legal moves: ..."
        step("... \\boxed{Ke9}")  -> illegal: White forfeits -> done, env_rewards={"score": 0.0}
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, game: ChessGame, color: chess.Color) -> None:
        self._game = game
        self._color = color

    async def init(self) -> MessageEnvInitOutput:
        """Show the rules, the board, and the legal moves for this player's first move."""
        rules = (
            f"You are playing chess as {_COLOR_NAMES[self._color]}. Play to win.\n\n"
            "Each turn you see the board and your legal moves. Think about the position, then end "
            "your reply with one of your legal moves, written exactly as listed, inside \\boxed{}. "
            "An illegal or missing move loses the game."
        )
        return MessageEnvInitOutput(
            init_prompt_messages=[
                {"role": "user", "content": f"{rules}\n\n{self._game.turn_message()}"}
            ]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        """Play this player's move, wait for the reply, and show the new board, or end the game."""
        matches = _BOXED_RE.findall(completion_message.get("content") or "")
        await self._game.play(self._color, matches[-1] if matches else None)
        await self._game.wait_for_turn(self._color)
        if self._game.is_over:
            return MessageEnvStepOutput(
                done=True, env_rewards={"score": self._game.scores[self._color]}
            )
        return MessageEnvStepOutput(
            env_messages=[{"role": "user", "content": self._game.turn_message()}]
        )


class ChessGame:
    """The board of one game, shared by both players' `ChessPlayerEnv`s, and the turn hand-off between them.

    A player plays its move with `play` and then waits in `wait_for_turn` until the other player has
    moved or the game is over. Against a `StockfishBot`, `play` makes the bot's reply itself; call
    `start` once before play, so the bot makes the first move when it is to move.

    A game ends on:
    (a) checkmate, stalemate, insufficient material, or threefold repetition;
    (b) `max_plies` plies played, scored by material (see `material_score`);
    (c) an illegal or missing move, or `forfeit`: that color loses.

    Example (self-play):

        game = ChessGame(sample=ChessSample(fen=chess.STARTING_FEN, opponent="self"), max_plies=40, seed=0)
        await game.start()
        await game.play(chess.WHITE, "e4")
        await game.wait_for_turn(chess.BLACK)  # returns at once: Black to move
        await game.play(chess.BLACK, "Ke9")    # illegal -> game.scores == {WHITE: 1.0, BLACK: 0.0}
    """

    def __init__(
        self,
        *,
        sample: ChessSample,
        max_plies: int,
        seed: int,
        bot: StockfishBot | None = None,
    ) -> None:
        self.board = chess.Board(sample.fen)
        self.scores: dict[chess.Color, float] | None = None
        """Each color's score once the game is over (win 1, draw 0.5, loss 0); `None` while it runs."""
        self.end_reason: str | None = None
        """Why the game ended, e.g. "checkmate", "max_plies", "illegal_move"; `None` while it runs."""
        self._max_plies = max_plies
        self._bot = bot
        self._bot_color = None if bot is None else not sample.policy_color
        # Shuffles each turn's legal-move list.
        self._rng = random.Random(seed)
        self._last_move_san: str | None = None
        self._turn_changed = asyncio.Condition()
        self._end_if_over()

    @property
    def is_over(self) -> bool:
        return self.scores is not None

    @property
    def num_plies(self) -> int:
        """Plies played in this game, not counting the opening already in the start FEN."""
        return len(self.board.move_stack)

    async def start(self) -> None:
        """Let the bot make the first move if it is to move."""
        async with self._turn_changed:
            if not self.is_over and self.board.turn == self._bot_color:
                self._push(await self._bot.move(self.board))
            self._turn_changed.notify_all()

    async def wait_for_turn(self, color: chess.Color) -> None:
        """Return once `color` is to move or the game is over."""
        async with self._turn_changed:
            await self._turn_changed.wait_for(
                lambda: self.is_over or self.board.turn == color
            )

    async def play(self, color: chess.Color, move_text: str | None) -> None:
        """Play `color`'s move, given in SAN (`Nf3`) or UCI (`g1f3`); an illegal or missing move forfeits.
        Does nothing once the game is over, e.g. after the other player forfeited."""
        async with self._turn_changed:
            if self.is_over:
                return
            move = None if move_text is None else _parse_move(self.board, move_text)
            if move is None:
                self._end(winner=not color, reason="illegal_move")
            else:
                self._push(move)
                if not self.is_over and self._bot is not None:
                    self._push(await self._bot.move(self.board))
            self._turn_changed.notify_all()

    async def forfeit(self, color: chess.Color, *, reason: str) -> None:
        """End the game as a loss for `color`, unless it is already over."""
        async with self._turn_changed:
            if not self.is_over:
                self._end(winner=not color, reason=reason)
            self._turn_changed.notify_all()

    def turn_message(self) -> str:
        """The user message for the color to move: the opponent's last move, the board, the legal
        moves in random order, and the request for a move, last so the model reads it last."""
        board = self.board
        legal_moves = [board.san(move) for move in board.legal_moves]
        # In python-chess's order the first listed move is always legal, and always playing it beats a random mover.
        self._rng.shuffle(legal_moves)
        lines = []
        if self._last_move_san is not None:
            lines.append(
                f"{_COLOR_NAMES[not board.turn]} played {self._last_move_san}.\n"
            )
        lines += [
            "Board (uppercase is White, lowercase is Black; rank 8 at the top):",
            str(board),
            "",
            f"FEN: {board.fen()}",
            "Legal moves: " + " ".join(legal_moves),
            "",
            f"Your move as {_COLOR_NAMES[board.turn]}. Think briefly, then write one legal move "
            "inside \\boxed{}.",
        ]
        return "\n".join(lines)

    def _push(self, move: chess.Move) -> None:
        self._last_move_san = self.board.san(move)
        self.board.push(move)
        self._end_if_over()

    def _end_if_over(self) -> None:
        outcome = self.board.outcome()
        # `outcome(claim_draw=True)` also ends the game when the side to move could repeat a
        # position a third time; end it only once the repetition has happened.
        if outcome is None and self.board.is_repetition(3):
            outcome = chess.Outcome(chess.Termination.THREEFOLD_REPETITION, winner=None)
        if outcome is not None:
            self._end(winner=outcome.winner, reason=outcome.termination.name.lower())
        elif self.num_plies >= self._max_plies:
            white_score = material_score(self.board)
            self.scores = {chess.WHITE: white_score, chess.BLACK: 1.0 - white_score}
            self.end_reason = "max_plies"

    def _end(self, *, winner: chess.Color | None, reason: str) -> None:
        if winner is None:
            self.scores = {chess.WHITE: 0.5, chess.BLACK: 0.5}
        else:
            self.scores = {winner: 1.0, not winner: 0.0}
        self.end_reason = reason


def material_score(board: chess.Board) -> float:
    """White's expected score from the material balance in pawns, after the side to move plays out
    (or declines) its captures: `1 / (1 + exp(-pawns / 4))`.

    Example:

        material_score(chess.Board())  # 0.5; even material
        # White up a knight (+3) -> 0.68; up a queen (+9) -> 0.90
        # Black to move with White's queen en prise -> scored as if Black had taken it, 0.10
    """
    pawns = _resolve_captures(board, alpha=-math.inf, beta=math.inf, depth=4)
    white_pawns = pawns if board.turn == chess.WHITE else -pawns
    return 1.0 / (1.0 + math.exp(-white_pawns / 4))


def _resolve_captures(
    board: chess.Board, *, alpha: float, beta: float, depth: int
) -> float:
    """Material in pawns for the side to move, after it plays out or declines its captures (negamax)."""
    material = sum(
        value
        * (
            len(board.pieces(piece, chess.WHITE))
            - len(board.pieces(piece, chess.BLACK))
        )
        for piece, value in _PIECE_VALUES.items()
    )
    stand_pat = material if board.turn == chess.WHITE else -material
    if depth == 0 or stand_pat >= beta:
        return stand_pat
    alpha = max(alpha, stand_pat)
    for move in board.generate_legal_captures():
        board.push(move)
        score = -_resolve_captures(board, alpha=-beta, beta=-alpha, depth=depth - 1)
        board.pop()
        if score >= beta:
            return score
        alpha = max(alpha, score)
    return alpha


def _parse_move(board: chess.Board, move_text: str) -> chess.Move | None:
    """Return the legal move that `move_text` names in SAN (`Nf3`) or UCI (`g1f3`), else `None`."""
    try:
        move = board.parse_san(move_text.strip().rstrip("!?"))
    except ValueError:
        return None
    # `parse_san` also reads UCI, and reads "--" as a null move, which is not legal
    return move if board.is_legal(move) else None
