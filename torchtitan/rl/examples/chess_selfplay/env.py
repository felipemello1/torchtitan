# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import json
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
_PIECE_ORDER = (
    chess.KING,
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
    chess.PAWN,
)
_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
# Training rewards (see `ChessGame.rewards`). A checkmate against you costs its full reward on the
# first ply, shrinking to 0 at `max_plies`; a draw grows to 0.5 by then. A forfeit costs its full
# reward on the first ply and half at `max_plies`, so it is below any loss.
_FORFEIT_REWARD = -1.0
_CHECKMATED_REWARD = -0.25
# A checkmate pays 10 (a win was 1), so a mate dominates its group's advantages.
_CHECKMATE_REWARD = 10.0
# Share of the material score in an unfinished game's reward: 0.5 keeps it within [0.25, 0.75].
_MATERIAL_WEIGHT = 0.5


class ChessPlayerEnv(MessageEnv):
    """One player's half of a `ChessGame`, as a chat: each user message shows the board and the legal
    moves, and each assistant reply ends with a move in `\\boxed{}`.

    `step` plays the move, then waits for the other player's reply. The rollout ends when the game
    does, and the last step's `env_rewards["score"]` is this player's reward (see `ChessGame.rewards`).

    Example (self-play; White's view):

        init:                    "You are playing chess as White ... Legal moves: c3 Nf3 ... e4 ..."
        step("... \\boxed{e4}")   -> waits for Black's move -> "Black played c5. <board> Legal moves: ..."
        step("... \\boxed{Ke9}")  -> illegal: White forfeits -> done, env_rewards={"score": -0.975}  (max_plies=40)
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
            "Each turn you see your pieces and your opponent's pieces, keyed by piece letter and "
            "square (Ke1 is a king on e1; P is a pawn), each with its legal moves. The current "
            "positions and legal moves are already given: avoid restating them. Analyze which move "
            "is best, then end your reply with that move, written exactly as listed, inside "
            '\\boxed{}. For example, "Pe2": ["e4"] means \\boxed{e4}, not \\boxed{Pe4}; '
            '"Nb1": ["Nbd2"] means \\boxed{Nbd2}, not \\boxed{Nd2}. An x marks a capture: '
            '"Nf3": ["Nxe5"] means \\boxed{Nxe5}. An illegal or missing move loses the game.\n\n'
            f"Checkmating your opponent scores 10. The game stops after {self._game.max_plies} plies "
            "(a ply is one move by either side); a game that reaches that limit without checkmate "
            "scores 0.25 to 0.75 by material (queen 9, rook 5, bishop and knight 3, pawn 1), and a "
            "stalemate scores at most 0.5."
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
                done=True, env_rewards={"score": self._game.rewards[self._color]}
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
    (a) checkmate, stalemate, or insufficient material (a repetition plays on, see `_end_if_over`);
    (b) `max_plies` plies played: a draw, adjusted by material (see `material_score`);
    (c) an illegal or missing move, or `forfeit`: that color loses.

    `scores` is the chess result, used for the Elo metrics; `rewards` is what training uses.

    Example (self-play):

        game = ChessGame(sample=ChessSample(fen=chess.STARTING_FEN, opponent="self"), max_plies=40, seed=0)
        await game.start()
        await game.play(chess.WHITE, "e4")
        await game.wait_for_turn(chess.BLACK)  # returns at once: Black to move
        await game.play(chess.BLACK, "Ke9")    # illegal -> game.scores == {WHITE: 1.0, BLACK: 0.0}
                                               #            game.rewards == {WHITE: 0.5, BLACK: -0.9875}
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
        self.forfeiter: chess.Color | None = None
        """The color that lost by forfeit (an illegal move or `forfeit`); `None` otherwise."""
        self._max_plies = max_plies
        self._bot = bot
        self._bot_color = None if bot is None else not sample.policy_color
        # Shuffles each piece's legal moves.
        self._rng = random.Random(seed)
        self._last_move_san: str | None = None
        self._turn_changed = asyncio.Condition()
        self._end_if_over()

    @property
    def is_over(self) -> bool:
        return self.scores is not None

    @property
    def rewards(self) -> dict[chess.Color, float]:
        """Each color's training reward once the game is over, with `played` the share of `max_plies`
        played, so it works for any `max_plies`:
        (a) checkmate: 10 for the winner however long it took, -0.25 * (1 - played) for the loser;
        (b) stalemate or insufficient material: 0.5 * played each;
        (c) `max_plies` plies: a draw, 0.5, moved halfway toward the material score, so within [0.25, 0.75];
        (d) a forfeit: -1 * (1 - played / 2), so within [-1, -0.5], below being checkmated at any ply;
            the other color is scored as in (c), not as a win.

        Example (max_plies=60):

            max_plies, White up a knight              -> {WHITE: 0.59, BLACK: 0.41}
            White checkmates on ply 19                -> {WHITE: 10.0, BLACK: -0.25 * (1 - 19 / 60) = -0.17}
            Black forfeits at ply 20, up a queen      -> {WHITE: 0.30, BLACK: -1 * (1 - 20 / 120) = -0.83}
            Black forfeits at ply 1, even material    -> {WHITE: 0.5, BLACK: -1 * (1 - 1 / 120) = -0.99}
        """
        played = self.num_plies / self._max_plies
        if self.end_reason == "checkmate":
            loser = self.board.turn
            return {
                not loser: _CHECKMATE_REWARD,
                loser: _CHECKMATED_REWARD * (1.0 - played),
            }
        if self.end_reason != "max_plies" and self.forfeiter is None:
            return {chess.WHITE: 0.5 * played, chess.BLACK: 0.5 * played}
        rewards = self.material_rewards
        if self.forfeiter is not None:
            rewards[self.forfeiter] = _FORFEIT_REWARD * (1.0 - played / 2)
        return rewards

    @property
    def material_rewards(self) -> dict[chess.Color, float]:
        """Each color's reward if the game ended at `max_plies` now: a draw, 0.5, moved halfway toward
        the material score."""
        white_reward = 0.5 + _MATERIAL_WEIGHT * (material_score(self.board) - 0.5)
        return {chess.WHITE: white_reward, chess.BLACK: 1.0 - white_reward}

    @property
    def max_plies(self) -> int:
        return self._max_plies

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
        """Play `color`'s move, written as the prompt lists it (`Nf3`); any other text forfeits.
        Does nothing once the game is over, e.g. after the other player forfeited."""
        async with self._turn_changed:
            if self.is_over:
                return
            move = None if move_text is None else _parse_move(self.board, move_text)
            if move is None:
                self._forfeit(color, reason="illegal_move")
            else:
                self._push(move)
                if not self.is_over and self._bot is not None:
                    self._push(await self._bot.move(self.board))
            self._turn_changed.notify_all()

    async def forfeit(self, color: chess.Color, *, reason: str) -> None:
        """End the game as a loss for `color`, unless it is already over."""
        async with self._turn_changed:
            if not self.is_over:
                self._forfeit(color, reason=reason)
            self._turn_changed.notify_all()

    def turn_message(self) -> str:
        """The user message for the color to move: the opponent's last move, each side's pieces with
        their moves, and the request for a move, last so the model reads it last."""
        board = self.board
        me, opponent = _COLOR_NAMES[board.turn], _COLOR_NAMES[not board.turn]
        lines = []
        if self._last_move_san is not None:
            lines.append(f"{opponent} played {self._last_move_san}.\n")
        # Pieces keyed by square instead of a board: on a drawn board Qwen3.5-35B spent most of its
        # thinking counting cells to name squares (no FEN either: it re-parsed it rank by rank). The
        # opponent's moves cost legal moves in a local probe (86% vs 93%); kept to show its threats.
        lines += [
            f"Your pieces ({me}) and their legal moves:",
            _moves_by_piece(board, board.turn, self._rng),
            "",
            f"Opponent pieces ({opponent}) and the moves they could make on their turn:",
            _moves_by_piece(board, not board.turn, self._rng),
        ]
        if board.is_check():
            lines += ["", f"{me} is in check."]
        lines += [
            "",
            f"Your move as {me} (ply {self.num_plies + 1} of {self._max_plies}). Write your best "
            "legal move inside \\boxed{}.",
        ]
        return "\n".join(lines)

    def _push(self, move: chess.Move) -> None:
        self._last_move_san = self.board.san(move)
        self.board.push(move)
        self._end_if_over()

    def _end_if_over(self) -> None:
        outcome = self.board.outcome()
        # A repetition plays on to `max_plies`: a repetition draw would lock in a reward while playing
        # on risks a forfeit, so self-play could learn to repeat moves instead of playing.
        if (
            outcome is not None
            and outcome.termination == chess.Termination.FIVEFOLD_REPETITION
        ):
            outcome = None
        if outcome is not None:
            self._end(winner=outcome.winner, reason=outcome.termination.name.lower())
        elif self.num_plies >= self._max_plies:
            white_score = material_score(self.board)
            self.scores = {chess.WHITE: white_score, chess.BLACK: 1.0 - white_score}
            self.end_reason = "max_plies"

    def _forfeit(self, color: chess.Color, *, reason: str) -> None:
        self.forfeiter = color
        self._end(winner=not color, reason=reason)

    def _end(self, *, winner: chess.Color | None, reason: str) -> None:
        if winner is None:
            self.scores = {chess.WHITE: 0.5, chess.BLACK: 0.5}
        else:
            self.scores = {winner: 1.0, not winner: 0.0}
        self.end_reason = reason


def _moves_by_piece(board: chess.Board, color: chess.Color, rng: random.Random) -> str:
    """Each piece of `color`, keyed by letter and square, with the moves it could make on its turn: a
    pinned or blocked piece shows `[]`, a captured one is absent.

    The side to move's moves are shuffled with `rng`: in python-chess's order the first listed move
    is always legal, and always playing it beats a random mover. The other side's moves are sorted,
    without king captures, and without check marks when the side to move is in check (every move
    would read as check).

    Example:

        {
          "Ke1": ["Ke2"],
          "Qd1": ["Qh5", "Qe2", "Qg4", "Qf3"],
          "Pd2": ["d4", "d3"],
          ...
        }
    """
    to_move = color == board.turn
    strip_checks = not to_move and board.is_check()
    if not to_move:
        board = board.copy()
        board.push(chess.Move.null())
    moves: dict[str, list[str]] = {
        chess.piece_symbol(piece_type).upper() + chess.square_name(square): []
        for piece_type in _PIECE_ORDER
        for square in board.pieces(piece_type, color)
    }
    for move in board.legal_moves:
        if board.piece_type_at(move.to_square) == chess.KING:
            continue
        piece = board.piece_at(move.from_square)
        san = board.san(move)
        key = piece.symbol().upper() + chess.square_name(move.from_square)
        moves[key].append(san.rstrip("+#") if strip_checks else san)
    rows = []
    for key, listed in moves.items():
        if to_move:
            rng.shuffle(listed)
        else:
            listed.sort()
        rows.append(f'  "{key}": {json.dumps(listed)}')
    return "{\n" + ",\n".join(rows) + "\n}"


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
    """Return the legal move whose SAN, as the prompt lists it, is `move_text`, else `None`.
    The check mark (plain or LaTeX-escaped) and any `x` are ignored: `Qxf1#`, `Qf1\\#` and `Qf1`
    all match `Qxf1#`, and `Nxf3` matches `Nf3`. No two legal moves differ only by an `x`: a move
    to a square either captures or it doesn't."""
    text = re.sub(r"\\?[+#]$", "", move_text.strip()).replace("x", "")
    for move in board.legal_moves:
        if board.san(move).rstrip("+#").replace("x", "") == text:
            return move
    return None
