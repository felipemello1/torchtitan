# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import ClassVar, TYPE_CHECKING

import chess

from torchtitan.rl.examples.chess_selfplay.bots import (
    BOTS,
    centipawn_losses,
    StockfishBot,
)
from torchtitan.rl.examples.chess_selfplay.data import ChessSample
from torchtitan.rl.examples.chess_selfplay.env import ChessGame
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.rollouter import RolloutWorker
from torchtitan.rl.rollout.types import GenerateFn, Rollout, RolloutGroup, RolloutStatus

if TYPE_CHECKING:
    from torchtitan.rl.generator import SamplingConfig

_END_REASONS = (
    "checkmate",
    "stalemate",
    "insufficient_material",
    "max_plies",
    "illegal_move",
    # a player that stopped without a move forfeits, with its rollout status as the reason
    *(
        status.value
        for status in RolloutStatus
        if status.is_truncated() or status.is_error()
    ),
)
# End reasons caused by exactly one bad reply: one per game that ended this way.
_REPLY_FORFEITS = ("illegal_move", "truncated_length", "error_parse")


class ChessSelfPlayWorker(RolloutWorker):
    """Plays `group_size` chess games from one start position. Each player's half of a game is one
    multi-turn rollout, so a self-play group of 8 games has up to 16 rollouts.

    Both players are the policy and both are trained. Advantages are centered per color (White
    against White, Black against Black), so the player that moves first does not get a free
    positive advantage. Against a Stockfish bot, only the policy's player is a rollout.

    Example (group_size=2, self-play):

        game 0: White mates on ply 31 of 60          -> rewards White 1.0, Black -0.25 * (1 - 31 / 60) = -0.12
        game 1: the 60-ply cap at even material      -> rewards White 0.5, Black 0.5
        advantages: White [+0.25, -0.25], Black [-0.31, +0.31]   (each color's mean is subtracted)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RolloutWorker.Config):
        max_plies: int
        """Plies (half-moves, both players) after which a game ends as a draw, adjusted by material.
        Rewards scale with the share of it played (see `ChessGame.rewards`), so any value works."""

        stockfish_path: str | None = None
        """Stockfish binary for bot games and for scoring the policy's moves (centipawn loss).
        `None`: self-play only, and no move scoring."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._max_plies = config.max_plies
        self._stockfish_path = config.stockfish_path

    async def run_group(
        self,
        *,
        generate_fn: GenerateFn,
        sample: ChessSample,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Play `group_size` games from `sample`, score each player, and center advantages per color.

        Args:
            generate_fn: Async callable that returns a Completion given a prompt.
            sample: The start position and opponent shared by the group.
            group_id: Stable group id.
            group_size: Number of games.
            sampling: Sampling config for every generate call in the group.

        Returns:
            One scored `RolloutGroup`, one rollout per player that made at least one move.
        """
        bots = [
            None
            if sample.opponent == "self"
            else StockfishBot(
                BOTS[sample.opponent],
                name=sample.opponent,
                engine_path=self._stockfish_path,
                seed=sample.seed + game_idx,
            )
            for game_idx in range(group_size)
        ]
        games = [
            ChessGame(
                sample=sample,
                max_plies=self._max_plies,
                seed=sample.seed + game_idx,
                bot=bot,
            )
            for game_idx, bot in enumerate(bots)
        ]
        policy_colors = (
            (chess.WHITE, chess.BLACK)
            if sample.opponent == "self"
            else (sample.policy_color,)
        )
        players = [(game, color) for game in games for color in policy_colors]
        try:
            await asyncio.gather(*(game.start() for game in games))
            maybe_rollouts = await asyncio.gather(
                *(
                    self._run_player(
                        generate_fn=generate_fn,
                        game=game,
                        color=color,
                        sampling=(
                            sampling
                            if sampling.seed is None
                            else replace(sampling, seed=sampling.seed + player_idx)
                        ),
                        group_id=group_id,
                        rollout_id=player_idx,
                    )
                    for player_idx, (game, color) in enumerate(players)
                )
            )
        finally:
            for bot in bots:
                if bot is not None:
                    bot.close()
        # Drop players with no turns: the game ended before their first move (e.g. White's first
        # move was illegal), or the first prompt was too long. The trainer would drop the whole group.
        played = [
            (rollout, color)
            for rollout, (_, color) in zip(maybe_rollouts, players, strict=True)
            if rollout is not None and rollout.turns
        ]
        rollouts = [rollout for rollout, _ in played]

        # score
        outputs = await self.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        # Post-scoring: center each color against its own mean.
        for color in policy_colors:
            color_rollouts = [rollout for rollout, c in played if c == color]
            if not color_rollouts:
                continue
            advantages = self.advantage_estimator(
                RolloutGroup(group_id=group_id, rollouts=color_rollouts)
            )
            for rollout, advantage in zip(color_rollouts, advantages, strict=True):
                rollout.advantage = advantage

        # Score the policy's moves with Stockfish: a strength measure that does not depend on the opponent.
        losses_per_game = []
        if self._stockfish_path is not None:
            scored_color = None if sample.opponent == "self" else sample.policy_color
            losses_per_game = await asyncio.gather(
                *(
                    asyncio.to_thread(
                        centipawn_losses,
                        fen=sample.fen,
                        moves=list(game.board.move_stack),
                        scored_color=scored_color,
                        engine_path=self._stockfish_path,
                    )
                    for game in games
                )
            )

        # Game metrics ride on a rollout turn: the controller logs every turn's metrics.
        # TODO: move them to RolloutGroup.metrics once the controller appends to it instead of
        # overwriting it in the rollout loop, and validation logs it too.
        if rollouts:
            rollouts[0].turns[-1].metrics.extend(
                _game_metrics(
                    games, rollouts, sample=sample, losses_per_game=losses_per_game
                )
            )
        return RolloutGroup(group_id=group_id, rollouts=rollouts)

    async def _run_player(
        self,
        *,
        generate_fn: GenerateFn,
        game: ChessGame,
        color: chess.Color,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout | None:
        """Play `color`'s half of `game` as one rollout; `None` if the game ends before its first move."""
        await game.wait_for_turn(color)
        if game.is_over:
            return None
        env = self._token_env_config.build(
            message_env=self._message_env_config.build(game=game, color=color),
            renderer=self._renderer,
        )
        rollout = None
        try:
            rollout = await self._run_single_rollout(
                generate_fn=generate_fn,
                env=env,
                sampling=sampling,
                group_id=group_id,
                rollout_id=rollout_id,
            )
            return rollout
        finally:
            # A player can stop without moving: its reply hit `max_tokens`, its history outgrew
            # `max_rollout_tokens`, or it errored. It forfeits with that status, so the other
            # player stops waiting; a no-op if the game already ended.
            # TODO: a game lost to an infra error (status "error") counts as a forfeit; dropping
            # both players' rollouts would keep infra failures out of the reward.
            await game.forfeit(
                color, reason="error" if rollout is None else rollout.status.value
            )
            if (
                rollout is not None
                and rollout.turns
                and not rollout.turns[-1].env_rewards
            ):
                # the env never stepped the reply that stopped this player: score the forfeit here
                rollout.turns[-1].env_rewards = {"score": game.rewards[color]}
            await env.close()


class EloFit:
    """Pools (bot Elo, policy score) pairs across a step and reduces them to one Elo: the rating
    whose expected score matches the actual one, `sum(score) = sum(1 / (1 + 10 ** ((bot_elo - R) / 400)))`.

    Draws and material-scored games count as fractional scores. All losses clamp to 0, all wins to 3000.

    Example:

        EloFit.reduce([EloFit([(1500, 1.0), (1500, 1.0), (1500, 1.0), (1500, 0.0)])])  # {"fit": 1690.8}
    """

    output_suffix: ClassVar[str] = "fit"

    def __init__(self, results: list[tuple[float, float]]) -> None:
        self.results = results

    @classmethod
    def from_list(cls, values: Sequence[tuple[float, float]]) -> EloFit:
        return cls(list(values))

    @classmethod
    def reduce(cls, metrics: Sequence[EloFit]) -> dict[str, float]:
        results = [result for metric in metrics for result in metric.results]
        actual = sum(score for _, score in results)
        low, high = 0.0, 3000.0
        for _ in range(60):
            mid = (low + high) / 2
            expected = sum(
                1 / (1 + 10 ** ((bot_elo - mid) / 400)) for bot_elo, _ in results
            )
            low, high = (mid, high) if expected < actual else (low, mid)
        return {cls.output_suffix: (low + high) / 2}


def _game_metrics(
    games: list[ChessGame],
    rollouts: list[Rollout],
    *,
    sample: ChessSample,
    losses_per_game: list[list[int]],
) -> list[m.Metric]:
    """Per-game end-reason rates and length, the share of policy replies that forfeit, the policy's
    centipawn loss per move, and, against a bot, its mean score and the step's Elo fit."""
    scope = "chess" if sample.split == "train" else "val_chess"
    prefix = f"{scope}_{sample.opponent}"
    metrics = [
        m.Metric(
            f"{prefix}/end_reason/{reason}",
            m.Mean.from_list([float(game.end_reason == reason) for game in games]),
        )
        for reason in _END_REASONS
    ]
    num_forfeiting_replies = sum(game.end_reason in _REPLY_FORFEITS for game in games)
    num_replies = sum(len(rollout.turns) for rollout in rollouts)
    metrics += [
        m.Metric(
            f"{prefix}/num_plies", m.Mean.from_list([game.num_plies for game in games])
        ),
        m.Metric(
            f"{prefix}/forfeit_rate_per_reply",
            m.Mean(float(num_forfeiting_replies), count=float(num_replies)),
        ),
    ]
    losses = [loss for game_losses in losses_per_game for loss in game_losses]
    if losses:
        metrics.append(m.Metric(f"{prefix}/acpl", m.Mean.from_list(losses)))
    if sample.opponent != "self":
        scores = [game.scores[sample.policy_color] for game in games]
        bot_elo = BOTS[sample.opponent].elo
        metrics += [
            m.Metric(f"{prefix}/policy_score", m.Mean.from_list(scores)),
            m.Metric(
                f"{scope}_bot/elo", EloFit([(bot_elo, score) for score in scores])
            ),
        ]
    return metrics
