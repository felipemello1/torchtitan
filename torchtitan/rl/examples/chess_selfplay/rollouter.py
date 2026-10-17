# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import shutil
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
from torchtitan.rl.rollout.types import (
    GenerateFn,
    ReleaseSessionFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
)

if TYPE_CHECKING:
    from torchtitan.rl.generator.generator import SamplingConfig

# `ChessGame.end_reason`s grouped for the end-of-game metrics; any other reason (an error status,
# `truncated_max_turns`) logs as "error".
_END_GROUPS = {
    "checkmate": ("checkmate",),
    "draw": ("stalemate", "insufficient_material"),
    "ply_limit": ("max_plies",),
    "illegal_move": ("illegal_move",),
    "reply_too_long": (RolloutStatus.TRUNCATED_LENGTH.value,),
    "context_full": (RolloutStatus.TRUNCATED_PROMPT_TOO_LONG.value,),
}
# End reasons caused by exactly one bad reply: one per game that ended this way.
_REPLY_FORFEITS = (
    "illegal_move",
    RolloutStatus.TRUNCATED_LENGTH.value,
    RolloutStatus.ERROR_PARSE.value,
)
# End reasons caused by the infrastructure, not the policy: scored as if the game hit `max_plies`.
_INFRA_ERRORS = (
    RolloutStatus.ERROR.value,
    RolloutStatus.ERROR_ABORT.value,
    RolloutStatus.ERROR_TIMEOUT.value,
)


class ChessSelfPlayWorker(RolloutWorker):
    """Plays `group_size` chess games from one start position. Each player's half of a game is one
    multi-turn rollout, so a self-play group of 8 games has up to 16 rollouts.

    Advantages are centered per color; a forfeit caused by one reply is charged to that turn alone
    (see the example's README, "Rewards and advantages"). Against a Stockfish bot, only the
    policy's player is a rollout.

    Example (group_size=2, self-play, max_plies=60):

        game 0: White mates on ply 31                   -> rewards White 1.0, Black -0.25 * (1 - 31 / 60) = -0.12
        game 1: Black forfeits on ply 41, even material -> rewards White 0.5, Black -1 * (1 - 41 / 60 / 2) = -0.66
        advantages: White [+0.25, -0.25], Black [-0.31, +0.31]   (each color's mean is subtracted;
                    Black's forfeit counts as the 0.5 it had at the cap)
        Black's forfeiting turn: +0.31 - (0.5 + 0.66) = -0.85
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RolloutWorker.Config):
        max_plies: int
        """Plies (half-moves, both players) after which a game ends as a draw, adjusted by material.
        Rewards scale with the share of it played (see `ChessGame.rewards`), so any value works."""

        stockfish_path: str | None = "stockfish"
        """Stockfish binary, a path or a name on `PATH`, for bot games and for scoring the policy's
        moves (centipawn loss). `None`: self-play only, and no move scoring."""

        checkmate_reward: float = 1.0
        """The winner's reward for a checkmate, at any length; the prompt states it.
            1:   a mate beats the best score of a game cut at `max_plies` (0.75)
            10:  a mate dominates its group's advantages"""

        checkmated_reward: float | None = None
        """The loser's reward for a checkmate; the prompt states it. A forfeit scores 0.5 to 1 below it.
            None:                  -0.25 x the unplayed share of `max_plies`; a forfeit -0.5 to -1
            1 - checkmate_reward:  a mate is zero-sum around a draw (10 -> -9: +9.5 / -9.5 vs 0.5);
                                   a forfeiting turn then pays ~10, not ~1.5"""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._max_plies = config.max_plies
        self._checkmate_reward = config.checkmate_reward
        self._checkmated_reward = config.checkmated_reward
        if (
            config.stockfish_path is not None
            and shutil.which(config.stockfish_path) is None
        ):
            raise ValueError(
                f"Stockfish not found at {config.stockfish_path!r}; install it, or set "
                "stockfish_path=None for self-play without validation games or move scoring"
            )
        self._stockfish_path = config.stockfish_path

    async def run_group(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        sample: ChessSample,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Play `group_size` games from `sample`, score each player, and center advantages per color.

        Args:
            generate_fn: Async callable that returns a Completion given a prompt.
            release_session_fn: Async callable each player's rollout calls after its last turn,
                to release its routing session.
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
                engine_path=self._stockfish_path,
                seed=sample.seed + game_idx,
            )
            for game_idx in range(group_size)
        ]
        max_plies = self._max_plies if sample.max_plies is None else sample.max_plies
        games = [
            ChessGame(
                sample=sample,
                max_plies=max_plies,
                seed=sample.seed + game_idx,
                bot=bot,
                checkmate_reward=self._checkmate_reward,
                checkmated_reward=self._checkmated_reward,
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
                        release_session_fn=release_session_fn,
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
            (rollout, game, color)
            for rollout, (game, color) in zip(maybe_rollouts, players, strict=True)
            if rollout is not None and rollout.turns
        ]
        rollouts = [rollout for rollout, _, _ in played]
        for rollout in rollouts:
            rollout.logs["opponent"] = sample.opponent

        # score
        outputs = await self.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        # Post-scoring: center each color on its rewards with every reply forfeit and infra error scored
        # as if its game had ended at `max_plies`; a reply forfeit's last turn alone pays the forfeit.
        for color in policy_colors:
            color_players = [
                (rollout, game)
                for rollout, game, player_color in played
                if player_color == color
            ]
            if not color_players:
                continue
            forfeit_costs = [
                game.material_rewards[color] - game.rewards[color]
                if game.forfeiter == color
                and game.end_reason in (*_REPLY_FORFEITS, *_INFRA_ERRORS)
                else 0.0
                for _, game in color_players
            ]
            capped_rollouts = [
                replace(rollout, reward=rollout.reward + forfeit_cost)
                for (rollout, _), forfeit_cost in zip(
                    color_players, forfeit_costs, strict=True
                )
            ]
            advantages = self.advantage_estimator(
                RolloutGroup(group_id=group_id, rollouts=capped_rollouts)
            )
            for (rollout, game), forfeit_cost, advantage in zip(
                color_players, forfeit_costs, advantages, strict=True
            ):
                rollout.advantage = advantage
                if game.forfeiter == color and game.end_reason in _REPLY_FORFEITS:
                    # assumes mean-centered advantages (the recipe's `should_std_normalize=False`)
                    rollout.turns[-1].advantage = advantage - forfeit_cost

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
                    games,
                    rollouts,
                    sample=sample,
                    losses_per_game=losses_per_game,
                    is_validation=group_id < 0,
                )
            )
        return RolloutGroup(group_id=group_id, rollouts=rollouts)

    async def _run_player(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
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
                release_session_fn=release_session_fn,
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
            # TODO: an infra error still counts as a forfeit in the reward and the Elo metrics (the
            # advantages already treat it as a game that hit `max_plies`).
            await game.forfeit(
                color, reason="error" if rollout is None else rollout.status.value
            )
            if (
                rollout is not None
                and rollout.turns
                and not rollout.turns[-1].env_rewards
            ):
                # the env never stepped the reply that stopped this player: score the forfeit here
                rollout.turns[-1].env_rewards = {
                    "score": game.rewards[color],
                    "won": float(game.scores[color] == 1.0),
                }
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
    is_validation: bool,
) -> list[m.Metric]:
    """Metrics in two sections, split by self-play and bot games where both apply.

    Example (training; validation prefixes each section with "validation_"):

        chess_strength/elo                          Elo fitted to the step's bot games
        chess_strength/score_vs_<bot>               the policy's mean chess result against <bot>
        chess_strength/acpl_{self_play,vs_bot}      Stockfish centipawn loss of the policy's moves
        chess_games/reward_{self_play,vs_bot}       mean training reward
        chess_games/plies_{self_play,vs_bot}        game length
        chess_games/max_plies_{self_play,vs_bot}    the games' ply cap
        chess_games/forfeits_per_reply_{self_play,vs_bot}
        chess_games/fallback_moves_per_reply_{self_play,vs_bot}   illegal forced answers saved by the thinking's box
        chess_games/end_{self_play,vs_bot}/<end>    share of games ending in checkmate, draw,
                                                    ply_limit, illegal_move, reply_too_long, context_full, error
    """
    prefix = "validation_" if is_validation else ""
    strength, games_section = f"{prefix}chess_strength", f"{prefix}chess_games"
    kind = "self_play" if sample.opponent == "self" else "vs_bot"
    end_groups = [
        next(
            (
                group
                for group, reasons in _END_GROUPS.items()
                if game.end_reason in reasons
            ),
            "error",
        )
        for game in games
    ]
    metrics = [
        m.Metric(
            f"{games_section}/end_{kind}/{group}",
            m.Mean.from_list([float(end == group) for end in end_groups]),
        )
        for group in [*_END_GROUPS, "error"]
    ]
    num_forfeiting_replies = sum(game.end_reason in _REPLY_FORFEITS for game in games)
    num_replies = sum(len(rollout.turns) for rollout in rollouts)
    metrics += [
        m.Metric(
            f"{games_section}/reward_{kind}",
            m.Mean.from_list([rollout.reward for rollout in rollouts]),
        ),
        m.Metric(
            f"{games_section}/plies_{kind}",
            m.Mean.from_list([game.num_plies for game in games]),
        ),
        m.Metric(
            f"{games_section}/max_plies_{kind}",
            m.Mean.from_list([game.max_plies for game in games]),
        ),
        m.Metric(
            f"{games_section}/forfeits_per_reply_{kind}",
            m.Mean(float(num_forfeiting_replies), count=float(num_replies)),
        ),
        m.Metric(
            f"{games_section}/fallback_moves_per_reply_{kind}",
            m.Mean(
                float(sum(game.num_fallback_moves for game in games)),
                count=float(num_replies),
            ),
        ),
    ]
    losses = [loss for game_losses in losses_per_game for loss in game_losses]
    if losses:
        metrics.append(m.Metric(f"{strength}/acpl_{kind}", m.Mean.from_list(losses)))
    if sample.opponent != "self":
        scores = [game.scores[sample.policy_color] for game in games]
        bot_elo = BOTS[sample.opponent].elo
        metrics += [
            m.Metric(
                f"{strength}/score_vs_{sample.opponent}", m.Mean.from_list(scores)
            ),
            m.Metric(f"{strength}/elo", EloFit([(bot_elo, score) for score in scores])),
        ]
    return metrics
