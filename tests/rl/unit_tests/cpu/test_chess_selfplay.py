# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the chess self-play example."""

import asyncio
import os
import shutil

import chess
import pytest
from renderers import Qwen3RendererConfig

from torchtitan.components.renderer import from_renderers
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.rl.examples.chess_selfplay import (
    BOTS,
    BotSpec,
    ChessGame,
    ChessPlayerEnv,
    ChessSample,
    ChessSelfPlayDataset,
    ChessSelfPlayWorker,
    ChessVsBotDataset,
    material_score,
    RewardChessScore,
    StockfishBot,
)
from torchtitan.rl.examples.chess_selfplay.bots import centipawn_losses
from torchtitan.rl.examples.chess_selfplay.rollouter import EloFit
from torchtitan.rl.generator import SamplingConfig
from torchtitan.rl.observability.controller import compute_rollout_metrics
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout import RolloutStatus
from torchtitan.rl.rollout.environment import TokenEnv
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.types import Completion

_SELF_PLAY = ChessSample(fen=chess.STARTING_FEN, opponent="self")
_TOKENIZER_PATH = "tests/assets/tokenizer"
_STOCKFISH = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
_needs_stockfish = pytest.mark.skipif(
    _STOCKFISH is None, reason="needs a Stockfish binary"
)
# A bot that only plays random moves, so it never starts Stockfish.
_RANDOM_BOT = BotSpec(elo=360, random_move_prob=1.0)


def _new_game(*, max_plies: int = 40, sample: ChessSample = _SELF_PLAY) -> ChessGame:
    return ChessGame(sample=sample, max_plies=max_plies, seed=0)


async def _play_moves(game: ChessGame, moves: list[str]) -> None:
    for move in moves:
        await game.play(game.board.turn, move)


def test_checkmate_ends_the_game() -> None:
    async def run() -> None:
        game = _new_game()
        await _play_moves(game, ["f3", "e5", "g4", "Qh4#"])
        assert game.end_reason == "checkmate"
        assert game.scores == {chess.WHITE: 0.0, chess.BLACK: 1.0}

    asyncio.run(run())


@pytest.mark.parametrize("move_text", ["Ke9", "e5", "--", "", None])
def test_illegal_or_missing_move_forfeits(move_text) -> None:
    async def run() -> None:
        game = _new_game()
        await game.play(chess.WHITE, move_text)
        assert game.end_reason == "illegal_move"
        assert game.scores == {chess.WHITE: 0.0, chess.BLACK: 1.0}

    asyncio.run(run())


@pytest.mark.parametrize("move_text", ["Nf3", "g1f3", " Nf3 ", "Nf3!?"])
def test_san_and_uci_are_accepted(move_text) -> None:
    async def run() -> None:
        game = _new_game()
        await game.play(chess.WHITE, move_text)
        assert not game.is_over
        assert game.board.peek() == chess.Move.from_uci("g1f3")

    asyncio.run(run())


def test_a_move_after_the_game_ended_changes_nothing() -> None:
    async def run() -> None:
        game = _new_game()
        await game.play(chess.WHITE, "e4")
        await game.forfeit(chess.WHITE, reason="truncated_length")
        await game.play(chess.BLACK, "Ke9")
        assert game.end_reason == "truncated_length"
        assert game.scores == {chess.WHITE: 0.0, chess.BLACK: 1.0}
        assert game.num_plies == 1

    asyncio.run(run())


def test_repetition_plays_on_to_the_ply_cap() -> None:
    async def run() -> None:
        game = _new_game(max_plies=20)
        # the start position occurs a fifth time on ply 16
        await _play_moves(game, ["Nf3", "Nf6", "Ng1", "Ng8"] * 4)
        assert game.board.is_fivefold_repetition() and not game.is_over
        await _play_moves(game, ["Nf3", "Nf6", "Ng1", "Ng8"])
        assert game.end_reason == "max_plies"
        assert game.rewards == {chess.WHITE: 0.5, chess.BLACK: 0.5}

    asyncio.run(run())


def test_max_plies_scores_by_material() -> None:
    async def run() -> None:
        game = _new_game(max_plies=4)
        # 1. e4 d5 2. exd5 Nf6: White is a pawn up when the cap hits
        await _play_moves(game, ["e4", "d5", "exd5", "Nf6"])
        assert game.end_reason == "max_plies"
        assert game.scores[chess.WHITE] == pytest.approx(material_score(game.board))
        assert game.scores[chess.WHITE] + game.scores[chess.BLACK] == pytest.approx(1.0)
        assert game.scores[chess.WHITE] > 0.5

    asyncio.run(run())


def test_rewards_make_a_forfeit_cost_more_than_a_loss_and_no_free_win() -> None:
    async def run() -> None:
        # White is checkmated after 2 of its 20 moves
        mate = _new_game()
        await _play_moves(mate, ["f3", "e5", "g4", "Qh4#"])
        assert mate.rewards == {
            chess.WHITE: pytest.approx(-0.25 * (1 - 2 / 20)),
            chess.BLACK: 1.0,
        }

        # 1. e4 d5 2. exd5 Nf6: White is a pawn up
        capped = _new_game(max_plies=4)
        await _play_moves(capped, ["e4", "d5", "exd5", "Nf6"])
        white_reward = 0.5 + 0.5 * (material_score(capped.board) - 0.5)
        assert 0.5 < white_reward < 0.75
        assert capped.rewards == {
            chess.WHITE: pytest.approx(white_reward),
            chess.BLACK: pytest.approx(1.0 - white_reward),
        }

        # same position, but White forfeits after 2 moves: Black, a pawn down, gets the capped
        # reward, not a win
        forfeited = _new_game()
        await _play_moves(forfeited, ["e4", "d5", "exd5", "Nf6", "Ke9"])
        assert forfeited.scores == {chess.WHITE: 0.0, chess.BLACK: 1.0}
        assert forfeited.rewards == {
            chess.WHITE: pytest.approx(-0.5 * (1 - 2 / 20)),
            chess.BLACK: pytest.approx(1.0 - white_reward),
        }

        # forfeiting before the first move costs the full -0.5
        early = _new_game()
        await early.play(chess.WHITE, "Ke9")
        assert early.rewards == {chess.WHITE: -0.5, chess.BLACK: 0.5}

        # White stalemates Black on its first move: a draw is worth 0.5 times the moves played
        draw = _new_game(
            sample=ChessSample(fen="7k/4Q3/8/8/8/8/8/K7 w - - 0 1", opponent="self")
        )
        await draw.play(chess.WHITE, "Qf7")
        assert draw.end_reason == "stalemate"
        assert draw.rewards == {
            chess.WHITE: pytest.approx(0.5 * 1 / 20),
            chess.BLACK: 0.0,
        }

    asyncio.run(run())


def test_material_score_plays_out_captures() -> None:
    assert material_score(chess.Board()) == 0.5
    white_up_a_queen = chess.Board(
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    assert material_score(white_up_a_queen) == pytest.approx(0.905, abs=1e-3)
    # even material, but Black to move takes White's queen on d5 with the e6 pawn
    queen_en_prise = chess.Board(
        "rnbqkbnr/pppp1ppp/4p3/3Q4/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    )
    assert material_score(queen_en_prise) == pytest.approx(
        1 / (1 + 2.718281828 ** (9 / 4)), abs=1e-3
    )
    assert (
        queen_en_prise.fen()
        == "rnbqkbnr/pppp1ppp/4p3/3Q4/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    )


def test_forfeit_wakes_the_waiting_player() -> None:
    async def run() -> None:
        game = _new_game()
        black_waits = asyncio.create_task(game.wait_for_turn(chess.BLACK))
        await asyncio.sleep(0)
        assert not black_waits.done()
        await game.forfeit(chess.WHITE, reason="truncated_length")
        await asyncio.wait_for(black_waits, timeout=1)
        assert game.scores == {chess.WHITE: 0.0, chess.BLACK: 1.0}
        # a second forfeit does not overwrite the result
        await game.forfeit(chess.BLACK, reason="truncated_length")
        assert game.scores == {chess.WHITE: 0.0, chess.BLACK: 1.0}

    asyncio.run(run())


def _bot_game(
    policy_color: chess.Color, *, spec: BotSpec = _RANDOM_BOT, engine_path=None
):
    sample = ChessSample(
        fen=chess.STARTING_FEN, opponent="test_bot", policy_color=policy_color, seed=3
    )
    bot = StockfishBot(spec, name="test_bot", engine_path=engine_path, seed=3)
    return ChessGame(sample=sample, max_plies=40, seed=0, bot=bot), bot


def test_bot_opens_on_start_when_policy_plays_black() -> None:
    async def run() -> None:
        game, _ = _bot_game(chess.BLACK)
        assert game.num_plies == 0
        await game.start()
        assert game.board.turn == chess.BLACK
        assert game.num_plies == 1
        assert "White played" in game.turn_message()

    asyncio.run(run())


def test_bot_replies_inside_play() -> None:
    async def run() -> None:
        game, _ = _bot_game(chess.WHITE)
        await game.start()  # White to move: the bot waits
        assert game.num_plies == 0
        await game.play(chess.WHITE, "e4")
        assert game.board.turn == chess.WHITE
        assert game.num_plies == 2

    asyncio.run(run())


@_needs_stockfish
def test_stockfish_bot_at_uci_elo_plays_legal_moves() -> None:
    async def run() -> None:
        game, bot = _bot_game(
            chess.WHITE, spec=BOTS["sf_elo1320"], engine_path=_STOCKFISH
        )
        for move in ["e4", "Nf3", "Bc4"]:
            await game.play(chess.WHITE, move)
            if game.is_over:
                break
        bot.close()
        assert game.num_plies == 6

    asyncio.run(run())


@_needs_stockfish
def test_centipawn_losses_rank_moves() -> None:
    moves = [chess.Move.from_uci(uci) for uci in ["e2e4", "e7e5", "f2f3"]]
    losses = centipawn_losses(
        fen=chess.STARTING_FEN, moves=moves, scored_color=None, engine_path=_STOCKFISH
    )
    assert len(losses) == 3
    assert losses[2] > losses[0]  # f3 is worse than e4
    white_only = centipawn_losses(
        fen=chess.STARTING_FEN,
        moves=moves,
        scored_color=chess.WHITE,
        engine_path=_STOCKFISH,
    )
    assert len(white_only) == 2


def test_elo_fit() -> None:
    # 3 wins and a loss against a 1500 bot: 1500 + 400 * log10(3)
    assert EloFit.reduce([EloFit([(1500, 1.0)] * 3), EloFit([(1500, 0.0)])])[
        "fit"
    ] == pytest.approx(1690.8, abs=0.1)
    assert EloFit.reduce([EloFit([(1500, 0.0)])])["fit"] == pytest.approx(0.0, abs=1e-6)


def test_player_env_shows_the_board_and_scores_the_end() -> None:
    async def run() -> None:
        game = _new_game()
        white = ChessPlayerEnv(ChessPlayerEnv.Config(), game=game, color=chess.WHITE)
        init = await white.init()
        prompt = init.init_prompt_messages[0]["content"]
        assert "You are playing chess as White" in prompt
        assert "Legal moves: " in prompt
        assert prompt.endswith(
            "Think briefly, then write one legal move inside \\boxed{}."
        )
        # no move written as an example, so copying the prompt never plays a move
        assert "\\boxed{e4}" not in prompt and "\\boxed{Nf3}" not in prompt

        white_step = asyncio.create_task(
            white.step({"role": "assistant", "content": "I open. \\boxed{e4}"})
        )
        await asyncio.sleep(0)
        assert not white_step.done()  # waits for Black's move
        await game.play(chess.BLACK, "e5")
        step_output = await asyncio.wait_for(white_step, timeout=1)
        assert not step_output.done
        assert step_output.env_messages[0]["content"].startswith("Black played e5.")

        step_output = await white.step({"role": "assistant", "content": "\\boxed{Ke9}"})
        assert step_output.done
        # White forfeits after 1 of its 20 moves
        assert step_output.env_rewards == {"score": pytest.approx(-0.5 * (1 - 1 / 20))}

    asyncio.run(run())


def test_legal_moves_are_shuffled_per_game() -> None:
    def listed_moves(seed: int) -> list[str]:
        game = ChessGame(sample=_SELF_PLAY, max_plies=40, seed=seed)
        line = next(
            line
            for line in game.turn_message().splitlines()
            if line.startswith("Legal moves: ")
        )
        return line.removeprefix("Legal moves: ").split()

    board = chess.Board()
    in_generation_order = [board.san(move) for move in board.legal_moves]
    assert sorted(listed_moves(0)) == sorted(in_generation_order)
    assert listed_moves(0) == listed_moves(0)
    assert listed_moves(0) != listed_moves(1)
    assert listed_moves(0) != in_generation_order


def test_datasets_are_deterministic_and_resumable() -> None:
    first = ChessSelfPlayDataset.Config(seed=1, max_opening_plies=3).build()
    second = ChessSelfPlayDataset.Config(seed=1, max_opening_plies=3).build()
    assert [next(first) for _ in range(5)] == [next(second) for _ in range(5)]
    state = first.state_dict()
    expected = [next(first) for _ in range(3)]
    first.load_state_dict(state)
    assert [next(first) for _ in range(3)] == expected

    validation = ChessVsBotDataset.Config(
        num_games=4, opponents=("sf_eps90", "sf_eps50")
    ).build()
    samples = [next(validation) for _ in range(8)]
    assert samples[:4] == samples[4:]
    assert [(sample.opponent, sample.policy_color) for sample in samples[:4]] == [
        ("sf_eps90", True),
        ("sf_eps90", False),
        ("sf_eps50", True),
        ("sf_eps50", False),
    ]
    assert all(sample.split == "validation" for sample in samples)


def test_self_play_dataset_mixes_in_bot_groups() -> None:
    dataset = ChessSelfPlayDataset.Config(
        bots=("sf_eps90", "sf_eps50"), bot_fraction=0.5
    ).build()
    opponents = [next(dataset).opponent for _ in range(100)]
    assert opponents[0::2] == ["self"] * 50
    assert set(opponents[1::2]) == {"sf_eps90", "sf_eps50"}
    assert all(
        next(ChessSelfPlayDataset.Config().build()).opponent == "self" for _ in range(3)
    )


# ======== Worker: plays both players of each game against a scripted generate_fn ========


class _ScriptedPolicy:
    """Answers each player's turns from a script keyed by rollout id (even = White, odd = Black)."""

    def __init__(
        self, tokenizer, moves_by_rollout: dict[int, list[str]], *, truncate_at=None
    ) -> None:
        self._tokenizer = tokenizer
        self._moves_by_rollout = moves_by_rollout
        self._truncate_at = (
            truncate_at  # (rollout_id, turn_id) whose reply hits max_tokens
        )

    async def __call__(self, prompt_token_ids, *, request_id, **kwargs) -> Completion:
        fields = dict(part.split("=") for part in request_id.split("/"))
        rollout_id, turn_id = int(fields["rollout"]), int(fields["turn"])
        text = f"\\boxed{{{self._moves_by_rollout[rollout_id][turn_id]}}}"
        token_ids = self._tokenizer.encode(text, add_bos=False, add_eos=False)
        finish_reason = "stop"
        if (rollout_id, turn_id) == self._truncate_at:
            finish_reason = "length"
        else:
            token_ids.append(self._tokenizer.tokenizer.token_to_id("<|im_end|>"))
        return Completion(
            min_policy_version=0,
            max_policy_version=0,
            request_id=request_id,
            token_ids=token_ids,
            token_logprobs=[-0.1] * len(token_ids),
            finish_reason=finish_reason,
        )


async def _run_group(
    moves_by_rollout, *, group_size, sample=_SELF_PLAY, truncate_at=None
):
    worker = ChessSelfPlayWorker.Config(
        rubric=Rubric.Config(
            reward_fns=[RewardChessScore.Config()],
        ),
        message_env=ChessPlayerEnv.Config(),
        token_env=TokenEnv.Config(step_timeout_s=None),
        max_plies=40,
    ).build()
    tokenizer_config = HuggingFaceTokenizer.Config()
    await worker.setup_async(
        tokenizer_config=tokenizer_config,
        renderer_config=from_renderers(
            Qwen3RendererConfig(enable_thinking=True, thinking_retention="all")
        ),
        hf_assets_path=_TOKENIZER_PATH,
    )
    policy = _ScriptedPolicy(
        tokenizer_config.build(tokenizer_path=_TOKENIZER_PATH),
        moves_by_rollout,
        truncate_at=truncate_at,
    )
    return await asyncio.wait_for(
        worker.run_group(
            generate_fn=policy,
            sample=sample,
            group_id=0,
            group_size=group_size,
            sampling=SamplingConfig(),
        ),
        timeout=10,
    )


def _reduced_metrics(rollouts, prefix="rollout"):
    return MetricsProcessor._aggregate_metrics(
        compute_rollout_metrics(prefix=prefix, rollouts=rollouts)
    )


def test_worker_trains_both_colors_with_per_color_advantages() -> None:
    async def run() -> None:
        group = await _run_group(
            {
                0: ["f3", "g4"],  # game 0: fool's mate, Black wins
                1: ["e5", "Qh4#"],
                2: ["e4"],  # game 1: Black forfeits on its first move
                3: ["Ke9"],
            },
            group_size=2,
        )
        by_id = {rollout.rollout_id: rollout for rollout in group.rollouts}
        assert sorted(by_id) == [0, 1, 2, 3]
        assert all(r.status == RolloutStatus.COMPLETED for r in group.rollouts)
        # White is mated after 2 of its 20 moves; Black's forfeit before its first move costs -0.5
        # and gives White a draw's 0.5 at even material, not a win
        assert [by_id[i].reward for i in range(4)] == pytest.approx(
            [-0.225, 1.0, 0.5, -0.5]
        )
        # each color is centered on its own mean (White 0.1375, Black 0.25)
        assert [by_id[i].advantage for i in range(4)] == pytest.approx(
            [-0.3625, 0.75, 0.3625, -0.75]
        )
        assert [len(by_id[i].turns) for i in range(4)] == [2, 2, 1, 1]

        reduced = _reduced_metrics(group.rollouts)
        assert reduced["chess_self/end_reason/checkmate/mean"] == 0.5
        assert reduced["chess_self/end_reason/illegal_move/mean"] == 0.5
        # 6 policy replies, 1 of them illegal
        assert reduced["chess_self/forfeit_rate_per_reply/mean"] == pytest.approx(1 / 6)

    asyncio.run(run())


def test_worker_drops_a_player_that_never_moved() -> None:
    async def run() -> None:
        group = await _run_group({0: ["Ke9"], 1: []}, group_size=1)
        assert [rollout.rollout_id for rollout in group.rollouts] == [0]
        assert group.rollouts[0].reward == -0.5

    asyncio.run(run())


def test_worker_forfeits_a_player_that_stops_mid_game() -> None:
    async def run() -> None:
        # White's second reply hits max_tokens: the TokenEnv stops White without stepping the
        # env, so the worker must forfeit for White or Black would wait forever.
        group = await _run_group(
            {0: ["e4", "d4"], 1: ["e5", "d5"]}, group_size=1, truncate_at=(0, 1)
        )
        white, black = group.rollouts
        assert white.status == RolloutStatus.TRUNCATED_LENGTH
        assert black.status == RolloutStatus.COMPLETED
        # White forfeits after 1 of its 20 moves; Black gets the even-material 0.5
        assert (white.reward, black.reward) == pytest.approx((-0.475, 0.5))
        assert white.turns[-1].env_rewards == {"score": pytest.approx(-0.475)}
        assert black.turns[-1].env_rewards == {"score": 0.5}
        assert (
            _reduced_metrics(group.rollouts)[
                "chess_self/end_reason/truncated_length/mean"
            ]
            == 1.0
        )

    asyncio.run(run())


def test_worker_plays_only_the_policy_against_a_bot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(BOTS, "test_bot", _RANDOM_BOT)

    async def run() -> None:
        sample = ChessSample(
            fen=chess.STARTING_FEN,
            opponent="test_bot",
            policy_color=chess.BLACK,
            seed=3,
            split="validation",
        )
        group = await _run_group({0: ["Ke9"]}, group_size=1, sample=sample)
        assert [rollout.rollout_id for rollout in group.rollouts] == [0]
        rollout = group.rollouts[0]
        assert rollout.reward == -0.5
        # Black's first prompt shows the bot's opening move
        assert "White played" in rollout.turns[0].prompt_messages[-1]["content"]

        reduced = _reduced_metrics(group.rollouts, prefix="validation")
        assert reduced["validation_reward/_mean"] == -0.5
        # the Elo metrics use the chess result: a forfeit is a loss
        assert reduced["val_chess_test_bot/policy_score/mean"] == 0.0
        assert reduced["val_chess_test_bot/forfeit_rate_per_reply/mean"] == 1.0
        assert (
            reduced["val_chess_test_bot/num_plies/mean"] == 1.0
        )  # the bot's opening move
        assert reduced["val_chess_bot/elo/fit"] == pytest.approx(
            0.0, abs=1e-6
        )  # one loss

    asyncio.run(run())
