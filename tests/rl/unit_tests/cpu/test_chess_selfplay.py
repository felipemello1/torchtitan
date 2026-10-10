# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the chess self-play example."""

import asyncio
import json
import os
import shutil

import chess
import pytest
import torch
from renderers import Qwen3RendererConfig

from torchtitan.components.renderer import from_renderers
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.components.batcher import Batcher
from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
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
from torchtitan.rl.examples.chess_selfplay.bots import (
    centipawn_losses,
    executable_stockfish,
)
from torchtitan.rl.examples.chess_selfplay.openings import OPENINGS
from torchtitan.rl.examples.chess_selfplay.rollouter import EloFit
from torchtitan.rl.generator import SamplingConfig
from torchtitan.rl.observability.controller import compute_rollout_metrics
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.environment import TokenEnv
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.types import Completion, RolloutTurnID

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


@pytest.mark.parametrize(
    "move_text",
    ["Ke9", "e5", "--", "", None, "g1f3", "Pe4", "Ng1-Nf3", "Ngf3", "Nf3!?"],
)
def test_illegal_or_missing_move_forfeits(move_text) -> None:
    async def run() -> None:
        game = _new_game()
        await game.play(chess.WHITE, move_text)
        assert game.end_reason == "illegal_move"
        assert game.scores == {chess.WHITE: 0.0, chess.BLACK: 1.0}

    asyncio.run(run())


@pytest.mark.parametrize("move_text", ["Nf3", " Nf3 ", "Nf3+", "Nf3\\#"])
def test_the_listed_move_is_accepted(move_text) -> None:
    async def run() -> None:
        game = _new_game()
        await game.play(chess.WHITE, move_text)
        assert not game.is_over
        assert game.board.peek() == chess.Move.from_uci("g1f3")

    asyncio.run(run())


@pytest.mark.parametrize("move_text", ["Qxf7+", "Qf7+", "Qf7"])
def test_a_capture_is_accepted_with_or_without_its_x(move_text) -> None:
    async def run() -> None:
        game = _new_game()
        await _play_moves(game, ["e4", "e5", "Qh5", "Nc6", move_text])
        assert not game.is_over
        assert game.board.peek() == chess.Move.from_uci("h5f7")

    asyncio.run(run())


def test_an_x_on_a_move_that_does_not_capture_is_ignored() -> None:
    async def run() -> None:
        game = _new_game()
        await game.play(chess.WHITE, "Nxf3")
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
        # White is checkmated on ply 4 of 40
        mate = _new_game()
        await _play_moves(mate, ["f3", "e5", "g4", "Qh4#"])
        assert mate.rewards == {
            chess.WHITE: pytest.approx(-0.25 * (1 - 4 / 40)),
            chess.BLACK: 10.0,
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

        # same position, but White forfeits at ply 4: Black, a pawn down, gets the capped reward,
        # not a win
        forfeited = _new_game()
        await _play_moves(forfeited, ["e4", "d5", "exd5", "Nf6", "Ke9"])
        assert forfeited.scores == {chess.WHITE: 0.0, chess.BLACK: 1.0}
        assert forfeited.rewards == {
            chess.WHITE: pytest.approx(-1.0 * (1 - 4 / 80)),
            chess.BLACK: pytest.approx(1.0 - white_reward),
        }
        # a forfeit, however late, costs more than a checkmate, however early
        assert forfeited.rewards[chess.WHITE] < mate.rewards[chess.WHITE]

        # forfeiting before the first move costs the full -1
        early = _new_game()
        await early.play(chess.WHITE, "Ke9")
        assert early.rewards == {chess.WHITE: -1.0, chess.BLACK: 0.5}

        # White stalemates Black on ply 1: a draw is worth 0.5 times the share of plies played
        draw = _new_game(
            sample=ChessSample(fen="7k/4Q3/8/8/8/8/8/K7 w - - 0 1", opponent="self")
        )
        await draw.play(chess.WHITE, "Qf7")
        assert draw.end_reason == "stalemate"
        assert draw.rewards == {
            chess.WHITE: pytest.approx(0.5 * 1 / 40),
            chess.BLACK: pytest.approx(0.5 * 1 / 40),
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
        assert "(ply 2 of 40)" in game.turn_message()

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
def test_centipawn_losses_ignore_repetitions() -> None:
    # White, a queen up, shuffles its king until the position repeats a third time: each move keeps
    # the win, so it costs little, not the ~2000 of a position Stockfish would call a draw.
    moves = [
        chess.Move.from_uci(uci)
        for uci in ["e1d1", "e8d8", "d1e1", "d8e8", "e1d1", "e8d8", "d1e1", "d8e8"]
    ]
    losses = centipawn_losses(
        fen="4k3/8/8/8/8/8/8/Q3K3 w - - 0 1",
        moves=moves,
        scored_color=chess.WHITE,
        engine_path=_STOCKFISH,
    )
    assert len(losses) == 4
    assert max(losses) < 200


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


def test_executable_stockfish_copies_a_binary_without_the_execute_bit(tmp_path) -> None:
    staged = tmp_path / "stockfish"
    staged.write_bytes(b"binary")
    staged.chmod(0o444)
    copy = executable_stockfish(str(staged))
    assert copy != str(staged)
    assert os.access(copy, os.X_OK) and open(copy, "rb").read() == b"binary"
    staged.chmod(0o755)
    assert executable_stockfish(str(staged)) == str(staged)
    assert executable_stockfish("stockfish") == "stockfish"
    assert executable_stockfish(None) is None


def test_reward_loses_the_share_of_force_closed_turns() -> None:
    def turn(
        turn_id: int, loss_mask: list[bool] | None, score: float | None = None
    ) -> RolloutTurn:
        return RolloutTurn(
            rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=turn_id),
            prompt_prefix_len=0,
            prompt_delta_token_ids=[1],
            completion_token_ids=[2, 3],
            completion_logprobs=[-0.1, -0.1],
            completion_loss_mask=loss_mask,
            env_rewards={} if score is None else {"score": score},
            min_policy_version=0,
            max_policy_version=0,
        )

    # 1 of 4 turns had its thinking force-closed (its forced tokens are masked out of the loss)
    rollout = Rollout(
        group_id=0,
        rollout_id=0,
        status=RolloutStatus.COMPLETED,
        turns=[
            turn(0, None),
            turn(1, [True, False]),
            turn(2, [True, True]),
            turn(3, None, score=1.0),
        ],
    )
    penalized = RewardChessScore.Config(forced_close_penalty=0.1).build()
    assert asyncio.run(penalized(rollout, _SELF_PLAY)) == pytest.approx(1.0 - 0.1 / 4)
    assert asyncio.run(RewardChessScore.Config().build()(rollout, _SELF_PLAY)) == 1.0


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
        assert '"Pe2": [' in prompt and "Opponent pieces (Black)" in prompt
        assert prompt.endswith("Write your best legal move inside \\boxed{}.")
        # one right and one wrong way to write a move
        assert "means \\boxed{Nbd2}, not \\boxed{Nd2}" in prompt
        assert "An x marks a capture" in prompt
        assert "stops after 40 plies" in prompt and "(ply 1 of 40)" in prompt

        white_step = asyncio.create_task(
            white.step({"role": "assistant", "content": "I open. \\boxed{e4}"})
        )
        await asyncio.sleep(0)
        assert not white_step.done()  # waits for Black's move
        await game.play(chess.BLACK, "e5")
        step_output = await asyncio.wait_for(white_step, timeout=1)
        assert not step_output.done
        assert step_output.env_messages[0]["content"].startswith("Black played e5.")
        assert "(ply 3 of 40)" in step_output.env_messages[0]["content"]

        step_output = await white.step({"role": "assistant", "content": "\\boxed{Ke9}"})
        assert step_output.done
        # White forfeits at ply 2 of 40
        assert step_output.env_rewards == {"score": pytest.approx(-1.0 * (1 - 2 / 80))}

    asyncio.run(run())


def test_legal_moves_are_shuffled_per_game() -> None:
    def listed_moves(seed: int) -> list[str]:
        game = ChessGame(sample=_SELF_PLAY, max_plies=40, seed=seed)
        text = game.turn_message()
        by_piece = json.loads(text[text.index("{") : text.index("}") + 1])
        return [move for moves in by_piece.values() for move in moves]

    board = chess.Board()
    in_generation_order = [board.san(move) for move in board.legal_moves]
    assert sorted(listed_moves(0)) == sorted(in_generation_order)
    assert listed_moves(0) == listed_moves(0)
    assert listed_moves(0) != listed_moves(1)


def test_turn_message_shows_pinned_pieces_without_moves() -> None:
    # White's bishop on b5 pins Black's knight on c6 to the king
    sample = ChessSample(
        fen="r1bqkbnr/pp2pppp/2n5/1B6/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 0 1",
        opponent="self",
    )
    text = ChessGame(sample=sample, max_plies=40, seed=0).turn_message()
    by_piece = json.loads(text[text.index("{") : text.index("}") + 1])
    assert by_piece["Nc6"] == [] and "Pd7" not in by_piece
    block = text.split("Opponent pieces (White)")[1]
    opponent = json.loads(block[block.index("{") : block.index("}") + 1])
    assert (
        "Bxc6+" in opponent["Bb5"]
    )  # Black is not in check, so White's threats keep their marks


def test_opponent_moves_drop_king_captures_and_false_checks() -> None:
    # Black to move, in check from White's bishop on b5
    sample = ChessSample(
        fen="rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 3",
        opponent="self",
    )
    text = ChessGame(sample=sample, max_plies=40, seed=0).turn_message()
    block = text.split("Opponent pieces (White)")[1]
    opponent = json.loads(block[block.index("{") : block.index("}") + 1])
    assert "Bxe8" not in opponent["Bb5"] and "Bxe8+" not in opponent["Bb5"]
    assert not any(move.endswith("+") for moves in opponent.values() for move in moves)


def test_start_positions_come_from_the_opening_book() -> None:
    for _, line in OPENINGS:
        board = chess.Board()
        for san in line.split():
            board.push_san(san)  # every book move is legal
    dataset = ChessSelfPlayDataset.Config().build()
    starts = {next(dataset).fen for _ in range(50)}
    assert len(starts) > 25  # many book positions, not only the standard one


def test_datasets_are_deterministic_and_resumable() -> None:
    first = ChessSelfPlayDataset.Config(seed=1).build()
    second = ChessSelfPlayDataset.Config(seed=1).build()
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
    """Answers each player's turns from a script keyed by rollout id (even = White, odd = Black).
    An exception in the script is raised instead, like a failed generator call. With `num_topk=2`,
    each token's top-k rows are [token, 1000 + turn_id] with logprobs [-0.1, -2.0]."""

    def __init__(
        self,
        tokenizer,
        moves_by_rollout: dict[int, list[str]],
        *,
        truncate_at=None,
        num_topk: int = 0,
    ) -> None:
        self._tokenizer = tokenizer
        self._moves_by_rollout = moves_by_rollout
        self._truncate_at = (
            truncate_at  # (rollout_id, turn_id) whose reply hits max_tokens
        )
        self._num_topk = num_topk

    async def __call__(self, prompt_token_ids, *, request_id, **kwargs) -> Completion:
        fields = dict(part.split("=") for part in request_id.split("/"))
        rollout_id, turn_id = int(fields["rollout"]), int(fields["turn"])
        move = self._moves_by_rollout[rollout_id][turn_id]
        if isinstance(move, Exception):
            raise move
        text = f"\\boxed{{{move}}}"
        token_ids = self._tokenizer.encode(text, add_bos=False, add_eos=False)
        finish_reason = "stop"
        if (rollout_id, turn_id) == self._truncate_at:
            finish_reason = "length"
        else:
            token_ids.append(self._tokenizer.tokenizer.token_to_id("<|im_end|>"))
        topk_token_ids = topk_logprobs = None
        if self._num_topk:
            topk_token_ids = torch.tensor(
                [[token_id, 1000 + turn_id] for token_id in token_ids],
                dtype=torch.int32,
            )
            topk_logprobs = torch.tensor([[-0.1, -2.0]] * len(token_ids))
        return Completion(
            min_policy_version=0,
            max_policy_version=0,
            request_id=request_id,
            token_ids=token_ids,
            token_logprobs=[-0.1] * len(token_ids),
            topk_token_ids=topk_token_ids,
            topk_logprobs=topk_logprobs,
            finish_reason=finish_reason,
        )


async def _run_group(
    moves_by_rollout,
    *,
    group_size,
    sample=_SELF_PLAY,
    truncate_at=None,
    worker=None,
    num_topk=0,
):
    worker = (
        worker
        or ChessSelfPlayWorker.Config(
            rubric=Rubric.Config(
                reward_fns=[RewardChessScore.Config()],
            ),
            message_env=ChessPlayerEnv.Config(),
            token_env=TokenEnv.Config(step_timeout_s=None),
            max_plies=40,
        ).build()
    )
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
        num_topk=num_topk,
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
        # White is mated on ply 4 of 40; Black's forfeit at ply 1 costs -1 * (1 - 1 / 80) and
        # gives White a draw's 0.5 at even material, not a win
        assert [by_id[i].reward for i in range(4)] == pytest.approx(
            [-0.225, 10.0, 0.5, -0.9875]
        )
        # each color is centered on its own mean, Black's forfeit counted as its 0.5 at the cap
        # (White 0.1375, Black 5.25); Black's forfeiting turn alone pays -4.75 - (0.5 + 0.9875)
        assert [by_id[i].advantage for i in range(4)] == pytest.approx(
            [-0.3625, 4.75, 0.3625, -4.75]
        )
        assert [len(by_id[i].turns) for i in range(4)] == [2, 2, 1, 1]
        assert [[turn.advantage for turn in by_id[i].turns] for i in range(4)] == [
            [None, None],
            [None, None],
            [None],
            [pytest.approx(-6.2375)],
        ]

        reduced = _reduced_metrics(group.rollouts)
        assert reduced["chess_games/end_self_play/checkmate/mean"] == 0.5
        assert reduced["chess_games/end_self_play/illegal_move/mean"] == 0.5
        # 6 policy replies, 1 of them illegal
        assert reduced[
            "chess_games/forfeits_per_reply_self_play/mean"
        ] == pytest.approx(1 / 6)

    asyncio.run(run())


def test_worker_carries_each_turns_topk_rows_into_the_packed_training_sample() -> None:
    async def run() -> None:
        # fool's mate: two turns per player, the opponent's move and the next board between them
        group = await _run_group(
            {0: ["f3", "g4"], 1: ["e5", "Qh4#"]}, group_size=1, num_topk=2
        )
        builder = TrainingSampleBuilder.Config().build()
        samples = []
        for rollout in group.rollouts:
            [sample] = builder.rollout_to_training_samples(rollout)
            samples.append(sample)
            loss_mask = sample.loss_mask
            assert sample.topk_token_ids.shape == (len(sample.token_ids), 2)
            # completion tokens: the policy's rows, from the turn that sampled them
            assert torch.equal(
                sample.topk_token_ids[loss_mask, 0], sample.token_ids[loss_mask]
            )
            assert sample.topk_token_ids[loss_mask, 1].tolist() == [
                1000 + turn_id
                for turn_id, turn in enumerate(rollout.turns)
                for _ in turn.completion_token_ids
            ]
            torch.testing.assert_close(
                sample.topk_logprobs[loss_mask],
                torch.tensor([[-0.1, -2.0]] * int(loss_mask.sum())),
            )
            # prompt, opponent move and board tokens: zero rows
            assert not sample.topk_token_ids[~loss_mask].any()
            assert not sample.topk_logprobs[~loss_mask].any()

        # Packed: row t holds the top-k that sampled labels[t], zero rows elsewhere.
        num_tokens = 64 * -(-sum(len(sample.token_ids) for sample in samples) // 64)
        batcher = Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=num_tokens,
            max_context_length=num_tokens,
            num_prompts_per_train_step=1,
            dp_degree=1,
            pad_id=0,
            temperature=1.0,
            num_topk_logprobs=2,
        )
        microbatch = batcher._pack_training_samples(samples)
        topk_token_ids = microbatch.generator_topk_token_ids
        loss_mask = microbatch.loss_mask
        assert topk_token_ids.shape == (num_tokens, 2)
        assert torch.equal(topk_token_ids[loss_mask, 0], microbatch.labels[loss_mask])
        assert not topk_token_ids[~loss_mask].any()
        assert not microbatch.generator_topk_logprobs[~loss_mask].any()

    asyncio.run(run())


def test_worker_centers_each_color_as_if_its_forfeits_ended_at_the_cap() -> None:
    async def run() -> None:
        group = await _run_group(
            {
                0: ["f3", "g4"],  # game 0: White is mated on ply 4
                1: ["e5", "Qh4#"],
                2: ["e4", "exd5", "Ke9"],  # game 1: White forfeits on ply 4, a pawn up
                3: ["d5", "Nf6"],
                4: ["d4", "Ke9"],  # game 2: White forfeits on ply 2, even material
                5: ["d5"],
            },
            group_size=3,
        )
        by_id = {rollout.rollout_id: rollout for rollout in group.rollouts}
        white, black = (0, 2, 4), (1, 3, 5)
        assert [by_id[i].reward for i in white] == pytest.approx(
            [-0.225, -0.95, -0.975]
        )
        # White centers on [-0.225, 0.5311, 0.5]: each forfeit counts as White's reward at the cap,
        # 0.5 + 0.5 * (material_score - 0.5) with a pawn up, then 0.5 at even material; mean 0.2687
        assert [by_id[i].advantage for i in white] == pytest.approx(
            [-0.4937, 0.2624, 0.2313], abs=1e-4
        )
        # each forfeiting turn alone pays: its reward against that mean, -0.95 - 0.2687, -0.975 - 0.2687
        assert [[turn.advantage for turn in by_id[i].turns] for i in white] == [
            [None, None],
            [None, None, pytest.approx(-1.2187, abs=1e-4)],
            [None, pytest.approx(-1.2437, abs=1e-4)],
        ]
        # Black never forfeits: plain centering on [10.0, 0.4689, 0.5], and no turn overrides
        assert [by_id[i].advantage for i in black] == pytest.approx(
            [6.3437, -3.1874, -3.1563], abs=1e-4
        )
        assert all(turn.advantage is None for i in black for turn in by_id[i].turns)

    asyncio.run(run())


def test_worker_centers_an_infra_error_as_if_its_game_ended_at_the_cap() -> None:
    async def run() -> None:
        group = await _run_group(
            {
                0: ["f3", "g4"],  # game 0: White is mated on ply 4
                1: ["e5", "Qh4#"],
                # game 1: White's third generator call fails on ply 4, at even material
                2: ["e4", "Nf3", RuntimeError("generator lost")],
                3: ["e5", "Nc6"],
            },
            group_size=2,
        )
        by_id = {rollout.rollout_id: rollout for rollout in group.rollouts}
        white = by_id[2]
        assert white.status == RolloutStatus.ERROR
        assert white.reward == pytest.approx(-1 * (1 - 4 / 80))
        # it counts as its 0.5 at the cap, White centers on [-0.225, 0.5], and no turn pays
        assert [by_id[0].advantage, white.advantage] == pytest.approx([-0.3625, 0.3625])
        assert [turn.advantage for turn in white.turns] == [None, None]
        assert (
            _reduced_metrics(group.rollouts)["chess_games/end_self_play/error/mean"]
            == 0.5
        )

    asyncio.run(run())


def test_worker_drops_a_player_that_never_moved() -> None:
    async def run() -> None:
        group = await _run_group({0: ["Ke9"], 1: []}, group_size=1)
        assert [rollout.rollout_id for rollout in group.rollouts] == [0]
        assert group.rollouts[0].reward == -1.0

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
        # White forfeits at ply 2 of 40; Black gets the even-material 0.5
        assert (white.reward, black.reward) == pytest.approx((-0.975, 0.5))
        assert white.turns[-1].env_rewards == {"score": pytest.approx(-0.975)}
        assert black.turns[-1].env_rewards == {"score": 0.5}
        # a reply cut at max_tokens is a reply forfeit too: White centers on its 0.5 at the cap,
        # and the cut turn alone pays -0.975 - 0.5
        assert white.advantage == pytest.approx(0.0)
        assert [turn.advantage for turn in white.turns] == [None, pytest.approx(-1.475)]
        assert [turn.advantage for turn in black.turns] == [None]
        assert (
            _reduced_metrics(group.rollouts)[
                "chess_games/end_self_play/reply_too_long/mean"
            ]
            == 1.0
        )

    asyncio.run(run())


def test_worker_moves_up_the_bot_curriculum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(BOTS, "bot_a", BotSpec(elo=100, random_move_prob=1.0))
    monkeypatch.setitem(BOTS, "bot_b", BotSpec(elo=300, random_move_prob=1.0))

    async def run(win_rate: float) -> list[float]:
        worker = ChessSelfPlayWorker.Config(
            rubric=Rubric.Config(reward_fns=[RewardChessScore.Config()]),
            message_env=ChessPlayerEnv.Config(),
            token_env=TokenEnv.Config(step_timeout_s=None),
            max_plies=40,
            bot_curriculum=("bot_a", "bot_b"),
            curriculum_win_rate=win_rate,
            curriculum_games=1,
        ).build()
        sample = ChessSample(
            fen=chess.STARTING_FEN,
            opponent="curriculum",
            policy_color=chess.BLACK,
            seed=3,
        )
        elos = []
        for _ in range(3):
            group = await _run_group(
                {0: ["Ke9"]}, group_size=1, sample=sample, worker=worker
            )
            elos.append(
                _reduced_metrics(group.rollouts)[
                    "chess_strength/curriculum_bot_elo/mean"
                ]
            )
        return elos

    # each game is a forfeit, a loss: the worker stays on bot_a
    assert asyncio.run(run(win_rate=0.6)) == [100, 100, 100]
    # any win rate clears -1: it moves up after the first game, then stays on the last bot
    assert asyncio.run(run(win_rate=-1.0)) == [100, 300, 300]


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
        # Black forfeits at ply 1, after the bot's opening move
        assert rollout.reward == pytest.approx(-1.0 * (1 - 1 / 80))
        # Black's first prompt shows the bot's opening move
        assert "White played" in rollout.turns[0].prompt_messages[-1]["content"]

        reduced = _reduced_metrics(group.rollouts, prefix="validation")
        assert reduced["validation_reward/_mean"] == pytest.approx(-0.9875)
        # the Elo metrics use the chess result: a forfeit is a loss
        assert reduced["val_chess_strength/score_vs_test_bot/mean"] == 0.0
        assert reduced["val_chess_games/end_vs_bot/checkmate_by_policy/mean"] == 0.0
        assert reduced["val_chess_games/forfeits_per_reply_vs_bot/mean"] == 1.0
        assert (
            reduced["val_chess_games/plies_vs_bot/mean"] == 1.0
        )  # the bot's opening move
        assert reduced["val_chess_strength/elo/fit"] == pytest.approx(
            0.0, abs=1e-6
        )  # one loss

    asyncio.run(run())


def test_worker_puts_a_bot_game_forfeit_on_its_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(BOTS, "test_bot", _RANDOM_BOT)

    async def run() -> None:
        sample = ChessSample(fen=chess.STARTING_FEN, opponent="test_bot", seed=1)
        group = await _run_group({0: ["e4", "Nf3", "Ke9"]}, group_size=1, sample=sample)
        (rollout,) = group.rollouts
        # the bot answers 1. e4 Nc6 2. Nf3 h5, and White forfeits on ply 4 at even material
        assert "Black played h5." in rollout.turns[2].prompt_messages[-1]["content"]
        assert rollout.reward == pytest.approx(-0.95)
        # alone in its group, it centers on its 0.5 at the cap; the forfeiting turn pays -0.95 - 0.5
        assert rollout.advantage == pytest.approx(0.0)
        assert [turn.advantage for turn in rollout.turns] == [
            None,
            None,
            pytest.approx(-1.45),
        ]

    asyncio.run(run())


@pytest.mark.parametrize(
    "step,expected",
    [(None, 40), (10, 50), (50, 50), (100, 100), (149, 149), (400, 150)],
)
def test_max_plies_schedule_follows_the_train_step(monkeypatch, step, expected) -> None:
    worker = ChessSelfPlayWorker.Config(
        rubric=Rubric.Config(reward_fns=[RewardChessScore.Config()]),
        message_env=ChessPlayerEnv.Config(),
        token_env=TokenEnv.Config(step_timeout_s=None),
        max_plies=40,
        max_plies_schedule=((50, 50), (150, 150)),
    ).build()
    monkeypatch.setattr(sl, "get_step", lambda: step)
    assert worker._scheduled_max_plies() == expected
