# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5-4B chess recipes: self-play mixed with games against Stockfish bots."""

from __future__ import annotations

import os

from renderers import Qwen35RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LMHeadFP32OutputConverter
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.components.data import IterableRLDataLoader
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.distributed.routing import (
    InterGeneratorRouter,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.examples.chess_selfplay import (
    ChessPlayerEnv,
    ChessSelfPlayDataset,
    ChessSelfPlayWorker,
    ChessVsBotDataset,
    RewardChessScore,
)
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import DAPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.environment import TokenEnv
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan.rl.rollout.thinking_budget import ThinkingBudget
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.trainer import Trainer

# TODO: Enable CUDA graphs for RL trainers after eager/graph numerics parity is
# verified.


def rl_chess_qwen3_5_4b(
    max_plies: int = 60,
    max_rollout_tokens: int = 28672,
    max_response_tokens: int = 1024,
) -> Controller.Config:
    """`max_plies`-ply games on one node: an FSDP=4 trainer and four TP=1 generators.

    A player's history grows ~600 tokens per turn (a ~500-token turn message plus its reply), so
    `max_rollout_tokens` must cover `max_plies / 2` turns: 28k for 60 plies, 48k for 120.

    Each step trains on 16 start positions x 8 games. Half the groups are self-play (up to 2 rollouts per
    game, one per color); the other half play a Stockfish bot drawn from the ladder in `bots.BOTS`
    (1 rollout per game). Validation plays 64 greedy-decoded games against the same ladder before
    and after training.
    """
    ladder = ("sf_eps90", "sf_eps75", "sf_eps50", "sf_eps25", "sf_elo1320")
    max_total_tokens = max_rollout_tokens + max_response_tokens
    num_validation_games = 64
    model_config = build_model_config(
        "4B",
        seq_len=max_total_tokens,
        attn_backend="varlen",
        # Compute vocabulary logits in fp32; the rest of the forward uses bf16.
        converters=[LMHeadFP32OutputConverter.Config()],
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.5-4B",
        dump_folder="outputs/rl/qwen3_5_4b_chess",
        async_loop=AsyncLoopConfig(
            num_training_steps=150,
            num_prompts_per_train_step=16,
            # Games per start position. Only one player of a game generates at a time, so this
            # is also the group's generation concurrency, which sizes the generators' max_num_seqs.
            num_samples_per_prompt=8,
            # A 60-ply game spans several policy versions, so the observed policy age runs above this target.
            target_offpolicy_steps=4,
            validation=ValidationConfig(steps=num_validation_games),
        ),
        rollouter=Rollouter.Config(
            training_dataloader=IterableRLDataLoader.Config(
                dataset=ChessSelfPlayDataset.Config(bots=ladder, bot_fraction=0.5)
            ),
            validation_dataset=ChessVsBotDataset.Config(
                num_games=num_validation_games, opponents=ladder
            ),
            # Threads run renderer calls, bot moves, and Stockfish move scoring.
            num_threads_per_worker=16,
            worker=ChessSelfPlayWorker.Config(
                rubric=Rubric.Config(
                    reward_fns=[RewardChessScore.Config()],
                    # No truncation_reward / error_reward: a player that stops mid-game forfeits, and
                    # the worker scores the forfeit by the moves it lasted (see `ChessGame.rewards`).
                ),
                message_env=ChessPlayerEnv.Config(),
                token_env=TokenEnv.Config(
                    max_rollout_tokens=max_rollout_tokens,
                    # A player's step waits for the other player's move. A timeout there cannot end
                    # the group, which still waits for the slow player, and would forfeit the waiting one.
                    step_timeout_s=None,
                ),
                advantage=AdvantageEstimator.Config(should_std_normalize=False),
                max_plies=max_plies,
                stockfish_path="stockfish",
            ),
        ),
        # Thinking off: with thinking on, Qwen3.5-4B thinks past 4,096 tokens on every chess move.
        # Its reasoning goes in the reply instead, and each turn bridges onto the previous tokens.
        renderer=from_renderers(Qwen35RendererConfig(enable_thinking=False)),
        num_generators=4,
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            console_log_keys_train=MetricsProcessor.Config().console_log_keys_train
            + [
                "chess_strength/elo",
                "chess_strength/acpl_self_play",
                "chess_games/plies_self_play",
                "chess_games/forfeits_per_reply_self_play",
            ],
            console_log_keys_validation=[
                "validation_reward/_mean",
                "val_chess_strength/elo",
                "val_chess_strength/score_vs_.*",
                "timing/validate",
            ],
        ),
        trainer=Trainer.Config(
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*",
                            lr=1e-6,
                            betas=(0.9, 0.98),
                            weight_decay=0.1,
                        )
                    ]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0,
                    min_lr_factor=1.0,
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_total_tokens,
                max_context_length=max_total_tokens,
            ),
            # FSDP only: a 4B model does not need tensor parallelism.
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=4,
                tensor_parallel_degree=1,
            ),
            # With full activation checkpointing the trainer reserved 33 of 95 GB (FSDP=2 x TP=2).
            activation_checkpoint=SelectiveAC.Config(),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                # Qwen3.5's 248k vocabulary makes fp32 logits large; 16 chunks keep the trainer under 95 GB.
                num_chunks=16,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
            ),
        ),
    )


# TODO: add the Qwen3.5-35B-A3B variant once the trainer alone can use Dist-MoE experts.
def rl_chess_qwen3_5_4b_gb300(
    max_plies: int = 120,
    max_thinking_tokens: int = 1024,
    opening_max_thinking_tokens: int = 2048,
    max_context_tokens: int = 131072,
) -> Controller.Config:
    """Qwen3.5-4B (instruct) with thinking on, 150 steps on three GB300 hosts, 192 positions x 8 games
    per step. Host 0 trains with FSDP 4; hosts 1-2 run eight one-GPU generators.

    A turn thinks up to `max_thinking_tokens` (a player's first 5 turns up to
    `opening_max_thinking_tokens`); then `ThinkingBudget` closes the thinking and starts
    the answer with "\\boxed{", and the reward loses up to 0.1 for force-closed turns. A player keeps
    its own past thinking in its history (never the opponent's), so the history grows up to
    ~`max_thinking_tokens` per turn: 120 plies x 1k thinking need ~100k of `max_context_tokens`
    (a ~600-token turn message plus a ~1,050-token reply per turn). 120 plies covers over 90% of
    Lichess games below 2000 Elo.
    """
    # Room for the answer after the thinking ends, forced or not. The opening budget sets every turn's
    # cap: by default 2,560 instead of 1,536, and `max_rollout_tokens` drops to 128,512.
    max_response_tokens = max(max_thinking_tokens, opening_max_thinking_tokens) + 512
    # `max_context_tokens` must be a multiple of 1,024 so ChunkedLossWrapper splits each TP rank's
    # sequence into equal chunks.
    max_rollout_tokens = max_context_tokens - max_response_tokens
    config = rl_chess_qwen3_5_4b(
        max_plies=max_plies,
        max_rollout_tokens=max_rollout_tokens,
        max_response_tokens=max_response_tokens,
    )
    config.renderer = from_renderers(
        Qwen35RendererConfig(enable_thinking=True, thinking_retention="all")
    )
    # Bot groups start against a random mover and move ~200 Elo up once the policy wins over 40%:
    # only checkmates count, and most games the policy leads still end at the ply cap.
    config.rollouter.training_dataloader.dataset.bots = ("curriculum",)
    worker = config.rollouter.worker
    worker.curriculum_win_rate = 0.4
    worker.bot_curriculum = (
        "sf_random",
        "sf_eps75",
        "sf_eps50",
        "sf_eps25",
        "sf_elo1320",
        "sf_elo1500",
        "sf_elo1700",
        "sf_elo1900",
        "sf_elo2100",
        "sf_elo2300",
        "sf_elo2500",
    )
    # The answer ends at the box's closing brace, so a forced move is never lost to the token cap.
    # A player's first 5 turns think longer: at 1,024 tokens, 76-100% of them were force-closed.
    worker.thinking_budget = ThinkingBudget.Config(
        max_thinking_tokens=max_thinking_tokens,
        opening_max_thinking_tokens=opening_max_thinking_tokens,
        opening_turns=5,
        answer_prefix="\\boxed{",
        answer_end_text="}",
    )
    worker.rubric.reward_fns = [RewardChessScore.Config(forced_close_penalty=0.1)]
    config.dump_folder = "outputs/rl/qwen3_5_4b_chess_gb300"
    # The bot games and move scoring track progress, so validation is off.
    config.async_loop.validation.steps = 0
    # The launcher sets the path of a Stockfish build for the host's architecture.
    worker.stockfish_path = os.environ.get("CHESS_STOCKFISH_PATH", "stockfish")

    config.num_generators = 8
    config.async_loop.num_prompts_per_train_step = 192
    config.async_loop.target_offpolicy_steps = 5
    config.generator.gpu_memory_limit = 0.9
    config.generator.max_num_batched_tokens = 8192
    # Up to 18,432 players are live (1,152 groups x 16), over the default 4,096 sessions: pin them all.
    config.generator_router = InterGeneratorRouter.Config(
        strategy=StickySessionRoutingStrategy.Config(max_sessions=65536)
    )
    # TODO: admit groups by generator KV room (router admission); 1,152 in flight outgrow the KV cache.
    # TODO: hold each player's KV between its turns (session KV holding, 5% free-block floor).
    # TODO: pass vLLM's watermark (3%) through, so running requests keep room to grow.
    return config
