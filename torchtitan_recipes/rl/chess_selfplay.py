# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-node Qwen3.5-4B chess recipe: self-play mixed with games against Stockfish bots."""

from __future__ import annotations

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
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.components.data import IterableRLDataLoader
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
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

    Each turn adds a ~530-token turn message plus the reply to a player's history. A player plays
    `max_plies / 2` turns, so 28,672 `max_rollout_tokens` fit 30 turns with replies averaging up to
    ~400 tokens.

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
                dataset=ChessSelfPlayDataset.Config(bots=ladder)
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
            # Selective AC filled 73 of 79 GiB per H100 here, with allocator retries, and was no faster.
            activation_checkpoint=FullAC.Config(),
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
