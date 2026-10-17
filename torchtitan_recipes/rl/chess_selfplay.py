# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 chess self-play recipe: 4B-Base from scratch on one node, thinking on, validated against a
ladder of Stockfish bots."""

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
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.examples.chess_selfplay import (
    ChessCurriculum,
    ChessPlayerEnv,
    ChessSelfPlayDataset,
    ChessSelfPlayWorker,
    ChessVsBotDataset,
    RewardChessScore,
)
from torchtitan.rl.generator.generator import (
    SamplingConfig,
    VLLMCudaGraphConfig,
    VLLMGenerator,
)
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.vllm.session_kv_holder import VLLMSessionKVHolder
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.data import IterableRLDataLoader
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.rollout.thinking_budget import ThinkingBudget
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.trainer.losses import DAPOLoss
from torchtitan.rl.trainer.trainer import Trainer

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

# TODO: Enable CUDA graphs for RL trainers after eager/graph numerics parity is
# verified.


def rl_chess_qwen3_5_4b_base(
    *, max_plies: int = 90, max_context_tokens: int = 131072
) -> Controller.Config:
    """Qwen3.5-4B-Base chess self-play from scratch on one node:

        GPUs 0-3: the trainer, FSDP 4
        GPUs 4-7: four TP1 generators

    Each move thinks up to 2,048 tokens, then gets a forced close and a boxed move. Training games'
    ply cap grows from 60 at step 0 to `max_plies` at step 200. Validation plays 8 games against
    each of 7 Stockfish bots before and after training.

    Never run in this form: sized for GPUs with ~180 GB each. A 90-ply game with every reply at the
    thinking cap holds ~111K tokens. The trainer packs `max_context_tokens` per microbatch, so on
    80 GB GPUs lower `max_context_tokens`, and `max_plies` with it.
    """
    # TODO: a multi-host version once score centering, the adaptive rollout buffer and CPU KV
    # offload land; the runs behind the example's README used them.
    max_thinking_tokens = 2048
    # After the thinking: the forced close and "\boxed{" (~28 tokens) then the move, or a self-closed
    # answer's box and a short line.
    max_answer_tokens = 128
    max_response_tokens = max_thinking_tokens + max_answer_tokens
    num_prompts_per_train_step = 12
    target_offpolicy_steps = 4
    # 8 fixed games (4 per color) against each of 7 bots.
    ladder = (
        "sf_random",
        "sf_eps90",
        "sf_eps75",
        "sf_eps50",
        "sf_eps25",
        "sf_elo1320",
        "sf_elo1500",
    )
    num_validation_games = 8 * len(ladder)
    # FSDP only: a 4B model does not need tensor parallelism.
    trainer_parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=4,
        tensor_parallel_degree=1,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=1,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "4B",
        seq_len=max_context_tokens,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    thinking_budget = ThinkingBudget.Config(
        max_thinking_tokens=max_thinking_tokens,
        answer_prefix="\\boxed{",
        # The answer ends at the box's closing brace, so a forced move is never lost to the token cap.
        answer_end_text="}",
        # A reply that stops inside thinking plays the box it wrote.
        force_close_on_stop=False,
        # After the model's own </think>, the answer is the box alone: without it, answers repeated
        # "</think> \boxed{X}" to the token cap and forfeited up to 15% of games.
        answer_after_close=True,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3.5-4B-Base",
        dump_folder="outputs/rl/qwen3_5_4b_base_chess",
        async_loop=AsyncLoopConfig(
            # The ply cap reaches `max_plies` at step 200.
            num_training_steps=200,
            # 12 positions x 8 games x 2 colors = 192 rollouts per step.
            num_prompts_per_train_step=num_prompts_per_train_step,
            # Games per start position. Only one player of a game generates at a time, so this
            # is also the group's generation concurrency, which sizes the generators' max_num_seqs.
            num_samples_per_prompt=8,
            # A 90-ply game spans several policy versions, so the observed policy age runs above this target.
            target_offpolicy_steps=target_offpolicy_steps,
            # Caps every trained group's policy age at target_offpolicy_steps + 6 = 10 steps; the
            # trainer waits for a straggler instead of dropping it.
            windowed_fifo_batches=6,
            validation=ValidationConfig(steps=num_validation_games),
        ),
        rollouter=Rollouter.Config(
            # Self-play only: the ladder validation measures strength.
            training_dataloader=IterableRLDataLoader.Config(
                dataset=ChessSelfPlayDataset.Config()
            ),
            validation_dataset=ChessVsBotDataset.Config(
                num_games=num_validation_games, opponents=ladder
            ),
            # Training games' ply cap: 60 at step 0, growing to `max_plies` at step 200.
            curriculum=ChessCurriculum.Config(
                start_max_plies=60,
                end_max_plies=max_plies,
                max_plies_ramp_steps=200,
            ),
            # Threads run renderer calls, bot moves, and Stockfish move scoring.
            num_threads_per_worker=16,
            worker=ChessSelfPlayWorker.Config(
                rubric=Rubric.Config(
                    reward_fns=[RewardChessScore.Config()],
                    # No truncation_reward / error_reward: a player that stops mid-game forfeits, and
                    # the worker scores the forfeit by the moves it lasted (see `ChessGame.rewards`).
                ),
                # After a forced close, an illegal answer plays the thinking's last box.
                message_env=ChessPlayerEnv.Config(
                    forced_close_text=thinking_budget.close_text
                ),
                token_env=TokenEnv.Config(
                    max_rollout_tokens=max_context_tokens - max_response_tokens,
                    # A player's step waits for the other player's move. A timeout there cannot end
                    # the group, which still waits for the slow player, and would forfeit the waiting one.
                    step_timeout_s=None,
                ),
                advantage=AdvantageEstimator.Config(should_std_normalize=False),
                thinking_budget=thinking_budget,
                max_plies=max_plies,
                # A mate pays 10 and costs the mated side 9: zero-sum around a draw's ~0.5.
                checkmate_reward=10.0,
                checkmated_reward=-9.0,
            ),
        ),
        # A player keeps its own past thinking, never its opponent's.
        renderer=from_renderers(
            Qwen35RendererConfig(enable_thinking=True, thinking_retention="all")
        ),
        num_generators=4,
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_chess_strength/elo",
                "validation_chess_strength/score_vs_.*",
                "timing/validate",
            ],
        ),
        trainer=Trainer.Config(
            model=trainer_model,
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
                num_tokens_per_microbatch_per_dp_rank=max_context_tokens,
                max_context_length=max_context_tokens,
            ),
            parallelism=trainer_parallelism,
            activation_remat=FullActivationRematConfig(),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                # Qwen3.5's 248k vocabulary makes fp32 logits large; 16 chunks keep them small.
                # `max_context_tokens` must be divisible by them.
                num_chunks=16,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(trainer_model),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            checkpointer=None,
            # Hold each player's prefix (attention + GDN state) between its turns.
            session_kv_holder=VLLMSessionKVHolder.Config(),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
            ),
        ),
    )
