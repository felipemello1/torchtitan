# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-node Qwen3.5-4B chess recipe: self-play mixed with games against Stockfish bots."""

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
from torchtitan.config import OverrideConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import RegionAC, SelectiveAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.distributed.routing import admission
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
            validation=ValidationConfig(num_samples=num_validation_games),
        ),
        rollouter=Rollouter.Config(
            train_dataset=ChessSelfPlayDataset.Config(bots=ladder, bot_fraction=0.5),
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


def rl_chess_qwen3_5_35b_a3b(
    max_plies: int = 120,
    max_thinking_tokens: int = 1024,
    max_context_tokens: int = 131072,
) -> Controller.Config:
    """Qwen3.5-35B-A3B (instruct) with thinking on, 150 steps on three GB300 hosts, 192 positions x 8
    games per step.

    A turn thinks up to `max_thinking_tokens`; then `ThinkingBudget` closes the thinking and starts
    the answer with "\\boxed{", and the reward loses up to 0.1 for force-closed turns. A player keeps
    its own past thinking in its history (never the opponent's), so the history grows up to
    ~`max_thinking_tokens` per turn: 120 plies x 1k thinking need ~100k of `max_context_tokens`
    (a ~600-token turn message plus a ~1,050-token reply per turn). 120 plies covers p90 of human games below 2000 Elo
    (research/game_length_by_elo.md in discussion 118).

    Host 0 trains: FSDP 2 x TP 2 x EP 4 with Dist-MoE experts, the layout of the 35B Terminal-Bench
    runs. Hosts 1-2 run eight one-GPU generators, each with every expert, FULL CUDA graphs: with
    thinking on, generation is the bottleneck.
    """
    # room for the answer after the thinking ends, forced or not
    max_response_tokens = max_thinking_tokens + 512
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
    # Bot groups start against a random mover and move ~200 Elo up once the policy wins over 60%.
    config.rollouter.train_dataset.bots = ("curriculum",)
    worker = config.rollouter.worker
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
    worker.thinking_budget = ThinkingBudget.Config(
        max_thinking_tokens=max_thinking_tokens,
        answer_prefix="\\boxed{",
        answer_end_text="}",
    )
    worker.rubric.reward_fns = [RewardChessScore.Config(forced_close_penalty=0.1)]
    config.model = build_model_config(
        "35B-A3B", seq_len=max_context_tokens, attn_backend="varlen"
    )
    config.hf_assets_path = "torchtitan/rl/example_checkpoint/Qwen3.5-35B-A3B"
    config.dump_folder = "outputs/rl/qwen3_5_35b_a3b_chess"
    # The bot games and move scoring track progress, so validation is off.
    config.async_loop.validation.num_samples = 0
    # The launcher sets the path of a Stockfish build for the host's architecture.
    config.rollouter.worker.stockfish_path = os.environ.get(
        "CHESS_STOCKFISH_PATH", "stockfish"
    )

    trainer = config.trainer
    expert_parallel_degree = 4
    trainer.parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=expert_parallel_degree,
    )
    # fp32 master weights; FSDP unshards bf16 params for compute.
    trainer.training.dtype = "float32"
    trainer.training.mixed_precision_param = "bfloat16"
    # Recompute every op in the block except the Dist-MoE call, which is never recomputed.
    trainer.activation_checkpoint = RegionAC.Config(save_regions=[])
    # Dist-MoE experts on the trainer's model copy only; generators keep stock experts.
    trainer.override = OverrideConfig(
        imports=["torchtitan_recipes.overrides.dist_moe.dist_moe_routed_experts"]
    )
    # Worst case, every EP rank routes all its tokens to one rank; a smaller scratch buffer is an
    # illegal memory access.
    trainer.dist_moe = DistMoeRuntime.Config(
        scratch_capacity_factor=float(expert_parallel_degree)
    )
    trainer.loss.loss_fn.global_vocab_size = decoder_vocab_size(config.model)
    # DOME uploads every saved step and prunes, so torchtitan keeps them all.
    trainer.checkpointer = CheckpointManager.Config(
        initial_load_in_hf=True,
        interval=25,
        enable_first_step_checkpoint=True,
        async_mode="async",
        keep_latest_k=0,
        last_save_in_hf=False,
        last_save_model_only=False,
    )

    config.num_generators = 8
    config.async_loop.num_prompts_per_train_step = 192
    config.async_loop.target_offpolicy_steps = 5
    config.generator.gpu_memory_limit = 0.9
    config.generator.max_num_batched_tokens = 8192
    # 1,152 groups in flight outgrow the generators' KV cache, so a waiting player's history was
    # evicted before its next turn. Start new games only while the live ones, each grown to its
    # expected final size, still fit. The other modes, as a one-line switch:
    #   admission.KVEstimateAdmission.Config(limit=0.9, sessions_per_group=16)  # current size only
    #   admission.KVUsageAdmission.Config(initial_inflight=512)  # vLLM's measured KV usage
    # 1.8, not the default 1.6: a bot group opens 8 of the 12 seats a new group reserves.
    config.generator_router.admission = admission.KVGrowthEstimateAdmission.Config(
        limit=1.8
    )
    # Hold each player's prefix (attention + GDN state) between its turns; below 5% free blocks,
    # release the sessions idle longest. The watermark keeps 3% free for running requests to grow.
    config.generator.hold_session_kv = True
    config.generator.session_kv_free_floor = 0.05
    config.generator.watermark = 0.03
    return config


def rl_chess_qwen3_5_4b_gb300(
    max_plies: int = 50,
    max_thinking_tokens: int = 1024,
    max_context_tokens: int = 131072,
) -> Controller.Config:
    """The 35B GB300 recipe with Qwen3.5-4B (instruct): same thinking budget, context, and eight
    one-GPU generators on hosts 1-2; host 0 trains the dense 4B with FSDP 4. A step trains 96
    positions x 8 games against the curriculum bot (no self-play), and the ply cap grows from 50 to 150
    over steps 70-150.

    The 4B's fp32 weights, grads and Adam state take ~16 GB per trainer GPU (the 35B's ~140 GB), so
    the trainer spends the memory on selective activation checkpointing instead of full recompute.
    """
    config = rl_chess_qwen3_5_35b_a3b(
        max_plies=max_plies,
        max_thinking_tokens=max_thinking_tokens,
        max_context_tokens=max_context_tokens,
    )
    config.model = build_model_config(
        "4B", seq_len=max_context_tokens, attn_backend="varlen"
    )
    config.hf_assets_path = "torchtitan/rl/example_checkpoint/Qwen3.5-4B"
    config.async_loop.num_prompts_per_train_step = 96
    # The ply cap grows from 50 at step 70 (the resume point) to 150 at step 150.
    config.rollouter.worker.max_plies_schedule = ((70, 50), (150, 150))
    # Bot games only: late self-play games were both sides walking their kings to the ply cap.
    config.rollouter.train_dataset.bot_fraction = 1.0
    # Start at sf_eps75, which the resumed policy already plays: the level isn't checkpointed.
    config.rollouter.worker.bot_curriculum = config.rollouter.worker.bot_curriculum[1:]
    config.dump_folder = "outputs/rl/qwen3_5_4b_chess_gb300"
    trainer = config.trainer
    trainer.parallelism = ParallelismConfig(data_parallel_shard_degree=4)
    trainer.activation_checkpoint = SelectiveAC.Config()
    trainer.override = OverrideConfig()
    trainer.dist_moe = None
    trainer.loss.loss_fn.global_vocab_size = decoder_vocab_size(config.model)
    return config
