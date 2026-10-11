# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5-4B chess recipes: one node with self-play mixed with games against Stockfish bots, and
three GB300 hosts with thinking on and self-play only."""

from __future__ import annotations

import os
import shutil
import tempfile

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
from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.components.data import IterableRLDataLoader
from torchtitan.rl.components.work_buffer import AdaptiveRolloutGroupWorkBuffer
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.distributed.routing.admission import KVGrowthEstimateAdmission
from torchtitan.rl.examples.chess_selfplay import (
    ChessPlayerEnv,
    ChessRollouter,
    ChessSelfPlayDataset,
    ChessSelfPlayWorker,
    ChessVsBotDataset,
    RewardChessScore,
)
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import DAPOLoss, ScoreCenteringLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.environment import TokenEnv
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
        rollouter=ChessRollouter.Config(
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


def rl_chess_qwen3_5_4b_gb300(
    max_plies: int = 60,
    max_thinking_tokens: int = 1024,
    max_context_tokens: int = 131072,
) -> Controller.Config:
    """Qwen3.5-4B (instruct) self-play with thinking on, on three GB300 hosts:

        host 0:     the trainer, FSDP 4
        hosts 1-2:  eight one-GPU generators

    A player keeps its own past thinking, so its history grows ~1.6k tokens a turn.
    """
    # room for the answer after the thinking ends, forced or not
    max_response_tokens = max_thinking_tokens + 512
    # ChunkedLossWrapper needs `max_context_tokens` divisible by its 16 chunks.
    config = rl_chess_qwen3_5_4b(
        max_plies=max_plies,
        max_rollout_tokens=max_context_tokens - max_response_tokens,
        max_response_tokens=max_response_tokens,
    )
    config.dump_folder = "outputs/rl/qwen3_5_4b_chess_gb300"
    config.renderer = from_renderers(
        Qwen35RendererConfig(enable_thinking=True, thinking_retention="all")
    )

    # ======== Rollouts ========
    rollouter = config.rollouter
    # Self-play only: the ladder validation below measures strength.
    rollouter.training_dataloader.dataset.bots = ()
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
    rollouter.validation_dataset = ChessVsBotDataset.Config(
        num_games=num_validation_games, opponents=ladder
    )
    worker = rollouter.worker
    # The launcher points this at a Stockfish build for the host's architecture. DOME stages it
    # read-only without the execute bit, so run a copy (the rollout workers share this host).
    stockfish = os.environ.get("CHESS_STOCKFISH_PATH", "stockfish")
    if os.path.isfile(stockfish) and not os.access(stockfish, os.X_OK):
        executable = os.path.join(tempfile.mkdtemp(prefix="stockfish_"), "stockfish")
        shutil.copy(stockfish, executable)
        os.chmod(executable, 0o755)
        stockfish = executable
    worker.stockfish_path = stockfish
    # A mate pays 10; a game cut at the ply cap pays at most 0.75, so a mate dominates its group's
    # advantages.
    worker.checkmate_reward = 10.0
    # The answer ends at the box's closing brace, so a forced move is never lost to the token cap.
    worker.thinking_budget = ThinkingBudget.Config(
        max_thinking_tokens=max_thinking_tokens,
        answer_prefix="\\boxed{",
        answer_end_text="}",
    )
    worker.rubric.reward_fns = [RewardChessScore.Config(forced_close_penalty=0.1)]

    # ======== Async loop ========
    loop = config.async_loop
    # 48 positions x 8 games x 2 colors = 768 rollouts per step.
    loop.num_prompts_per_train_step = 48
    # Generation paces itself: the mean policy age stays under 6 steps, every trained group under
    # 12. At most 6 batches generate at once, which bounds the rollout workers, vLLM's
    # max_num_seqs, and the Stockfish processes on host 0.
    loop.group_buffer = AdaptiveRolloutGroupWorkBuffer.Config(
        max_offpolicy_steps=12,
        target_offpolicy_steps=6,
        generation_capacity=6 * loop.num_prompts_per_train_step,
    )
    # Under the adaptive buffer this only sets train_batch/pct_samples_over_target_age.
    loop.target_offpolicy_steps = loop.group_buffer.target_offpolicy_steps
    # Validate at step 0, every 5 steps and after the last, without pausing training.
    loop.validation = ValidationConfig(
        steps=num_validation_games, freq=5, overlap_training=True
    )

    # ======== Trainer ========
    trainer = config.trainer
    # The 4B's fp32 weights, grads and Adam state take ~16 GB of a GB300's memory, so the trainer saves
    # the attention and FFN outputs instead of recomputing everything. Main's SelectiveAC saves every
    # region, including GDN's, which OOMs at 131K tokens.
    trainer.activation_checkpoint = RegionAC.Config(
        save_regions=["attn.inner_attention", "feed_forward.w2.linear"]
    )
    # Score centering instead of DAPO's clipped objective: REINFORCE weighted by min(p / q, 2), with
    # the generator's top-16 logprobs as each token's baseline.
    trainer.loss.loss_fn = ScoreCenteringLoss.Config(
        max_ratio=2.0, global_vocab_size=decoder_vocab_size(config.model)
    )
    # The launcher uploads every saved step and prunes, so torchtitan keeps them all.
    trainer.checkpointer = CheckpointManager.Config(
        initial_load_in_hf=True,
        interval=25,
        enable_first_step_checkpoint=True,
        async_mode="async",
        keep_latest_k=0,
        last_save_in_hf=False,
        last_save_model_only=False,
    )

    # ======== Generators ========
    config.num_generators = 8
    generator = config.generator
    generator.max_num_batched_tokens = 8192
    generator.sampling.num_topk_logprobs = 16
    # Hold each player's prefix (attention + GDN state) between its turns; below 5% free blocks
    # (`session_kv_free_floor`), release the sessions idle longest.
    generator.hold_session_kv = True
    # Admit a request only while 3% of KV blocks stay free, so running requests have room to grow.
    generator.extra_vllm_engine_args = {"watermark": 0.03}
    # Start a new group only while every placed group, at its expected final size, fits the
    # generators' KV cache; `ChessRollouter` books two sessions per self-play game.
    config.generator_router.admission = KVGrowthEstimateAdmission.Config()
    return config


def rl_chess_qwen3_5_4b_gb300_resume_old_stack() -> Controller.Config:
    """`rl_chess_qwen3_5_4b_gb300` for one resume from a checkpoint saved before the new stack, which
    kept the data stream in a separate file: restore the model, optimizer and train state, and start
    the controller's data stream fresh. Later resumes use `rl_chess_qwen3_5_4b_gb300`."""
    config = rl_chess_qwen3_5_4b_gb300()
    config.trainer.checkpointer.exclude_from_loading = ["controller"]
    return config
