# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 chess recipes: 4B on one node, self-play mixed with games against Stockfish bots; 4B and
35B-A3B-Base on three GB300 hosts, thinking on, self-play only."""

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
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.components.data import IterableRLDataLoader
from torchtitan.rl.components.work_buffer import AdaptiveRolloutGroupWorkBuffer
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.examples.chess_selfplay import (
    ChessCurriculum,
    ChessPlayerEnv,
    ChessSelfPlayDataset,
    ChessSelfPlayWorker,
    ChessVsBotDataset,
    RewardChessScore,
)
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import DAPOLoss, ScoreCenteringLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.rollout.thinking_budget import ThinkingBudget
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.trainer import Trainer

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

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
        seq_len=max_total_tokens,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3.5-4B",
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
                num_tokens_per_microbatch_per_dp_rank=max_total_tokens,
                max_context_length=max_total_tokens,
            ),
            parallelism=trainer_parallelism,
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
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
            ),
        ),
    )


def rl_chess_qwen3_5_4b_gb300(
    max_plies: int = 120,
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
        # A reply that stops inside thinking plays the box it wrote. Force-closed under an advantage cap
        # at 0, such turns grew to 13.5% of turns (chess118gcq, steps 289-302).
        force_close_on_stop=False,
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
    # the attention and FFN outputs instead of recomputing everything. SelectiveAC saves every
    # region, including GDN's, which OOMs at 131K tokens.
    trainer.activation_checkpoint = RegionAC.Config(
        save_regions=["attn.inner_attention", "feed_forward.w2.linear"]
    )
    # Score centering instead of DAPO's clipped objective: REINFORCE weighted by min(p / q, 2), with
    # the generator's top-16 logprobs as each token's baseline.
    trainer.loss.loss_fn = ScoreCenteringLoss.Config(
        max_ratio=2.0, global_vocab_size=decoder_vocab_size(trainer.model)
    )
    # The launcher uploads every saved step and prunes, so torchtitan keeps them all.
    trainer.checkpointer = CheckpointManager.Config(
        initial_load_in_hf=True,
        interval=10,
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
    return config


def rl_chess_qwen3_5_35b_a3b_base_gb300(
    max_plies: int = 120,
    max_thinking_tokens: int = 1024,
    max_context_tokens: int = 131072,
    num_prompts_per_train_step: int = 24,
) -> Controller.Config:
    """Qwen3.5-35B-A3B-Base self-play from scratch on five GB300 hosts; games, thinking and validation
    as in `rl_chess_qwen3_5_4b_gb300`, with a zero-sum mate (10 / -9):

        host 0:     the trainer, FSDP 2 x TP 2 x EP 4 with Dist-MoE experts
        hosts 1-4:  sixteen TP1 generators; idle session KV goes to CPU

    Training games' ply cap grows from 60 at step 0 to `max_plies` at step 200; validation games
    always play `max_plies`.
    """
    expert_parallel_degree = 4
    config = rl_chess_qwen3_5_4b_gb300(
        max_plies=max_plies,
        max_thinking_tokens=max_thinking_tokens,
        max_context_tokens=max_context_tokens,
    )
    config.hf_assets_path = "torchtitan/rl/example_checkpoint/Qwen3.5-35B-A3B-Base"
    config.dump_folder = "outputs/rl/qwen3_5_35b_a3b_base_chess_gb300"
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=expert_parallel_degree,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "35B-A3B",
        seq_len=max_context_tokens,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=config.generator.parallelism,
    )

    # ======== Rollouts ========
    config.rollouter.curriculum = ChessCurriculum.Config(
        start_max_plies=60, end_max_plies=max_plies, max_plies_ramp_steps=200
    )
    # Zero-sum mate: with the mated side losing only ~0.25, self-play stopped defending (82% of mates
    # followed a one-move blunder another move avoided).
    worker = config.rollouter.worker
    worker.checkmated_reward = 1.0 - worker.checkmate_reward

    # ======== Async loop ========
    loop = config.async_loop
    loop.num_training_steps = 1000
    # 24 positions x 8 games x 2 colors = 384 rollouts per step. At 48 positions, host 0 ran out of RAM
    # at step 100 (chess118gk): the buffer holds up to (6 + 1) batches of games.
    loop.num_prompts_per_train_step = num_prompts_per_train_step
    loop.group_buffer = AdaptiveRolloutGroupWorkBuffer.Config(
        # The mean policy age of trained groups stays under 6 steps, every one under 10.
        target_offpolicy_steps=6,
        max_offpolicy_steps=10,
        # Generate up to (6 + 1) batches at once from the first step, the most a mean age of 6
        # allows; sizes vLLM's max_num_seqs.
        generation_capacity=(6 + 1) * num_prompts_per_train_step,
        start_batches=6 + 1,
    )
    loop.target_offpolicy_steps = loop.group_buffer.target_offpolicy_steps
    loop.validation.freq = 5

    # ======== Trainer ========
    trainer = config.trainer
    trainer.model = trainer_model
    trainer.parallelism = trainer_parallelism
    # Recompute every op in the block except the Dist-MoE call, which is never recomputed.
    trainer.activation_checkpoint = RegionAC.Config(save_regions=[])
    # Worst case, every EP rank routes all its tokens to one rank; a smaller scratch
    # buffer is an illegal memory access.
    trainer.dist_moe_runtime = DistMoeRuntime.Config(
        scratch_capacity_factor=float(expert_parallel_degree)
    )
    # Constant 3e-6, no warmup: at 5e-6 self-play move quality slid from step ~23.
    trainer.optim.optimizer.optimizers[0].lr = 3e-6
    trainer.optim.lr_scheduler.warmup_steps = 0
    # Synchronous saves: an async save's CPU copy of the 35B state (~420 GB) on top of the batch
    # can exceed the trainer host's job memory.
    trainer.checkpointer.async_mode = "disabled"
    # Dist-MoE experts on the trainer's model only; the generators keep stock experts. The
    # functional WGRAD path, as the GB300 Terminal-Bench 35B runs use.
    config.trainer = apply_transforms(
        trainer, [DistMoeTransform(inplace_wgrad_accum=False)]
    )

    # ======== Generators ========
    config.num_generators = 16
    generator = config.generator
    generator.model = generator_model
    # Router replay: the trainer routes each token to the experts the generator picked.
    generator.return_routed_experts = True
    # ScoreCenteringLoss's top-k: 32 matched the full vocabulary in arXiv 2609.20807; 16 halves the
    # host RAM the top-k rows take, after host 0 ran out of RAM (chess118gk).
    generator.sampling.num_topk_logprobs = 16
    # Admit a request only while 10% of KV blocks stay free. Below 15% free, release the sessions idle
    # longest; their KV stays in CPU memory.
    generator.extra_vllm_engine_args = {"watermark": 0.1}
    generator.session_kv_free_floor = 0.15
    # Host RAM per engine for vLLM's CPU KV offload: 4 engines x 96 GiB fit a GB300 host's /dev/shm
    # (half of ~954 GB) with margin.
    generator.kv_offload_gib = 96.0
    return config


def rl_chess_qwen3_5_35b_a3b_base_gb300_lr2e6() -> Controller.Config:
    """`rl_chess_qwen3_5_35b_a3b_base_gb300` at lr 2e-6, to resume a 5e-6 run. At 5e-6, entropy fell
    0.74 -> 0.46 and self-play ACPL rose 146 -> 185 over steps 20-29."""
    config = rl_chess_qwen3_5_35b_a3b_base_gb300()
    config.trainer.optim.optimizer.optimizers[0].lr = 2e-6
    return config


def rl_chess_qwen3_5_35b_a3b_base_gb300_lr1e6() -> Controller.Config:
    """`rl_chess_qwen3_5_35b_a3b_base_gb300` at lr 1e-6, the 4B's stable rate, to resume a faster run.
    At 2e-6, from step 20 self-play ACPL rose 162 -> 330 and validation Elo fell 787 -> 620 by step 38."""
    config = rl_chess_qwen3_5_35b_a3b_base_gb300()
    config.trainer.optim.optimizer.optimizers[0].lr = 1e-6
    return config
