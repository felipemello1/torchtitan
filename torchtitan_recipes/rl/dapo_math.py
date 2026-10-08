# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO-Math recipes: verified single-node Qwen3-4B-Base, and multi-host Qwen3.5-Base."""

from __future__ import annotations

import os

from renderers import Qwen35RendererConfig, Qwen3RendererConfig

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
from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan.models.qwen3 import build_model_config
from torchtitan.models.qwen3_5 import build_model_config as build_qwen3_5_model_config
from torchtitan.rl.components.batcher import Batcher
from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.examples.dapo_math.data import (
    AIME2025Dataset,
    DapoMathDataset,
    Intellect3MathDataset,
)
from torchtitan.rl.examples.dapo_math.env import DapoMathEnv
from torchtitan.rl.examples.dapo_math.rubric import RewardMathVerify
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import DAPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.environment import TokenEnv
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.thinking_budget import ThinkingBudget
from torchtitan.rl.rubric import CorrectLengthPenalty, Rubric
from torchtitan.rl.trainer import Trainer

# TODO: Enable CUDA graphs for RL trainers after eager/graph numerics parity is
# verified.


def _dapo_math_rollouter_config(
    *,
    validation_dataset: AIME2025Dataset.Config,
    token_env: TokenEnv.Config,
) -> Rollouter.Config:
    return Rollouter.Config(
        train_dataset=DapoMathDataset.Config(),
        validation_dataset=validation_dataset,
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(
                reward_fns=[RewardMathVerify.Config(weight=1.0)],
                error_reward=0.0,
            ),
            message_env=DapoMathEnv.Config(),
            token_env=token_env,
            advantage=AdvantageEstimator.Config(should_std_normalize=False),
        ),
    )


def _qwen3_4b_dapo_math_config(
    *,
    max_response_tokens: int,
    max_total_tokens: int,
    dump_folder: str,
) -> Controller.Config:
    """Build the shared Qwen3-4B DAPO-Math configuration."""
    num_validation_samples = 30
    validation_dataset = AIME2025Dataset.Config(
        num_samples=num_validation_samples,
    )
    model_config = build_model_config(
        "4B",
        seq_len=max_total_tokens,
        attn_backend="varlen",
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-4B-Base",
        dump_folder=dump_folder,
        async_loop=AsyncLoopConfig(
            num_training_steps=150,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=16,
            target_offpolicy_steps=4,
            validation=ValidationConfig(
                num_samples=num_validation_samples,
            ),
        ),
        rollouter=_dapo_math_rollouter_config(
            validation_dataset=validation_dataset,
            token_env=TokenEnv.Config(
                max_rollout_tokens=max_total_tokens,
                max_num_turns=1,
            ),
        ),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=True)),
        num_generators=6,
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
                "validation/response_length/mean",
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
            # A minimum factor of 1 keeps the learning rate constant.
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_total_tokens,
                max_context_length=max_total_tokens,
            ),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=100,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
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


def rl_dapo_qwen3_4b_math_8k() -> Controller.Config:
    """Run 8K responses on one node: one TP=2 trainer and six TP=1 generators."""
    return _qwen3_4b_dapo_math_config(
        max_response_tokens=8192,
        max_total_tokens=10240,
        dump_folder="outputs/rl/qwen3_4b_dapo_math_8k",
    )


def rl_dapo_qwen3_4b_math_32k() -> Controller.Config:
    """Run 32K responses on one node: one TP=2 trainer and six TP=1 generators."""
    return _qwen3_4b_dapo_math_config(
        max_response_tokens=32768,
        max_total_tokens=34816,
        dump_folder="outputs/rl/qwen3_4b_dapo_math_32k",
    )


def rl_dapo_qwen3_5_4b_base_math() -> Controller.Config:
    """Qwen3.5-4B-Base, thinking off, 8K responses: a short run that checks checkpoint,
    resume and eval on the 2-host trainer layout.

    12 GPUs on 3 hosts: trainer FSDP 8 on two, four TP1 generators on the third. 4 prompts
    x 4 samples per step, AIME 2025 greedy every 3 steps. `DOME_V2_PROMPTS` and
    `DOME_V2_MICROBATCH_ROWS` (rows of 10,240 tokens, default 1) set the batch at launch,
    so a resumed job can change it without a new commit.
    """
    return _qwen3_5_base_dapo_math_config(
        flavor="4B",
        enable_thinking=False,
        max_response_tokens=8192,
        max_total_tokens=10240,
        num_prompts_per_train_step=int(os.environ.get("DOME_V2_PROMPTS", 4)),
        num_samples_per_prompt=4,
        microbatch_rows=int(os.environ.get("DOME_V2_MICROBATCH_ROWS", 1)),
        validation_interval_steps=3,
        num_generators=4,
        parallelism=ParallelismConfig(data_parallel_shard_degree=8),
        num_loss_chunks=8,
        dump_folder="outputs/rl/qwen3_5_4b_base_dapo_math",
    )


def rl_dapo_qwen3_5_35b_a3b_base_math() -> Controller.Config:
    """Qwen3.5-35B-A3B-Base, thinking on, 32K responses, 150 steps.

    16 GB300 GPUs on 4 hosts. Trainer on two: FSDP 4 x TP 2 x EP 4 with Dist-MoE experts.
    Generators: eight TP1 engines, each with every expert, FULL CUDA graphs. 16 prompts x
    16 samples per step, AIME 2025 greedy every 25 steps. `DOME_V2_PROMPTS`,
    `DOME_V2_MICROBATCH_ROWS` (rows of max_total_tokens, default 5) and
    `DOME_V2_MAX_RESPONSE_TOKENS` (default 32,768) set the batch and budget at launch, so a
    resumed job can change them without a new commit.
    """
    return _qwen3_5_35b_a3b_base_dapo_math_config(
        max_response_tokens=int(os.environ.get("DOME_V2_MAX_RESPONSE_TOKENS", 32768)),
        num_prompts_per_train_step=int(os.environ.get("DOME_V2_PROMPTS", 16)),
        microbatch_rows=int(os.environ.get("DOME_V2_MICROBATCH_ROWS", 5)),
        dump_folder="outputs/rl/qwen3_5_35b_a3b_base_dapo_math",
    )


def rl_dapo_qwen3_5_35b_a3b_base_intellect3_math() -> Controller.Config:
    """Qwen3.5-35B-A3B-Base on harder problems: the 10,805 INTELLECT-3 RL math problems that
    Qwen3-4B-Thinking-2507 solves in 1-7 of 8 tries, instead of DAPO-Math-17k.

    The `rl_dapo_qwen3_5_35b_a3b_base_math` layout with 32 prompts x 16 samples, 131K
    responses with a forced answer at the cap and a length penalty (`_apply_length_control`),
    and 1-row microbatches. Truncated rollouts score `DOME_V2_TRUNCATION_REWARD` (default 0).
    No online validation: the checkpoints are evaluated offline.
    """
    return _intellect3_math_config(default_prompts=32, default_microbatch_rows=1)


def rl_dapo_qwen3_5_35b_a3b_base_intellect3_math_12_generators() -> Controller.Config:
    """`rl_dapo_qwen3_5_35b_a3b_base_intellect3_math` with one trainer host and three
    generator hosts: more sequences in flight, which is what bounds the step.

    Trainer on 4 GPUs: FSDP 2 x TP 2 x EP 4, 1-row microbatches (fp32 state and Adam take
    ~125 GiB per GPU). Twelve TP1 engines; 64 prompts x 16 samples per step;
    target_offpolicy_steps 5.
    """
    return _intellect3_math_config(
        default_prompts=64,
        default_microbatch_rows=1,
        data_parallel_shard_degree=2,
        num_generators=12,
        target_offpolicy_steps=5,
    )


def rl_dapo_qwen3_5_35b_a3b_base_intellect3_math_8_generators() -> Controller.Config:
    """`rl_dapo_qwen3_5_35b_a3b_base_intellect3_math_12_generators` with two generator hosts
    instead of three: 1 trainer host (4 GPUs) + 8 TP1 engines. Set the batch with
    `DOME_V2_PROMPTS` at launch.
    """
    return _intellect3_math_config(
        default_prompts=64,
        default_microbatch_rows=1,
        data_parallel_shard_degree=2,
        num_generators=8,
        target_offpolicy_steps=5,
    )


def _intellect3_math_config(
    *, default_prompts: int, default_microbatch_rows: int, **layout
) -> Controller.Config:
    """Build the INTELLECT-3 run; `layout` goes to `_qwen3_5_35b_a3b_base_dapo_math_config`."""
    max_response_tokens = int(os.environ.get("DOME_V2_MAX_RESPONSE_TOKENS", 131072))
    config = _qwen3_5_35b_a3b_base_dapo_math_config(
        max_response_tokens=max_response_tokens,
        num_prompts_per_train_step=int(
            os.environ.get("DOME_V2_PROMPTS", default_prompts)
        ),
        microbatch_rows=int(
            os.environ.get("DOME_V2_MICROBATCH_ROWS", default_microbatch_rows)
        ),
        dump_folder="outputs/rl/qwen3_5_35b_a3b_base_intellect3_math",
        **layout,
    )
    config.rollouter.train_dataset = Intellect3MathDataset.Config()
    # A truncated rollout has no final answer; grading the last \boxed{} of its unfinished
    # reasoning would reward a guess.
    config.rollouter.worker.rubric.truncation_reward = float(
        os.environ.get("DOME_V2_TRUNCATION_REWARD", 0.0)
    )
    _apply_length_control(config, max_response_tokens=max_response_tokens)
    config.async_loop.validation = ValidationConfig(num_samples=0)
    return config


def _apply_length_control(
    config: Controller.Config, *, max_response_tokens: int
) -> None:
    """Force an answer at the response cap and charge long correct answers.

    Thinking still open 2,048 tokens before the cap gets a forced close plus
    `Answer: \\boxed{`, and a correct forced answer is worth half. `DOME_V2_FORCED_ANSWER=0`
    turns that off; `DOME_V2_LENGTH_PENALTY=none` turns off the `CorrectLengthPenalty`.
    """
    worker = config.rollouter.worker
    if os.environ.get("DOME_V2_FORCED_ANSWER", "1") == "1":
        worker.thinking_budget = ThinkingBudget.Config(
            max_thinking_tokens=max_response_tokens - 2048,
            answer_prefix="Answer: \\boxed{",
        )
        worker.rubric.forced_answer_scale = 0.5
    if os.environ.get("DOME_V2_LENGTH_PENALTY", "correct") == "correct":
        worker.rubric.length_penalty = CorrectLengthPenalty.Config(
            max_tokens=max_response_tokens
        )


def _qwen3_5_35b_a3b_base_dapo_math_config(
    *,
    max_response_tokens: int,
    num_prompts_per_train_step: int,
    microbatch_rows: int,
    dump_folder: str,
    data_parallel_shard_degree: int = 4,
    num_generators: int = 8,
    target_offpolicy_steps: int = 4,
) -> Controller.Config:
    """Build the 16-GPU Qwen3.5-35B-A3B-Base DAPO run with a Dist-MoE trainer.

    Args:
        data_parallel_shard_degree: Trainer FSDP degree; with TP 2 the trainer takes
            2 x this many GPUs (4 -> 8 GPUs on 2 hosts, 2 -> 4 GPUs on 1 host).
        num_generators: One-GPU vLLM engines.
    """
    expert_parallel_degree = 4
    config = _qwen3_5_base_dapo_math_config(
        flavor="35B-A3B",
        enable_thinking=True,
        max_response_tokens=max_response_tokens,
        # 2,048 tokens for the prompt.
        max_total_tokens=max_response_tokens + 2048,
        num_prompts_per_train_step=num_prompts_per_train_step,
        num_samples_per_prompt=16,
        microbatch_rows=microbatch_rows,
        validation_interval_steps=25,
        num_generators=num_generators,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=data_parallel_shard_degree,
            tensor_parallel_degree=2,
            expert_parallel_degree=expert_parallel_degree,
        ),
        num_loss_chunks=16,
        dump_folder=dump_folder,
        target_offpolicy_steps=target_offpolicy_steps,
    )
    trainer = config.trainer
    # Recompute every op in the block except the Dist-MoE call, which is never recomputed.
    trainer.activation_checkpoint = RegionAC.Config(save_regions=[])
    # Dist-MoE experts on the trainer's model copy only; generators keep stock experts.
    trainer.override = OverrideConfig(
        imports=["torchtitan_recipes.overrides.dist_moe.dist_moe_routed_experts"]
    )
    # Worst case, every EP rank routes all its tokens to one rank; a smaller scratch
    # buffer is an illegal memory access.
    trainer.dist_moe = DistMoeRuntime.Config(
        scratch_capacity_factor=float(expert_parallel_degree)
    )
    return config


def _qwen3_5_base_dapo_math_config(
    *,
    flavor: str,
    enable_thinking: bool,
    max_response_tokens: int,
    max_total_tokens: int,
    num_prompts_per_train_step: int,
    num_samples_per_prompt: int,
    microbatch_rows: int,
    validation_interval_steps: int,
    num_generators: int,
    parallelism: ParallelismConfig,
    num_loss_chunks: int,
    dump_folder: str,
    target_offpolicy_steps: int = 4,
) -> Controller.Config:
    """Build a Qwen3.5-Base DAPO-Math run that saves resumable checkpoints.

    Args:
        microbatch_rows: Tokens per trainer microbatch, in rows of `max_total_tokens`.
    """
    num_validation_samples = 30
    model_config = build_qwen3_5_model_config(
        flavor,
        seq_len=max_total_tokens,
        attn_backend="varlen",
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path=f"torchtitan/rl/example_checkpoint/Qwen3.5-{flavor}-Base",
        dump_folder=dump_folder,
        async_loop=AsyncLoopConfig(
            num_training_steps=150,
            num_prompts_per_train_step=num_prompts_per_train_step,
            num_samples_per_prompt=num_samples_per_prompt,
            target_offpolicy_steps=target_offpolicy_steps,
            windowed_fifo_batches=None,
            validation=ValidationConfig(
                num_samples=num_validation_samples,
                interval_steps=validation_interval_steps,
                greedy=True,
            ),
            # Launch knobs for a resumed job: keep zero-std groups (advantage 0, no wait
            # for a replacement), or tolerate longer runs of them before failing.
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=os.environ.get("DOME_V2_DROP_ZERO_STD", "1")
                == "1"
            ),
            batcher=Batcher.Config(
                max_consecutive_untrainable_batches=int(
                    os.environ.get("DOME_V2_MAX_UNTRAINABLE_BATCHES", 10)
                )
            ),
        ),
        rollouter=_dapo_math_rollouter_config(
            validation_dataset=AIME2025Dataset.Config(
                num_samples=num_validation_samples
            ),
            token_env=TokenEnv.Config(
                max_rollout_tokens=max_total_tokens,
                max_num_turns=1,
            ),
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(
                enable_thinking=enable_thinking, thinking_retention="all"
            )
        ),
        num_generators=num_generators,
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
                "validation/response_length/mean",
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
                # A minimum factor of 1 keeps the learning rate constant.
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0,
                    min_lr_factor=1.0,
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=microbatch_rows
                * max_total_tokens,
                max_context_length=max_total_tokens,
                # fp32 master weights; FSDP unshards bf16 params for compute.
                dtype="float32",
                mixed_precision_param="bfloat16",
            ),
            parallelism=parallelism,
            # Every save is a full resumable DCP; DOME sets the interval and uploads
            # the steps, so torchtitan purges nothing.
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=25,
                enable_first_step_checkpoint=True,
                async_mode="async",
                keep_latest_k=0,
                last_save_in_hf=False,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=num_loss_chunks,
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
            gpu_memory_limit=0.9,
            max_num_batched_tokens=8192,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
            ),
        ),
    )
