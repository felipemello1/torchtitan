# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5/3.6 SWE-rebench V2 recipes, with the agent outside the sandbox.

Both recipes read ``SWE_REBENCH_V2_SANDBOX`` (``docker`` or ``sandoq``, default
``docker``) for where task containers run; see ``swe_rebench_v2_rollouter_config``.

Budgets: 30 agent turns of up to 16,384 tokens each. Qwen3.5 thinks before
every tool call, and a turn cut off mid-thought has no tool call, which ends the
rollout, so turns get more room than the 4,096 the Qwen3 receipts used. The
rollout context (prompt, every turn and every tool result, with all thinking
kept) caps a rollout's total length.
"""

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
from torchtitan.config import CommConfig, OverrideConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LMHeadCastConverter
from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.qwen3_5 import model_registry as qwen3_5_model_registry
from torchtitan.models.qwen3_6 import model_registry as qwen3_6_model_registry
from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.experiments.verifiers.swe_rebench_v2.rollouter import (
    swe_rebench_v2_rollouter_config,
)
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import GRPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.observability.rollout_recorder import (
    KeepExtremeRewardsFilter,
    RolloutSampleRecorder,
)
from torchtitan.rl.trainer import Trainer

MAX_TOKENS_PER_TURN = 16384


def rl_grpo_qwen35_9b_swe_rebench_v2_smoke() -> Controller.Config:
    """Qwen3.5-9B smoke run: 3 steps of 2 tasks x 8 samples, 65,536-token rollouts.

    8 GPUs: a trainer (FSDP=2, TP=2) and one TP=4 generator, each filling a
    4-GPU host.
    """
    max_context_length = 65536
    model_config = qwen3_5_model_registry(
        "9B",
        seq_len=max_context_length,
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.5-9B",
        dump_folder="outputs/rl/qwen35_9b_swe_rebench_v2_smoke",
        async_loop=AsyncLoopConfig(
            num_training_steps=3,
            num_prompts_per_train_step=2,
            num_samples_per_prompt=8,
            target_offpolicy_steps=1,
            validation=ValidationConfig(num_samples=0),
            # The F2P/P2P reward is dense, but an all-fail group is common
            # early; keep it so a weak model still makes steps.
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False
            ),
        ),
        compile=None,
        rollouter=swe_rebench_v2_rollouter_config(
            sandbox=os.environ.get("SWE_REBENCH_V2_SANDBOX", "docker"),
            max_rollout_tokens=max_context_length,
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(enable_thinking=True, thinking_retention="all")
        ),
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(k=1),
            log_tensors=False,
            log_logprobs=False,
        ),
        num_generators=1,
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*",
                            lr=1e-6,
                            betas=(0.9, 0.999),
                            weight_decay=0.01,
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
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
                max_context_length=max_context_length,
                # fp32 master weights: bf16 rounds most 1e-6 updates away.
                dtype="float32",
                mixed_precision_param="bfloat16",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
            ),
            activation_checkpoint=FullAC.Config(),
            comm=CommConfig(init_timeout_seconds=1800),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
                keep_latest_k=2,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            # Admit two full-size turns per scheduler step.
            max_num_batched_tokens=2 * MAX_TOKENS_PER_TURN,
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            cuda_graph=VLLMCudaGraphConfig(mode="FULL_DECODE_ONLY"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=MAX_TOKENS_PER_TURN,
            ),
        ),
    )


def rl_grpo_qwen36_35b_a3b_swe_rebench_v2_dist_moe() -> Controller.Config:
    """Qwen3.6-35B-A3B with Dist-MoE in the trainer only (SM100+, 2 x 4 GPUs).

    Trainer: FSDP=2 x TP=2 with EP=4 on one host. Generator: one DP=2 x TP=2,
    EP=4 replica on a second host that keeps the stock ``RoutedExperts``. EP stays
    inside a host, so Dist-MoE's symmetric memory never crosses hosts. 64 rollouts
    (4 tasks x 16 samples) of up to 131,072 tokens per step.
    """
    max_context_length = 131072
    expert_parallel_degree = 4
    model_config = qwen3_6_model_registry(
        "35B-A3B",
        seq_len=max_context_length,
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.6-35B-A3B",
        dump_folder="outputs/rl/qwen36_35b_a3b_swe_rebench_v2_dist_moe",
        async_loop=AsyncLoopConfig(
            num_training_steps=100,
            num_prompts_per_train_step=4,
            num_samples_per_prompt=16,
            target_offpolicy_steps=1,
            validation=ValidationConfig(num_samples=0),
            # The F2P/P2P reward is dense, but an all-fail group is common
            # early; keep it so a weak model still makes steps.
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False
            ),
        ),
        # Dist-MoE supports make_fx and CUDA graphs, not full torch.compile.
        compile=None,
        rollouter=swe_rebench_v2_rollouter_config(
            sandbox=os.environ.get("SWE_REBENCH_V2_SANDBOX", "docker"),
            max_rollout_tokens=max_context_length,
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(enable_thinking=True, thinking_retention="all")
        ),
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(k=1),
            log_tensors=False,
            log_logprobs=False,
        ),
        num_generators=1,
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[
                        AdamW.Config(
                            pattern=r".*",
                            lr=1e-6,
                            betas=(0.9, 0.999),
                            weight_decay=0.01,
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
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
                max_context_length=max_context_length,
                # fp32 master weights; Dist-MoE consumes the bf16 FSDP unshard.
                dtype="float32",
                mixed_precision_param="bfloat16",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=expert_parallel_degree,
            ),
            # Keep the Dist-MoE call out of recompute: its region is
            # recompute=False, and everything else in the block is recomputed.
            activation_checkpoint=RegionAC.Config(save_regions=[]),
            # Swap in Dist-MoE experts on the trainer's model copy only.
            override=OverrideConfig(
                imports=["torchtitan.overrides.dist_moe.dist_moe_routed_experts"]
            ),
            # Scratch factor EP covers every rank routing all tokens to one
            # rank; scratch overflow is an illegal memory access.
            dist_moe=DistMoeRuntime.Config(
                device_scratch_capacity_factor=float(expert_parallel_degree)
            ),
            comm=CommConfig(init_timeout_seconds=1800),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
                keep_latest_k=2,
            ),
            loss=ChunkedLossWrapper.Config(
                # Twice the 65,536-token recipes' chunks keeps per-chunk fp32
                # logits the same size.
                num_chunks=16,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            # Admit two full-size turns per scheduler step.
            max_num_batched_tokens=2 * MAX_TOKENS_PER_TURN,
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=expert_parallel_degree,
            ),
            # The stock all-to-all dispatcher syncs with the host, so the
            # generator cannot capture CUDA graphs.
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=MAX_TOKENS_PER_TURN,
            ),
        ),
    )
