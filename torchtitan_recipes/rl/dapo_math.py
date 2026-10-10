# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO-Math recipes: verified single-node Qwen3-4B-Base, and multi-host Qwen3.5-Base."""

from __future__ import annotations

import dataclasses

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
from torchtitan.rl.components.data import IterableRLDataLoader
from torchtitan.rl.components.work_buffer import AdaptiveRolloutGroupWorkBuffer
from torchtitan.rl.controller import (
    AsyncLoopConfig,
    Controller,
    RLModelDefaults,
    ValidationConfig,
)
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.examples.dapo_math.data import (
    DapoMathDataset,
    Intellect3MathDataset,
    MathEvalDataset,
)
from torchtitan.rl.examples.dapo_math.env import DapoMathEnv
from torchtitan.rl.examples.dapo_math.rubric import PerBenchmarkRubric, RewardMathVerify
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import DAPOLoss, ScoreCenteringLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.environment import TokenEnv
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.thinking_budget import ThinkingBudget
from torchtitan.rl.trainer import Trainer

# TODO: Enable CUDA graphs for RL trainers after eager/graph numerics parity is
# verified.


def _dapo_math_rollouter_config(
    *,
    validation_dataset: MathEvalDataset.Config,
    token_env: TokenEnv.Config,
) -> Rollouter.Config:
    return Rollouter.Config(
        training_dataloader=IterableRLDataLoader.Config(
            dataset=DapoMathDataset.Config()
        ),
        validation_dataset=validation_dataset,
        worker=RolloutWorker.Config(
            rubric=PerBenchmarkRubric.Config(
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
    # avg@4: each of `MathEvalDataset`'s 240 problems (193 core, 47 hard), sampled like training.
    # TODO: the hard tier reads near 0 at 8K/32K and needs >=64K responses. Not wired: validation
    #   shares training's max_tokens, and the generators share the trainer's max_context_length.
    num_validation_samples = 4 * 240
    validation_dataset = MathEvalDataset.Config()
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
                steps=num_validation_samples,
                greedy=False,
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
                "validation_reward/component/core/mean",
                "validation_reward/component/hard/mean",
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


def rl_dapo_qwen3_5_35b_a3b_base_intellect3_math_131k() -> Controller.Config:
    """Qwen3.5-35B-A3B-Base on INTELLECT-3 math with 131K responses; every value is set here,
    with no environment knobs.

    12 GPUs: the trainer on 4 (FSDP 2 x TP 2 x EP 4, Dist-MoE experts, 1-row microbatches)
    and eight TP1 vLLM engines (vLLM watermark 0.03). 64 prompts x 16 samples per step, from
    the problems Qwen3-4B-Thinking-2507 solves in 1-6 of 8 tries. Thinking still open at
    126,976 tokens is closed and answered, and a correct forced answer scores 0.5. No length
    reward. Validates at avg@4 on the 240-problem `MathEvalDataset`, sampled like training.
    """
    max_response_tokens = 131072
    # 2,048 tokens for the prompt.
    max_total_tokens = max_response_tokens + 2048
    expert_parallel_degree = 4
    model_config = build_qwen3_5_model_config(
        "35B-A3B",
        seq_len=max_total_tokens,
        attn_backend="varlen",
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.5-35B-A3B-Base",
        dump_folder="outputs/rl/qwen3_5_35b_a3b_base_intellect3_math_131k",
        async_loop=AsyncLoopConfig(
            num_training_steps=150,
            num_prompts_per_train_step=64,
            num_samples_per_prompt=16,
            target_offpolicy_steps=6,
            # avg@4: MathEvalDataset cycles in order, so 4 * 240 draws grade each problem 4 times.
            validation=ValidationConfig(steps=4 * 240, greedy=False),
        ),
        rollouter=Rollouter.Config(
            training_dataloader=IterableRLDataLoader.Config(
                dataset=Intellect3MathDataset.Config(max_pass_rate=0.75)
            ),
            validation_dataset=MathEvalDataset.Config(),
            worker=RolloutWorker.Config(
                rubric=PerBenchmarkRubric.Config(
                    reward_fns=[RewardMathVerify.Config(weight=1.0)],
                    error_reward=0.0,
                    # A truncated rollout has no final answer; grading the last \boxed{} of
                    # its unfinished reasoning would reward a guess.
                    truncation_reward=0.0,
                    forced_answer_scale=0.5,
                ),
                message_env=DapoMathEnv.Config(),
                token_env=TokenEnv.Config(
                    max_rollout_tokens=max_total_tokens,
                    max_num_turns=1,
                ),
                advantage=AdvantageEstimator.Config(should_std_normalize=False),
                # Thinking still open 4,096 tokens before the response cap is closed and
                # answered.
                thinking_budget=ThinkingBudget.Config(
                    max_thinking_tokens=max_response_tokens - 4096,
                    answer_prefix="Answer: \\boxed{",
                ),
            ),
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(enable_thinking=True, thinking_retention="all")
        ),
        num_generators=8,
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            console_log_keys_validation=[
                "validation_reward/component/core/mean",
                "validation_reward/component/hard/mean",
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
                num_tokens_per_microbatch_per_dp_rank=max_total_tokens,
                max_context_length=max_total_tokens,
                # fp32 master weights; FSDP unshards bf16 params for compute.
                dtype="float32",
                mixed_precision_param="bfloat16",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=expert_parallel_degree,
            ),
            # Recompute every op in the block except the Dist-MoE call, which is never
            # recomputed.
            activation_checkpoint=RegionAC.Config(save_regions=[]),
            # Dist-MoE experts on the trainer's model copy only; generators keep stock experts.
            override=OverrideConfig(
                imports=[
                    "torchtitan_recipes.overrides.dist_moe.dist_moe_routed_experts"
                ]
            ),
            # Worst case, every EP rank routes all its tokens to one rank; a smaller scratch
            # buffer is an illegal memory access.
            dist_moe=DistMoeRuntime.Config(
                scratch_capacity_factor=float(expert_parallel_degree)
            ),
            # A full resumable checkpoint (~420 GB) every 10 steps, all kept for offline eval.
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                enable_first_step_checkpoint=True,
                async_mode="async",
                keep_latest_k=0,
                last_save_in_hf=False,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
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
            gpu_memory_limit=0.9,
            max_num_batched_tokens=8192,
            # At the default 512 max_num_seqs the engines fill their KV cache; keep 3% of KV
            # blocks free at admission, so new requests preempt running ones less often.
            extra_vllm_engine_args={"watermark": 0.03},
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_response_tokens,
            ),
        ),
    )


def rl_dapo_qwen3_5_35b_a3b_base_intellect3_math_131k_kimi() -> Controller.Config:
    """`rl_dapo_qwen3_5_35b_a3b_base_intellect3_math_131k` plus Kimi k1.5's length reward at
    weight 0.1.

    k1.5 turns it on after plain-RL steps: resume a step of the base recipe with
    `--resume-step N`; the two recipes share a dump folder.
    """
    config = rl_dapo_qwen3_5_35b_a3b_base_intellect3_math_131k()
    config.rollouter.worker.rubric.length_reward_weight = 0.1
    return config


def rl_dapo_qwen3_5_35b_a3b_base_intellect3_math_131k_tonight() -> Controller.Config:
    """`rl_dapo_qwen3_5_35b_a3b_base_intellect3_math_131k` as a fresh 300-step run with score
    centering, the adaptive rollout buffer, and validation beside training.

    Same 12 GPUs, 64 x 16 batch, 131K budget and forced answer; no length reward.
    """
    config = rl_dapo_qwen3_5_35b_a3b_base_intellect3_math_131k()
    config.dump_folder = "outputs/rl/qwen3_5_35b_a3b_base_intellect3_math_131k_tonight"
    # fp32 logits on the trainer and the generator, and every expert bias kept at its loaded value.
    config.model_defaults = RLModelDefaults(fp32_lm_head=True, freeze_expert_bias=True)
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=300,
        # Under the adaptive buffer, this only sets train_batch/pct_samples_over_target_age.
        target_offpolicy_steps=6,
        group_buffer=AdaptiveRolloutGroupWorkBuffer.Config(
            # The demand keeps the mean policy age of trained groups under 6 steps.
            target_offpolicy_steps=6,
            # A group older than 12 steps when consumed is dropped and its prompt not retried.
            max_offpolicy_steps=12,
            # The engines are KV-bound at ~2,850 running sequences (~285 groups at ~10 live seats).
            # 640 binds only when over ~30% of groups have zero reward variance, and then it still
            # keeps a vLLM queue; each idle group worker costs controller time on buffer changes.
            generation_capacity=640,
            # Start at the fixed buffer's (6 + 1) x 64 groups: from 3 x 64, the first steps underfeed
            # the engines while the demand rule ramps up.
            start_batches=7,
        ),
        # Groups whose 16 rewards are equal carry no gradient: the batch takes the next group.
        training_sample_builder=dataclasses.replace(
            config.async_loop.training_sample_builder,
            drop_zero_std_reward_groups=True,
        ),
        # avg@4 on the 240 problems every 10 steps, beside training (the first at step 0). A
        # pass took 3.2-3.6 h on the last run, so the next one starts when the previous ends.
        validation=ValidationConfig(
            steps=4 * 240, greedy=False, freq=10, overlap_training=True
        ),
    )
    config.rollouter.training_dataloader.dataset = dataclasses.replace(
        config.rollouter.training_dataloader.dataset,
        # A problem whose group comes back with every reward above 0.9 (all 16 correct
        # without a forced answer) is skipped in later epochs.
        skip_solved_prompts=True,
    )
    config.trainer = dataclasses.replace(
        config.trainer,
        # No router replay for now: its routed experts are ~2/3 of each step's batch, which every
        # trainer rank receives; at step 3 (~30 GB) delivery outlasted monarch's 60 s supervision poll.
        replay_routed_experts=False,
        # Synchronous saves: an async save's ~420 GB CPU copy, on top of the 4 trainer ranks each
        # unpacking the 13-21 GB batch at 2-3x, can exceed the trainer host's 900 GB job memory.
        checkpointer=dataclasses.replace(
            config.trainer.checkpointer, async_mode="disabled"
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        lr=1e-5,
                        betas=(0.9, 0.98),
                        weight_decay=0.1,
                    )
                ]
            ),
            # Fresh weights: ramp linearly to 1e-5 over 10 steps, then hold it.
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=10,
                min_lr_factor=1.0,
            ),
        ),
        loss=ChunkedLossWrapper.Config(
            num_chunks=16,
            loss_fn=ScoreCenteringLoss.Config(
                # Truncated IS at 2, the paper's TIS+SC arm: better than SC alone at high staleness.
                max_ratio=2.0,
                global_vocab_size=decoder_vocab_size(config.model),
            ),
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        return_routed_experts=False,
        sampling=dataclasses.replace(
            config.generator.sampling,
            # ScoreCenteringLoss's top-k. 32 is the smallest k arXiv 2609.20807 tested, and it
            # matched the full vocabulary (Fig. 5); 256 bytes per generated token.
            num_topk_logprobs=32,
        ),
    )
    return config
