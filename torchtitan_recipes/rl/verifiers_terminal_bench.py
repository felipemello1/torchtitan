# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal-agent RL recipes: train on Harbor tasks, validate on Terminal-Bench 2.1.

The environment (bash-tool agent, Harbor taskset) is the framework-agnostic
``torchtitan_recipes.rl.terminal_bench`` package; this module wires it into
TitanRL. ``TERMINAL_BENCH_SANDBOX`` picks where each rollout's commands run:
``docker`` (default) starts task containers on this host, ``sandoq`` leases a
remote Firecracker VM per rollout.
"""

import os
from typing import Literal

import verifiers.v1 as vf
from renderers import Qwen36RendererConfig

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
from torchtitan.config.transform import LMHeadCastConverter
from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.models.qwen3_6 import build_model_config
from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.examples.verifiers import (
    GenerationServer,
    RewardFromVerifiers,
    VerifiersEnvServer,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import GRPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.observability.vllm import VllmOtelStatLogger
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.trainer import Trainer
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig
from verifiers.v1.tasksets.harbor import HarborEnvConfig

from torchtitan_recipes.rl.terminal_bench.agent_outside import (
    AgentOutsideHarnessConfig,
    HARNESS_ID,
)
from torchtitan_recipes.rl.terminal_bench.taskset import (
    TASKSET_ID,
    TerminalTasksetConfig,
)

# Harbor dataset ids, read from $HOME/.cache/harbor/<id with "/" and "@" as "_">.
# 1,024 moderate and complex TMax-15K tasks (a seeded sample).
TMAX_1K = "tmax-mc1024@27de1c1b"
# Terminal-Bench 2.1 (harbor-framework/terminal-bench-2-1@7131e437) without
# qemu-alpine-ssh, qemu-startup and protein-assembly: 86 tasks.
TERMINAL_BENCH_2_1_86 = "tb21-86@7131e437"

MAX_ROLLOUT_TOKENS = 32768
MAX_TOKENS_PER_TURN = 4096


def rl_grpo_qwen3_6_35b_a3b_terminal_bench() -> Controller.Config:
    """Qwen3.6-35B-A3B: 50 steps on TMax tasks, Terminal-Bench 2.1 pass@1 at steps 0, 25, 50.

    8 GB300 GPUs on two hosts. Trainer: FSDP=2 x TP=2, EP=4, with Dist-MoE experts
    (SM100+). Generator: one DP=2 x TP=2, EP=4 replica with FULL CUDA graphs and
    the stock experts. 16 tasks x 8 rollouts per step, each rollout up to 20 turns
    and 32,768 tokens.
    """
    expert_parallel_degree = 4
    model_config = build_model_config(
        "35B-A3B",
        seq_len=MAX_ROLLOUT_TOKENS,
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.6-35B-A3B",
        dump_folder="outputs/rl/qwen3_6_35b_a3b_terminal_bench",
        async_loop=AsyncLoopConfig(
            num_training_steps=50,
            num_prompts_per_train_step=16,
            num_samples_per_prompt=8,
            target_offpolicy_steps=1,
            validation=ValidationConfig(
                num_samples=86, interval_steps=25, greedy=False
            ),
            # A group whose rollouts all score the same has zero advantage, so it
            # adds no gradient; keeping it means a step never waits for a replacement.
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False
            ),
        ),
        rollouter=_terminal_bench_rollouter_config(
            train_dataset=TMAX_1K,
            validation_dataset=TERMINAL_BENCH_2_1_86,
            sandbox=os.environ.get("TERMINAL_BENCH_SANDBOX", "docker"),
        ),
        renderer=from_renderers(Qwen36RendererConfig(enable_thinking=False)),
        num_generators=1,
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            console_log_keys_validation=[
                "validation_reward/_mean",
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
                            betas=(0.9, 0.999),
                            weight_decay=0.0,
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
                num_tokens_per_microbatch_per_dp_rank=MAX_ROLLOUT_TOKENS,
                max_context_length=MAX_ROLLOUT_TOKENS,
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
                imports=[
                    "torchtitan_recipes.overrides.dist_moe.dist_moe_routed_experts"
                ]
            ),
            # Scratch factor EP covers every rank routing all tokens to one
            # rank; scratch overflow is an illegal memory access.
            dist_moe=DistMoeRuntime.Config(
                device_scratch_capacity_factor=float(expert_parallel_degree)
            ),
            checkpointer=CheckpointManager.Config(initial_load_in_hf=True),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    clip_eps=0.2,
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=expert_parallel_degree,
            ),
            # Each turn prefills its new tool output (up to 16,384 chars); fit one
            # in a single engine step instead of vLLM's default 2,048 tokens.
            max_num_batched_tokens=8192,
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=MAX_TOKENS_PER_TURN,
            ),
            # vLLM engine stats (KV-cache usage, running requests) when an exporter is set.
            vllm_stat_logger=(
                VllmOtelStatLogger.Config()
                if os.environ.get("OTEL_METRICS_EXPORTER")
                else None
            ),
        ),
    )


def rl_grpo_qwen3_6_35b_a3b_terminal_bench_smoke() -> Controller.Config:
    """``rl_grpo_qwen3_6_35b_a3b_terminal_bench`` without validation, for short runs."""
    config = rl_grpo_qwen3_6_35b_a3b_terminal_bench()
    config.async_loop.validation = ValidationConfig(num_samples=0)
    return config


def _terminal_bench_rollouter_config(
    *,
    train_dataset: str,
    validation_dataset: str,
    sandbox: Literal["docker", "sandoq"],
) -> VerifiersRollouter.Config:
    """Run each rollout's agent loop in the env server and its commands in a sandbox.

    Args:
        train_dataset: Harbor dataset id to train on.
        validation_dataset: Harbor dataset id to validate on; must differ.
        sandbox: ``"docker"`` runs each task's container on this host;
            ``"sandoq"`` leases a Sandoq Firecracker VM per rollout.
    """
    if train_dataset == validation_dataset:
        raise ValueError("Training and validation must use different datasets")

    if sandbox == "docker":
        runtime = vf.DockerConfig()
    elif sandbox == "sandoq":
        # The Sandoq provider (`sandoq_provider`) serves `vf.PrimeConfig` only
        # when selected. Each task's test.sh installs pytest over the network, so
        # a VM without host networking silently scores every rollout 0.
        if os.environ.get("VF_SANDBOX_PROVIDER") != "oci-runner":
            raise ValueError("sandbox='sandoq' needs VF_SANDBOX_PROVIDER=oci-runner")
        if os.environ.get("OCI_RUNNER_TASK_NETWORK") != "host":
            raise ValueError(
                "sandbox='sandoq' needs OCI_RUNNER_TASK_NETWORK=host, or every "
                "reward is 0"
            )
        runtime = vf.PrimeConfig(idle_timeout=3600)
    else:
        raise ValueError(f"unknown sandbox {sandbox!r}")

    return VerifiersRollouter.Config(
        train_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=TerminalTasksetConfig(
                id=TASKSET_ID, dataset=train_dataset
            ),
            shuffle=False,
        ),
        validation_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=TerminalTasksetConfig(
                id=TASKSET_ID, dataset=validation_dataset
            ),
            shuffle=False,
        ),
        verifiers_env_server=VerifiersEnvServer.Config(
            environment=HarborEnvConfig(
                agent=vf.AgentConfig(
                    harness=AgentOutsideHarnessConfig(
                        id=HARNESS_ID,
                        command_timeout_sec=300,
                        max_tool_output_chars=16384,
                    ),
                    runtime=runtime,
                    max_turns=20,
                    timeout=AgentTimeoutConfig(setup=1500, rollout=1800, scoring=1500),
                ),
            ),
            # 16 x 24 = 384 rollouts in flight: 256 training (2 steps of 16 x 8)
            # plus a validation pass. Size the sandbox pool to match.
            serve=vf.ServeConfig(
                pool=vf.StaticPoolConfig(num_workers=16),
                max_concurrent=24,
                address="tcp://127.0.0.1:0",
            ),
        ),
        rubric=Rubric.Config(
            reward_fns=[RewardFromVerifiers.Config(weight=1.0)],
            error_reward=0.0,
        ),
        generation_server=GenerationServer.Config(
            max_rollout_tokens=MAX_ROLLOUT_TOKENS
        ),
    )
