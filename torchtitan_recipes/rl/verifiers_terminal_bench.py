# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 terminal-agent recipes using Verifiers and TitanRL.

Both recipes train on ``local/tmax@v1``, the TMax tasks exported by
``prepare_tmax.py`` into ``~/.cache/harbor/local_tmax_v1``, and validate on the
Terminal-Bench 2.1 Harbor dataset, which the ``harbor`` CLI downloads into
``~/.cache/harbor`` on first use. Edit the ids below to use other datasets.
"""

import dataclasses
import math
import os

import verifiers.v1 as vf

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
from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
    ValidationLoopMode,
)
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.examples.verifiers import (
    GenerationServer,
    RewardFromVerifiers,
    VerifiersEnvServer,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    register_harness_alias,
)
from torchtitan.rl.examples.verifiers.terminal_bench.taskset import (
    TerminalTasksetConfig,
)
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import DAPOLoss, GRPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.observability.rollout_recorder import (
    KeepExtremeRewardsFilter,
    RolloutSampleRecorder,
)
from torchtitan.rl.rollout.thinking_budget import ThinkingBudget
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.trainer import Trainer
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.harnesses.terminus_2 import Terminus2HarnessConfig
from verifiers.v1.tasksets.harbor import HarborEnvConfig

_ENV_SERVER_WORKERS = 16


def _terminal_bench_rollouter_config(
    train_dataset: str,
    validation_dataset: str,
    *,
    max_context_length: int,
    max_turns: int,
    max_concurrent_rollouts: int,
    num_env_workers: int = _ENV_SERVER_WORKERS,
) -> VerifiersRollouter.Config:
    """Select Harbor datasets by id.

    ``max_context_length`` is the generator's sequence length; the generation
    server caps each prompt one token below it. ``max_turns`` is the agent turn limit, which
    Verifiers enforces. ``max_concurrent_rollouts`` sizes the env server; set it
    to the number of rollouts the controller keeps in flight, or the excess
    queues in the env server and the generators idle. The env server splits it
    over ``num_env_workers`` processes.
    """
    if train_dataset == validation_dataset:
        raise ValueError(
            "Training and Terminal-Bench evaluation must use different datasets"
        )

    taskset_id = register_local_taskset_alias(TerminalTasksetConfig.__module__)
    return VerifiersRollouter.Config(
        train_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=TerminalTasksetConfig(
                id=taskset_id, dataset=train_dataset
            ),
            seed=42,
            shuffle=True,
        ),
        validation_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=TerminalTasksetConfig(
                id=taskset_id, dataset=validation_dataset
            ),
            seed=99,
            shuffle=False,
        ),
        verifiers_env_server=VerifiersEnvServer.Config(
            environment=HarborEnvConfig(
                agent=vf.AgentConfig(
                    harness=Terminus2HarnessConfig(
                        id=register_harness_alias(), version="0.22.0"
                    ),
                    runtime=vf.DockerConfig(),
                    max_turns=max_turns,
                    timeout=AgentTimeoutConfig(
                        setup=600,
                        rollout=7200,
                        scoring=12000,
                    ),
                ),
            ),
            serve=vf.ServeConfig(
                pool=vf.StaticPoolConfig(num_workers=num_env_workers),
                max_concurrent=math.ceil(max_concurrent_rollouts / num_env_workers),
                address="tcp://127.0.0.1:0",
            ),
        ),
        rubric=Rubric.Config(
            reward_fns=[RewardFromVerifiers.Config(weight=1.0)],
            error_reward=0.0,
        ),
        # vLLM needs room for one output token. A prompt of max_context_length tokens
        # would pass Verifiers' check (prompt > cap), then crash the engine loop.
        generation_server=GenerationServer.Config(
            max_rollout_tokens=max_context_length - 1
        ),
        connection_timeout_sec=1800.0,
    )


def _on_sandoq(
    rollouter: VerifiersRollouter.Config,
    *,
    interleaved_thinking: bool,
    rollout_log_dir: str,
) -> VerifiersRollouter.Config:
    """Run the rollouter's Terminus-2 in the env server, with a Sandoq VM per rollout.

    Only the agent's shell commands go to the VM; with ``interleaved_thinking``, each turn's
    reasoning goes back to the policy.
    Needs ``torchtitan_recipes/rl/verifiers_plugins`` on PYTHONPATH and the oci-runner provider
    env (DOME's ``SANDOQ_ENV``).
    """
    # Imported here: Verifiers imports plugin ids as top-level modules, and the plugin
    # directory is on PYTHONPATH only in Sandoq jobs.
    from terminal_bench_sandoq import (
        PLUGIN_ID,
        sandbox_runtime,
        StockTerminusOutsideConfig,
    )

    def on_plugin(dataset: VerifiersTaskDataset.Config) -> VerifiersTaskDataset.Config:
        # Keep each task's declared timeouts; the agent-level rollout timeout still wins.
        taskset = TerminalTasksetConfig(
            id=PLUGIN_ID,
            dataset=dataset.verifiers_taskset.dataset,
            ignore_timeouts=False,
        )
        return dataclasses.replace(dataset, verifiers_taskset=taskset)

    train_dataset = on_plugin(rollouter.train_dataset)
    env_server = rollouter.verifiers_env_server
    environment = env_server.environment
    agent = environment.agent.model_copy(
        update={
            "harness": StockTerminusOutsideConfig(
                id=PLUGIN_ID,
                interleaved_thinking=interleaved_thinking,
                rollout_log_dir=rollout_log_dir,
            ),
            "runtime": sandbox_runtime(),
            # SandoqTerminalTaskset.load sets each task's scoring timeout instead.
            "timeout": environment.agent.timeout.model_copy(update={"scoring": None}),
            # A lost exec channel (Sandoq transport, VM gone) gets one more try on a fresh VM.
            "retries": RetryConfig(max_retries=1, include=["SandboxError"]),
        }
    )
    return dataclasses.replace(
        rollouter,
        train_dataset=train_dataset,
        validation_dataset=on_plugin(rollouter.validation_dataset),
        verifiers_env_server=dataclasses.replace(
            env_server,
            # The taskset plugin also supplies the env, which leases each rollout's VM.
            environment=environment.model_copy(
                update={"agent": agent, "taskset": train_dataset.verifiers_taskset}
            ),
        ),
    )


def rl_grpo_qwen35_9b_terminal_bench() -> Controller.Config:
    """Qwen3.5-9B: train on TMax (``local/tmax@v1``), validate on Terminal-Bench 2.1.

    16 GPUs: 8 trainer (FSDP=8) and 8 one-GPU generators.
    """
    # Agent turns average about 1.1K tokens (completion plus terminal output),
    # so 120 turns need about 128K.
    max_context_length = 131072
    async_loop = AsyncLoopConfig(
        num_training_steps=100,
        num_prompts_per_train_step=12,
        num_samples_per_prompt=32,
        target_offpolicy_steps=3,
        validation=ValidationConfig(num_samples=89),
    )
    model_config = build_model_config(
        "9B",
        seq_len=max_context_length,
        attn_backend="varlen",
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.5-9B",
        dump_folder="outputs/rl/qwen35_9b_terminal_bench",
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
        ),
        async_loop=async_loop,
        rollouter=_terminal_bench_rollouter_config(
            train_dataset="local/tmax@v1",
            validation_dataset="terminal-bench/terminal-bench-2-1",
            max_context_length=max_context_length,
            max_turns=120,
            max_concurrent_rollouts=async_loop.max_active_rollout_groups
            * async_loop.num_samples_per_prompt,
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(
                enable_thinking=True,
                thinking_retention="all",
            )
        ),
        num_generators=8,
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
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
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
                max_context_length=max_context_length,
                dtype="float32",
            ),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=8,
                tensor_parallel_degree=1,
            ),
            activation_checkpoint=FullAC.Config(),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=20,
                keep_latest_k=3,
                async_mode="async",
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=32,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cuda_graph=VLLMCudaGraphConfig(mode="FULL_DECODE_ONLY"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=16384,
            ),
        ),
    )


def rl_grpo_qwen35_35b_a3b_terminal_bench() -> Controller.Config:
    """Qwen3.5-35B-A3B: train on TMax (``local/tmax@v1``), validate on Terminal-Bench 2.1.

    16 GPUs: 8 trainer (FSDP=4, TP=2, EP=8) and 2 generators of 4 GPUs
    (DP=2, TP=2, EP=4).

    The layout is constrained from both sides. The model has 2 KV heads, so TP
    is at most 2 in either role. Trainer EP must be at least TP, divide the 256
    experts and divide ``dp_shard * tp``; EP=8 spans the whole sparse region, so
    each rank holds 32 experts. The generator's DP axis only supplies ranks for
    expert parallelism, so its EP equals DP x TP; 256 experts over 4 ranks is 64
    each.

    The trainer keeps fp32 master weights, the default. That is about 70 GB of
    model states per GPU across 8 GPUs before activations, so it needs GPUs with
    well over 80 GB of memory.

    Generator CUDA graphs are off because of the standard all-to-all MoE token
    dispatcher, not DistMoE. That dispatcher copies the split sizes to the host,
    which CUDA graph capture does not allow
    ("Cannot copy between CPU and CUDA tensors during CUDA graph capture"). Turn
    capture back on together with a dispatcher that avoids the host read, such as
    HybridEP with ``non_blocking_capacity_factor``.

    TODO: migrate to DistMoE and capture generator CUDA graphs in ``FULL`` mode.
    """
    max_context_length = 65536
    async_loop = AsyncLoopConfig(
        num_training_steps=100,
        num_prompts_per_train_step=8,
        num_samples_per_prompt=32,
        target_offpolicy_steps=4,
        validation=ValidationConfig(num_samples=89),
    )
    # TODO: update to distMoE model config
    model_config = build_model_config(
        "35B-A3B",
        seq_len=max_context_length,
        attn_backend="varlen",
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.5-35B-A3B",
        dump_folder="outputs/rl/qwen35_35b_a3b_terminal_bench",
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
        ),
        async_loop=async_loop,
        rollouter=_terminal_bench_rollouter_config(
            train_dataset="local/tmax@v1",
            validation_dataset="terminal-bench/terminal-bench-2-1",
            max_context_length=max_context_length,
            max_turns=120,
            max_concurrent_rollouts=async_loop.max_active_rollout_groups
            * async_loop.num_samples_per_prompt,
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(
                enable_thinking=True,
                thinking_retention="all",
            )
        ),
        num_generators=2,
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
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
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
                max_context_length=max_context_length,
                dtype="float32",
            ),
            parallelism=ParallelismConfig(
                data_parallel_replicate_degree=1,
                data_parallel_shard_degree=4,
                tensor_parallel_degree=2,
                expert_parallel_degree=8,
            ),
            activation_checkpoint=FullAC.Config(),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=20,
                keep_latest_k=3,
                async_mode="async",
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=32,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=4,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=16384,
            ),
        ),
    )


def rl_grpo_qwen3_5_35b_a3b_base_terminal_bench() -> Controller.Config:
    """Qwen3.5-35B-A3B-Base: train on all of TMax-15K; no online validation (the 78
    Terminal-Bench 2.1 tasks that fit a small Sandoq VM are evaluated offline).

    16 GB300 GPUs on 4 hosts. Trainer on two: FSDP 4 x TP 2 x EP 4 with Dist-MoE experts.
    Generators: eight TP1 engines, each with every expert, FULL CUDA graphs. 150 turns of
    up to 16,384 tokens, 131,072 per rollout; Terminus-2 runs in the env server and each
    rollout's shell commands go to a Sandoq VM (`_on_sandoq`).

    `DOME_SANDOQ_POOL` (required) is the number of sandboxes the run may hold. It sets the
    rollouts in the env server and, by the pool >= 3 batches rule, the prompts per step:
    pool 1,000 -> 16 x 16, 600 -> 12 x 16, 400 -> 8 x 16. `DOME_V2_PROMPTS` and
    `DOME_V2_MICROBATCH_ROWS` (rows of 131,072 tokens, default 1) override the batch at
    launch, so a resumed job can change it without a new commit.
    """
    # Read at load time, not import, so tests and other recipes import this module.
    sandbox_pool = int(os.environ["DOME_SANDOQ_POOL"])
    expert_parallel_degree = 4
    num_samples_per_prompt = 16
    config = _qwen3_5_base_terminal_bench_config(
        flavor="35B-A3B",
        num_prompts_per_train_step=int(
            os.environ.get(
                "DOME_V2_PROMPTS", min(16, sandbox_pool // (3 * num_samples_per_prompt))
            )
        ),
        num_samples_per_prompt=num_samples_per_prompt,
        microbatch_rows=int(os.environ.get("DOME_V2_MICROBATCH_ROWS", 1)),
        max_turns=150,
        sandbox_pool=sandbox_pool,
        # At most 24 rollouts per env-server worker: each waiting rollout holds one of the
        # worker's 32 executor threads, and Sandoq needs a free thread to ready a VM.
        num_env_workers=math.ceil(sandbox_pool / 24),
        # No online eval (Felipe, 2026-10-07): the checkpoints are evaluated offline.
        num_validation_samples=0,
        num_generators=8,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=4,
            tensor_parallel_degree=2,
            expert_parallel_degree=expert_parallel_degree,
        ),
        dump_folder="outputs/rl/qwen3_5_35b_a3b_base_terminal_bench",
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


def rl_grpo_qwen3_5_35b_a3b_base_terminal_bench_1x2() -> Controller.Config:
    """`rl_grpo_qwen3_5_35b_a3b_base_terminal_bench` on 3 hosts: 1 trainer host + 2 generator hosts.

    12 GB300 GPUs. Trainer on one host: FSDP 2 x TP 2 x EP 4 with Dist-MoE experts, 1-row
    microbatches. Generators: eight TP1 engines, FULL CUDA graphs, vLLM watermark 0.03.
    12 prompts x 16 samples per step, up to 5 steps off-policy. Every 10 steps, a greedy pass
    over the 78 Terminal-Bench 2.1 tasks runs beside training on the same generators
    (`ValidationLoopMode.OVERLAP_TRAINING`). A turn still thinking at 12,288 of its 16,384
    tokens gets a forced end of thinking. Kimi k1.5's length reward, weight 0.1, favors the
    shorter rollouts in each group (`kimi_length_rewards`).

    `DOME_SANDOQ_POOL` (required) is the number of sandboxes the run may hold; 920 is 80% of
    the 1,152 training rollouts in flight, (5 + 1) x 12 x 16. `DOME_V2_PROMPTS` overrides the prompts,
    `DOME_V2_THINKING_BUDGET` the thinking tokens per turn (0 turns the budget off) and
    `DOME_V2_LENGTH_REWARD_WEIGHT` the length reward's weight (0 turns it off).
    """
    # Read at load time, not import, so tests and other recipes import this module.
    sandbox_pool = int(os.environ["DOME_SANDOQ_POOL"])
    expert_parallel_degree = 4
    config = _qwen3_5_base_terminal_bench_config(
        flavor="35B-A3B",
        num_prompts_per_train_step=int(os.environ.get("DOME_V2_PROMPTS", 12)),
        num_samples_per_prompt=16,
        microbatch_rows=1,
        max_turns=150,
        sandbox_pool=sandbox_pool,
        num_env_workers=math.ceil(sandbox_pool / 24),
        num_validation_samples=78,
        num_generators=8,
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            expert_parallel_degree=expert_parallel_degree,
        ),
        dump_folder="outputs/rl/qwen3_5_35b_a3b_base_terminal_bench_1x2",
        target_offpolicy_steps=5,
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
    # Admit a request only while 3% of KV blocks stay free, so running requests have room to
    # grow before vLLM preempts one.
    config.generator.watermark = 0.03
    max_thinking_tokens = int(os.environ.get("DOME_V2_THINKING_BUDGET", 12288))
    if max_thinking_tokens > 0:
        config.rollouter.thinking_budget = ThinkingBudget.Config(
            max_thinking_tokens=max_thinking_tokens
        )
    config.rollouter.rubric.length_reward_weight = float(
        os.environ.get("DOME_V2_LENGTH_REWARD_WEIGHT", 0.1)
    )
    config.async_loop.validation.interval_steps = 25
    config.async_loop.validation.loop_mode = ValidationLoopMode.OVERLAP_TRAINING
    return config


def rl_grpo_qwen3_5_4b_base_terminal_bench_dev() -> Controller.Config:
    """Qwen3.5-4B-Base dev check of the 35B-A3B recipe's Terminal-Bench path on Sandoq.

    8 GB300 GPUs on 2 hosts: trainer FSDP 4 on one, four TP1 generators on the other.
    2 prompts x 4 samples per step, 20 turns, a pool of 16 sandboxes over two env-server
    workers, no validation. `DOME_V2_THINKING_BUDGET` (default 0, off) caps thinking per turn,
    so a short run can check the forced end of thinking.
    """
    config = _qwen3_5_base_terminal_bench_config(
        flavor="4B",
        num_prompts_per_train_step=2,
        num_samples_per_prompt=4,
        microbatch_rows=1,
        max_turns=20,
        sandbox_pool=16,
        num_env_workers=2,
        num_validation_samples=0,
        num_generators=4,
        parallelism=ParallelismConfig(data_parallel_shard_degree=4),
        dump_folder="outputs/rl/qwen3_5_4b_base_terminal_bench_dev",
    )
    max_thinking_tokens = int(os.environ.get("DOME_V2_THINKING_BUDGET", 0))
    if max_thinking_tokens > 0:
        config.rollouter.thinking_budget = ThinkingBudget.Config(
            max_thinking_tokens=max_thinking_tokens
        )
    return config


def rl_grpo_qwen3_5_9b_base_terminal_bench_fast() -> Controller.Config:
    """Qwen3.5-9B-Base with thinking off: fast Terminal-Bench steps on Sandoq.

    12 GB300 GPUs on 3 hosts: trainer FSDP 4 on one, eight TP1 generators on the other two.
    8 samples per prompt, 150 turns, up to 6 steps off-policy, no validation.

    `DOME_SANDOQ_POOL` (default 1,434) is the number of sandboxes the run may hold. It sets the
    rollouts in the env server and the prompts per step, sized so the pool is 80% of the
    rollouts in flight, (6 + 1) x prompts x 8: pool 1,434 -> 32 x 8, 717 -> 16 x 8.
    `DOME_V2_PROMPTS` overrides the prompts.
    """
    # Read at load time, not import, so tests and other recipes import this module.
    sandbox_pool = int(os.environ.get("DOME_SANDOQ_POOL", 1434))
    num_samples_per_prompt = 8
    target_offpolicy_steps = 6
    rollouts_in_flight_per_prompt = (
        target_offpolicy_steps + 1
    ) * num_samples_per_prompt
    return _qwen3_5_base_terminal_bench_config(
        flavor="9B",
        num_prompts_per_train_step=int(
            os.environ.get(
                "DOME_V2_PROMPTS",
                round(sandbox_pool / (0.8 * rollouts_in_flight_per_prompt)),
            )
        ),
        num_samples_per_prompt=num_samples_per_prompt,
        microbatch_rows=1,
        max_turns=150,
        sandbox_pool=sandbox_pool,
        num_env_workers=math.ceil(sandbox_pool / 24),
        num_validation_samples=0,
        num_generators=8,
        parallelism=ParallelismConfig(data_parallel_shard_degree=4),
        dump_folder="outputs/rl/qwen3_5_9b_base_terminal_bench_fast",
        enable_thinking=False,
        # The trainer waits for most of a step; more groups in flight shorten the wait.
        target_offpolicy_steps=target_offpolicy_steps,
    )


def _qwen3_5_base_terminal_bench_config(
    *,
    flavor: str,
    num_prompts_per_train_step: int,
    num_samples_per_prompt: int,
    microbatch_rows: int,
    max_turns: int,
    sandbox_pool: int,
    num_env_workers: int,
    num_validation_samples: int,
    num_generators: int,
    parallelism: ParallelismConfig,
    dump_folder: str,
    enable_thinking: bool = True,
    target_offpolicy_steps: int = 4,
) -> Controller.Config:
    """Build a Qwen3.5-Base Terminal-Bench run on Sandoq that saves resumable checkpoints.

    Args:
        enable_thinking: Qwen3.5 thinking; Terminus-2 then also sends each turn's reasoning
            back. Off, the renderer prefills an empty think block.
        microbatch_rows: Tokens per trainer microbatch, in rows of 131,072 tokens.
        sandbox_pool: Sandoq sessions the run may hold; the env server runs this many
            rollouts at once.
        target_offpolicy_steps: How many steps a group's policy may trail the trainer; the
            controller keeps (this + 1) x prompts groups in flight.
    """
    max_context_length = 131072
    model_config = build_model_config(
        flavor,
        seq_len=max_context_length,
        attn_backend="varlen",
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path=f"torchtitan/rl/example_checkpoint/Qwen3.5-{flavor}-Base",
        dump_folder=dump_folder,
        # k = the group size keeps every scored rollout; keep_errors adds the errored ones.
        rollout_recorder=RolloutSampleRecorder.Config(
            filter=KeepExtremeRewardsFilter.Config(
                k=num_samples_per_prompt, keep_errors=True
            )
        ),
        async_loop=AsyncLoopConfig(
            num_training_steps=150,
            num_prompts_per_train_step=num_prompts_per_train_step,
            num_samples_per_prompt=num_samples_per_prompt,
            target_offpolicy_steps=target_offpolicy_steps,
            windowed_fifo_batches=None,
            validation=ValidationConfig(
                num_samples=num_validation_samples, interval_steps=25, greedy=True
            ),
        ),
        rollouter=_on_sandoq(
            _terminal_bench_rollouter_config(
                train_dataset="tmax-15k@7b090eca",
                validation_dataset="tb21-78@7131e437",
                max_context_length=max_context_length,
                max_turns=max_turns,
                # The controller admits more groups than the pool holds; the excess waits
                # in the env server, not in Sandoq.
                max_concurrent_rollouts=sandbox_pool,
                num_env_workers=num_env_workers,
            ),
            interleaved_thinking=enable_thinking,
            # DOME sets dump_folder with --output-dir $DOME_TTRL_LOCAL_OUT_DIR after this
            # returns, and the env server, which writes these logs, never sees dump_folder.
            # Off DOME, --output-dir moves rollout_samples.jsonl but not these logs.
            rollout_log_dir=os.path.join(
                os.environ.get("DOME_TTRL_LOCAL_OUT_DIR", dump_folder), "rollout_logs"
            ),
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(
                enable_thinking=enable_thinking,
                thinking_retention="all",
            )
        ),
        num_generators=num_generators,
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
                "validation/launch_step",
                "validation/min_policy_version/min",
                "validation/max_policy_version/max",
                "validation/mixed_policy_rollouts/mean",
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
                num_tokens_per_microbatch_per_dp_rank=microbatch_rows
                * max_context_length,
                max_context_length=max_context_length,
                # fp32 master weights; FSDP unshards bf16 params for compute.
                dtype="float32",
                mixed_precision_param="bfloat16",
            ),
            parallelism=parallelism,
            activation_checkpoint=FullAC.Config(),
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
                num_chunks=32,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            gpu_memory_limit=0.9,
            max_num_batched_tokens=8192,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=16384,
            ),
        ),
    )
