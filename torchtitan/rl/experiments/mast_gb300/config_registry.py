# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RL recipes for two 4-GPU hosts: the trainer fills one host, one generator the other.

Checkpoints are read from ``/mnt/torchtrain_datasets/tree/<family>/<model>``; the
launcher mounts that directory on every host. W&B is off because the hosts have no
API key; TensorBoard events go to ``dump_folder``. Qwen3.5 renders in non-thinking mode:
thinking responses run ~20K tokens, so an 8K cap truncates nearly all of them.
"""

from __future__ import annotations

import dataclasses

from renderers import Qwen35RendererConfig

from torchtitan.components.renderer import from_renderers
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LMHeadCastConverter
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import model_registry as qwen3_5_model_registry
from torchtitan.models.qwen3_6 import model_registry as qwen3_6_model_registry
from torchtitan.rl.controller import Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.examples.dapo_math.config_registry import (
    rl_dapo_qwen3_4b_math_8k,
    rl_dapo_qwen3_6_35b_a3b_math_dist_moe,
)
from torchtitan.rl.examples.verifiers import VerifiersRollouter
from torchtitan.rl.experiments.verifiers.swe_rebench_v2.config_registry import (
    rl_grpo_qwen35_9b_swe_rebench_v2_smoke,
    rl_grpo_qwen36_35b_a3b_swe_rebench_v2_dist_moe,
)
from torchtitan.rl.experiments.verifiers.terminal_bench.config_registry import (
    rl_grpo_qwen35_35b_a3b_terminal_bench,
    rl_grpo_qwen35_9b_terminal_bench,
)
from torchtitan.rl.generator import VLLMCudaGraphConfig

_CHECKPOINT_ROOT = "/mnt/torchtrain_datasets/tree"
# Terminal-Bench 2.1 tasks whose images publish linux/arm64 (4 of 89), minus hf-model-inference:
# it serves Flask on the fixed port 5000, which sibling rollouts on --network=host would share.
_TB_ARM64_TASKS = ["mteb-leaderboard", "mteb-retrieve", "pytorch-model-recovery"]


def dapo_qwen35_9b() -> Controller.Config:
    """Dense Qwen3.5-9B: trainer FSDP=2 x TP=2 on host 0, generator TP=4 on host 1."""
    config = rl_dapo_qwen3_4b_math_8k()
    config.model = qwen3_5_model_registry(
        "9B",
        seq_len=config.trainer.training.max_context_length,
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    config.trainer.loss.loss_fn.global_vocab_size = decoder_vocab_size(config.model)
    config.renderer = from_renderers(Qwen35RendererConfig(enable_thinking=False))
    config.hf_assets_path = f"{_CHECKPOINT_ROOT}/qwen3_5/Qwen3.5-9B"
    config.dump_folder = "outputs/rl/dapo_qwen35_9b"
    config.async_loop.num_training_steps = 10
    config.num_generators = 1
    config.trainer.parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
    )
    config.generator.parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=4,
    )
    config.metrics.enable_wandb = False
    config.metrics.enable_tensorboard = True
    return config


def dapo_qwen36_35b_a3b() -> Controller.Config:
    """MoE Qwen3.6-35B-A3B: trainer FSDP=2 x TP=2 x EP=4, generator DP=2 x TP=2 x EP=4."""
    config = dapo_qwen35_9b()
    config.model = qwen3_6_model_registry(
        "35B-A3B",
        seq_len=config.trainer.training.max_context_length,
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    config.trainer.loss.loss_fn.global_vocab_size = decoder_vocab_size(config.model)
    config.hf_assets_path = f"{_CHECKPOINT_ROOT}/qwen3_5/Qwen3.6-35B-A3B"
    config.dump_folder = "outputs/rl/dapo_qwen36_35b_a3b"
    config.trainer.parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
    )
    config.generator.parallelism = InferenceParallelismConfig(
        data_parallel_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
    )
    return config


def dapo_qwen36_35b_a3b_h100() -> Controller.Config:
    """``dapo_qwen36_35b_a3b`` for 8-GPU H100 hosts: a bf16 FSDP=8 x EP=8 trainer.

    fp32 master weights for 35B do not fit 80 GiB GPUs at EP=4.
    """
    config = dapo_qwen36_35b_a3b()
    config.trainer.training.dtype = "bfloat16"
    config.trainer.parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        expert_parallel_degree=8,
    )
    return config


def dapo_qwen36_35b_a3b_dist_moe() -> Controller.Config:
    """``rl_dapo_qwen3_6_35b_a3b_math_dist_moe`` with the mounted checkpoint and 5 steps."""
    config = rl_dapo_qwen3_6_35b_a3b_math_dist_moe()
    config.renderer = from_renderers(Qwen35RendererConfig(enable_thinking=False))
    config.hf_assets_path = f"{_CHECKPOINT_ROOT}/qwen3_5/Qwen3.6-35B-A3B"
    config.async_loop.num_training_steps = 5
    config.metrics.enable_wandb = False
    config.metrics.enable_tensorboard = True
    return config


def tb_qwen35_35b_a3b_smoke() -> Controller.Config:
    """Terminal-Bench smoke: 2 prompts x 4 samples, trainer FSDP=2 x TP=2 x EP=4, generator EP=4.

    Reads ``TERMINAL_BENCH_{TRAIN,EVAL}_DATASET`` and ``TERMINAL_BENCH_SANDBOX`` like the
    base recipe. The Qwen3.6-35B-A3B checkpoint has the Qwen3.5-35B-A3B architecture.
    """
    config = rl_grpo_qwen35_35b_a3b_terminal_bench()
    config.hf_assets_path = f"{_CHECKPOINT_ROOT}/qwen3_5/Qwen3.6-35B-A3B"
    config.dump_folder = "outputs/rl/tb_qwen35_35b_a3b_smoke"
    config.async_loop.num_prompts_per_train_step = 2
    config.async_loop.num_samples_per_prompt = 4
    config.async_loop.validation = ValidationConfig(num_samples=0)
    config.num_generators = 1
    config.trainer.parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
    )
    config.generator.cuda_graph = VLLMCudaGraphConfig(mode="FULL")
    config.metrics.enable_wandb = False
    config.metrics.enable_tensorboard = True
    return config


def tb_qwen35_9b_docker_arm64_smoke() -> Controller.Config:
    """Terminal-Bench with task containers on host 0: one step of 2 prompts x 4 samples.

    Set ``TERMINAL_BENCH_SANDBOX=docker`` and a docker CLI on ``DOCKER_HOST``; GB300 hosts are
    aarch64, so only ``_TB_ARM64_TASKS`` run. Trainer FSDP=2 x TP=2 on host 0, one TP=4
    generator on host 1. All-zero groups are kept so a model that solves nothing still steps.
    """
    config = rl_grpo_qwen35_9b_terminal_bench()
    config.hf_assets_path = f"{_CHECKPOINT_ROOT}/qwen3_5/Qwen3.5-9B"
    config.dump_folder = "outputs/rl/tb_qwen35_9b_docker_arm64_smoke"
    config.rollouter = _only_tasks(config.rollouter, _TB_ARM64_TASKS)
    config.async_loop.num_training_steps = 1
    config.async_loop.num_prompts_per_train_step = 2
    config.async_loop.num_samples_per_prompt = 4
    config.async_loop.validation = ValidationConfig(num_samples=0)
    config.async_loop.training_sample_builder.drop_zero_std_reward_groups = False
    config.num_generators = 1
    config.trainer.parallelism = ParallelismConfig(
        data_parallel_shard_degree=2, tensor_parallel_degree=2
    )
    config.generator.parallelism = InferenceParallelismConfig(
        data_parallel_degree=1, tensor_parallel_degree=4
    )
    config.metrics.enable_wandb = False
    config.metrics.enable_tensorboard = True
    return config


def _only_tasks(
    rollouter: VerifiersRollouter.Config, tasks: list[str]
) -> VerifiersRollouter.Config:
    """Keep only these task names in the train and validation tasksets."""
    train = rollouter.train_dataset.verifiers_taskset.model_copy(
        update={"tasks": tasks}
    )
    validation = rollouter.validation_dataset.verifiers_taskset.model_copy(
        update={"tasks": tasks}
    )
    env_server = rollouter.verifiers_env_server
    return dataclasses.replace(
        rollouter,
        train_dataset=dataclasses.replace(
            rollouter.train_dataset, verifiers_taskset=train
        ),
        validation_dataset=dataclasses.replace(
            rollouter.validation_dataset, verifiers_taskset=validation
        ),
        # The env server's taskset is derived from the training one and must match it.
        verifiers_env_server=dataclasses.replace(
            env_server,
            environment=env_server.environment.model_copy(update={"taskset": train}),
        ),
    )


def swe_rebench_v2_qwen35_9b_smoke() -> Controller.Config:
    """``rl_grpo_qwen35_9b_swe_rebench_v2_smoke`` with the mounted checkpoint."""
    config = rl_grpo_qwen35_9b_swe_rebench_v2_smoke()
    config.hf_assets_path = f"{_CHECKPOINT_ROOT}/qwen3_5/Qwen3.5-9B"
    config.metrics.enable_wandb = False
    config.metrics.enable_tensorboard = True
    return config


def swe_rebench_v2_qwen36_35b_a3b_dist_moe() -> Controller.Config:
    """``rl_grpo_qwen36_35b_a3b_swe_rebench_v2_dist_moe`` with the mounted checkpoint and FULL graphs."""
    config = rl_grpo_qwen36_35b_a3b_swe_rebench_v2_dist_moe()
    config.hf_assets_path = f"{_CHECKPOINT_ROOT}/qwen3_5/Qwen3.6-35B-A3B"
    config.generator.cuda_graph = VLLMCudaGraphConfig(mode="FULL")
    config.metrics.enable_wandb = False
    config.metrics.enable_tensorboard = True
    return config
