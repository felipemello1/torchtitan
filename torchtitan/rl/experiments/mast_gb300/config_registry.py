# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO-Math recipes for two 4-GPU hosts: the trainer fills one host, one generator the other.

Checkpoints are read from ``/mnt/torchtrain_datasets/tree/<family>/<model>``; the
launcher mounts that directory on every host. W&B is off because the hosts have no
API key; TensorBoard events go to ``dump_folder``. Qwen3.5 renders in non-thinking mode:
thinking responses run ~20K tokens, so an 8K cap truncates nearly all of them.
"""

from __future__ import annotations

from renderers import Qwen35RendererConfig

from torchtitan.components.renderer import from_renderers
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LMHeadCastConverter
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import model_registry as qwen3_5_model_registry
from torchtitan.models.qwen3_6 import model_registry as qwen3_6_model_registry
from torchtitan.rl.controller import Controller
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.examples.dapo_math.config_registry import (
    rl_dapo_qwen3_4b_math_8k,
    rl_dapo_qwen3_6_35b_a3b_math_dist_moe,
)

_CHECKPOINT_ROOT = "/mnt/torchtrain_datasets/tree"


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
