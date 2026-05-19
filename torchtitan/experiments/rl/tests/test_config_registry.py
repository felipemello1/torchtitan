# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import pytest

from torchtitan.experiments.rl.config_registry import (
    rl_dapo_qwen3_1_7b_alphabet_sort_2gpu,
    rl_dapo_qwen3_4b_alphabet_sort,
    rl_dapo_qwen3_4b_alphabet_sort_2gpu,
    rl_grpo_qwen3_0_6b_alphabet_sort,
    rl_grpo_qwen3_1_7b,
    rl_grpo_qwen3_1_7b_alphabet_sort,
    rl_grpo_qwen3_4b_alphabet_sort,
)
from torchtitan.experiments.rl.loss import DAPOLoss


def test_1_7b_configs_disable_compile():
    assert not rl_grpo_qwen3_1_7b().compile.enable
    cfg = rl_grpo_qwen3_1_7b_alphabet_sort()
    assert not cfg.compile.enable
    assert cfg.trainer.max_microbatch_samples == 4
    assert cfg.trainer.lr_scheduler.warmup_steps == 0
    assert cfg.trainer.lr_scheduler.decay_ratio == 0.0
    assert cfg.generator.sampling.top_p == 1.0


def test_alphabet_sort_1_7b_dapo_2gpu_config_stays_on_two_gpus():
    cfg = rl_dapo_qwen3_1_7b_alphabet_sort_2gpu()

    assert cfg.model_spec.flavor == "1.7B"
    assert cfg.num_steps == 50
    assert cfg.num_prompts_per_step == 4
    assert cfg.async_rollout_groups == 4
    assert cfg.replay_buffer_groups == 8
    assert cfg.trainer.parallelism.tensor_parallel_degree == 1
    assert cfg.generator.parallelism.tensor_parallel_degree == 1
    assert cfg.generator.sampling.max_tokens == 512
    assert not cfg.compile.enable
    assert cfg.trainer.max_microbatch_samples == 4
    assert cfg.trainer.lr_scheduler.warmup_steps == 0
    assert cfg.trainer.lr_scheduler.decay_ratio == 0.0
    assert cfg.generator.sampling.top_p == 1.0
    assert isinstance(cfg.trainer.loss, DAPOLoss.Config)


def test_alphabet_sort_0_6b_defaults_keep_compile_and_disable_thinking():
    cfg = rl_grpo_qwen3_0_6b_alphabet_sort()

    assert cfg.compile.enable
    assert cfg.renderer.name == "qwen3"
    assert cfg.renderer.enable_thinking is False
    assert cfg.generator.sampling.top_p == 1.0
    assert cfg.trainer.max_microbatch_samples == 8
    assert cfg.trainer.lr_scheduler.warmup_steps == 0
    assert cfg.trainer.lr_scheduler.decay_ratio == 0.0
    assert cfg.train_dataset.max_turns == 3
    assert cfg.train_dataset.max_names_per_turn == 4


def test_alphabet_sort_4b_uses_8_gpu_async_recipe():
    cfg = rl_grpo_qwen3_4b_alphabet_sort()

    assert (
        cfg.hf_assets_path
        == "torchtitan/experiments/rl/example_checkpoint/Qwen3-4B-Instruct-2507"
    )
    assert cfg.model_spec.flavor == "4B-Instruct-2507"
    assert cfg.model_spec.model.rope.max_seq_len == 262144
    assert cfg.model_spec.model.rope.theta == 5000000.0
    assert not cfg.compile.enable
    assert cfg.num_steps == 100
    assert cfg.num_prompts_per_step == 16
    assert cfg.rollout_group_size == 8
    assert cfg.async_rollout_groups == 16
    assert cfg.replay_buffer_groups == 32
    assert cfg.max_offpolicy_steps == 1
    assert cfg.num_validation_samples == 64
    assert cfg.trainer.optimizer.lr == 5e-7
    assert cfg.trainer.max_microbatch_samples == 1
    assert cfg.trainer.lr_scheduler.warmup_steps == 0
    assert cfg.trainer.lr_scheduler.decay_ratio == 0.0
    assert cfg.trainer.parallelism.tensor_parallel_degree == 4
    assert cfg.generator.parallelism.tensor_parallel_degree == 4
    assert cfg.generator.gpu_memory_limit == 0.85
    assert cfg.generator.sampling.temperature == 1.0
    assert cfg.generator.sampling.top_p == 1.0
    assert cfg.generator.sampling.max_tokens == 768


def test_alphabet_sort_4b_dapo_uses_clip_high_recipe():
    cfg = rl_dapo_qwen3_4b_alphabet_sort()

    assert cfg.num_steps == 100
    assert isinstance(cfg.trainer.loss, DAPOLoss.Config)
    assert cfg.trainer.loss.clip_low == 0.2
    assert cfg.trainer.loss.clip_high == 0.28
    assert cfg.trainer.loss.dual_clip_c == 3.0


def test_alphabet_sort_4b_dapo_2gpu_config_stays_on_two_gpus():
    cfg = rl_dapo_qwen3_4b_alphabet_sort_2gpu()

    assert cfg.model_spec.flavor == "4B-Instruct-2507"
    assert cfg.num_steps == 25
    assert cfg.num_prompts_per_step == 4
    assert cfg.async_rollout_groups == 4
    assert cfg.replay_buffer_groups == 8
    assert cfg.trainer.parallelism.tensor_parallel_degree == 1
    assert cfg.generator.parallelism.tensor_parallel_degree == 1
    assert cfg.trainer.lr_scheduler.warmup_steps == 0
    assert cfg.trainer.lr_scheduler.decay_ratio == 0.0
    assert cfg.generator.sampling.temperature == 1.0
    assert cfg.generator.sampling.top_p == 1.0


def test_alphabet_sort_configs_reject_unsupported_top_p():
    cfg = rl_dapo_qwen3_1_7b_alphabet_sort_2gpu()
    generator = dataclasses.replace(
        cfg.generator,
        sampling=dataclasses.replace(cfg.generator.sampling, top_p=0.95),
    )

    with pytest.raises(ValueError, match="top_p=1.0"):
        dataclasses.replace(cfg, generator=generator)
