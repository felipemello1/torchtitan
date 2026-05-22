# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config entry points for the RL/unified experiment.

Each function returns a complete ``RLTrainer.Config`` and is discoverable by
``ConfigManager`` via ``--module rl --config <function_name>``.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    BatchConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.alphabet_sort import (
    AlphabetSortBuilder,
    AlphabetSortDataset,
)
from torchtitan.experiments.rl.config_derivation import AsyncPipelineConfig
from torchtitan.experiments.rl.grpo import Batcher, RLTrainer
from torchtitan.experiments.rl.loss import DAPOLoss
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.sum_digits import SumDigitsBuilder, SumDigitsDataset
from torchtitan.models.qwen3 import model_registry


def _alphabet_sort_dataset(seed: int) -> AlphabetSortDataset.Config:
    return AlphabetSortDataset.Config(
        seed=seed,
        min_turns=3,
        max_turns=5,
        min_names_per_turn=1,
        max_names_per_turn=4,
    )


def _alphabet_sort_builder() -> AlphabetSortBuilder.Config:
    return AlphabetSortBuilder.Config(
        similarity_power=4,
        power_per_turn=False,
    )


def _alphabet_sort_config(
    *,
    model_size: str,
    hf_assets_path: str,
    lr: float,
    batch: BatchConfig,
    num_validation_prompts: int,
    num_steps: int = 50,
    trainer_tensor_parallel_degree: int = 2,
    generator_tensor_parallel_degree: int = 2,
    group_size: int = 8,
    num_generator_instances: int = 1,
    max_offpolicy_steps: int = 1,
    async_pipeline: AsyncPipelineConfig | None = None,
    compile: CompileConfig | None = None,
    trainer_max_microbatch_samples: int | None = 8,
) -> RLTrainer.Config:
    return RLTrainer.Config(
        model_spec=model_registry(model_size, attn_backend="varlen"),
        hf_assets_path=hf_assets_path,
        num_steps=num_steps,
        group_size=group_size,
        max_rollout_turns=5,
        num_validation_prompts=num_validation_prompts,
        save_rollout_samples=True,
        num_generator_instances=num_generator_instances,
        max_offpolicy_steps=max_offpolicy_steps,
        async_pipeline=async_pipeline or AsyncPipelineConfig(),
        compile=compile
        if compile is not None
        else CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=_alphabet_sort_dataset(seed=142857),
        train_env_builder=_alphabet_sort_builder(),
        validation_dataset=_alphabet_sort_dataset(seed=314159),
        validation_env_builder=_alphabet_sort_builder(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(batch=batch),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=lr),
            max_microbatch_samples=trainer_max_microbatch_samples,
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                decay_ratio=0.0,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=trainer_tensor_parallel_degree,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=25,
                last_save_model_only=False,
            ),
            loss=DAPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            gpu_memory_limit=0.85,
            parallelism=ParallelismConfig(
                tensor_parallel_degree=generator_tensor_parallel_degree,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=768,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        group_size=group_size,
        num_validation_prompts=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_env_builder=SumDigitsBuilder.Config(
            correctness_reward=1.0, format_reward=0.3
        ),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_env_builder=SumDigitsBuilder.Config(
            correctness_reward=1.0, format_reward=0.3
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=DAPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=1.0,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=10,
        group_size=group_size,
        num_validation_prompts=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_env_builder=SumDigitsBuilder.Config(
            correctness_reward=1.0, format_reward=0.3
        ),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_env_builder=SumDigitsBuilder.Config(
            correctness_reward=1.0, format_reward=0.3
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=DAPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=1.0,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_14b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-14B (16 GPUs: 8 gen + 8 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("14B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-14B",
        num_steps=10,
        group_size=group_size,
        num_validation_prompts=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_env_builder=SumDigitsBuilder.Config(
            correctness_reward=1.0, format_reward=0.3
        ),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_env_builder=SumDigitsBuilder.Config(
            correctness_reward=1.0, format_reward=0.3
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=8,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=DAPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=8,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=1.0,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_batch_invariant() -> RLTrainer.Config:
    """On-policy GRPO config for Qwen3-0.6B under same parallelism (4 GPUs: 2 gen + 2 train).

    Enables deterministic + batch-invariant mode for true on-policy RL training.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        group_size=group_size,
        num_validation_prompts=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_env_builder=SumDigitsBuilder.Config(
            correctness_reward=1.0, format_reward=0.3
        ),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_env_builder=SumDigitsBuilder.Config(
            correctness_reward=1.0, format_reward=0.3
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            # bfloat16 is needed for trainer to align with generator dtype
            # TODO: replace bfloat16 enablement with FSDP2+TP2
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            debug=batch_invariant_config,
            loss=DAPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=1.0,
                max_tokens=100,
            ),
            debug=batch_invariant_config,
        ),
    )


def rl_dapo_qwen3_0_6b_alphabet_sort() -> RLTrainer.Config:
    """Multi-turn AlphabetSort DAPO config for Qwen3-0.6B."""
    return _alphabet_sort_config(
        model_size="0.6B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        lr=1e-6,
        batch=BatchConfig(local_batch_size=8, global_batch_size=64, seq_len=2048),
        num_validation_prompts=32,
    )


def rl_dapo_qwen3_1_7b_alphabet_sort_2gpu() -> RLTrainer.Config:
    """Two-GPU AlphabetSort DAPO config for Qwen3-1.7B."""
    return _alphabet_sort_config(
        model_size="1.7B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        lr=1e-6,
        batch=BatchConfig(local_batch_size=4, global_batch_size=128, seq_len=2048),
        num_validation_prompts=16,
        trainer_tensor_parallel_degree=1,
        generator_tensor_parallel_degree=1,
        trainer_max_microbatch_samples=4,
        compile=CompileConfig(enable=False),
    )


def rl_dapo_qwen3_1_7b_alphabet_sort_3gpu_multigen() -> RLTrainer.Config:
    """Three-GPU AlphabetSort DAPO config with two generator instances."""
    return _alphabet_sort_config(
        model_size="1.7B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        lr=1e-6,
        batch=BatchConfig(local_batch_size=4, global_batch_size=128, seq_len=2048),
        num_validation_prompts=16,
        trainer_tensor_parallel_degree=1,
        generator_tensor_parallel_degree=1,
        num_generator_instances=2,
        trainer_max_microbatch_samples=4,
        compile=CompileConfig(enable=False),
    )


def rl_dapo_qwen3_1_7b_alphabet_sort_2gpu_acceptance() -> RLTrainer.Config:
    """Reward-up acceptance recipe for Qwen3-1.7B AlphabetSort."""
    return _alphabet_sort_config(
        model_size="1.7B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        lr=1e-5,
        num_steps=100,
        batch=BatchConfig(local_batch_size=8, global_batch_size=128, seq_len=2048),
        num_validation_prompts=64,
        trainer_tensor_parallel_degree=1,
        generator_tensor_parallel_degree=1,
        async_pipeline=AsyncPipelineConfig(rollout_concurrency_groups=16),
        trainer_max_microbatch_samples=16,
        compile=CompileConfig(enable=False),
    )
