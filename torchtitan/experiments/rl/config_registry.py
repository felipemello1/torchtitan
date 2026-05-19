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
from torchtitan.experiments.rl.grpo import RLTrainer
from torchtitan.experiments.rl.loss import DAPOLoss, GRPOLoss
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.sum_digits import SumDigitsBuilder, SumDigitsDataset
from torchtitan.models.qwen3 import model_registry


def _alphabet_sort_dataset(seed: int) -> AlphabetSortDataset.Config:
    """Shared AlphabetSort task parameters for train/validation splits."""
    return AlphabetSortDataset.Config(
        seed=seed,
        min_turns=3,
        max_turns=3,
        min_names_per_turn=1,
        max_names_per_turn=4,
    )


def _alphabet_sort_builder() -> AlphabetSortBuilder.Config:
    """Shared AlphabetSort scoring parameters."""
    return AlphabetSortBuilder.Config(
        similarity_power=8,
        power_per_turn=False,
    )


def _dapo_loss() -> DAPOLoss.Config:
    """DAPO clip-higher and dual-clip settings used by explicit DAPO configs."""
    return DAPOLoss.Config(clip_low=0.2, clip_high=0.28, dual_clip_c=3.0)


def _alphabet_sort_config(
    *,
    model_size: str,
    hf_assets_path: str,
    lr: float,
    num_prompts_per_step: int,
    num_validation_samples: int,
    num_steps: int = 50,
    trainer_tensor_parallel_degree: int = 2,
    generator_tensor_parallel_degree: int = 2,
    async_rollout_groups: int = 2,
    replay_buffer_groups: int | None = None,
    generator_max_tokens: int = 256,
    generator_gpu_memory_limit: float = 0.75,
    generator_temperature: float = 0.8,
    generator_top_p: float = 1.0,
    compile_config: CompileConfig | None = None,
    trainer_max_microbatch_samples: int | None = 8,
    loss: GRPOLoss.Config | DAPOLoss.Config | None = None,
) -> RLTrainer.Config:
    """Shared AlphabetSort config with explicit trainer/generator TP sizing."""
    group_size = 8
    if replay_buffer_groups is None:
        replay_buffer_groups = num_prompts_per_step
    return RLTrainer.Config(
        model_spec=model_registry(model_size, attn_backend="varlen"),
        hf_assets_path=hf_assets_path,
        num_steps=num_steps,
        num_prompts_per_step=num_prompts_per_step,
        rollout_group_size=group_size,
        num_validation_samples=num_validation_samples,
        max_rollout_turns=5,
        max_trajectory_tokens=2048,
        async_rollout_groups=async_rollout_groups,
        replay_buffer_groups=replay_buffer_groups,
        max_offpolicy_steps=1,
        compile=(
            compile_config
            if compile_config is not None
            else CompileConfig(enable=True, backend="aot_eager")
        ),
        train_dataset=_alphabet_sort_dataset(seed=142857),
        train_env_builder=_alphabet_sort_builder(),
        validation_dataset=_alphabet_sort_dataset(seed=314159),
        validation_env_builder=_alphabet_sort_builder(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
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
            loss=loss if loss is not None else GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            gpu_memory_limit=generator_gpu_memory_limit,
            parallelism=ParallelismConfig(
                tensor_parallel_degree=generator_tensor_parallel_degree,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=generator_temperature,
                top_p=generator_top_p,
                max_tokens=generator_max_tokens,
            ),
        ),
    )


def _sum_digits_smoke_config(
    *,
    model_size: str,
    hf_assets_path: str,
    lr: float,
    trainer_tensor_parallel_degree: int,
    generator_tensor_parallel_degree: int,
    trainer_dtype: str | None = None,
    trainer_enable_sequence_parallel: bool | None = None,
    trainer_debug: DebugConfig | None = None,
    generator_debug: DebugConfig | None = None,
    generator_data_parallel_shard_degree: int | None = None,
    compile_config: CompileConfig | None = None,
) -> RLTrainer.Config:
    """Shared 10-step SumDigits smoke config."""
    group_size = 8
    trainer_parallelism_kwargs = {}
    if trainer_enable_sequence_parallel is not None:
        trainer_parallelism_kwargs[
            "enable_sequence_parallel"
        ] = trainer_enable_sequence_parallel
    generator_parallelism_kwargs = {}
    if generator_data_parallel_shard_degree is not None:
        generator_parallelism_kwargs[
            "data_parallel_shard_degree"
        ] = generator_data_parallel_shard_degree
    trainer_kwargs = {}
    if trainer_debug is not None:
        trainer_kwargs["debug"] = trainer_debug
    generator_kwargs = {}
    if generator_debug is not None:
        generator_kwargs["debug"] = generator_debug

    return RLTrainer.Config(
        model_spec=model_registry(model_size, attn_backend="varlen"),
        hf_assets_path=hf_assets_path,
        num_steps=10,
        num_prompts_per_step=5,
        rollout_group_size=group_size,
        num_validation_samples=20,
        compile=(
            compile_config
            if compile_config is not None
            else CompileConfig(enable=True, backend="aot_eager")
        ),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_env_builder=SumDigitsBuilder.Config(
            correctness_reward=1.0,
            format_reward=0.3,
        ),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_env_builder=SumDigitsBuilder.Config(
            correctness_reward=1.0,
            format_reward=0.3,
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=lr),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=(
                TrainingConfig(dtype=trainer_dtype)
                if trainer_dtype is not None
                else TrainingConfig()
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=trainer_tensor_parallel_degree,
                disable_loss_parallel=True,
                **trainer_parallelism_kwargs,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
            **trainer_kwargs,
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=generator_tensor_parallel_degree,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
                **generator_parallelism_kwargs,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=1.0,
                max_tokens=100,
            ),
            **generator_kwargs,
        ),
    )


def rl_grpo_qwen3_0_6b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    return _sum_digits_smoke_config(
        model_size="0.6B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        lr=2e-6,
        trainer_tensor_parallel_degree=2,
        generator_tensor_parallel_degree=4,
    )


def rl_grpo_qwen3_0_6b_alphabet_sort() -> RLTrainer.Config:
    """Multi-turn AlphabetSort GRPO config for Qwen3-0.6B (4 GPUs)."""
    return _alphabet_sort_config(
        model_size="0.6B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        lr=1e-6,
        num_prompts_per_step=8,
        num_validation_samples=32,
    )


def rl_dapo_qwen3_0_6b_alphabet_sort() -> RLTrainer.Config:
    """Multi-turn AlphabetSort DAPO config for Qwen3-0.6B (4 GPUs)."""
    return _alphabet_sort_config(
        model_size="0.6B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        lr=1e-6,
        num_prompts_per_step=8,
        num_validation_samples=32,
        loss=_dapo_loss(),
    )


def rl_grpo_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train)."""
    return _sum_digits_smoke_config(
        model_size="1.7B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        lr=2e-6,
        trainer_tensor_parallel_degree=2,
        generator_tensor_parallel_degree=4,
        generator_data_parallel_shard_degree=1,
        compile_config=CompileConfig(enable=False),
    )


def rl_grpo_qwen3_1_7b_alphabet_sort() -> RLTrainer.Config:
    """Multi-turn AlphabetSort GRPO config for Qwen3-1.7B (4 GPUs)."""
    return _alphabet_sort_config(
        model_size="1.7B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        lr=1e-6,
        num_prompts_per_step=4,
        num_validation_samples=16,
        compile_config=CompileConfig(enable=False),
        trainer_max_microbatch_samples=4,
    )


def rl_dapo_qwen3_1_7b_alphabet_sort_2gpu() -> RLTrainer.Config:
    """Two-GPU AlphabetSort DAPO config for Qwen3-1.7B (1 gen + 1 train)."""
    return _alphabet_sort_config(
        model_size="1.7B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        lr=1e-6,
        num_steps=50,
        num_prompts_per_step=16,
        num_validation_samples=16,
        trainer_tensor_parallel_degree=1,
        generator_tensor_parallel_degree=1,
        async_rollout_groups=16,
        replay_buffer_groups=32,
        generator_max_tokens=512,
        compile_config=CompileConfig(enable=False),
        trainer_max_microbatch_samples=4,
        loss=_dapo_loss(),
    )


def _alphabet_sort_4b_config(
    *, loss: GRPOLoss.Config | DAPOLoss.Config | None = None
) -> RLTrainer.Config:
    """Eight-GPU AlphabetSort config for Qwen3-4B (4 gen + 4 train)."""
    return _alphabet_sort_config(
        model_size="4B-Instruct-2507",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-4B-Instruct-2507",
        lr=5e-7,
        num_steps=100,
        num_prompts_per_step=16,
        num_validation_samples=64,
        trainer_tensor_parallel_degree=4,
        generator_tensor_parallel_degree=4,
        async_rollout_groups=16,
        replay_buffer_groups=32,
        generator_max_tokens=768,
        generator_gpu_memory_limit=0.85,
        generator_temperature=1.0,
        generator_top_p=1.0,
        compile_config=CompileConfig(enable=False),
        trainer_max_microbatch_samples=1,
        loss=loss,
    )


def rl_grpo_qwen3_4b_alphabet_sort() -> RLTrainer.Config:
    """Eight-GPU AlphabetSort GRPO config for Qwen3-4B (4 gen + 4 train)."""
    return _alphabet_sort_4b_config()


def rl_dapo_qwen3_4b_alphabet_sort() -> RLTrainer.Config:
    """Eight-GPU AlphabetSort DAPO config for Qwen3-4B (4 gen + 4 train)."""
    return _alphabet_sort_4b_config(loss=_dapo_loss())


def diagnostic_rl_dapo_qwen3_4b_alphabet_sort_2gpu() -> RLTrainer.Config:
    """Diagnostic Qwen3-4B DAPO config.

    This is known to hit a direct-sync timeout on this devserver. Keep it for
    model-sync investigation, not as the accepted runnable recipe.
    """
    return _alphabet_sort_config(
        model_size="4B-Instruct-2507",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-4B-Instruct-2507",
        lr=3e-7,
        num_steps=25,
        num_prompts_per_step=4,
        num_validation_samples=16,
        trainer_tensor_parallel_degree=1,
        generator_tensor_parallel_degree=1,
        async_rollout_groups=4,
        replay_buffer_groups=8,
        generator_max_tokens=768,
        generator_gpu_memory_limit=0.80,
        generator_temperature=1.0,
        generator_top_p=1.0,
        compile_config=CompileConfig(enable=False),
        trainer_max_microbatch_samples=1,
        loss=_dapo_loss(),
    )


def rl_grpo_qwen3_14b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-14B (16 GPUs: 8 gen + 8 train)."""
    return _sum_digits_smoke_config(
        model_size="14B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-14B",
        lr=1e-6,
        trainer_tensor_parallel_degree=8,
        generator_tensor_parallel_degree=8,
        trainer_dtype="bfloat16",
    )


def rl_grpo_qwen3_0_6b_batch_invariant() -> RLTrainer.Config:
    """On-policy GRPO config for Qwen3-0.6B under same parallelism (4 GPUs: 2 gen + 2 train).

    Enables deterministic + batch-invariant mode for true on-policy RL training.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    return _sum_digits_smoke_config(
        model_size="0.6B",
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        lr=2e-6,
        trainer_tensor_parallel_degree=2,
        generator_tensor_parallel_degree=2,
        # bfloat16 keeps trainer and generator dtype behavior aligned for this
        # batch-invariant debugging config.
        trainer_dtype="bfloat16",
        trainer_enable_sequence_parallel=False,
        trainer_debug=batch_invariant_config,
        generator_debug=batch_invariant_config,
    )
