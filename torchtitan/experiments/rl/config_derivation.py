# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capacity derivation for the RL replay pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.config import ParallelismConfig


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _trainer_dp_degree(parallelism: ParallelismConfig) -> int:
    dp_shard = max(parallelism.data_parallel_shard_degree, 1)
    return parallelism.data_parallel_replicate_degree * dp_shard


def compute_world_size(parallelism: ParallelismConfig) -> int:
    """Compute world size from all parallel dimensions."""
    dp_shard = max(parallelism.data_parallel_shard_degree, 1)
    return (
        parallelism.data_parallel_replicate_degree
        * dp_shard
        * parallelism.tensor_parallel_degree
        * parallelism.pipeline_parallel_degree
        * parallelism.context_parallel_degree
    )


@dataclass(kw_only=True, slots=True)
class AsyncPipelineConfig:
    """Capacity overrides for async rollout producers and replay."""

    rollout_concurrency: int | None = None
    """Prompt groups concurrently producing rollouts. ``None`` derives from batch size."""

    replay_buffer_samples: int | None = None
    """Replay FIFO capacity in samples. ``None`` derives from batch size."""

    max_admitted_generation_prompts: int | None = None
    """Controller-side cap on generation prompts admitted to vLLM."""


@dataclass(frozen=True, slots=True)
class DerivedRLConfig:
    """Resolved capacity view computed from ``BatchConfig`` and DP degree."""

    global_batch_rows: int
    local_batch_rows: int
    seq_len: int
    trainer_dp_degree: int
    gradient_accumulation_steps: int
    rollout_token_target: int
    rollout_concurrency: int
    replay_buffer_samples: int
    max_admitted_generation_prompts: int


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def derived_capacity(cfg: "RLTrainer.Config") -> DerivedRLConfig:
    """Resolve async replay capacity from the user config.

    Example::

        batch = cfg.batcher.batch
        derived = derived_capacity(cfg)
        # derived.rollout_token_target == derived.global_batch_rows * batch.seq_len
    """
    batch = cfg.batcher.batch
    dp = _trainer_dp_degree(cfg.trainer.parallelism)
    _validate_positive("BatchConfig.local_batch_size", batch.local_batch_size)
    _validate_positive("BatchConfig.seq_len", batch.seq_len)
    _validate_positive("trainer_dp_degree", dp)

    global_batch_rows = (
        batch.global_batch_size
        if batch.global_batch_size > 0
        else batch.local_batch_size * dp
    )
    rows_per_grad_accum = batch.local_batch_size * dp
    if global_batch_rows <= 0:
        raise ValueError(f"global_batch_rows must be positive, got {global_batch_rows}")
    if global_batch_rows % rows_per_grad_accum != 0:
        raise ValueError(
            "BatchConfig.global_batch_size must be divisible by "
            "local_batch_size * trainer_dp_degree; got "
            f"global_batch_size={global_batch_rows}, "
            f"local_batch_size={batch.local_batch_size}, dp={dp}"
        )
    gradient_accumulation_steps = global_batch_rows // rows_per_grad_accum
    rollout_token_target = global_batch_rows * batch.seq_len

    prompt_groups_per_batch = _ceil_div(global_batch_rows, cfg.group_size)
    default_rollout_concurrency = prompt_groups_per_batch * (
        cfg.max_offpolicy_steps + 1
    )
    rollout_concurrency = (
        cfg.async_pipeline.rollout_concurrency
        if cfg.async_pipeline.rollout_concurrency is not None
        else default_rollout_concurrency
    )

    replay_buffer_samples = (
        cfg.async_pipeline.replay_buffer_samples
        if cfg.async_pipeline.replay_buffer_samples is not None
        else max(global_batch_rows, rollout_concurrency * cfg.group_size)
    )

    default_max_admitted = max(
        rollout_concurrency * cfg.group_size,
        cfg.num_validation_samples,
    )
    max_admitted_generation_prompts = (
        cfg.async_pipeline.max_admitted_generation_prompts
        if cfg.async_pipeline.max_admitted_generation_prompts is not None
        else default_max_admitted
    )

    return DerivedRLConfig(
        global_batch_rows=global_batch_rows,
        local_batch_rows=batch.local_batch_size,
        seq_len=batch.seq_len,
        trainer_dp_degree=dp,
        gradient_accumulation_steps=gradient_accumulation_steps,
        rollout_token_target=rollout_token_target,
        rollout_concurrency=rollout_concurrency,
        replay_buffer_samples=replay_buffer_samples,
        max_admitted_generation_prompts=max_admitted_generation_prompts,
    )


def compute_generator_max_num_seqs(cfg: "RLTrainer.Config") -> int:
    """Resolve the vLLM running-sequence cap for the generator actor."""
    derived = cfg.derived
    return max(derived.max_admitted_generation_prompts, cfg.num_validation_samples)


def format_resolved_config(cfg: "RLTrainer.Config") -> str:
    """Format the resolved async replay capacity view."""
    derived = cfg.derived
    batch = cfg.batcher.batch
    global_batch_origin = (
        "user-set"
        if batch.global_batch_size > 0
        else "local_batch_size * trainer_dp_degree"
    )

    rows = [
        ("batch.global_batch_size", f"{derived.global_batch_rows} rows"),
        ("batch.global_batch_origin", global_batch_origin),
        ("batch.local_batch_size", f"{derived.local_batch_rows} rows"),
        ("batch.seq_len", f"{derived.seq_len} tokens"),
        ("trainer_dp_degree", str(derived.trainer_dp_degree)),
        ("gradient_accumulation_steps", str(derived.gradient_accumulation_steps)),
        ("rollout_token_target", f"{derived.rollout_token_target} tokens"),
        ("rollout_concurrency", f"{derived.rollout_concurrency} groups"),
        ("replay_buffer_samples", f"{derived.replay_buffer_samples} samples"),
        (
            "max_admitted_generation_prompts",
            f"{derived.max_admitted_generation_prompts} prompts",
        ),
        ("num_generator_instances", str(cfg.num_generator_instances)),
        ("max_offpolicy_steps", str(cfg.max_offpolicy_steps)),
    ]

    lines = ["[RL config resolved]"]
    lines.extend(f"  {key:<34} = {value}" for key, value in rows)
    return "\n".join(lines)
