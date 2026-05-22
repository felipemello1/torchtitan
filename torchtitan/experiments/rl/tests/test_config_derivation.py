# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for derived RL async replay capacity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from torchtitan.config import BatchConfig, ConfigManager, ParallelismConfig
from torchtitan.experiments.rl.config_derivation import (
    AsyncPipelineConfig,
    compute_generator_max_num_seqs,
    derived_capacity,
    format_resolved_config,
)


def _cfg(
    *,
    batch: BatchConfig | None = None,
    parallelism: ParallelismConfig | None = None,
    max_microbatch_samples: int | None = None,
    group_size: int = 4,
    max_offpolicy_steps: int = 1,
    async_pipeline: AsyncPipelineConfig | None = None,
    num_validation_prompts: int = 8,
    num_generator_instances: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        batcher=SimpleNamespace(
            batch=batch
            or BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=16)
        ),
        trainer=SimpleNamespace(
            parallelism=parallelism or ParallelismConfig(),
            max_microbatch_samples=max_microbatch_samples,
        ),
        group_size=group_size,
        max_offpolicy_steps=max_offpolicy_steps,
        async_pipeline=async_pipeline or AsyncPipelineConfig(),
        num_validation_prompts=num_validation_prompts,
        num_generator_instances=num_generator_instances,
    )


def test_derived_capacity_defaults_from_batch_and_pipeline() -> None:
    cfg = _cfg()

    derived = derived_capacity(cfg)

    assert derived.global_batch_rows == 8
    assert derived.local_batch_rows == 2
    assert derived.trainer_microbatch_rows is None
    assert derived.seq_len == 16
    assert derived.trainer_dp_degree == 1
    assert derived.gradient_accumulation_steps == 4
    assert derived.rollout_token_target == 128
    assert derived.prompt_groups_per_batch == 2
    assert derived.rollout_concurrency_groups == 4
    assert derived.replay_buffer_samples == 16
    assert derived.max_admitted_generation_prompts == 16


def test_derived_capacity_uses_explicit_async_overrides() -> None:
    cfg = _cfg(
        async_pipeline=AsyncPipelineConfig(
            rollout_concurrency_groups=3,
            replay_buffer_samples=11,
            max_admitted_generation_prompts=13,
        )
    )

    derived = derived_capacity(cfg)

    assert derived.rollout_concurrency_groups == 3
    assert derived.replay_buffer_samples == 11
    assert derived.max_admitted_generation_prompts == 13


def test_derived_capacity_derives_global_batch_from_local_batch_and_dp() -> None:
    cfg = _cfg(
        batch=BatchConfig(local_batch_size=3, global_batch_size=-1, seq_len=7),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=2,
        ),
        group_size=5,
    )

    derived = derived_capacity(cfg)

    assert derived.trainer_dp_degree == 4
    assert derived.global_batch_rows == 12
    assert derived.gradient_accumulation_steps == 1
    assert derived.rollout_token_target == 84


def test_derived_capacity_ceil_rounds_microbatch_count() -> None:
    cfg = _cfg(batch=BatchConfig(local_batch_size=3, global_batch_size=10))

    derived = derived_capacity(cfg)

    assert derived.gradient_accumulation_steps == 4


def test_derived_capacity_uses_trainer_microbatch_samples() -> None:
    cfg = _cfg(
        batch=BatchConfig(local_batch_size=8, global_batch_size=128),
        max_microbatch_samples=16,
    )

    derived = derived_capacity(cfg)

    assert derived.trainer_microbatch_rows == 16
    assert derived.gradient_accumulation_steps == 8


def test_compute_generator_max_num_seqs_uses_derived_admission_cap() -> None:
    cfg = _cfg(
        async_pipeline=AsyncPipelineConfig(max_admitted_generation_prompts=17),
        num_validation_prompts=5,
    )
    cfg.derived = derived_capacity(cfg)

    assert compute_generator_max_num_seqs(cfg) == 17


def test_format_resolved_config_shows_batchconfig_origin() -> None:
    cfg = _cfg(
        batch=BatchConfig(local_batch_size=2, global_batch_size=-1, seq_len=16),
        parallelism=ParallelismConfig(data_parallel_replicate_degree=2),
    )
    cfg.derived = derived_capacity(cfg)

    formatted = format_resolved_config(cfg)

    assert "batch.global_batch_size" in formatted
    assert "= 4 rows" in formatted
    assert "batch.global_batch_origin" in formatted
    assert "= local_batch_size * trainer_dp_degree" in formatted
    assert "trainer.max_microbatch_samples" in formatted
    assert "prompt_groups_per_batch" in formatted
    assert "rollout_concurrency_groups" in formatted
    assert "num_validation_prompts" in formatted


@pytest.mark.parametrize(
    ("override", "migration"),
    [
        ("--num_prompts_per_step=5", "BatchConfig"),
        ("--rollout_group_size=8", "group_size"),
        ("--num_validation_samples=20", "num_validation_prompts"),
        ("--async_rollout_groups=4", "rollout_concurrency_groups"),
        (
            "--async_pipeline.rollout_concurrency=4",
            "async_pipeline.rollout_concurrency_groups",
        ),
        ("--replay_buffer_groups=8", "replay_buffer_samples"),
        (
            "--max_admitted_generation_prompts=32",
            "async_pipeline.max_admitted_generation_prompts",
        ),
    ],
)
def test_removed_rl_capacity_overrides_raise_migration_message(
    override: str,
    migration: str,
) -> None:
    with pytest.raises(ValueError, match=migration):
        ConfigManager().parse_args(
            ["--module", "rl", "--config", "rl_grpo_qwen3_0_6b", override]
        )
