# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for derived RL async replay capacity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from torchtitan.config import BatchConfig, ParallelismConfig
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
    group_size: int = 4,
    max_offpolicy_steps: int = 1,
    async_pipeline: AsyncPipelineConfig | None = None,
    num_validation_samples: int = 8,
    num_generator_instances: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        batcher=SimpleNamespace(
            batch=batch
            or BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=16)
        ),
        trainer=SimpleNamespace(parallelism=parallelism or ParallelismConfig()),
        group_size=group_size,
        max_offpolicy_steps=max_offpolicy_steps,
        async_pipeline=async_pipeline or AsyncPipelineConfig(),
        num_validation_samples=num_validation_samples,
        num_generator_instances=num_generator_instances,
    )


def test_derived_capacity_defaults_from_batch_and_pipeline() -> None:
    cfg = _cfg()

    derived = derived_capacity(cfg)

    assert derived.global_batch_rows == 8
    assert derived.local_batch_rows == 2
    assert derived.seq_len == 16
    assert derived.trainer_dp_degree == 1
    assert derived.gradient_accumulation_steps == 4
    assert derived.rollout_token_target == 128
    assert derived.rollout_concurrency == 4
    assert derived.replay_buffer_samples == 16
    assert derived.max_admitted_generation_prompts == 16


def test_derived_capacity_uses_explicit_async_overrides() -> None:
    cfg = _cfg(
        async_pipeline=AsyncPipelineConfig(
            rollout_concurrency=3,
            replay_buffer_samples=11,
            max_admitted_generation_prompts=13,
        )
    )

    derived = derived_capacity(cfg)

    assert derived.rollout_concurrency == 3
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


def test_derived_capacity_rejects_global_batch_not_divisible_by_dp_rows() -> None:
    cfg = _cfg(batch=BatchConfig(local_batch_size=3, global_batch_size=10))

    with pytest.raises(ValueError, match="must be divisible"):
        derived_capacity(cfg)


def test_compute_generator_max_num_seqs_uses_derived_admission_cap() -> None:
    cfg = _cfg(
        async_pipeline=AsyncPipelineConfig(max_admitted_generation_prompts=17),
        num_validation_samples=5,
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
