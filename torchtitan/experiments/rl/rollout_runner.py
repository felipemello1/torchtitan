# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared rollout-group execution for validation and training producers."""

from __future__ import annotations

from dataclasses import dataclass

from renderers import Renderer

from torchtitan.experiments.rl.envs import EnvBuilder, EnvExample, TokenEnvConfig
from torchtitan.experiments.rl.rollouts import CompletionFn, do_rollout_group
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.types import RolloutOutput


@dataclass(frozen=True, kw_only=True, slots=True)
class RolloutGroupResult:
    """Completed rollout group plus behavior-version summary."""

    rollouts: list[RolloutOutput]
    behavior_version: int | None
    max_behavior_version: int | None


async def run_rollout_group(
    *,
    env_builder: EnvBuilder,
    example: EnvExample,
    group_size: int,
    renderer: Renderer,
    completion_fn: CompletionFn,
    sampling: SamplingConfig,
    max_turns: int,
    token_env_config: TokenEnvConfig,
) -> RolloutGroupResult:
    """Run one GRPO group."""
    rollouts = await do_rollout_group(
        envs=[env_builder.build(example=example) for _ in range(group_size)],
        renderer=renderer,
        completion_fn=completion_fn,
        sampling=sampling,
        group_id=example.group_id,
        max_turns=max_turns,
        token_env_config=token_env_config,
    )
    versioned_rollouts = [rollout for rollout in rollouts if rollout.turns]
    behavior_version = (
        min(rollout.behavior_version for rollout in versioned_rollouts)
        if versioned_rollouts
        else None
    )
    max_behavior_version = (
        max(rollout.max_behavior_version for rollout in versioned_rollouts)
        if versioned_rollouts
        else None
    )
    return RolloutGroupResult(
        rollouts=rollouts,
        behavior_version=behavior_version,
        max_behavior_version=max_behavior_version,
    )
