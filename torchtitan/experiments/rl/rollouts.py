# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async rollout drivers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Protocol

from renderers import Renderer

from torchtitan.experiments.rl.envs import EnvBuilder, EnvExample
from torchtitan.experiments.rl.envs.token_env import TokenEnv, TokenEnvConfig
from torchtitan.experiments.rl.envs.types import MessageEnv
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.types import (
    Completion,
    RolloutOutput,
    RolloutStatus,
    RolloutTurn,
    validate_rollout_output,
)


class CompletionFn(Protocol):
    """Async policy callable used by rollout drivers."""

    def __call__(
        self,
        *,
        prompt_token_ids: list[int],
        sampling: SamplingConfig,
        request_id: str,
    ) -> Awaitable[Completion]:
        ...


async def do_single_rollout(
    *,
    token_env: TokenEnv,
    completion_fn: CompletionFn,
    sampling: SamplingConfig,
    group_id: str,
    sample_idx: int,
    max_turns: int,
) -> RolloutOutput:
    """Drive one env until it finishes or reaches `max_turns`."""
    prompt = await token_env.initial_prompt()
    turns: list[RolloutTurn] = []
    reward: float | None = None
    reward_components: dict[str, float] = {}
    status = RolloutStatus.TRUNCATED

    for turn_idx in range(max_turns):
        request_id = f"{group_id}:sample={sample_idx}:turn={turn_idx}"
        completion = await completion_fn(
            prompt_token_ids=prompt.token_ids,
            sampling=sampling,
            request_id=request_id,
        )
        token_step = await token_env.step(completion)
        env_step = token_step.env_step

        if env_step.done and env_step.reward is not None:
            reward = float(env_step.reward)
            reward_components = dict(env_step.reward_components)

        turns.append(
            RolloutTurn(
                prompt_token_ids=list(prompt.token_ids),
                response_token_ids=(
                    [] if completion.error is not None else list(completion.token_ids)
                ),
                response_logprobs=(
                    []
                    if completion.error is not None
                    else list(completion.token_logprobs)
                ),
                prompt_messages=list(prompt.messages),
                response_messages=list(token_step.response_messages),
                policy_version=completion.policy_version,
                finish_reason=completion.finish_reason,
            )
        )

        if env_step.done:
            status = env_step.status or RolloutStatus.COMPLETED
            break
        if token_step.next_prompt is None:
            raise ValueError(
                "non-terminal EnvStep did not produce a next prompt "
                f"(group_id={group_id}, sample_idx={sample_idx}, turn={turn_idx})"
            )
        prompt = token_step.next_prompt

    if status == RolloutStatus.TRUNCATED and reward is None:
        reward = 0.0
        reward_components = {"max_turns": 1.0}

    output = RolloutOutput(
        group_id=group_id,
        sample_idx=sample_idx,
        turns=turns,
        status=status,
        reward=reward,
        reward_components=reward_components,
    )
    validate_rollout_output(output)
    return output


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
) -> list[RolloutOutput]:
    """Run one rollout group from a concrete dataset example."""
    envs: list[MessageEnv] = []
    try:
        for _ in range(group_size):
            envs.append(env_builder.build(example=example))
    except BaseException:
        await asyncio.gather(
            *(env.close() for env in envs),
            return_exceptions=True,
        )
        raise

    token_envs = [TokenEnv(env, renderer, token_env_config) for env in envs]
    tasks = [
        asyncio.create_task(
            do_single_rollout(
                token_env=token_env,
                completion_fn=completion_fn,
                sampling=sampling,
                group_id=example.group_id,
                sample_idx=sample_idx,
                max_turns=max_turns,
            )
        )
        for sample_idx, token_env in enumerate(token_envs)
    ]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        await asyncio.gather(
            *(token_env.close() for token_env in token_envs),
            return_exceptions=True,
        )
