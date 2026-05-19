# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async rollout drivers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from renderers import Renderer

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


CompletionFn = Callable[[list[int], SamplingConfig, str], Awaitable[Completion]]


async def do_single_rollout(
    *,
    token_env: TokenEnv,
    completion_fn: CompletionFn,
    sampling: SamplingConfig,
    group_id: str,
    sample_idx: int,
    max_turns: int,
) -> RolloutOutput:
    """Drive one env until it finishes or reaches ``max_turns``.

    Args:
        token_env: Single-use token env wrapping a message env.
        completion_fn: Awaitable policy call. It receives prompt token IDs,
            sampling config, and a required request ID.
        sampling: Sampling parameters for each turn. Multiturn rollout calls
            use ``n=1``; group fanout owns sibling sampling.
        group_id: Stable group ID used for GRPO advantage centering.
        sample_idx: Sibling index inside the group.
        max_turns: Hard cap on assistant turns.
    """
    prompt = await token_env.initial_prompt()
    turns: list[RolloutTurn] = []
    reward: float | None = None
    reward_components: dict[str, float] = {}
    status = RolloutStatus.TRUNCATED

    for turn_idx in range(max_turns):
        request_id = f"{group_id}:sample={sample_idx}:turn={turn_idx}"
        completion = await completion_fn(prompt.token_ids, sampling, request_id)
        token_step = await token_env.step(completion)
        env_step = token_step.env_step

        if env_step.done and env_step.reward is not None:
            reward = float(env_step.reward)
            reward_components = dict(env_step.reward_components)

        turn_status = env_step.status or RolloutStatus.COMPLETED
        turns.append(
            RolloutTurn(
                prompt_token_ids=list(prompt.token_ids),
                response_token_ids=list(completion.token_ids),
                response_logprobs=list(completion.token_logprobs),
                prompt_messages=list(prompt.messages),
                response_messages=list(token_step.response_messages),
                policy_version=completion.policy_version,
                finish_reason=completion.finish_reason,
                status=turn_status,
            )
        )

        if env_step.done:
            status = turn_status
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


async def do_rollout_group(
    *,
    envs: list[MessageEnv],
    renderer: Renderer,
    completion_fn: CompletionFn,
    sampling: SamplingConfig,
    group_id: str,
    max_turns: int,
    token_env_config: TokenEnvConfig,
) -> list[RolloutOutput]:
    """Run one GRPO group concurrently and close every env.

    The caller owns env construction so config-based envs, dataset builders,
    and future remote-env proxies all use the same group rollout path.
    """
    token_envs = [TokenEnv(env, renderer, token_env_config) for env in envs]
    tasks = [
        asyncio.create_task(
            do_single_rollout(
                token_env=token_env,
                completion_fn=completion_fn,
                sampling=sampling,
                group_id=group_id,
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
