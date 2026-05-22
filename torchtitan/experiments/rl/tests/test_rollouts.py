# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Focused CPU tests for renderer, TokenEnv, and rollout group wiring."""

from __future__ import annotations

import asyncio

import pytest
from renderers import Message

from torchtitan.experiments.rl.envs import EnvExample, EnvReset, EnvStep
from torchtitan.experiments.rl.envs.token_env import TokenEnv, TokenEnvConfig
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollouts import run_rollout_group
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.types import Completion, RolloutStatus


MODEL_PATH = "torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B"


def _renderer():
    return RendererConfig(name="qwen3", enable_thinking=False).build(
        model_path=MODEL_PATH
    )


def _response_ids(renderer, text: str) -> list[int]:
    return list(renderer._tokenizer.encode(text, add_special_tokens=False))


class _EchoEnv:
    async def reset(self) -> EnvReset:
        return EnvReset(messages=[{"role": "user", "content": "say ok"}])

    async def step(self, assistant_message: Message) -> EnvStep:
        content = str(assistant_message.get("content") or "")
        reward = 1.0 if content.strip() == "ok" else 0.0
        return EnvStep(
            reward=reward,
            reward_components={"exact": reward},
            done=True,
            status=RolloutStatus.COMPLETED,
        )

    async def close(self) -> None:
        return None


class _EchoBuilder:
    def build(self, *, example: EnvExample) -> _EchoEnv:
        return _EchoEnv()


def test_renderer_parse_response_round_trip() -> None:
    renderer = _renderer()
    parsed = renderer.parse_response(_response_ids(renderer, "ok"))

    assert parsed.content == "ok"


def test_token_env_reset_and_step_happy_path() -> None:
    async def scenario():
        renderer = _renderer()
        token_env = TokenEnv(_EchoEnv(), renderer)
        prompt = await token_env.initial_prompt()
        token_ids = _response_ids(renderer, "ok")
        token_step = await token_env.step(
            Completion(
                policy_version=0,
                token_ids=token_ids,
                token_logprobs=[0.0] * len(token_ids),
                finish_reason="stop",
            )
        )
        return prompt, token_step

    prompt, token_step = asyncio.run(scenario())
    assert prompt.token_ids
    assert token_step.env_step.reward == 1.0
    assert token_step.env_step.done is True


def test_token_env_initial_context_overflow() -> None:
    async def scenario():
        token_env = TokenEnv(
            _EchoEnv(),
            _renderer(),
            TokenEnvConfig(max_trajectory_tokens=1, max_generation_tokens=1),
        )
        await token_env.initial_prompt()

    with pytest.raises(ValueError, match="max_trajectory_tokens"):
        asyncio.run(scenario())


def test_run_rollout_group_single_turn_sumdigits_shape() -> None:
    async def scenario():
        renderer = _renderer()

        async def completion_fn(
            *,
            prompt_token_ids: list[int],
            sampling: SamplingConfig,
            request_id: str,
        ) -> Completion:
            token_ids = _response_ids(renderer, "ok")
            return Completion(
                policy_version=5,
                token_ids=token_ids,
                token_logprobs=[0.0] * len(token_ids),
                finish_reason="stop",
            )

        return await run_rollout_group(
            env_builder=_EchoBuilder(),
            example=EnvExample(group_id="g0", sample_step=0, group_idx=0),
            group_size=2,
            renderer=renderer,
            completion_fn=completion_fn,
            sampling=SamplingConfig(),
            max_turns=1,
            token_env_config=TokenEnvConfig(),
        )

    rollouts = asyncio.run(scenario())
    assert [rollout.sample_idx for rollout in rollouts] == [0, 1]
    assert {rollout.group_id for rollout in rollouts} == {"g0"}
    assert [rollout.reward for rollout in rollouts] == [1.0, 1.0]
    assert all(rollout.turns[0].policy_version == 5 for rollout in rollouts)


def test_run_rollout_group_generation_error_is_terminal() -> None:
    async def scenario():
        async def completion_fn(
            *,
            prompt_token_ids: list[int],
            sampling: SamplingConfig,
            request_id: str,
        ) -> Completion:
            return Completion(
                policy_version=5,
                token_ids=[99],
                token_logprobs=[],
                finish_reason="abort",
                error="aborted",
            )

        return await run_rollout_group(
            env_builder=_EchoBuilder(),
            example=EnvExample(group_id="g0", sample_step=0, group_idx=0),
            group_size=1,
            renderer=_renderer(),
            completion_fn=completion_fn,
            sampling=SamplingConfig(),
            max_turns=1,
            token_env_config=TokenEnvConfig(),
        )

    rollout = asyncio.run(scenario())[0]
    assert rollout.status == RolloutStatus.ERROR
    assert rollout.turns[0].response_token_ids == []
