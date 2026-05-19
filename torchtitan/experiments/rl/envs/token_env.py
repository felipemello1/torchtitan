# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter from message envs to token-level rollout prompts."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from renderers import Message, ParsedResponse, Renderer, ToolSpec

from torchtitan.experiments.rl.envs.types import EnvReset, EnvStep, MessageEnv
from torchtitan.experiments.rl.types import Completion, RolloutStatus

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class TokenEnvConfig:
    """Operational policy for :class:`TokenEnv`.

    Args:
        error_reward: Reward used when parsing or env stepping fails.
        truncation_reward: Fallback reward for generation length stops when
            the env did not stamp a reward, and for controller-side context caps.
        max_trajectory_tokens: Optional prompt-token cap before the next generation.
        max_generation_tokens: Reserve included in context-cap checks.
        step_timeout_s: Optional timeout for one ``MessageEnv.step`` call.
    """

    error_reward: float = 0.0
    truncation_reward: float = 0.0
    max_trajectory_tokens: int | None = None
    max_generation_tokens: int | None = None
    step_timeout_s: float | None = 1800.0


@dataclass(kw_only=True, slots=True)
class PromptState:
    """Token prompt and message snapshot for one generation call."""

    token_ids: list[int]
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)


@dataclass(kw_only=True, slots=True)
class TokenStep:
    """Result of applying one token completion to a :class:`TokenEnv`."""

    env_step: EnvStep
    response_messages: list[Message]
    next_prompt: PromptState | None = None


class TokenEnv:
    """Message env plus renderer state for one rollout."""

    def __init__(
        self,
        env: MessageEnv,
        renderer: Renderer,
        config: TokenEnvConfig | None = None,
    ):
        self._env = env
        self._renderer = renderer
        self._config = config or TokenEnvConfig()
        self._messages: list[Message] = []
        self._tools: list[ToolSpec] = []
        self._previous_prompt_ids: list[int] | None = None
        self._previous_completion_ids: list[int] | None = None

    async def initial_prompt(self) -> PromptState:
        """Reset the env and render the first model prompt."""
        reset: EnvReset = await self._env.reset()
        self._messages = list(reset.messages)
        self._tools = list(reset.tools)
        prompt = PromptState(
            token_ids=await self._render_prompt(),
            messages=list(self._messages),
            tools=list(self._tools),
        )
        self._raise_if_context_exceeded(prompt)
        return prompt

    async def step(self, completion: Completion) -> TokenStep:
        """Parse completion tokens, step the env, and render the next prompt."""
        try:
            parsed: ParsedResponse = self._renderer.parse_response(completion.token_ids)
        except Exception as exc:
            logger.warning("renderer.parse_response failed: %s", exc, exc_info=False)
            return TokenStep(
                env_step=EnvStep(
                    reward=self._config.error_reward,
                    reward_components={"parse_error": 1.0},
                    done=True,
                    status=RolloutStatus.ERROR,
                ),
                response_messages=[],
            )

        assistant_message = _assistant_message_from(parsed)
        env_step = await self._call_env_step(assistant_message)
        response_messages = [assistant_message, *env_step.messages]
        self._messages.extend(response_messages)
        self._previous_completion_ids = list(completion.token_ids)

        if completion.finish_reason == "length":
            reward = (
                env_step.reward
                if env_step.reward is not None
                else self._config.truncation_reward
            )
            reward_components = (
                dict(env_step.reward_components)
                if env_step.reward is not None
                else {**env_step.reward_components, "length_stop": 1.0}
            )
            return TokenStep(
                env_step=EnvStep(
                    messages=env_step.messages,
                    reward=reward,
                    reward_components=reward_components,
                    done=True,
                    status=RolloutStatus.TRUNCATED,
                ),
                response_messages=response_messages,
            )

        if env_step.done:
            return TokenStep(env_step=env_step, response_messages=response_messages)

        next_prompt = PromptState(
            token_ids=await self._render_next_prompt(response_messages),
            messages=list(self._messages),
            tools=list(self._tools),
        )
        if self._context_exceeded(next_prompt):
            return TokenStep(
                env_step=EnvStep(
                    reward=self._config.truncation_reward,
                    reward_components={
                        **env_step.reward_components,
                        "context_overflow": 1.0,
                    },
                    done=True,
                    status=RolloutStatus.TRUNCATED,
                ),
                response_messages=response_messages,
            )
        return TokenStep(
            env_step=env_step,
            response_messages=response_messages,
            next_prompt=next_prompt,
        )

    async def close(self) -> None:
        await self._env.close()

    async def _call_env_step(self, assistant_message: Message) -> EnvStep:
        step_coro = self._env.step(assistant_message)
        if self._config.step_timeout_s is None:
            return await step_coro
        try:
            return await asyncio.wait_for(
                step_coro, timeout=self._config.step_timeout_s
            )
        except asyncio.TimeoutError:
            return EnvStep(
                reward=self._config.error_reward,
                reward_components={"step_timeout": 1.0},
                done=True,
                status=RolloutStatus.ERROR,
            )

    async def _render_prompt(self) -> list[int]:
        rendered = await asyncio.to_thread(
            self._renderer.render_ids,
            self._messages,
            tools=self._tools or None,
            add_generation_prompt=True,
        )
        self._previous_prompt_ids = list(rendered)
        return list(rendered)

    async def _render_next_prompt(self, response_messages: list[Message]) -> list[int]:
        if (
            self._previous_prompt_ids is not None
            and self._previous_completion_ids is not None
        ):
            env_messages = response_messages[1:]
            bridged = self._renderer.bridge_to_next_turn(
                self._previous_prompt_ids,
                self._previous_completion_ids,
                env_messages,
                tools=self._tools or None,
            )
            if bridged is not None:
                self._previous_prompt_ids = list(bridged.token_ids)
                return list(bridged.token_ids)
        return await self._render_prompt()

    def _raise_if_context_exceeded(self, prompt: PromptState) -> None:
        if self._context_exceeded(prompt):
            raise ValueError(
                "initial prompt plus generation reserve exceeds "
                f"max_trajectory_tokens={self._config.max_trajectory_tokens}"
            )

    def _context_exceeded(self, prompt: PromptState) -> bool:
        if self._config.max_trajectory_tokens is None:
            return False
        reserve = self._config.max_generation_tokens or 0
        return len(prompt.token_ids) + reserve > self._config.max_trajectory_tokens


def _assistant_message_from(parsed: ParsedResponse) -> Message:
    message: Message = {"role": "assistant", "content": parsed.content}
    if parsed.reasoning_content:
        message["reasoning_content"] = parsed.reasoning_content
    if parsed.tool_calls:
        message["tool_calls"] = parsed.tool_calls
    return message
