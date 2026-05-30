# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass

from renderers import Renderer

from torchtitan.config import Configurable
from torchtitan.experiments.rl.actors.generators.types import GenerateFn, SamplingConfig
from torchtitan.experiments.rl.env_types.renderer_env import RendererEnv
from torchtitan.experiments.rl.rollouts.types import (
    DatasetOutput,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rubrics import Reward, Rubric

logger = logging.getLogger(__name__)

# Runaway guard for multi-turn rollouts; envs normally terminate first (env
# `done`, or prompt overflow via RendererEnvConfig.max_rollout_tokens).
_MAX_TURNS = 100


# TODO: investigate whether Task should also hold its own dataset
# instead of dataset living on RLTrainer.
class Task(Configurable, abc.ABC):
    """Per-task bundle: a `Rubric` + env construction + group scoring. The
    controller owns the dataset and the rollout loop; `score_group` defaults
    to per-rollout `rubric.score_group` (override for cross-sibling scoring).

    Example:
        class MyTask(Task):
            @dataclass(kw_only=True, slots=True)
            class Config(Task.Config):
                rubric: MyRubric.Config = field(
                    default_factory=MyRubric.Config
                )
                renderer_env_config: RendererEnvConfig = field(
                    default_factory=RendererEnvConfig
                )

            def __init__(self, config: Config) -> None:
                self.rubric = config.rubric.build()
                self.renderer_env_config = config.renderer_env_config

            def make_envs(self, *, example, group_size, renderer):
                return [RendererEnv(...) for _ in range(group_size)]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rubric: Rubric.Config

    rubric: Rubric
    """Built by each subclass's `__init__` from `config.rubric`; used by `score_group`."""

    # TODO: revisit the Renderer being injected into `make_envs` once we
    # know whether Task should own a Renderer (per-task chat templates).
    @abc.abstractmethod
    def make_envs(
        self,
        *,
        example: DatasetOutput,
        group_size: int,
        renderer: Renderer,
    ) -> list[RendererEnv]:
        """Construct `group_size` single-use envs from one dataset example.

        Args:
            example: Dataset row sampled from the controller's dataset.
            group_size: Number of sibling envs for this prompt group.
            renderer: Renderer shared by the rollout controller.

        Returns:
            `group_size` `RendererEnv` instances, each ready for one rollout.
        """

    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[Reward]:
        """Score one group's rollouts; the controller applies the rewards.

        Default impl delegates to `self.rubric.score_group`. Override for
        cross-sibling scoring (judge, pairwise, diversity) or partial-credit
        reward shaping.

        Args:
            rollouts: Sibling rollouts in one prompt group, already stepped.
            env_input: Dataset payload shared by the group.

        Returns:
            One `Reward` per rollout, in input order.
        """
        return await self.rubric.score_group(rollouts, env_input)

    async def do_group_rollout(
        self,
        *,
        example: DatasetOutput,
        group_id: str,
        group_size: int,
        renderer: Renderer,
        sampling_config: SamplingConfig,
        generate: GenerateFn,
    ) -> RolloutGroup:
        """Roll out and score one prompt group.

        Builds `group_size` sibling envs and drives them concurrently. All
        siblings share one `generate` (the controller routes a whole group to
        one generator), so the prompt they share stays warm in that engine's
        prefix cache. Scores the group via `score_group`.

        Args:
            example: Dataset row for this group.
            group_id: Stable group id; siblings share it for advantage centering.
            group_size: Number of sibling rollouts.
            renderer: Renderer shared by the group's envs.
            sampling_config: Sampling for every generate call in the group.
            generate: Monarch-free async callable bound to the routed generator.

        Returns:
            One scored `RolloutGroup`.
        """
        envs = self.make_envs(example=example, group_size=group_size, renderer=renderer)
        try:
            rollouts = await asyncio.gather(
                *(
                    self.do_single_rollout(
                        group_id=group_id,
                        sample_idx=i,
                        env=env,
                        sampling_config=sampling_config,
                        generate=generate,
                    )
                    for i, env in enumerate(envs)
                )
            )
        finally:
            await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)

        rewards = await self.score_group(rollouts, example.env_input)
        for rollout, reward in zip(rollouts, rewards, strict=True):
            rollout.reward = reward.reward
            rollout.reward_components = reward.components
        return RolloutGroup(
            group_id=group_id, env_input=example.env_input, rollouts=rollouts
        )

    async def do_single_rollout(
        self,
        *,
        group_id: str,
        sample_idx: int,
        env: RendererEnv,
        sampling_config: SamplingConfig,
        generate: GenerateFn,
    ) -> Rollout:
        """Drive one env to a terminal state via its own generate calls.

        Loops `generate -> env.step_completion` until the env reports terminal
        (env `done`, length / parse / timeout, or prompt overflow). On any error
        the rollout keeps the turns gathered so far and is marked ERROR; the
        controller scores it afterward via `score_group`.

        Args:
            group_id: Group id, prefixed onto each turn's `request_id` so all of
                a group's turns route to the same generator.
            sample_idx: Sample index within the group (0..group_size-1).
            env: The env for this rollout.
            sampling_config: Sampling for every generate call.
            generate: Monarch-free async callable bound to the routed generator.

        Returns:
            One unscored `Rollout` (reward filled later by the controller).
        """
        turns: list[RolloutTurn] = []
        status = RolloutStatus.ERROR
        try:
            step = await env.initial_prompt()
            while not step.status.is_terminal() and len(turns) < _MAX_TURNS:
                completion = await generate(
                    step.next_prompt_token_ids,
                    request_id=f"{group_id}/sample={sample_idx}/turn={len(turns)}",
                    sampling_config=sampling_config,
                )
                next_step = await env.step_completion(completion)
                turns.append(
                    RolloutTurn(
                        prompt_token_ids=list(step.next_prompt_token_ids),
                        response_token_ids=list(completion.token_ids),
                        response_logprobs=list(completion.token_logprobs),
                        policy_version=completion.policy_version,
                        prompt_messages=list(step.next_prompt_messages),
                        assistant_message=next_step.assistant_message,
                        env_messages=list(next_step.env_messages),
                        reward_components=dict(next_step.env_reward_components),
                    )
                )
                step = next_step
            status = step.status
            if not status.is_terminal():
                logger.warning(
                    "rollout %s/sample=%d hit _MAX_TURNS=%d; truncating",
                    group_id,
                    sample_idx,
                    _MAX_TURNS,
                )
                status = RolloutStatus.TRUNCATED_LENGTH
        except Exception:
            logger.exception(
                "rollout %s/sample=%d failed after %d turn(s); marking ERROR",
                group_id,
                sample_idx,
                len(turns),
            )
            status = RolloutStatus.ERROR
        return Rollout(
            group_id=group_id, sample_idx=sample_idx, status=status, turns=turns
        )
