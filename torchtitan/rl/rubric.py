# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.rollout.types import Rollout


class RewardFn(Configurable, abc.ABC):
    """A single reward function, as a Configurable callable.

    Subclass and implement `__call__`. Its `Config` carries the `weight` used in
    the rubric's weighted sum, plus any args a stateful reward fn needs (a reward
    model path, an LLM-judge endpoint, a threshold, ...).

    Example:
        class RewardCorrect(RewardFn):
            @dataclass(kw_only=True, slots=True)
            class Config(RewardFn.Config):
                pass  # only needs `weight`

            async def __call__(self, rollout, env_input) -> float:
                ...
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        weight: float = 1.0
        """Relative weight in the rubric's weighted sum (normalized across fns)."""

    def __init__(self, config: Config) -> None:
        self.weight = config.weight

    @abc.abstractmethod
    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        """Return this fn's score for one rollout.

        Args:
            rollout: Rollout to score.
            env_input: Dataset payload used to build the env (target/metadata).
        """


@dataclass(frozen=True, kw_only=True, slots=True)
class RubricOutput:
    """One rollout's reward, as returned by a `Rubric`.

    Example:
        >>> RubricOutput(reward=0.5, reward_breakdown={"RewardCorrect": 1.0, "RewardFormat": 0.0})
    """

    reward: float
    """Final scalar reward for this rollout; assigned to `Rollout.reward`."""

    reward_breakdown: dict[str, float] = field(default_factory=dict)
    """Per-reward-fn outputs (unweighted), keyed by reward-fn class name.
    The default `Rubric` computes `reward` from these; callers may also use them for per-reward
    advantage, reweighting, metrics, or inspection."""


class Rubric(Configurable):
    """Scores rollouts with a set of weighted reward functions.

    The reward fns and their weights live in config (`reward_fns`), so common
    cases need no subclass. Subclass and override `score_group` for cross-sibling
    scoring (pairwise comparison, diversity, rank normalization).

    Setting `truncation_reward` / `error_reward` short-circuits the reward fns for
    rollouts whose status is truncated / errored.

    Example:
        rubric = Rubric.Config(
            reward_fns=[RewardCorrect.Config(weight=1.0), RewardFormat.Config(weight=0.3)],
            truncation_reward=0.0,
        ).build()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        reward_fns: list[RewardFn.Config] = field(default_factory=list)
        """The rubric's reward fns + weights; built and weight-normalized at init."""

        truncation_reward: float | None = None
        """Reward for a truncated rollout. If set, the reward fns are SKIPPED and this fixed
        reward is used. If None, the reward fns run on the truncated rollout."""

        error_reward: float | None = None
        """Reward for a errored rollout. If set, the reward fns are SKIPPED and this fixed
        reward is used. If None, the reward fns run on the errored rollout."""

        length_reward_weight: float = 0.0
        """Weight of Kimi k1.5's length reward (`kimi_length_rewards`), added once the group is
        graded; 0: off. Groups whose lengths differ are never zero-std, so all-correct and
        all-wrong groups pass `drop_zero_std_reward_groups` and train on length alone."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._reward_fns = [rwd_cfg.build() for rwd_cfg in config.reward_fns]

        # Sanity checks
        if not self._reward_fns:
            raise ValueError("Rubric.Config.reward_fns must not be empty")
        names = [type(fn).__name__ for fn in self._reward_fns]
        if len(names) != len(set(names)):
            raise ValueError(f"reward fn names must be unique; got {names}")
        self._weight_sum = sum(fn.weight for fn in self._reward_fns)
        if self._weight_sum <= 0:
            raise ValueError(
                f"rubric weights must sum to a positive value; got {self._weight_sum}"
            )

    @sl.log_trace_span("score_single_rollout")
    async def _score_single_rollout(
        self, rollout: Rollout, env_input: object
    ) -> RubricOutput:
        """Score one rollout. Short-circuits to `truncation_reward` /
        `error_reward` when those are set and the rollout truncated / errored.

        Args:
            rollout: Rollout to score.
            env_input: Dataset payload used to build the env (target/metadata).

        Returns:
            Final weighted reward + per-fn raw breakdown.
        """
        # Short-circuit on truncate / error and return the configured reward. The
        # breakdown records the short-circuit reason so it shows up in metrics.
        cfg = self._config
        if cfg.truncation_reward is not None and rollout.status.is_truncated():
            return RubricOutput(
                reward=cfg.truncation_reward,
                reward_breakdown={"truncated": cfg.truncation_reward},
            )
        if cfg.error_reward is not None and rollout.status.is_error():
            return RubricOutput(
                reward=cfg.error_reward,
                reward_breakdown={"errored": cfg.error_reward},
            )

        # Run all reward fns and weight-sum (weights normalized to sum to 1.0).
        per_fn_rewards = await asyncio.gather(
            *(fn(rollout, env_input) for fn in self._reward_fns)
        )

        reward_breakdown = {}
        total_reward = 0.0
        for fn, r in zip(self._reward_fns, per_fn_rewards, strict=True):
            reward_breakdown[type(fn).__name__] = r
            total_reward += (fn.weight / self._weight_sum) * r

        return RubricOutput(reward=total_reward, reward_breakdown=reward_breakdown)

    @sl.log_trace_span("score_group")
    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[RubricOutput]:
        """Score every rollout in one prompt group, then add the length reward.

        Override for cross-rollout rewards (pairwise comparison, diversity,
        rank normalization).

        Args:
            rollouts: Sibling rollouts sampled from one prompt group.
            env_input: Dataset payload originally used to construct the rollout env.

        Returns:
            One `RubricOutput` per rollout, in input order.
        """
        outputs = await asyncio.gather(
            *(self._score_single_rollout(r, env_input) for r in rollouts)
        )
        if self._config.length_reward_weight == 0:
            return outputs
        length_rewards = kimi_length_rewards(
            rollouts=rollouts,
            rewards=[output.reward for output in outputs],
            weight=self._config.length_reward_weight,
        )
        return [
            RubricOutput(
                reward=output.reward + length_reward,
                reward_breakdown={
                    **output.reward_breakdown,
                    "length_reward": length_reward,
                },
            )
            for output, length_reward in zip(outputs, length_rewards, strict=True)
        ]


def kimi_length_rewards(
    *, rollouts: list[Rollout], rewards: list[float], weight: float
) -> list[float]:
    """Kimi k1.5's length reward (arXiv 2501.12599, section 2.3.3), one per rollout in group order.

    Over the group, `lam = 0.5 - (len - min_len) / (max_len - min_len)`, with `len` the
    completion tokens of all turns. A correct rollout (reward > 0) gets `weight * lam`; a wrong
    one gets `weight * min(0, lam)`, so only its length above the group's midpoint costs it. All
    lengths equal: 0 for every rollout. An errored rollout gets 0 and is left out of `min_len`
    and `max_len`: the error, not the model, decided where it stopped.

    Assumes the mean baseline (`should_std_normalize=False`), as k1.5 does: with std
    normalization, a group that differs only in length trains at full advantage scale for any
    `weight`.

    Example:
        weight=0.1; lengths 1,000 / 2,000 / 1,000 / 3,000 tokens, the first two correct
        correct, 1,000 -> 0.1 * 0.5          = +0.05
        correct, 2,000 -> 0.1 * 0.0          =  0.0
        wrong,   1,000 -> 0.1 * min(0, 0.5)  =  0.0
        wrong,   3,000 -> 0.1 * min(0, -0.5) = -0.05
    """
    lengths = [
        sum(len(rollout_turn.completion_token_ids) for rollout_turn in rollout.turns)
        for rollout in rollouts
    ]
    scored_lengths = [
        length
        for length, rollout in zip(lengths, rollouts, strict=True)
        if not rollout.status.is_error()
    ]
    # No scored rollout, or all of one length.
    if len(set(scored_lengths)) < 2:
        return [0.0] * len(rollouts)
    min_len, max_len = min(scored_lengths), max(scored_lengths)
    length_rewards = []
    for rollout, length, reward in zip(rollouts, lengths, rewards, strict=True):
        if rollout.status.is_error():
            length_rewards.append(0.0)
            continue
        lam = 0.5 - (length - min_len) / (max_len - min_len)
        length_rewards.append(weight * (lam if reward > 0 else min(0.0, lam)))
    return length_rewards
