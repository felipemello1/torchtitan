# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Advantage estimator: a post-scoring step run by the ``Rollouter``.

After a group is scored, this turns the group's rewards into per-rollout advantages
(in group order) that the trainer/loss consumes directly.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.rl.rollout.types import RolloutGroup


class AdvantageEstimator(Configurable):
    """Group-relative advantage estimator: ``A_i = (r_i - mean(r)) / denom``.

    ``denom = std(r) + eps`` when ``should_std_normalize`` (standard GRPO), else
    ``1.0`` (Dr.GRPO mean-baseline). The name is intentionally general — other
    advantage schemes can live here later.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        should_std_normalize: bool = False
        """Divide the centered advantage by the group reward std (+eps) — standard GRPO.
        False (default) = mean-baseline only (Dr.GRPO)."""

    def __init__(self, config: Config) -> None:
        self.should_std_normalize: bool = config.should_std_normalize

    def __call__(self, group: RolloutGroup) -> list[float]:
        """Return per-rollout advantages, in group order. Errored rollouts (sandbox, harness,
        timeout) are left out of the mean/std and get 0.0: their fixed `error_reward` says nothing
        about the task. `TrainingSampleBuilder` drops them from the loss.

        Example (Dr.GRPO):
            status    [COMPLETED, COMPLETED, ERROR_TIMEOUT, COMPLETED]
            reward    [1.0,       0.0,       0.0,           1.0]
            mean over the 3 kept rewards = 2/3
            advantage [1/3,       -2/3,      0.0,           1/3]
            all errored -> all 0.0
        """
        rewards = [
            rollout.reward
            for rollout in group.rollouts
            if not rollout.status.is_error()
        ]
        if not rewards:
            return [0.0] * len(group.rollouts)
        group_mean = sum(rewards) / len(rewards)
        denom = (
            (statistics.pstdev(rewards) + 1e-6) if self.should_std_normalize else 1.0
        )
        return [
            0.0
            if rollout.status.is_error()
            else (rollout.reward - group_mean) / denom
            for rollout in group.rollouts
        ]
