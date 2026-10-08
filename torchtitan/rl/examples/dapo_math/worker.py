# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import statistics
from dataclasses import dataclass

from torchtitan.rl.examples.dapo_math.data import DapoMathSample
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.rollout.rollouter import RolloutWorker

# Lower edges are inclusive: a pass rate of 0.75 (6 of 8) logs under ge0.75.
_PASS_RATE_BINS = ("lt0.25", "0.25-0.5", "0.5-0.75", "ge0.75")


class DapoMathRolloutWorker(RolloutWorker):
    """`RolloutWorker` that also logs, per pass-rate bin, the fraction of groups all correct or all wrong.

    Each group logs 1 or 0 under its bin; 1 only if its rewards have zero std (it does not train).
    Only samples with a `pass_rate` log, i.e. `Intellect3MathDataset` ones.

    Example:
        # 4 groups with pass_rate 0.875: 3 all correct, 1 mixed
        # -> math/zero_std/all_correct/avg8_ge0.75/mean = 0.75
        #    math/zero_std/all_wrong/avg8_ge0.75/mean = 0.0
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RolloutWorker.Config):
        pass

    async def run_group(self, *, sample: DapoMathSample, **kwargs) -> RolloutGroup:
        group = await super().run_group(sample=sample, **kwargs)
        if sample.pass_rate is None:
            return group

        rewards = [rollout.reward for rollout in group.rollouts]
        is_zero_std = len(rewards) > 1 and statistics.pstdev(rewards) == 0.0
        # Correct = reward > 0, so an errored rollout (error_reward=0.0) counts as wrong.
        is_all_correct = is_zero_std and rewards[0] > 0.0
        bin_name = _PASS_RATE_BINS[min(int(sample.pass_rate * 4), 3)]
        group.metrics += [
            m.Metric(
                f"math/zero_std/all_correct/avg8_{bin_name}",
                m.Mean(1.0 if is_all_correct else 0.0),
            ),
            m.Metric(
                f"math/zero_std/all_wrong/avg8_{bin_name}",
                m.Mean(1.0 if is_zero_std and not is_all_correct else 0.0),
            ),
        ]
        return group
