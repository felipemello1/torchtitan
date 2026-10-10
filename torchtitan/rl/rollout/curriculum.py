# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.config import Configurable
from torchtitan.rl.rollout.types import RolloutGroup


class Curriculum(Stateful, Configurable):
    """Rewrites training samples (e.g. their difficulty) from the train step and trained groups' results.

    The `Rollouter` calls it for training groups only:
    1. `prepare` rewrites a sample right before its group is rolled out.
    2. `summarize` keeps what `update` needs from the finished group.
    3. `update` gets the summaries of the groups each train step trained, right before the checkpoint.
    All hooks run on the controller's event loop; keep them cheap.

    Example:

        # Promote to the next difficulty when a step's pass rate at the current one is > 0.8.
        class PassRateCurriculum(Curriculum):
            @dataclass(kw_only=True, slots=True)
            class Config(Curriculum.Config):
                promote_pass_rate: float = 0.8

            def __init__(self, config: Config) -> None:
                self.promote_pass_rate = config.promote_pass_rate
                self.level = 0

            def prepare(self, sample, *, step):
                # the env builds a task of this difficulty
                return replace(sample, level=self.level)

            def summarize(self, sample, group):
                num_passed = sum(rollout.reward == 1.0 for rollout in group.rollouts)
                return sample.level, num_passed / len(group.rollouts)

            def update(self, *, step, summaries):
                # groups prepared before the last promotion still arrive; skip them
                pass_rates = [rate for level, rate in summaries if level == self.level]
                if pass_rates and sum(pass_rates) / len(pass_rates) > self.promote_pass_rate:
                    self.level += 1

            def state_dict(self):
                return {"level": self.level}

            def load_state_dict(self, state_dict):
                self.level = state_dict["level"]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config

    def prepare(self, sample: object, *, step: int) -> object:
        """Return the sample to roll out at train step `step`, e.g. with its opponent set.

        Return a copy: the dataloader keeps the original `sample` to replay it after a resume.
        Before a run's first train step, `step` is the last finished one (0, or N after a resume).
        """
        return sample

    def summarize(self, sample: object, group: RolloutGroup) -> object:
        """Return the small result `update` needs from a finished training group; `sample` is what `prepare` returned.

        Don't change state here: a group not yet trained at a checkpoint is rolled out again after a resume.
        """
        return None

    def update(self, *, step: int, summaries: list[object]) -> None:
        """Update from the summaries of the groups trained at step `step`.

        Zero-std groups left out of the loss count as trained; a group whose rollout raised has no summary.
        """

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
