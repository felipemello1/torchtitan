# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, replace

from torchtitan.rl.examples.chess_selfplay.data import ChessSample
from torchtitan.rl.rollout.curriculum import Curriculum


class ChessCurriculum(Curriculum):
    """Grows training games' ply cap with the train step; validation games keep the worker's `max_plies`.

    Example (start_max_plies=60, end_max_plies=120, max_plies_ramp_steps=200):

        prepare(sample, step=0)    # max_plies=60
        prepare(sample, step=100)  # max_plies=90
        prepare(sample, step=300)  # max_plies=120
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Curriculum.Config):
        start_max_plies: int
        """Training games' ply cap at step 0."""

        end_max_plies: int
        """Training games' ply cap from step `max_plies_ramp_steps` on."""

        max_plies_ramp_steps: int = 200
        """Steps the cap takes to grow linearly from `start_max_plies` to `end_max_plies`."""

    def __init__(self, config: Config) -> None:
        self._start_max_plies = config.start_max_plies
        self._end_max_plies = config.end_max_plies
        self._max_plies_ramp_steps = config.max_plies_ramp_steps

    def prepare(self, sample: ChessSample, *, step: int) -> ChessSample:
        ramp_step = min(step, self._max_plies_ramp_steps)
        max_plies = (
            self._start_max_plies
            + (self._end_max_plies - self._start_max_plies)
            * ramp_step
            // self._max_plies_ramp_steps
        )
        return replace(sample, max_plies=max_plies)
