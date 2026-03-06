# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

from torchtitan.observability.common import clear_step_tags, set_step
from torchtitan.observability.logging_boundary import EveryNSteps


class MetricsProcessor:
    """Processes and logs training metrics.

    Manages step context and schedules for gating expensive operations
    like tensor metric reduction (all-reduce across ranks).
    """

    @dataclass
    class Config:
        log_tensor_metrics_freq: int = 10
        """How often to reduce tensor metrics (expensive all-reduce)."""

    def __init__(
        self, config: "MetricsProcessor.Config", *, dump_folder: str, rank: int = 0
    ):
        # Assumes init_observability() was already called by the caller.
        self._config = config
        self._tensor_metrics_schedule = EveryNSteps(
            every_n=config.log_tensor_metrics_freq, additional_steps={1}
        )

    def set_step(self, step: int) -> None:
        """Set current training step. Call before train_step().

        Sets the ContextVar so all subsequent record_span/record_event calls
        are tagged with this step in JSONL.
        """
        self._step = step
        set_step(step)
        clear_step_tags()

    def should_log_tensors(self, step: int) -> bool:
        """Returns True on tensor reduction steps. Caller uses this to gate replicate_to_host."""
        return self._tensor_metrics_schedule(step)

    def close(self) -> None:
        pass
