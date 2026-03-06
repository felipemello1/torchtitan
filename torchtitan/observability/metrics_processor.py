# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.observability.common import clear_step_tags, set_step


class MetricsProcessor:
    """Processes and logs training metrics.

    Manages step context so all record_span/record_event calls are tagged
    with the current training step in JSONL output.
    """

    class Config:
        pass

    def __init__(
        self, config: "MetricsProcessor.Config", *, dump_folder: str, rank: int = 0
    ):
        # Assumes init_observability() was already called by the caller.
        self._config = config

    def set_step(self, step: int) -> None:
        """Set current training step. Call before train_step().

        Sets the ContextVar so all subsequent record_span/record_event calls
        are tagged with this step in JSONL.
        """
        self._step = step
        set_step(step)
        clear_step_tags()

    def close(self) -> None:
        pass
