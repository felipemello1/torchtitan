# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Logging boundary: schedule + re-export of replicate_to_host.

Ported from reference schedule.py. Changes:
  OSS_COMPAT — removed Node base class and Config inner class.
"""


class EveryNSteps:
    """Returns True every N steps, with optional additional steps.

    Step 0 is excluded by default — use additional_steps={0} to include it.
    Ported from reference EveryNSteps with Node/Config removed.

    Args:
        every_n: Returns True if step > 0 and step % every_n == 0.
        additional_steps: Optional set of steps where action is also taken.

    Raises:
        ValueError: If every_n <= 0.
    """

    def __init__(self, every_n: int, additional_steps: set[int] | None = None) -> None:
        if every_n <= 0:
            raise ValueError("every_n must be positive")
        self.every_n = every_n
        self.additional_steps = additional_steps

    def __call__(self, step: int) -> bool:
        return (step > 0 and step % self.every_n == 0) or (
            self.additional_steps is not None and step in self.additional_steps
        )


# Re-export for convenience (PR2)
from torchtitan.observability.tensor_metrics import replicate_to_host  # noqa: E402, F401
