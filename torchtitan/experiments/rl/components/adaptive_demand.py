# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure demand-estimation helpers shared by the adaptive rollout controller."""

import math
from collections.abc import Sequence


def estimate_demand_target(
    *,
    unavailable_history: Sequence[int],
    num_prompts_per_train_step: int,
    stall_probability: float,
    max_active_rollout_groups: int,
) -> int:
    """Estimate total active demand from recent unavailable group counts."""
    if not unavailable_history:
        raise ValueError("unavailable_history must not be empty")
    history = sorted(unavailable_history)
    rank = max(0, math.ceil((1 - stall_probability) * len(history)) - 1)
    unavailable_bad_day = history[rank]
    estimate = 2 * num_prompts_per_train_step + unavailable_bad_day + 1
    return min(
        max_active_rollout_groups,
        max(2 * num_prompts_per_train_step, estimate),
    )
