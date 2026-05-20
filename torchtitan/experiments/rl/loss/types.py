# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared types for the RL loss modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["LossOutput"]


@dataclass(frozen=True, slots=True)
class LossOutput:
    """Return value for every loss in this package.

    Each loss emits one scalar tensor (``loss`` — backproppable) and
    two dicts of per-step metrics keyed by reduce-op. The trainer
    ``all_reduce``s them across DP ranks with SUM and MAX respectively
    before logging:

    - ``sum_metrics``: values to be SUMMED across ranks
      (e.g. ``local_sum / global_N`` → SUM-reduce gives global mean).
    - ``max_metrics``: values to be MAXed across ranks
      (e.g. per-rank max importance ratio).
    """

    loss: torch.Tensor
    sum_metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    max_metrics: dict[str, torch.Tensor] = field(default_factory=dict)
