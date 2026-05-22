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

    `loss` is the scalar tensor backward is called on. `sum_metrics` are
    pre-normalized so a SUM all-reduce across the loss mesh reconstructs
    the global mean; `max_metrics` are MAX-reduced.

    Example::

        # Loss is a scalar; metrics are scalar tensors carrying per-rank
        # contributions that the trainer all-reduces.
        out = LossOutput(
            loss=torch.tensor(0.42, requires_grad=True),
            sum_metrics={
                "loss/mean": torch.tensor(0.42),
                "loss/ratio/mean": torch.tensor(1.01),  # local_sum / global_N
            },
            max_metrics={"loss/ratio/max_abs": torch.tensor(1.18)},
        )
        out.loss.backward()
    """

    loss: torch.Tensor
    sum_metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    max_metrics: dict[str, torch.Tensor] = field(default_factory=dict)
