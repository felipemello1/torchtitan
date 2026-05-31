# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch

AggType = Literal["token_mean", "fixed_horizon", "sequence_mean"]
RatioType = Literal["token", "sequence"]
KLType = Literal["k1", "k2", "k3"]
Reduce = Literal["sum", "max"]


@dataclass(frozen=True, slots=True)
class LossNormalization:
    """Global denominators for one optimizer step.

    Computed once by the batcher and shared by every DP rank and every
    gradient-accumulation microbatch, so that pre-normalized per-token
    contributions SUM-reduce to the exact global value.

    Attributes:
        num_global_valid_tokens (int): Total response tokens (loss_mask == 1)
            across the whole global batch. Denominator for `token_mean` and for
            every pre-normalized metric.
        num_global_sequences (int): Total source episodes across the global batch.
            Denominator for `sequence_mean`.
        num_global_fixed_horizon_tokens (int): `num_global_sequences * seq_len`;
            a length-independent constant horizon. Denominator for `fixed_horizon`.
    """

    num_global_valid_tokens: int
    num_global_sequences: int
    num_global_fixed_horizon_tokens: int


@dataclass(frozen=True, slots=True)
class LossMetric:
    """A scalar metric and how it reduces across the loss mesh and gradient-
    accumulation microbatches.

    `sum` is for pre-normalized contributions (a value already divided by a
    global denominator, so summing the per-rank/per-microbatch shares
    reconstructs the global value); `max` is for running maxima. The op that
    produces the metric sets the reduction, so the loss and trainer never have
    to know it.

    Attributes:
        value (torch.Tensor): Scalar metric value.
        reduce (Reduce): How to combine shares across ranks/microbatches.
    """

    value: torch.Tensor
    reduce: Reduce = "sum"


@dataclass
class LossOutput:
    """Output from all loss functions.

    Attributes:
        loss (torch.Tensor): Scalar loss tensor for backpropagation.
        metrics (dict[str, LossMetric]): Metric name -> value + reduction. The
            trainer groups these by `LossMetric.reduce` to reduce them across the
            loss mesh and microbatches.
    """

    loss: torch.Tensor
    metrics: dict[str, LossMetric]
