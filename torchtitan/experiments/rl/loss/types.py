# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal

import torch

AggType = Literal["token_mean", "fixed_horizon", "sequence_mean"]
RatioType = Literal["token", "sequence"]
KLType = Literal["k1", "k2", "k3"]


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


@dataclass
class LossOutput:
    """Output from all loss functions.

    Attributes:
        loss (torch.Tensor): Scalar loss tensor for backpropagation.
        sum_metrics (dict[str, torch.Tensor]): Per-rank metric shares, pre-normalized
            so SUM-reduction across the loss mesh (and microbatches) reconstructs the
            global value.
        max_metrics (dict[str, torch.Tensor]): Metric shares reduced with MAX.
    """

    loss: torch.Tensor
    sum_metrics: dict[str, torch.Tensor]
    max_metrics: dict[str, torch.Tensor] = field(default_factory=dict)
