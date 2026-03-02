# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoggingGrad: identity in forward, logs gradient stats in backward.

backward() is never traced by torch.compile — safe inside compiled regions.
Under activation checkpointing: forward runs twice (identity, harmless),
backward runs once (correct).
"""

import torch
from torchtitan.observability.invocation_context import record_tensor_metric
from torchtitan.observability.tensor_metrics import MaxTMetric, MeanTMetric


class LoggingGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, prefix):
        ctx.prefix = prefix
        return x

    @staticmethod
    def backward(ctx, grad_out):
        record_tensor_metric(
            f"{ctx.prefix}/grad/abs_mean",
            MeanTMetric(mean=grad_out.abs().mean(), weight=1),
        )
        record_tensor_metric(
            f"{ctx.prefix}/grad/abs_max",
            MaxTMetric(grad_out.abs().max()),
        )
        return grad_out, None
