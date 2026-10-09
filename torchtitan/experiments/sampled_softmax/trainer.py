# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer that drives the sampled-softmax schedule and logs its metrics next to the standard ones."""

from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.sampled_softmax.loss import SampledSoftmaxChunkedLoss
from torchtitan.trainer import Trainer


class SampledSoftmaxTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        loss_fn = self.engine.loss_fn
        if not isinstance(loss_fn, SampledSoftmaxChunkedLoss):
            return
        # Keep the schedule's step fractions tied to --training.steps overrides.
        loss_fn.config.total_steps = config.training.steps
        base_log = self.metrics_processor.log
        parallel_dims = self.engine.parallel_dims

        def log(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
            extra: dict[str, Any] = dict(extra_metrics or {})
            loss_mesh = (
                parallel_dims.get_optional_mesh("loss")
                if parallel_dims.dp_cp_enabled
                else None
            )
            for key, value in loss_fn.step_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = (
                        dist_utils.dist_mean(value.detach().float(), loss_mesh)
                        if loss_mesh is not None
                        else float(value)
                    )
                extra[key] = value
            base_log(
                step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=extra
            )

        self.metrics_processor.log = log  # pyrefly: ignore[bad-assignment]

    def train_step(self, data_iterator):
        loss_fn = self.engine.loss_fn
        if isinstance(loss_fn, SampledSoftmaxChunkedLoss):
            loss_fn.step = self.engine.num_completed_steps + 1
        super().train_step(data_iterator)
