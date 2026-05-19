# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.config import Configurable
from torchtitan.experiments.rl.loss.ops import (
    clipped_policy_gradient_loss,
    validate_clip_bound,
    validate_max_log_ratio,
)


class DAPOLoss(Configurable):
    """DAPO clipped surrogate loss for selected replay tokens.

    This ports Forge's DAPO loss shape to TorchTitan's trainer contract:
    token-selected logprobs are passed in directly, and returned metrics are
    scalar tensor shares ready for the trainer's later all-reduce.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_low: float = 0.2
        """Lower PPO clipping epsilon for the probability ratio."""

        clip_high: float = 0.28
        """Upper PPO clipping epsilon for the probability ratio."""

        dual_clip_c: float = 3.0
        """DAPO dual-clip constant for negative-advantage tokens."""

        max_log_ratio: float = 10.0
        """Clamp ``policy_logprob - behavior_logprob`` before exponentiating."""

    def __init__(self, config: Config):
        validate_clip_bound("clip_low", config.clip_low)
        validate_clip_bound("clip_high", config.clip_high)
        if config.dual_clip_c <= 1:
            raise ValueError(
                f"dual_clip_c must be greater than 1, got {config.dual_clip_c}"
            )
        validate_max_log_ratio(config.max_log_ratio)
        self.clip_low = config.clip_low
        self.clip_high = config.clip_high
        self.dual_clip_c = config.dual_clip_c
        self.max_log_ratio = config.max_log_ratio

    def __call__(
        self,
        policy_logprobs: torch.Tensor,
        behavior_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        num_global_valid_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return clipped_policy_gradient_loss(
            policy_logprobs=policy_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            num_global_valid_tokens=num_global_valid_tokens,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            max_log_ratio=self.max_log_ratio,
            dual_clip_c=self.dual_clip_c,
        )
