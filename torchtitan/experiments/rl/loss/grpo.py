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


class GRPOLoss(Configurable):
    """Symmetric clipped GRPO surrogate loss for selected replay tokens."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_eps: float = 0.2
        """Symmetric PPO clipping epsilon for the probability ratio."""

        max_log_ratio: float = 10.0
        """Clamp ``policy_logprob - behavior_logprob`` before exponentiating."""

    def __init__(self, config: Config):
        validate_clip_bound("clip_eps", config.clip_eps)
        validate_max_log_ratio(config.max_log_ratio)
        self.clip_eps = config.clip_eps
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
            clip_low=self.clip_eps,
            clip_high=self.clip_eps,
            max_log_ratio=self.max_log_ratio,
        )
