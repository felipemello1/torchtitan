# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sampling configuration shared by RL rollouts and the vLLM actor."""

from dataclasses import dataclass, field


TRAINING_VLLM_LOGPROBS_MODE = "processed_logprobs"
"""vLLM logprob mode used for behavior-policy probabilities in training."""


@dataclass(kw_only=True, slots=True)
class SamplingConfig:
    """Sampling parameters passed to vLLM's ``SamplingParams``."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 1.0
    """Nucleus sampling threshold."""

    max_tokens: int = 100
    """Maximum number of tokens to generate per completion."""

    stop_token_ids: list[int] = field(default_factory=list)
    """Token IDs that stop generation at the assistant-turn boundary."""


@dataclass(frozen=True, slots=True)
class TrainingLogprobConfig:
    """Validated policy-logprob transform used by the RL loss.

    vLLM returns behavior logprobs after sampling processors when configured
    with ``TRAINING_VLLM_LOGPROBS_MODE``. The trainer must compute policy
    logprobs in the same space before forming importance ratios.
    """

    temperature: float
    """Temperature applied before trainer-side ``log_softmax``."""

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(
                "training logprob temperature must be positive, "
                f"got {self.temperature}"
            )

    @classmethod
    def from_sampling(cls, sampling: SamplingConfig) -> "TrainingLogprobConfig":
        """Build the trainer logprob contract for a generation sampling config."""
        if sampling.top_p != 1.0:
            raise ValueError(
                "trainer logprob correction currently supports top_p=1.0 only; "
                f"got {sampling.top_p}"
            )
        return cls(temperature=sampling.temperature)
