# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sampling configuration shared by RL rollouts and the vLLM actor."""

from dataclasses import dataclass, field


TRAINING_VLLM_LOGPROBS_MODE = "processed_logprobs"
"""vLLM logprob mode used for the reference-policy (`pi_old`) probabilities
recorded at sampling time.

Returns the logprob distribution after vLLM's sampling-temperature transform,
so the trainer can recover the same distribution by dividing its own logits
by the same temperature before `log_softmax`.
"""


@dataclass(kw_only=True, slots=True)
class SamplingConfig:
    """Sampling parameters passed to vLLM's `SamplingParams`.

    Example::

        SamplingConfig(
            temperature=0.8,
            top_p=1.0,
            max_tokens=100,
            stop_token_ids=[151645],
        )
    """

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
    """Trainer-side logprob correction parameters.

    Recreates the sampling-time logprob distribution inside the trainer so
    PPO/GRPO/DAPO importance ratios operate on the same probability space
    the rollouts came from. The generator emits reference logprobs through
    vLLM's `processed_logprobs` mode (see `TRAINING_VLLM_LOGPROBS_MODE`),
    which applies the sampling temperature; the trainer divides its own
    logits by the same temperature before `log_softmax` so the two log
    spaces line up.

    Example::

        # Sampling at temperature=0.8 with no nucleus truncation.
        sampling = SamplingConfig(temperature=0.8, top_p=1.0, max_tokens=100)
        logprob_config = TrainingLogprobConfig.from_sampling(sampling)
        # logprob_config.temperature == 0.8

        # Used by `compute_logprobs` to rebuild the sampling distribution:
        # logits.float().div_(logprob_config.temperature).log_softmax(-1)
    """

    temperature: float
    """Generator sampling temperature. Trainer logits are divided by this
    value before `log_softmax`."""

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(
                "training logprob temperature must be positive, "
                f"got {self.temperature}"
            )

    @classmethod
    def from_sampling(cls, sampling: SamplingConfig) -> "TrainingLogprobConfig":
        """Build a logprob-correction contract from the generator's sampling config.

        Currently restricted to `top_p == 1.0` because nucleus sampling
        renormalizes the distribution per token; the trainer cannot
        reconstruct the same support set without replaying the sort/truncate
        step. Temperature-only sampling is recoverable analytically.

        Example::

            sampling = SamplingConfig(temperature=0.7, top_p=1.0, max_tokens=128)
            TrainingLogprobConfig.from_sampling(sampling)
            # -> TrainingLogprobConfig(temperature=0.7)

            TrainingLogprobConfig.from_sampling(
                SamplingConfig(temperature=0.7, top_p=0.95, max_tokens=128)
            )
            # -> ValueError: trainer logprob correction supports top_p=1.0 only ...
        """
        if sampling.top_p != 1.0:
            raise ValueError(
                "trainer logprob correction supports top_p=1.0 only; "
                f"got top_p={sampling.top_p}"
            )
        return cls(temperature=sampling.temperature)
