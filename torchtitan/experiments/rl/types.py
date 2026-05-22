# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
from renderers import Message

if TYPE_CHECKING:
    from torchtitan.experiments.rl.observability.metrics import Metric


class RolloutStatus(StrEnum):
    """Terminal status for one rollout."""

    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ERROR = "error"


@dataclass(kw_only=True, slots=True)
class Completion:
    """One sampled assistant response.

    Args:
        policy_version: Generator weight version at request admission.
        token_ids: Generated token IDs, shaped `[response_tokens]`.
        token_logprobs: Reference-policy logprobs, shaped `[response_tokens]`.
        finish_reason: vLLM stop reason, for example `"stop"` or `"length"`.
        error: Failure description for oversized prompts, vLLM aborts, or
            completion-build failures. `None` for successful generations.

    Example::

        Completion(
            policy_version=3,
            token_ids=[29871, 29946],
            token_logprobs=[-0.2, -0.1],
            finish_reason="stop",
        )
    """

    policy_version: int
    token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str | None = None
    error: str | None = None


@dataclass(kw_only=True, slots=True)
class RolloutTurn:
    """One generator call and the env response to that call.

    Shape legend:
        `T_p`: prompt token length for this turn.
        `T_r`: response token length for this turn.
    """

    prompt_token_ids: list[int]  # [T_p]
    response_token_ids: list[int]  # [T_r]
    response_logprobs: list[float]  # [T_r]
    policy_version: int
    prompt_messages: list[Message] = field(default_factory=list)
    response_messages: list[Message] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass(kw_only=True, slots=True)
class RolloutOutput:
    """A complete rollout for one GRPO sample.

    Args:
        group_id: Stable ID for the prompt group used for advantage centering.
        sample_idx: Sibling index within the group.
        turns: Ordered rollout turns.
        status: Terminal status.
        reward: Final scalar reward. Error rollouts may leave it `None`.
        reward_components: Final decomposed reward metrics.

    Example::

        RolloutOutput(
            group_id="step=3/group=7",
            sample_idx=2,
            status=RolloutStatus.COMPLETED,
            turns=[turn0],
            reward=1.0,
            reward_components={"correctness": 1.0},
        )
    """

    group_id: str
    sample_idx: int
    status: RolloutStatus
    turns: list[RolloutTurn] = field(default_factory=list)
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)

    @property
    def behavior_version(self) -> int:
        """Earliest policy version used by any turn in this rollout."""
        if not self.turns:
            raise ValueError("empty rollout has no behavior version")
        return min(turn.policy_version for turn in self.turns)

    @property
    def max_behavior_version(self) -> int:
        """Latest policy version used by any turn in this rollout."""
        if not self.turns:
            raise ValueError("empty rollout has no behavior version")
        return max(turn.policy_version for turn in self.turns)


def validate_rollout_output(output: RolloutOutput) -> None:
    """Raise when rollout token/logprob shapes are inconsistent."""
    if output.status != RolloutStatus.ERROR and not output.turns:
        raise ValueError(
            f"rollout group_id={output.group_id!r} sample_idx={output.sample_idx} "
            f"has no turns but status is {output.status!r}"
        )
    for turn_idx, turn in enumerate(output.turns):
        if len(turn.response_token_ids) != len(turn.response_logprobs):
            raise ValueError(
                f"turn {turn_idx}: response_token_ids "
                f"[{len(turn.response_token_ids)}] != response_logprobs "
                f"[{len(turn.response_logprobs)}]"
            )
    if output.status != RolloutStatus.ERROR and output.turns and output.reward is None:
        raise ValueError(
            f"rollout group_id={output.group_id!r} sample_idx={output.sample_idx} "
            "finished without a reward"
        )


@dataclass(kw_only=True, slots=True)
class ReplaySample:
    """Token-aligned trainer row derived from one rollout segment.

    Example::

        ReplaySample(
            token_ids=[101, 102, 201, 202],
            loss_mask=[0, 0, 1, 1],
            ref_logprobs=[0.0, 0.0, -0.7, -0.2],
            advantage=0.4,
            group_id="step=0/group=1",
            sample_idx=3,
            behavior_version=12,
            reward=1.0,
        )
    """

    token_ids: list[int]
    loss_mask: list[int]
    ref_logprobs: list[float]
    advantage: float
    group_id: str
    sample_idx: int
    behavior_version: int
    reward: float
    reward_components: dict[str, float] = field(default_factory=dict)
    metrics: "tuple[Metric, ...]" = ()
    """Typed metrics that should be emitted only if this sample is consumed."""

    def __post_init__(self) -> None:
        lengths = {
            "token_ids": len(self.token_ids),
            "loss_mask": len(self.loss_mask),
            "ref_logprobs": len(self.ref_logprobs),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"ReplaySample fields must have equal lengths: {lengths}")
        if not any(self.loss_mask):
            raise ValueError("ReplaySample must contain at least one loss token")

    @property
    def num_loss_tokens(self) -> int:
        return sum(self.loss_mask)


@dataclass(kw_only=True, slots=True)
class TrainingBatch:
    """Packed trainer batch with sample boundaries preserved."""

    token_ids: torch.Tensor  # [1, T]
    seq_lens: list[int]
    ref_logprobs: torch.Tensor  # [1, T]; 0.0 for prompt tokens
    loss_mask: torch.Tensor  # [1, T]; 1.0 for response tokens
    advantages: torch.Tensor  # [1, T]; per-token, 0.0 for prompt tokens


@dataclass(frozen=True, slots=True)
class OptimStepOutput:
    """Result returned by ``PolicyTrainer.optim_step`` to the controller."""

    policy_version: int
    metrics: dict[str, float]
