# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core rollout and trainer carriers for the RL experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import torch
from renderers import Message


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
        token_ids: Generated token IDs, shaped ``[response_tokens]``.
        token_logprobs: Behavior-policy logprobs, shaped ``[response_tokens]``.
        finish_reason: vLLM stop reason, for example ``"stop"`` or ``"length"``.
        error: Human-readable failure description when generation could not
            produce a usable response (e.g. oversized prompt, vLLM-side
            abort). ``None`` for successful generations.
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
        ``T_p``: prompt token length for this turn.
        ``T_r``: response token length for this turn.
        ``M_p``: prompt message count, including full history rendered this turn.
        ``M_r``: response message count, assistant plus env messages.
    """

    prompt_token_ids: list[int]  # [T_p]
    response_token_ids: list[int]  # [T_r]
    response_logprobs: list[float]  # [T_r]
    policy_version: int
    prompt_messages: list[Message] = field(default_factory=list)  # [M_p]
    response_messages: list[Message] = field(default_factory=list)  # [M_r]
    finish_reason: str | None = None


@dataclass(kw_only=True, slots=True)
class RolloutOutput:
    """A complete rollout for one GRPO sample.

    Args:
        group_id: Stable ID for the prompt group used for advantage centering.
        sample_idx: Sibling index within the group.
        turns: Ordered rollout turns.
        status: Terminal status.
        reward: Final scalar reward. Terminal successful/truncated rollouts must
            stamp it; error rollouts may leave it ``None``.
        reward_components: Final decomposed reward metrics.

    Example::

        output = RolloutOutput(
            group_id="step=3/group=7",
            sample_idx=2,
            status=RolloutStatus.COMPLETED,
            turns=[turn0, turn1],
            reward=0.75,
            reward_components={"similarity": 0.75},
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
    """Raise ``ValueError`` when rollout token/logprob shapes are inconsistent.

    ERROR rollouts with no turns are allowed (failures before the first
    generator response). Every other status must have at least one turn.
    """
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

    ``loss_mask`` selects generated assistant tokens that GRPO trains on.
    Prompt, user, tool, and structural tokens stay in ``token_ids`` for
    context but have zero mask, zero behavior logprob, and zero advantage.

    Example::

        ReplaySample(
            token_ids=[101, 102, 201, 202],
            loss_mask=[0, 0, 1, 1],
            behavior_logprobs=[0.0, 0.0, -0.7, -0.2],
            advantage=0.4,
            group_id="step=0/group=1",
            sample_idx=3,
            behavior_version=12,
            reward=1.0,
        )
    """

    token_ids: list[int]
    loss_mask: list[int]
    behavior_logprobs: list[float]
    advantage: float
    group_id: str
    sample_idx: int
    behavior_version: int
    reward: float
    reward_components: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lengths = {
            "token_ids": len(self.token_ids),
            "loss_mask": len(self.loss_mask),
            "behavior_logprobs": len(self.behavior_logprobs),
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
    """Packed trainer batch with token-level GRPO inputs.

    All four tensors are length ``T`` so the microbatch slicer can use
    sample-aligned token offsets uniformly. The trainer shifts
    ``loss_mask``, ``behavior_logprobs``, ``advantages`` by one when
    aligning them with ``policy_logprobs`` (which is shape ``[1, T-1]``
    because position ``t`` predicts ``token_ids[t+1]``).
    """

    token_ids: torch.Tensor  # [1, T]
    seq_lens: list[int]  # [num_samples]
    loss_mask: torch.Tensor  # [1, T]
    behavior_logprobs: torch.Tensor  # [1, T]
    advantages: torch.Tensor  # [1, T]


@dataclass(frozen=True, slots=True)
class OptimStepOutput:
    """Result returned by ``PolicyTrainer.optim_step`` to the controller."""

    policy_version: int
    metrics: dict[str, float]
