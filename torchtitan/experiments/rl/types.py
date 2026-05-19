# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core rollout and trainer carriers for the RL experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

import torch
from renderers import Message


LogValue: TypeAlias = str | int | float | bool | None
Logs: TypeAlias = dict[str, LogValue]


class RolloutStatus(StrEnum):
    """Terminal status for one turn or one rollout."""

    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ERROR = "error"


@dataclass(kw_only=True, slots=True)
class Completion:
    """One sampled assistant response.

    Args:
        policy_version: Generator weight version at request admission.
        prompt_idx: Position of the source prompt in the submitted request batch.
        text: Decoded completion text from vLLM.
        token_ids: Generated token IDs, shaped ``[response_tokens]``.
        token_logprobs: Behavior-policy logprobs, shaped ``[response_tokens]``.
        finish_reason: vLLM stop reason, for example ``"stop"`` or ``"length"``.
    """

    policy_version: int
    prompt_idx: int
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str | None = None


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
    prompt_messages: list[Message] = field(default_factory=list)  # [M_p]
    response_messages: list[Message] = field(default_factory=list)  # [M_r]
    policy_version: int = 0
    finish_reason: str | None = None
    status: RolloutStatus = RolloutStatus.COMPLETED
    reward_components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    logs: Logs = field(default_factory=dict)


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
            turns=[turn0, turn1],
            status=RolloutStatus.COMPLETED,
            reward=0.75,
            reward_components={"similarity": 0.75},
        )
    """

    group_id: str
    sample_idx: int
    turns: list[RolloutTurn] = field(default_factory=list)
    status: RolloutStatus = RolloutStatus.COMPLETED
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    logs: Logs = field(default_factory=dict)

    @property
    def behavior_version(self) -> int:
        """Earliest policy version used by any turn in this rollout."""
        if not self.turns:
            return 0
        return min(turn.policy_version for turn in self.turns)


def rollout_messages(output: RolloutOutput) -> list[Message]:
    """Return the final conversation visible at rollout termination."""
    messages: list[Message] = []
    for turn in output.turns:
        if not messages:
            messages.extend(turn.prompt_messages)
        messages.extend(turn.response_messages)
    return messages


def validate_rollout_output(output: RolloutOutput) -> None:
    """Raise ``ValueError`` when rollout token/logprob shapes are inconsistent."""
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
            advantages=[0.0, 0.0, 0.4, 0.4],
            group_id="step=0/group=1",
            sample_idx=3,
            behavior_version=12,
            reward=1.0,
        )
    """

    token_ids: list[int]
    loss_mask: list[int]
    behavior_logprobs: list[float]
    advantages: list[float]
    group_id: str
    sample_idx: int
    behavior_version: int
    reward: float
    reward_components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lengths = {
            "token_ids": len(self.token_ids),
            "loss_mask": len(self.loss_mask),
            "behavior_logprobs": len(self.behavior_logprobs),
            "advantages": len(self.advantages),
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
    """Packed trainer batch with token-level GRPO inputs."""

    token_ids: torch.Tensor  # [1, total_tokens]
    seq_lens: list[int]  # [num_samples]
    loss_mask: torch.Tensor  # [1, total_tokens]
    behavior_logprobs: torch.Tensor  # [1, total_tokens]
    advantages: torch.Tensor  # [1, total_tokens]
    behavior_versions: torch.Tensor  # [num_samples]
    rewards: torch.Tensor  # [num_samples]


@dataclass(frozen=True, slots=True)
class OptimStepOutput:
    """Result returned by ``PolicyTrainer.optim_step`` to the controller."""

    policy_version: int
    metrics: dict[str, float]
