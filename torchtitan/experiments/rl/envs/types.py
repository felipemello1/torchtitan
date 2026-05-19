# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Message-level environment protocol for RL rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, TypeAlias

from renderers import Message, ToolSpec

from torchtitan.experiments.rl.actors.generator import SamplingConfig
from torchtitan.experiments.rl.types import RolloutStatus


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, kw_only=True, slots=True)
class EnvExample:
    """One dataset row used to build a GRPO rollout group.

    Args:
        group_id: Stable ID for logging and advantage centering.
        step: Trainer step that sampled the row.
        group_idx: Position within the step batch.
        payload: JSON-serializable task data consumed by an env builder.
        sampling: Optional row-specific sampling override.
        tags: Logging tags such as ``("alphabet_sort", "train")``.
    """

    group_id: str
    step: int
    group_idx: int
    payload: dict[str, JsonValue] = field(default_factory=dict)
    sampling: SamplingConfig | None = None
    tags: tuple[str, ...] = ()


@dataclass(kw_only=True, slots=True)
class EnvReset:
    """Initial message state returned by :meth:`MessageEnv.reset`."""

    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(kw_only=True, slots=True)
class EnvStep:
    """Environment response to one assistant message.

    Terminal steps should set ``done=True`` and stamp ``reward``. Non-terminal
    steps return the follow-up messages to append to the conversation.
    """

    messages: list[Message] = field(default_factory=list)
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    done: bool = False
    status: RolloutStatus | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    logs: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@runtime_checkable
class MessageEnv(Protocol):
    """Single-use message environment.

    Pure Python envs can implement these methods as async functions that return
    immediately; sandboxed/browser/tool envs can await their backing services.
    """

    async def reset(self) -> EnvReset:
        ...

    async def step(self, assistant_message: Message) -> EnvStep:
        ...

    async def close(self) -> None:
        ...


class EnvDataset(Protocol):
    """Dataset that yields one example per rollout group."""

    def sample_groups(self, *, step: int, num_groups: int) -> list[EnvExample]:
        ...


class EnvBuilder(Protocol):
    """Builds one single-use env for a sampled example and sibling index."""

    def build(self, *, example: EnvExample, sample_idx: int) -> MessageEnv:
        ...

    def logging_tags(self) -> tuple[str, ...]:
        ...
