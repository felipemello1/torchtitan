# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded rollout conversation logging for smoke/debug runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from renderers import Message

from torchtitan.experiments.rl.types import RolloutOutput


class RolloutSampleLogger:
    """Writes a capped JSONL sample of rollout conversations."""

    def __init__(
        self,
        output_dir: str,
        *,
        max_groups_per_step: int = 2,
        filename: str = "rollout_samples.jsonl",
    ) -> None:
        if max_groups_per_step < 0:
            raise ValueError(
                f"max_groups_per_step must be non-negative, got {max_groups_per_step}"
            )
        self._max_groups_per_step = max_groups_per_step
        self._path = Path(output_dir) / filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._counts: dict[tuple[str, int], int] = {}

    def write(
        self,
        *,
        step: int,
        phase: str,
        rollouts: list[RolloutOutput],
    ) -> None:
        if self._max_groups_per_step == 0 or not rollouts:
            return
        key = (phase, step)
        remaining = self._max_groups_per_step - self._counts.get(key, 0)
        if remaining <= 0:
            return

        selected_group_ids: list[str] = []
        seen_groups: set[str] = set()
        for rollout in rollouts:
            if rollout.group_id in seen_groups:
                continue
            selected_group_ids.append(rollout.group_id)
            seen_groups.add(rollout.group_id)
            if len(selected_group_ids) >= remaining:
                break
        self._counts[key] = self._counts.get(key, 0) + len(selected_group_ids)
        selected_group_set = set(selected_group_ids)

        with self._path.open("a", encoding="utf-8") as handle:
            for rollout in rollouts:
                if rollout.group_id not in selected_group_set:
                    continue
                handle.write(
                    json.dumps(
                        _rollout_record(step=step, phase=phase, rollout=rollout),
                        sort_keys=True,
                    )
                )
                handle.write("\n")


def _rollout_record(
    *,
    step: int,
    phase: str,
    rollout: RolloutOutput,
) -> dict[str, Any]:
    return {
        "step": step,
        "phase": phase,
        "group_id": rollout.group_id,
        "sample_idx": rollout.sample_idx,
        "status": str(rollout.status),
        "reward": rollout.reward,
        "reward_components": rollout.reward_components,
        "behavior_version": rollout.behavior_version if rollout.turns else None,
        "max_behavior_version": (
            rollout.max_behavior_version if rollout.turns else None
        ),
        "turns": [
            {
                "turn_idx": turn_idx,
                "policy_version": turn.policy_version,
                "finish_reason": turn.finish_reason,
                "prompt_token_count": len(turn.prompt_token_ids),
                "response_token_count": len(turn.response_token_ids),
                **_logprob_stats(turn.response_logprobs),
                "messages": _messages_for_json(
                    [*turn.prompt_messages, *turn.response_messages]
                ),
            }
            for turn_idx, turn in enumerate(rollout.turns)
        ],
    }


def _logprob_stats(logprobs: list[float]) -> dict[str, Any]:
    finite_logprobs = [value for value in logprobs if math.isfinite(value)]
    return {
        "response_logprob_count": len(logprobs),
        "response_logprob_nonfinite_count": len(logprobs) - len(finite_logprobs),
        "response_logprob_finite_min": (
            min(finite_logprobs) if finite_logprobs else None
        ),
        "response_logprob_finite_max": (
            max(finite_logprobs) if finite_logprobs else None
        ),
    }


def _messages_for_json(messages: list[Message]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in message.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        for message in messages
    ]
