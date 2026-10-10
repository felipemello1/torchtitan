# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.rl.observability import metrics as m


@dataclass(frozen=True, slots=True)
class RolloutTurnID:
    """A turn's id: (group, sibling rollout, turn index); renders to the generator request_id.

    Example:

        RolloutTurnID(group_id=5, rollout_id=2, turn_id=0).to_string()
        # -> "group=5/rollout=2/turn=0"
        RolloutTurnID(group_id=5, rollout_id=2, turn_id=0).to_string(include_turn=False)
        # -> "group=5/rollout=2"
    """

    group_id: int
    """Globally-unique GRPO group id; siblings share it (the sticky-routing key, sans turn)."""
    rollout_id: int
    """Sibling index within the group (0..group_size-1)."""
    turn_id: int
    """Turn index within the rollout; for a TrainingSample, the turn where begins.
    This is not 0 when a single rollout is split into multiple training samples."""

    def to_string(self, *, include_turn: bool = True) -> str:
        base = f"group={self.group_id}/rollout={self.rollout_id}"
        return f"{base}/turn={self.turn_id}" if include_turn else base


@dataclass(kw_only=True, slots=True)
class Completion:
    """A single generated sequence from the generator.

    Example:

        Completion(min_policy_version=7, max_policy_version=7, request_id="r0", token_ids=[12, 9],
                   token_logprobs=[-0.2, -1.1], finish_reason="stop",
                   metrics=[Metric("generator/queue_time_ms", ...)])
    """

    min_policy_version: int
    """Oldest policy version among this turn's decode."""
    max_policy_version: int
    """Newest policy version among this turn's decode."""
    # TODO(async-rl): for exact per-token version attribution, switch the engine to
    #   RequestOutputKind.CUMULATIVE and record (start_token, version) boundaries; today we keep only
    #   the per-turn min (min_policy_version) / max (max_policy_version).
    request_id: str
    """Echoes the id the caller passed to `generate`, so callers can validate
    ordered completions or map by id."""
    token_ids: list[int]
    token_logprobs: list[float]
    topk_token_ids: torch.Tensor | None = None
    """[num_tokens, k] int32 ids of the generator's k most likely tokens at each position;
    None unless `SamplingConfig.num_topk_logprobs` > 0."""
    topk_logprobs: torch.Tensor | None = None
    """[num_tokens, k] float32 generator logprobs of `topk_token_ids`."""
    finish_reason: str | None = None
    """vLLM `CompletionOutput.finish_reason` ("stop" | "length" | "abort")"""

    metrics: list[m.Metric] = field(default_factory=list)
    """Per-generation metrics measured by the generator (latencies); the
    controller attaches them to the rollout turn."""


@dataclass(kw_only=True, slots=True)
class TrainingSample:
    """A trainable token sequence from a rollout.

    Example:
        # Turn 0: prompt P0 -> assistant A0 -> env reply E0
        # Turn 1: prompt [P0, A0, E0] -> assistant A1
        TrainingSample(
            rollout_id=RolloutTurnID(group_id=3, rollout_id=1, turn_id=1),
            min_policy_version=7,
            max_policy_version=9,            # weights updated during rollout
            token_ids=P0 + A0 + E0 + A1,
            loss_mask=[False]*len(P0) + [True]*len(A0) + [False]*len(E0) + [True]*len(A1),
            logprobs=[0.0]*len(P0) + logprobs_A0 + [0.0]*len(E0) + logprobs_A1,
            advantage=[0.0]*len(P0) + [adv]*len(A0) + [0.0]*len(E0) + [adv]*len(A1),
        )
    """

    min_policy_version: int
    """Oldest policy version among this branch's trained turns."""
    max_policy_version: int
    """Newest policy version among this branch's trained turns."""
    rollout_id: RolloutTurnID
    """This sample identifier."""
    token_ids: list[int]
    """[L] packed prompt + completions + env replies."""
    loss_mask: list[bool]
    """[L] True on assistant tokens to train."""
    logprobs: list[float]
    """[L] generator logprobs; 0.0 where loss_mask is False."""
    advantage: list[float]
    """[L] advantage on assistant tokens, 0.0 elsewhere."""
    topk_token_ids: torch.Tensor | None = None
    """[num_loss_tokens, k] generator top-k token ids, one row per True in loss_mask. None
    unless `SamplingConfig.num_topk_logprobs` > 0."""
    topk_logprobs: torch.Tensor | None = None
    """[num_loss_tokens, k] generator logprobs of `topk_token_ids`."""


@dataclass(frozen=True, slots=True)
class TrainingSampleGroup:
    """The training samples + metrics built from one rollout group.

    Example:
        TrainingSampleGroup(group_id=3, training_samples=[], metrics=[failure_metric])
        # -> failed / filtered / zero-std group; metrics still reach the trainer logger
    """

    group_id: int
    training_samples: list[TrainingSample]
    metrics: list[m.Metric]


@dataclass(kw_only=True, slots=True)
class TrainingMicrobatch(TokenizedTrainingMicrobatch):
    """Packed training microbatch for the RL trainer.

    Each training_sample's raw tokens (length N) are split into
    ``token_ids = raw[:-1]`` and ``labels = raw[1:]`` (both length
    N-1), matching the pre-training dataloader convention.
    """

    generator_logprobs: torch.Tensor  # [T]
    temperature: torch.Tensor  # [T]
    loss_mask: torch.Tensor  # [T]
    advantages: torch.Tensor  # [T]
    generator_topk_token_ids: torch.Tensor | None = None  # [num_loss_tokens, k]
    """One row per True in loss_mask; `to_loss_kwargs` expands them to [T, k] for the loss."""
    generator_topk_logprobs: torch.Tensor | None = None  # [num_loss_tokens, k]

    def loss_kwargs(self) -> dict[str, torch.Tensor]:
        """Loss kwargs on the CPU; top-k rows stay one per loss token until `to_loss_kwargs`."""
        loss_kwargs = {
            "generator_logprobs": self.generator_logprobs,
            "temperature": self.temperature,
            "loss_mask": self.loss_mask,
            "advantages": self.advantages,
        }
        if self.generator_topk_token_ids is not None:
            loss_kwargs["generator_topk_token_ids"] = self.generator_topk_token_ids
            loss_kwargs["generator_topk_logprobs"] = self.generator_topk_logprobs
        return loss_kwargs

    def to_loss_kwargs(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> dict[str, torch.Tensor]:
        """Move the loss kwargs to `device`, then expand the top-k to one row per token.

        Expanding after the move copies only the loss-token rows to `device`.

        Example:

            loss_mask                = [False, True, True, False]
            generator_topk_token_ids = [[5, 6], [7, 8]]
            # -> generator_topk_token_ids = [[0, 0], [5, 6], [7, 8], [0, 0]]
        """
        # slots=True breaks zero-arg super(), so call the parent explicitly.
        loss_kwargs = TokenizedTrainingMicrobatch.to_loss_kwargs(
            self, device, non_blocking=non_blocking
        )
        if self.generator_topk_token_ids is not None:
            loss_mask = loss_kwargs["loss_mask"].unsqueeze(-1)
            for key in ("generator_topk_token_ids", "generator_topk_logprobs"):
                loss_rows = loss_kwargs[key]
                rows = loss_rows.new_zeros(loss_mask.shape[0], loss_rows.shape[1])
                # masked_scatter_, unlike indexing with the mask, needs no host sync on CUDA.
                loss_kwargs[key] = rows.masked_scatter_(loss_mask, loss_rows)
        return loss_kwargs


@dataclass(frozen=True, slots=True)
class TrainerStepBatch:
    """Packed microbatches for one optimizer step.

    Example:
        # 5 training samples, effective length 5 each; 2 rows/rank, dp_degree=1
        # next-fit rows -> [[s5, s5], [s5, s5], [s5]] = 3 rows; rows_per_microbatch = 2 * 1 = 2
        # -> 2 microbatches (3 rows padded to 4 with one pad-only row):
        #    microbatches = [[TrainingMicrobatch(input=[20])],
        #                    [TrainingMicrobatch(input=[20])]]
        # The second microbatch contains one real row and one pad row.
        # global_loss_token_counts = per-objective loss-token counts
        # global_routing_token_counts = per-depth non-padding token counts
    """

    microbatches: list[list[TrainingMicrobatch]]  # [num_microbatches][dp_degree]
    global_loss_token_counts: torch.Tensor
    global_routing_token_counts: torch.Tensor
    metrics: list[m.Metric]
    group_ids: list[int]
    """Every consumed rollout group, including metric-only groups."""
    # one per packed training_sample; trainer computes policy_age at consume time
    min_policy_versions: list[int]


@dataclass(frozen=True, slots=True)
class OptimizerStepOutput:
    """Result returned by `Trainer.optim_step` to the controller."""

    policy_version: int
    metrics: dict[str, float]
