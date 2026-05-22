# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Microbatch scheduling helpers for RL trainer replay batches."""

from dataclasses import dataclass

import torch

from torchtitan.experiments.rl.types import TrainingBatch


@dataclass(frozen=True, slots=True)
class ScheduledTrainingMicrobatch:
    batch: TrainingBatch
    is_real: bool


@dataclass(frozen=True, slots=True)
class TrainingMicrobatchSchedule:
    microbatches: list[ScheduledTrainingMicrobatch]
    max_microbatches: int
    max_seq_len: int


def slice_training_batch(
    batch: TrainingBatch,
    *,
    start_sample: int,
    end_sample: int,
    start_token: int,
    end_token: int,
) -> TrainingBatch:
    """Slice a packed batch on sample-aligned token offsets."""
    return TrainingBatch(
        token_ids=batch.token_ids[:, start_token:end_token].contiguous(),
        seq_lens=list(batch.seq_lens[start_sample:end_sample]),
        ref_logprobs=batch.ref_logprobs[:, start_token:end_token].contiguous(),
        loss_mask=batch.loss_mask[:, start_token:end_token].contiguous(),
        advantages=batch.advantages[:, start_token:end_token].contiguous(),
    )


def split_training_batch(
    batch: TrainingBatch,
    *,
    max_samples: int | None,
    max_tokens: int | None,
) -> list[TrainingBatch]:
    """Split a packed rank-local batch without cutting through a sample."""
    if not batch.seq_lens:
        raise ValueError("TrainingBatch.seq_lens must be non-empty")
    if max_samples is None and max_tokens is None:
        return [batch]

    splits: list[TrainingBatch] = []
    start_sample = 0
    start_token = 0
    current_tokens = 0

    for end_sample, seq_len in enumerate(batch.seq_lens):
        if seq_len <= 0:
            raise ValueError(f"seq_lens must be positive, got {seq_len}")
        current_samples = end_sample - start_sample
        should_flush = current_samples > 0 and (
            (max_samples is not None and current_samples >= max_samples)
            or (max_tokens is not None and current_tokens + seq_len > max_tokens)
        )
        if should_flush:
            end_token = start_token + current_tokens
            splits.append(
                slice_training_batch(
                    batch,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_token=start_token,
                    end_token=end_token,
                )
            )
            start_sample = end_sample
            start_token = end_token
            current_tokens = 0
        current_tokens += seq_len

    splits.append(
        slice_training_batch(
            batch,
            start_sample=start_sample,
            end_sample=len(batch.seq_lens),
            start_token=start_token,
            end_token=start_token + current_tokens,
        )
    )
    return splits


def split_training_batches_by_rank(
    batches: list[TrainingBatch],
    *,
    max_samples: int | None,
    max_tokens: int | None,
) -> tuple[list[list[TrainingBatch]], int]:
    if not batches:
        raise ValueError("train_data must contain at least one TrainingBatch")
    splits_by_rank = [
        split_training_batch(
            batch,
            max_samples=max_samples,
            max_tokens=max_tokens,
        )
        for batch in batches
    ]
    max_microbatches = max(len(splits) for splits in splits_by_rank)
    return splits_by_rank, max_microbatches


def has_loss_tokens(batch: TrainingBatch) -> bool:
    return bool(batch.loss_mask.any().item())


def zero_gradient_training_batch_like(reference: TrainingBatch) -> TrainingBatch:
    loss_mask = torch.zeros(
        (1, 2),
        dtype=torch.float32,
        device=reference.loss_mask.device,
    )
    loss_mask[:, 1] = 1.0
    return TrainingBatch(
        token_ids=reference.token_ids.new_zeros((1, 2)),
        seq_lens=[2],
        ref_logprobs=reference.ref_logprobs.new_zeros((1, 2)),
        loss_mask=loss_mask,
        advantages=reference.advantages.new_zeros((1, 2)),
    )


def schedule_training_microbatches(
    train_data: list[TrainingBatch],
    *,
    dp_rank: int,
    max_samples: int | None,
    max_tokens: int | None,
) -> TrainingMicrobatchSchedule:
    """Build the local schedule while keeping all DP ranks in lockstep."""
    splits_by_rank, max_microbatches = split_training_batches_by_rank(
        train_data,
        max_samples=max_samples,
        max_tokens=max_tokens,
    )
    if dp_rank < 0 or dp_rank >= len(splits_by_rank):
        raise ValueError(
            f"dp_rank {dp_rank} is out of range for {len(splits_by_rank)} shards"
        )

    local_splits = splits_by_rank[dp_rank]
    reference = local_splits[0]
    dummy = zero_gradient_training_batch_like(reference)
    microbatches: list[ScheduledTrainingMicrobatch] = []
    for microbatch_idx in range(max_microbatches):
        if microbatch_idx < len(local_splits):
            candidate = local_splits[microbatch_idx]
            if has_loss_tokens(candidate):
                microbatches.append(
                    ScheduledTrainingMicrobatch(batch=candidate, is_real=True)
                )
                continue
        microbatches.append(ScheduledTrainingMicrobatch(batch=dummy, is_real=False))

    max_seq_len = max(max(item.batch.seq_lens) for item in microbatches)
    return TrainingMicrobatchSchedule(
        microbatches=microbatches,
        max_microbatches=max_microbatches,
        max_seq_len=max_seq_len,
    )
