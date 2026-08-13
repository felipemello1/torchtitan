# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


_ALL_TOKENS = 0
_VALID_TOKENS = 1
_SEGMENT_LENGTH_SUM = 2
_SEGMENT_COUNT = 3
_SUM_LENGTH_SQUARED = 4
_BATCH_COUNT = 5
_STEP_COUNT = 6


class DataStatistics:
    """Accumulate exact data counters between tensor-logging publications.

    Example:

        stats.record_batch(labels, positions)
        stats.record_step_loss(loss, global_valid_tokens)
        metrics = stats.collect()  # WORLD-reduce, return metrics, then reset
    """

    def __init__(
        self,
        *,
        dataset_id: str,
        data_contributor: bool,
        loss_contributor: bool,
        step_contributor: bool,
        ignore_index: int,
        device: torch.device,
    ) -> None:
        self.dataset_id = dataset_id
        self.data_contributor = data_contributor
        self.loss_contributor = loss_contributor
        self.step_contributor = step_contributor
        self.ignore_index = ignore_index
        self.integers = torch.zeros(7, dtype=torch.int64, device=device)
        self.loss_sum = torch.zeros((), dtype=torch.float32, device=device)

    @torch.no_grad()
    def record_batch(
        self,
        labels: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> None:
        """Add token, batch, and packed-document counts from one local batch.

        Args:
            labels: Token labels with shape `[B, L]`; `ignore_index` marks masked tokens.
            positions: Optional packed-document positions with shape `[B, L]`, resetting to zero at each segment.
        """

        if not self.data_contributor:
            return

        # Token and batch counts apply whether or not document positions exist.
        self.integers[_ALL_TOKENS].add_(labels.numel())
        self.integers[_VALID_TOKENS].add_(
            torch.count_nonzero(labels != self.ignore_index)
        )
        self.integers[_BATCH_COUNT].add_(1)

        if positions is None:
            return

        # A segment ends immediately before the next zero position or at row end.
        segment_ends = torch.empty_like(positions, dtype=torch.bool)
        segment_ends[:, :-1] = positions[:, 1:] == 0
        segment_ends[:, -1] = True
        segment_lengths = positions.to(torch.int64) + 1
        self.integers[_SEGMENT_LENGTH_SUM].add_(
            torch.sum(segment_lengths * segment_ends)
        )
        self.integers[_SEGMENT_COUNT].add_(torch.count_nonzero(segment_ends))
        self.integers[_SUM_LENGTH_SQUARED].add_(
            torch.sum(segment_lengths.square() * segment_ends)
        )

    @torch.no_grad()
    def record_step_loss(
        self,
        normalized_loss: torch.Tensor,
        global_valid_tokens: float | torch.Tensor,
    ) -> None:
        """Accumulate a token-weighted loss numerator and one logical step."""

        if self.loss_contributor:
            local_loss = normalized_loss.detach()
            if isinstance(local_loss, DTensor):
                local_loss = local_loss.to_local()
            self.loss_sum.add_(local_loss.float() * global_valid_tokens)
        if self.step_contributor:
            self.integers[_STEP_COUNT].add_(1)

    @torch.no_grad()
    def collect(self) -> dict[str, int | float]:
        """WORLD-reduce the current window, derive metrics, and clear local state.

        Example:

            # After two batches and one optimizer step:
            metrics = stats.collect()
            # metrics["data/datasets.c4.window_steps"] == 1
            # A second collect() starts from an empty window.
        """

        # Counts and the weighted-loss numerator share one exact reduction.
        packed = torch.cat(
            [
                self.integers.to(torch.float64),
                self.loss_sum.to(torch.float64).view(1),
            ]
        )
        if dist.is_initialized():
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)

        # Publication owns the window boundary: a successful collect starts fresh.
        self.integers.zero_()
        self.loss_sum.zero_()

        (
            all_tokens,
            valid_tokens,
            segment_length_sum,
            segment_count,
            sum_length_squared,
            batch_count,
            step_count,
        ) = (int(value) for value in packed[:7].cpu())
        loss_sum = float(packed[7].cpu())
        # Ratios are derived only after reducing their numerators and denominators.
        metrics: dict[str, int | float] = {
            f"data/datasets.{self.dataset_id}.all_token_count": all_tokens,
            f"data/datasets.{self.dataset_id}.valid_token_count": valid_tokens,
            f"data/datasets.{self.dataset_id}.batch_count": batch_count,
            f"data/datasets.{self.dataset_id}.window_steps": step_count,
            "data/documents.segment_count": segment_count,
            "data/documents.window_steps": step_count,
            "data/block_causal.sum_length_squared": sum_length_squared,
            "data/block_causal.window_steps": step_count,
        }
        if all_tokens:
            metrics[f"data/datasets.{self.dataset_id}.masked_fraction"] = (
                all_tokens - valid_tokens
            ) / all_tokens
        if valid_tokens:
            metrics[f"data/datasets.{self.dataset_id}.loss_mean"] = (
                loss_sum / valid_tokens
            )
        if segment_count:
            metrics["data/documents.segment_length_mean"] = (
                segment_length_sum / segment_count
            )
        return metrics
