# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loss, document-segment, and block-causal statistics from trainer data."""

from dataclasses import dataclass

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.families import TensorMetricFamily
from torchtitan.observability.tensor_logging.statistics import reduce_sum


_ALL_TOKEN_COUNT = 0
_VALID_TOKEN_COUNT = 1
_SEGMENT_LENGTH_SUM = 2
_SEGMENT_COUNT = 3
_SUM_LENGTH_SQUARED = 4
_OBSERVATION_COUNT = 5
_INTEGER_FIELDS = 6


@dataclass(frozen=True, slots=True)
class DataStatisticsSnapshot:
    integers: torch.Tensor
    loss_sum: torch.Tensor | None
    local_error: Exception | None


class DataStatisticsRecorder:
    """Accumulates exact data-owned sufficient statistics between publications."""

    def __init__(
        self,
        *,
        parallel_dims: ParallelDims,
        families: tuple[TensorMetricFamily, ...],
        dataset_id: str | None,
        device: torch.device,
    ) -> None:
        self._families = families
        self._record_loss = TensorMetricFamily.DATASET_LOSS in families
        self._record_segments = any(
            family in families
            for family in (
                TensorMetricFamily.DOCUMENT_SEGMENTS,
                TensorMetricFamily.BLOCK_CAUSAL_MOMENTS,
            )
        )
        if self._record_loss and dataset_id is None:
            raise ValueError("dataset loss logging requires one stable dataset ID")
        self._dataset_id = dataset_id
        self._world_mesh = parallel_dims.world_mesh

        tp_mesh = parallel_dims.get_optional_mesh("tp")
        cp_mesh = parallel_dims.get_optional_mesh("cp")
        pp_mesh = parallel_dims.get_optional_mesh("pp")
        tp_representative = tp_mesh is None or tp_mesh.get_local_rank() == 0
        cp_representative = cp_mesh is None or cp_mesh.get_local_rank() == 0
        first_pipeline_stage = pp_mesh is None or pp_mesh.get_local_rank() == 0
        last_pipeline_stage = (
            pp_mesh is None or pp_mesh.get_local_rank() == pp_mesh.size() - 1
        )
        self._records_data = (
            tp_representative and cp_representative and first_pipeline_stage
        )
        self._records_loss = tp_representative and last_pipeline_stage

        self._integers = torch.zeros(
            _INTEGER_FIELDS,
            dtype=torch.int64,
            device=device,
        )
        self._loss_sum = torch.zeros(1, dtype=torch.float32, device=device)
        self._local_error: Exception | None = None

    def record_batch(
        self,
        *,
        labels: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> None:
        """Accumulate one pre-context-parallel batch from one model replica."""
        if not self._records_data:
            return
        try:
            self._integers[_ALL_TOKEN_COUNT].add_(labels.numel())
            self._integers[_VALID_TOKEN_COUNT].add_(
                torch.count_nonzero(labels != IGNORE_INDEX)
            )
            self._integers[_OBSERVATION_COUNT].add_(1)
            if not self._record_segments:
                return
            if positions is None:
                raise ValueError("block-causal data batch has no positions")
            if positions.shape != labels.shape or positions.ndim != 2:
                raise ValueError("positions and labels must have the same [B, S] shape")

            segment_ends = torch.empty_like(positions, dtype=torch.bool)
            segment_ends[:, :-1] = positions[:, 1:] == 0
            segment_ends[:, -1] = True
            segment_lengths = positions.to(torch.int64) + 1
            self._integers[_SEGMENT_LENGTH_SUM].add_(
                torch.sum(segment_lengths * segment_ends)
            )
            self._integers[_SEGMENT_COUNT].add_(torch.count_nonzero(segment_ends))
            self._integers[_SUM_LENGTH_SQUARED].add_(
                torch.sum(torch.square(segment_lengths) * segment_ends)
            )
        except Exception as error:
            if self._local_error is None:
                self._local_error = ValueError(f"invalid data tensor sample: {error}")

    def record_loss(
        self,
        *,
        normalized_loss: torch.Tensor,
        global_valid_tokens: float | torch.Tensor,
    ) -> None:
        """Accumulate the local loss numerator on the authoritative loss stage."""
        if not self._record_loss or not self._records_loss:
            return
        try:
            self._loss_sum.add_(
                (normalized_loss.detach() * global_valid_tokens).float().sum()
            )
        except Exception as error:
            if self._local_error is None:
                self._local_error = ValueError(f"invalid data loss sample: {error}")

    def collect(self) -> DataStatisticsSnapshot:
        """Reduce and clear the interval accumulated since the last publication."""
        integers = reduce_sum(self._integers.clone(), self._world_mesh)
        loss_sum = (
            reduce_sum(self._loss_sum.clone(), self._world_mesh)
            if self._record_loss
            else None
        )
        snapshot = DataStatisticsSnapshot(
            integers=integers,
            loss_sum=loss_sum,
            local_error=self._local_error,
        )
        self.reset()
        return snapshot

    def derive_metrics(
        self,
        snapshot: DataStatisticsSnapshot,
        *,
        window_steps: int,
    ) -> dict[str, int | float]:
        """Derive scalar data metrics from the completed interval sums."""
        (
            all_token_count,
            valid_token_count,
            segment_length_sum,
            segment_count,
            sum_length_squared,
            observation_count,
        ) = (int(value) for value in snapshot.integers.cpu().tolist())
        metrics: dict[str, int | float] = {}

        if self._record_loss:
            assert self._dataset_id is not None
            prefix = f"tensor_metrics/data/datasets.{self._dataset_id}"
            metrics[f"{prefix}.valid_token_count"] = valid_token_count
            metrics[f"{prefix}.all_token_count"] = all_token_count
            metrics[f"{prefix}.observation_count"] = observation_count
            metrics[f"{prefix}.window_steps"] = window_steps
            if all_token_count > 0:
                metrics[f"{prefix}.masked_fraction"] = (
                    all_token_count - valid_token_count
                ) / all_token_count
            if valid_token_count > 0:
                assert snapshot.loss_sum is not None
                metrics[f"{prefix}.loss_mean"] = (
                    float(snapshot.loss_sum.cpu().item()) / valid_token_count
                )

        if TensorMetricFamily.DOCUMENT_SEGMENTS in self._families:
            prefix = "tensor_metrics/data/documents"
            metrics[f"{prefix}.segment_count"] = segment_count
            metrics[f"{prefix}.observation_count"] = observation_count
            metrics[f"{prefix}.window_steps"] = window_steps
            if segment_count > 0:
                metrics[f"{prefix}.segment_length_mean"] = (
                    segment_length_sum / segment_count
                )

        if TensorMetricFamily.BLOCK_CAUSAL_MOMENTS in self._families:
            prefix = "tensor_metrics/data/block_causal"
            metrics[f"{prefix}.sum_length_squared"] = sum_length_squared
            metrics[f"{prefix}.observation_count"] = observation_count
            metrics[f"{prefix}.window_steps"] = window_steps
        return metrics

    def reset(self) -> None:
        self._integers.zero_()
        self._loss_sum.zero_()
        self._local_error = None
