# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import Configurable
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.observability.tensor_logging.expert_counts import (
    ExpertCountRecorder,
    ExpertCountSnapshot,
)
from torchtitan.observability.tensor_logging.families import (
    BOUNDARY_FAMILIES,
    INTERNAL_FAMILIES,
    PARAMETER_FAMILIES,
    resolve_families,
    TensorMetricFamily,
)
from torchtitan.observability.tensor_logging.output_batch import (
    OutputStatisticsBatch,
    OutputStatisticsSnapshot,
)
from torchtitan.observability.tensor_logging.parameter_batch import (
    ParameterStatisticsBatch,
    ParameterStatisticsSnapshot,
)
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import logger


@dataclass(frozen=True, slots=True)
class TensorLoggingSnapshot:
    """Reduced parameter and output statistics for one logging step."""

    parameter: ParameterStatisticsSnapshot | None
    output: OutputStatisticsSnapshot | None
    expert_counts: ExpertCountSnapshot | None
    local_error: Exception | None


class TensorLogging(Configurable):
    """Distributed logging for selected tensor-metric families.

    Example:

        config.tensor_logging = TensorLogging.Config(
            enable=True,
            families=(TensorMetricFamily.PARAMETER,),
            layer_ids=(0,),
        )
    """

    _parameter_batch: ParameterStatisticsBatch | None
    _output_batch: OutputStatisticsBatch | None
    _expert_count_recorder: ExpertCountRecorder | None
    _is_writer: bool
    _outcome_template: torch.Tensor
    _last_successful_publication_step: int

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Selects tensor-metric families and global decoder layers to log."""

        enable: bool = False
        """Whether to collect selected distributed tensor statistics."""

        families: tuple[TensorMetricFamily, ...] | None = None
        """Semantic metric families, or the documented supported default."""

        layer_ids: tuple[int, ...] = (0,)
        """Global decoder layer IDs to observe."""

    @staticmethod
    def validate_job_config(
        config: Config,
        *,
        trainer_config: Any,
        is_core_trainer: bool,
    ) -> None:
        """Fail unsupported public requests before distributed construction."""
        if not config.enable:
            return
        selected_families = resolve_families(config.families)
        boundary_selected = any(
            family in BOUNDARY_FAMILIES for family in selected_families
        )
        internal_selected = any(
            family in INTERNAL_FAMILIES for family in selected_families
        )
        if not config.layer_ids:
            raise ValueError("tensor_logging.layer_ids must not be empty")
        if len(set(config.layer_ids)) != len(config.layer_ids):
            raise ValueError("tensor_logging.layer_ids must not contain duplicates")
        if any(
            not isinstance(layer_id, int) or layer_id < 0
            for layer_id in config.layer_ids
        ):
            raise ValueError(
                "tensor_logging.layer_ids must contain nonnegative integers"
            )
        if not is_core_trainer:
            raise ValueError("tensor logging currently supports only the core Trainer")

        parallelism = trainer_config.parallelism
        unsupported_degrees = {
            "pipeline_parallel_degree": parallelism.pipeline_parallel_degree,
            "context_parallel_degree": parallelism.context_parallel_degree,
            "data_parallel_replicate_degree": (
                parallelism.data_parallel_replicate_degree
            ),
        }
        for name, degree in unsupported_degrees.items():
            if degree != 1:
                raise ValueError(f"tensor logging requires {name}=1, got {degree}")
        if parallelism.expert_parallel_degree != 1 and any(
            family not in INTERNAL_FAMILIES for family in selected_families
        ):
            raise ValueError(
                "tensor logging requires expert_parallel_degree=1 unless "
                "only internal MoE families are selected"
            )
        if (
            TensorMetricFamily.EXPERT_COMPUTE_ROWS in selected_families
            and parallelism.expert_parallel_degree == 1
        ):
            raise ValueError(
                "EXPERT_COMPUTE_ROWS requires expert_parallel_degree greater than 1"
            )
        if parallelism.spmd_backend != "default":
            raise ValueError("tensor logging currently requires spmd_backend='default'")
        if trainer_config.comm.mode != "default":
            raise ValueError("tensor logging currently requires comm.mode='default'")
        if trainer_config.training.enable_cpu_offload:
            raise ValueError("tensor logging does not yet support CPU offload")
        if trainer_config.metrics.save_for_all_ranks:
            raise ValueError(
                "tensor logging does not support metrics.save_for_all_ranks=True"
            )
        if not (
            trainer_config.metrics.enable_tensorboard
            or trainer_config.metrics.enable_wandb
        ):
            raise ValueError(
                "tensor logging requires metrics.enable_tensorboard or "
                "metrics.enable_wandb"
            )
        activation_checkpoint = trainer_config.activation_checkpoint
        if (
            activation_checkpoint is not None
            and type(activation_checkpoint) is not SelectiveAC.Config
        ):
            raise ValueError(
                "tensor logging currently supports activation checkpointing "
                "None or SelectiveAC"
            )
        compile_model = (
            trainer_config.compile.enable
            and "model" in trainer_config.compile.components
        )
        if compile_model and trainer_config.compile.backend != "inductor":
            raise ValueError(
                "tensor logging currently supports only the inductor compile backend"
            )
        if boundary_selected or internal_selected:
            if activation_checkpoint is not None:
                raise ValueError(
                    "tensor forward logging does not yet support activation checkpointing"
                )
            if compile_model:
                raise ValueError(
                    "tensor forward logging does not yet support model compilation"
                )
            if trainer_config.validator.enable:
                raise ValueError(
                    "tensor forward logging does not yet support validation-enabled jobs"
                )

    @staticmethod
    def validate_model_config(
        config: Config,
        *,
        model_spec: ModelSpec,
        model_config: Any,
    ) -> None:
        """Validate the exact model family and selected pre-build projections."""
        if not config.enable:
            return
        selected_families = resolve_families(config.families)
        boundary_selected = any(
            family in BOUNDARY_FAMILIES for family in selected_families
        )
        internal_selected = any(
            family in INTERNAL_FAMILIES for family in selected_families
        )
        expected_model_type: type
        expected_parallelize_fn: object
        if model_spec.name == "llama3":
            expected_model_type = Llama3Model.Config
            expected_parallelize_fn = parallelize_llama
        elif model_spec.name == "qwen3":
            expected_model_type = Qwen3Model.Config
            expected_parallelize_fn = parallelize_qwen3
        else:
            raise ValueError(
                "tensor logging currently supports only ordinary llama3 and qwen3"
            )
        if type(model_config) is not expected_model_type:
            raise ValueError("tensor logging requires an unconverted model config")
        if model_spec.parallelize_fn is not expected_parallelize_fn:
            raise ValueError("tensor logging requires the ordinary model parallelizer")
        if has_quantization(model_config):
            raise ValueError("tensor logging does not yet support quantized models")

        for layer_id in config.layer_ids:
            if layer_id >= len(model_config.layers):
                raise ValueError(
                    f"tensor_logging.layer_ids contains {layer_id}, but the model "
                    f"has {len(model_config.layers)} layers"
                )
            if boundary_selected:
                wo_config = model_config.layers[layer_id].attention.wo
                if type(wo_config) is not Linear.Config:
                    raise ValueError(
                        "tensor output logging requires an ordinary Linear.Config at "
                        f"layers.{layer_id}.attention.wo"
                    )
                feed_forward_config = model_config.layers[layer_id].feed_forward
                if type(feed_forward_config) is not FeedForward.Config:
                    raise ValueError(
                        "tensor output logging requires an ordinary "
                        f"FeedForward.Config at layers.{layer_id}.feed_forward"
                    )
                if type(feed_forward_config.w2) is not Linear.Config:
                    raise ValueError(
                        "tensor output logging requires an ordinary Linear.Config at "
                        f"layers.{layer_id}.feed_forward.w2"
                    )
            if internal_selected:
                if model_spec.name != "qwen3":
                    raise ValueError(
                        "internal MoE tensor logging currently requires Qwen3"
                    )
                moe_config = model_config.layers[layer_id].moe
                if type(moe_config) is not MoE.Config:
                    raise ValueError(
                        "internal MoE tensor logging requires an ordinary MoE.Config "
                        f"at layers.{layer_id}.moe"
                    )
                if type(moe_config.routed_experts.token_dispatcher) is not (
                    AllToAllTokenDispatcher.Config
                ):
                    raise ValueError(
                        "internal MoE tensor logging currently requires the standard "
                        "token dispatcher"
                    )

    def __init__(
        self,
        config: Config,
        *,
        model: nn.Module,
        parallel_dims: ParallelDims,
        metrics_processor: MetricsProcessor,
        device: torch.device,
    ) -> None:
        selected_families = resolve_families(config.families)
        parameter_families = tuple(
            family for family in selected_families if family in PARAMETER_FAMILIES
        )
        boundary_families = tuple(
            family for family in selected_families if family in BOUNDARY_FAMILIES
        )
        internal_families = tuple(
            family for family in selected_families if family in INTERNAL_FAMILIES
        )
        self._parameter_batch = (
            ParameterStatisticsBatch(
                model=model,
                parallel_dims=parallel_dims,
                layer_ids=config.layer_ids,
                families=parameter_families,
            )
            if parameter_families
            else None
        )
        self._output_batch = (
            OutputStatisticsBatch(
                model=model,
                parallel_dims=parallel_dims,
                layer_ids=config.layer_ids,
                families=boundary_families,
                device=device,
            )
            if boundary_families
            else None
        )
        self._expert_count_recorder = (
            ExpertCountRecorder(
                model=model,
                parallel_dims=parallel_dims,
                layer_ids=config.layer_ids,
                families=internal_families,
                device=device,
            )
            if internal_families
            else None
        )

        self._is_writer = metrics_processor.has_active_logger
        active_writer_count = torch.tensor(
            int(self._is_writer), dtype=torch.int32, device=device
        )
        dist.all_reduce(active_writer_count, op=dist.ReduceOp.SUM)
        if int(active_writer_count.item()) != 1:
            raise RuntimeError(
                "tensor logging requires exactly one active metrics writer"
            )

        self._outcome_template = torch.empty((), dtype=torch.int32, device=device)
        self._last_successful_publication_step = 0

        selected_names = ", ".join(family.name for family in selected_families)
        logger.info(f"Tensor logging selected families: {selected_names}")

    def begin_step(self, *, should_log: bool) -> None:
        if self._output_batch is not None:
            self._output_batch.begin_step(should_log=should_log)

    def collect(self, *, step: int) -> TensorLoggingSnapshot:
        parameter = (
            self._parameter_batch.collect(step=step)
            if self._parameter_batch is not None
            else None
        )
        output = (
            self._output_batch.collect() if self._output_batch is not None else None
        )
        expert_counts = (
            self._expert_count_recorder.collect()
            if self._expert_count_recorder is not None
            else None
        )
        local_error = None
        for snapshot in (parameter, output, expert_counts):
            if snapshot is not None and snapshot.local_error is not None:
                local_error = snapshot.local_error
                break
        return TensorLoggingSnapshot(
            parameter=parameter,
            output=output,
            expert_counts=expert_counts,
            local_error=local_error,
        )

    def derive_metrics(
        self,
        snapshot: TensorLoggingSnapshot,
        *,
        step: int,
    ) -> dict[str, int | float]:
        if not self._is_writer:
            return {}
        window_steps = step - self._last_successful_publication_step
        if window_steps <= 0:
            raise RuntimeError(
                f"tensor logging publication step {step} does not advance its window"
            )
        metrics: dict[str, int | float] = {}
        if snapshot.parameter is not None:
            assert self._parameter_batch is not None
            metrics.update(
                self._parameter_batch.derive_metrics(
                    snapshot.parameter,
                    window_steps=window_steps,
                )
            )
        if snapshot.output is not None:
            assert self._output_batch is not None
            metrics.update(
                self._output_batch.derive_metrics(
                    snapshot.output,
                    window_steps=window_steps,
                )
            )
        if snapshot.expert_counts is not None:
            assert self._expert_count_recorder is not None
            metrics.update(
                self._expert_count_recorder.derive_metrics(
                    snapshot.expert_counts,
                    window_steps=window_steps,
                )
            )
        return metrics

    def reset_after_checkpoint_load(self, *, step: int) -> None:
        self._last_successful_publication_step = step
        if self._output_batch is not None:
            self._output_batch.begin_step(should_log=False)

    def close(self) -> None:
        if self._output_batch is not None:
            self._output_batch.close()
        if self._expert_count_recorder is not None:
            self._expert_count_recorder.close()

    def complete_publication(
        self,
        *,
        step: int,
        local_error: Exception | None,
    ) -> None:
        outcome = self._outcome_template.new_tensor(int(local_error is None))
        dist.all_reduce(outcome, op=dist.ReduceOp.MIN)
        all_succeeded = bool(outcome.item())
        if local_error is not None:
            raise local_error.with_traceback(local_error.__traceback__)
        if not all_succeeded:
            raise RuntimeError("tensor metric publication failed on another rank")
        self._last_successful_publication_step = step
