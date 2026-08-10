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
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.observability.tensor_logging.parameter_batch import (
    ParameterStatisticsBatch,
    ParameterStatisticsSnapshot,
)
from torchtitan.observability.tensor_logging.sites import (
    resolve_parameter_sites,
    TensorMetricSite,
)
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import logger


class TensorLogging(Configurable):
    """Distributed logging for the selected built-in tensor sites."""

    _batch: ParameterStatisticsBatch
    _is_writer: bool
    _outcome_template: torch.Tensor
    _expected_contributors: int
    _last_successful_publication_step: int

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Whether to collect selected distributed tensor statistics."""

        sites: tuple[TensorMetricSite, ...] | None = None
        """Built-in semantic sites, or the documented supported default."""

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
        resolve_parameter_sites(config.sites)
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
            "expert_parallel_degree": parallelism.expert_parallel_degree,
            "data_parallel_replicate_degree": (
                parallelism.data_parallel_replicate_degree
            ),
        }
        for name, degree in unsupported_degrees.items():
            if degree != 1:
                raise ValueError(f"tensor logging requires {name}=1, got {degree}")
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
            wo_config = model_config.layers[layer_id].attention.wo
            if type(wo_config) is not Linear.Config:
                raise ValueError(
                    "tensor logging requires an ordinary Linear.Config at "
                    f"layers.{layer_id}.attention.wo"
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
        selected_sites, omitted_sites = resolve_parameter_sites(config.sites)
        self._batch = ParameterStatisticsBatch(
            model=model,
            parallel_dims=parallel_dims,
            layer_ids=config.layer_ids,
            sites=selected_sites,
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
        self._expected_contributors = dist.get_world_size()
        self._last_successful_publication_step = 0

        selected_names = ", ".join(site.name for site in selected_sites)
        omitted_names = ", ".join(
            f"{site.name} ({reason})" for site, reason in omitted_sites.items()
        )
        logger.info(
            f"Tensor logging selected: {selected_names}; omitted: {omitted_names}"
        )

    def collect(self, *, step: int) -> ParameterStatisticsSnapshot:
        return self._batch.collect(step=step)

    def derive_metrics(
        self,
        snapshot: ParameterStatisticsSnapshot,
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
        return self._batch.derive_metrics(
            snapshot,
            expected_contributors=self._expected_contributors,
            window_steps=window_steps,
        )

    def reset_after_checkpoint_load(self, *, step: int) -> None:
        self._last_successful_publication_step = step

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
