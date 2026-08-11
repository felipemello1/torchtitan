# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import Configurable
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.observability.tensor_logging.data_statistics import (
    DataStatisticsRecorder,
    DataStatisticsSnapshot,
)
from torchtitan.observability.tensor_logging.expert_bias import (
    ExpertBiasRecorder,
    ExpertBiasSnapshot,
)
from torchtitan.observability.tensor_logging.expert_counts import (
    ExpertCountRecorder,
    ExpertCountSnapshot,
)
from torchtitan.observability.tensor_logging.families import (
    BOUNDARY_FAMILIES,
    DATA_FAMILIES,
    EXPERT_COUNT_FAMILIES,
    INTERNAL_FAMILIES,
    JOB_FAMILIES,
    OPTIMIZER_FAMILIES,
    PARAMETER_FAMILIES,
    resolve_families,
    ROUTER_FAMILIES,
    TensorMetricFamily,
)
from torchtitan.observability.tensor_logging.optimizer_statistics import (
    AdamWStatisticsRecorder,
    OptimizerStatisticsSnapshot,
)
from torchtitan.observability.tensor_logging.output_batch import (
    OutputStatisticsBatch,
    OutputStatisticsSnapshot,
)
from torchtitan.observability.tensor_logging.parameter_batch import (
    ParameterStatisticsBatch,
    ParameterStatisticsSnapshot,
)
from torchtitan.observability.tensor_logging.router_statistics import (
    RouterStatisticsRecorder,
    RouterStatisticsSnapshot,
)
from torchtitan.observability.tensor_logging.whole_gradient import (
    WholeGradientSnapshot,
    WholeGradientStatistics,
)
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import logger


@dataclass(frozen=True, slots=True)
class TensorLoggingSnapshot:
    """Reduced parameter and output statistics for one logging step."""

    parameter: ParameterStatisticsSnapshot | None
    output: OutputStatisticsSnapshot | None
    expert_counts: ExpertCountSnapshot | None
    router: RouterStatisticsSnapshot | None
    whole_gradient: WholeGradientSnapshot | None
    expert_bias: ExpertBiasSnapshot | None
    optimizer: OptimizerStatisticsSnapshot | None
    data: DataStatisticsSnapshot | None
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
    _router_statistics_recorder: RouterStatisticsRecorder | None
    _whole_gradient_statistics: WholeGradientStatistics | None
    _expert_bias_recorder: ExpertBiasRecorder | None
    _optimizer_statistics_recorder: AdamWStatisticsRecorder | None
    _data_statistics_recorder: DataStatisticsRecorder | None
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
        adamw_statistics_selected = any(
            family
            in (
                TensorMetricFamily.OPTIMIZER_DISTRIBUTION,
                TensorMetricFamily.MOMENTUM_GRADIENT_COSINE,
            )
            for family in selected_families
        )
        non_data_selected = any(
            family not in DATA_FAMILIES for family in selected_families
        )
        layer_owned_selected = any(
            family not in JOB_FAMILIES + DATA_FAMILIES for family in selected_families
        )
        if not layer_owned_selected:
            if config.layer_ids != (0,):
                raise ValueError(
                    "tensor_logging.layer_ids applies only to layer-owned families"
                )
        else:
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
        if parallelism.pipeline_parallel_degree != 1:
            raise ValueError(
                "tensor logging currently requires pipeline_parallel_degree=1"
            )
        data_compatible_degrees = {
            "context_parallel_degree": parallelism.context_parallel_degree,
            "data_parallel_replicate_degree": (
                parallelism.data_parallel_replicate_degree
            ),
        }
        for name, degree in data_compatible_degrees.items():
            if degree != 1 and non_data_selected:
                raise ValueError(
                    f"tensor logging requires {name}=1 unless only data metric "
                    f"families are selected, got {degree}"
                )
        ep_families = (
            INTERNAL_FAMILIES
            + PARAMETER_FAMILIES
            + JOB_FAMILIES
            + OPTIMIZER_FAMILIES
            + DATA_FAMILIES
        )
        if parallelism.expert_parallel_degree != 1 and any(
            family not in ep_families for family in selected_families
        ):
            raise ValueError(
                "tensor logging requires expert_parallel_degree=1 unless "
                "only EP-compatible tensor metric families are selected"
            )
        if (
            TensorMetricFamily.EXPERT_COMPUTE_ROWS in selected_families
            and parallelism.expert_parallel_degree == 1
        ):
            raise ValueError(
                "EXPERT_COMPUTE_ROWS requires expert_parallel_degree greater than 1"
            )
        supported_spmd_backends = (
            ("default",) if non_data_selected else ("default", "spmd_types")
        )
        if parallelism.spmd_backend not in supported_spmd_backends:
            raise ValueError(
                "tensor logging does not support spmd_backend="
                f"'{parallelism.spmd_backend}' for the selected families"
            )
        if trainer_config.comm.mode != "default":
            raise ValueError("tensor logging currently requires comm.mode='default'")
        if trainer_config.training.enable_cpu_offload:
            raise ValueError("tensor logging does not yet support CPU offload")
        if (
            adamw_statistics_selected
            and trainer_config.optimizer.implementation == "fused_opt_states_bf16"
        ):
            raise ValueError(
                "optimizer tensor logging requires FP32 AdamW optimizer states"
            )
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
        dataloader_config = trainer_config.dataloader
        data_selected = any(family in DATA_FAMILIES for family in selected_families)
        if (
            data_selected
            and type(dataloader_config) is not HuggingFaceTextDataLoader.Config
        ):
            raise ValueError(
                "data tensor logging requires a HuggingFace text dataloader"
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
        layer_owned_selected = any(
            family not in JOB_FAMILIES + DATA_FAMILIES for family in selected_families
        )
        boundary_selected = any(
            family in BOUNDARY_FAMILIES for family in selected_families
        )
        internal_selected = any(
            family in INTERNAL_FAMILIES for family in selected_families
        )
        expert_bias_selected = TensorMetricFamily.EXPERT_BIAS in selected_families
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

        for layer_id in config.layer_ids if layer_owned_selected else ():
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
            if internal_selected or expert_bias_selected:
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
                if expert_bias_selected and moe_config.load_balance_coeff is None:
                    raise ValueError(
                        "expert-bias logging requires auxiliary-loss-free balancing "
                        f"at layers.{layer_id}.moe"
                    )
                if any(
                    family in EXPERT_COUNT_FAMILIES for family in selected_families
                ) and type(moe_config.routed_experts.token_dispatcher) is not (
                    AllToAllTokenDispatcher.Config
                ):
                    raise ValueError(
                        "internal MoE tensor logging currently requires the standard "
                        "token dispatcher"
                    )
                router_selected = any(
                    family in ROUTER_FAMILIES for family in selected_families
                )
                if router_selected:
                    router_config = moe_config.router
                    if type(router_config) is not TokenChoiceTopKRouter.Config:
                        raise ValueError(
                            "router tensor logging requires an ordinary "
                            "token-choice router"
                        )
                    if TensorMetricFamily.ROUTER_DISTRIBUTION in selected_families:
                        if router_config.num_expert_groups is not None:
                            raise ValueError(
                                "router distribution logging does not support "
                                "node-limited routing"
                            )
                        if router_config._debug_force_load_balance:
                            raise ValueError(
                                "router distribution logging does not support "
                                "forced routing"
                            )

    def __init__(
        self,
        config: Config,
        *,
        model: nn.Module,
        parallel_dims: ParallelDims,
        metrics_processor: MetricsProcessor,
        local_batch_size: int,
        dataloader_config: ParallelAwareDataloader.Config,
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
        count_families = tuple(
            family for family in internal_families if family in EXPERT_COUNT_FAMILIES
        )
        router_families = tuple(
            family for family in internal_families if family in ROUTER_FAMILIES
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
                families=count_families,
                device=device,
            )
            if count_families
            else None
        )
        self._router_statistics_recorder = (
            RouterStatisticsRecorder(
                model=model,
                parallel_dims=parallel_dims,
                layer_ids=config.layer_ids,
                families=router_families,
                local_batch_size=local_batch_size,
                device=device,
            )
            if router_families
            else None
        )
        self._whole_gradient_statistics = (
            WholeGradientStatistics(model=model, parallel_dims=parallel_dims)
            if TensorMetricFamily.WHOLE_GRADIENT in selected_families
            else None
        )
        self._expert_bias_recorder = (
            ExpertBiasRecorder(
                model=model,
                layer_ids=config.layer_ids,
                device=device,
            )
            if TensorMetricFamily.EXPERT_BIAS in selected_families
            else None
        )
        adamw_families = tuple(
            family
            for family in selected_families
            if family
            in (
                TensorMetricFamily.OPTIMIZER_DISTRIBUTION,
                TensorMetricFamily.MOMENTUM_GRADIENT_COSINE,
            )
        )
        self._optimizer_statistics_recorder = (
            AdamWStatisticsRecorder(
                model=model,
                parallel_dims=parallel_dims,
                layer_ids=config.layer_ids,
                families=adamw_families,
            )
            if adamw_families
            else None
        )
        data_families = tuple(
            family for family in selected_families if family in DATA_FAMILIES
        )
        dataset_id = (
            dataloader_config.dataset
            if type(dataloader_config) is HuggingFaceTextDataLoader.Config
            else None
        )
        self._data_statistics_recorder = (
            DataStatisticsRecorder(
                parallel_dims=parallel_dims,
                families=data_families,
                dataset_id=dataset_id,
                device=device,
            )
            if data_families
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
        if self._expert_bias_recorder is not None:
            self._expert_bias_recorder.begin_step(should_log=should_log)
        if self._optimizer_statistics_recorder is not None:
            self._optimizer_statistics_recorder.begin_step(should_log=should_log)

    def bind_optimizer(self, optimizer: OptimizersContainer) -> None:
        """Bind optimizer-owned producers after model hooks are installed."""
        if self._expert_bias_recorder is not None:
            self._expert_bias_recorder.bind_optimizer(optimizer)
        if self._optimizer_statistics_recorder is not None:
            self._optimizer_statistics_recorder.bind_optimizer(optimizer)

    def record_data_batch(
        self,
        *,
        labels: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> None:
        """Record one raw batch before context-parallel sharding."""
        if self._data_statistics_recorder is not None:
            self._data_statistics_recorder.record_batch(
                labels=labels,
                positions=positions,
            )

    def record_data_loss(
        self,
        *,
        normalized_loss: torch.Tensor,
        global_valid_tokens: float | torch.Tensor,
    ) -> None:
        """Record one loss numerator after forward/backward."""
        if self._data_statistics_recorder is not None:
            self._data_statistics_recorder.record_loss(
                normalized_loss=normalized_loss,
                global_valid_tokens=global_valid_tokens,
            )

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
        router = (
            self._router_statistics_recorder.collect()
            if self._router_statistics_recorder is not None
            else None
        )
        whole_gradient = (
            self._whole_gradient_statistics.collect(step=step)
            if self._whole_gradient_statistics is not None
            else None
        )
        data = (
            self._data_statistics_recorder.collect()
            if self._data_statistics_recorder is not None
            else None
        )
        local_error = None
        for snapshot in (
            parameter,
            output,
            expert_counts,
            router,
            whole_gradient,
            data,
        ):
            if snapshot is not None and snapshot.local_error is not None:
                local_error = snapshot.local_error
                break
        return TensorLoggingSnapshot(
            parameter=parameter,
            output=output,
            expert_counts=expert_counts,
            router=router,
            whole_gradient=whole_gradient,
            expert_bias=None,
            optimizer=None,
            data=data,
            local_error=local_error,
        )

    def collect_after_optimizer(
        self,
        snapshot: TensorLoggingSnapshot,
    ) -> TensorLoggingSnapshot:
        """Attach optimizer-owned point samples to a pre-optimizer snapshot."""
        if (
            self._expert_bias_recorder is None
            and self._optimizer_statistics_recorder is None
        ):
            return snapshot
        expert_bias = (
            self._expert_bias_recorder.collect()
            if self._expert_bias_recorder is not None
            else None
        )
        optimizer = (
            self._optimizer_statistics_recorder.collect()
            if self._optimizer_statistics_recorder is not None
            else None
        )
        local_error = snapshot.local_error
        if local_error is None and expert_bias is not None:
            local_error = expert_bias.local_error
        if local_error is None and optimizer is not None:
            local_error = optimizer.local_error
        return replace(
            snapshot,
            expert_bias=expert_bias,
            optimizer=optimizer,
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
        if snapshot.router is not None:
            assert self._router_statistics_recorder is not None
            metrics.update(
                self._router_statistics_recorder.derive_metrics(
                    snapshot.router,
                    window_steps=window_steps,
                )
            )
        if snapshot.whole_gradient is not None:
            assert self._whole_gradient_statistics is not None
            metrics.update(
                self._whole_gradient_statistics.derive_metrics(
                    snapshot.whole_gradient,
                    window_steps=window_steps,
                )
            )
        if snapshot.expert_bias is not None:
            assert self._expert_bias_recorder is not None
            metrics.update(
                self._expert_bias_recorder.derive_metrics(
                    snapshot.expert_bias,
                    window_steps=window_steps,
                )
            )
        if snapshot.optimizer is not None:
            assert self._optimizer_statistics_recorder is not None
            metrics.update(
                self._optimizer_statistics_recorder.derive_metrics(
                    snapshot.optimizer,
                    window_steps=window_steps,
                )
            )
        if snapshot.data is not None:
            assert self._data_statistics_recorder is not None
            metrics.update(
                self._data_statistics_recorder.derive_metrics(
                    snapshot.data,
                    window_steps=window_steps,
                )
            )
        return metrics

    def reset_after_checkpoint_load(self, *, step: int) -> None:
        self._last_successful_publication_step = step
        if self._output_batch is not None:
            self._output_batch.begin_step(should_log=False)
        if self._expert_bias_recorder is not None:
            self._expert_bias_recorder.begin_step(should_log=False)
        if self._optimizer_statistics_recorder is not None:
            self._optimizer_statistics_recorder.begin_step(should_log=False)
        if self._data_statistics_recorder is not None:
            self._data_statistics_recorder.reset()

    def close(self) -> None:
        if self._output_batch is not None:
            self._output_batch.close()
        if self._expert_count_recorder is not None:
            self._expert_count_recorder.close()
        if self._router_statistics_recorder is not None:
            self._router_statistics_recorder.close()
        if self._expert_bias_recorder is not None:
            self._expert_bias_recorder.close()
        if self._optimizer_statistics_recorder is not None:
            self._optimizer_statistics_recorder.close()

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
