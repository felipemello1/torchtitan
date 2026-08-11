# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.testing._internal.distributed.fake_pg  # noqa: F401

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.metrics import BaseLogger, LoggerContainer, MetricsProcessor
from torchtitan.config import ConfigManager, override
from torchtitan.distributed.activation_checkpoint import (
    FullAC,
    MemoryBudgetAC,
    SelectiveAC,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.llama3.config_registry import llama3_debugmodel
from torchtitan.models.qwen3.config_registry import qwen3_debugmodel, qwen3_moe_debug
from torchtitan.observability.tensor_logging import TensorLogging, TensorMetricFamily
from torchtitan.observability.tensor_logging.families import resolve_families
from torchtitan.trainer import Trainer


def _enabled_config():
    config = llama3_debugmodel()
    config.tensor_logging.enable = True
    config.metrics.enable_tensorboard = True
    return config


def test_family_selection_is_explicit_and_unique() -> None:
    selected = resolve_families(None)
    assert selected == (
        TensorMetricFamily.PARAMETER,
        TensorMetricFamily.PRECLIP_GRADIENT,
    )

    selected = resolve_families((TensorMetricFamily.PRECLIP_GRADIENT,))
    assert selected == (TensorMetricFamily.PRECLIP_GRADIENT,)

    with pytest.raises(ValueError, match="must not be empty"):
        resolve_families(())
    with pytest.raises(ValueError, match="duplicates"):
        resolve_families(
            (
                TensorMetricFamily.PARAMETER,
                TensorMetricFamily.PARAMETER,
            )
        )
    with pytest.raises(ValueError, match="TensorMetricFamily values"):
        resolve_families(cast(tuple[TensorMetricFamily, ...], ("parameter",)))


def test_recipe_parser_accepts_the_three_field_surface() -> None:
    config = ConfigManager().parse_args(
        [
            "--module",
            "llama3",
            "--config",
            "llama3_debugmodel",
            "--tensor-logging.enable",
            "--tensor-logging.families",
            "PARAMETER",
            "PRECLIP_GRADIENT",
            "--tensor-logging.layer-ids",
            "0",
            "2",
        ]
    )

    assert config.tensor_logging == TensorLogging.Config(
        enable=True,
        families=(
            TensorMetricFamily.PARAMETER,
            TensorMetricFamily.PRECLIP_GRADIENT,
        ),
        layer_ids=(0, 2),
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: setattr(config.parallelism, "pipeline_parallel_degree", 2),
            "pipeline_parallel_degree=1",
        ),
        (
            lambda config: setattr(config.parallelism, "context_parallel_degree", 2),
            "context_parallel_degree=1",
        ),
        (
            lambda config: setattr(
                config.parallelism, "data_parallel_replicate_degree", 2
            ),
            "data_parallel_replicate_degree=1",
        ),
        (
            lambda config: setattr(config.parallelism, "spmd_backend", "spmd_types"),
            "spmd_backend='spmd_types'",
        ),
        (
            lambda config: setattr(config.comm, "mode", "fake_backend"),
            "comm.mode='default'",
        ),
        (
            lambda config: setattr(config.training, "enable_cpu_offload", True),
            "CPU offload",
        ),
        (
            lambda config: setattr(config.metrics, "save_for_all_ranks", True),
            "save_for_all_ranks",
        ),
        (
            lambda config: setattr(config, "activation_checkpoint", FullAC.Config()),
            "None or SelectiveAC",
        ),
        (
            lambda config: setattr(
                config, "activation_checkpoint", MemoryBudgetAC.Config()
            ),
            "None or SelectiveAC",
        ),
        (
            lambda config: setattr(
                config,
                "compile",
                replace(config.compile, enable=True, backend="aot_eager"),
            ),
            "inductor compile backend",
        ),
    ],
)
def test_job_support_row_fails_closed(mutate, message: str) -> None:
    config = _enabled_config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        TensorLogging.validate_job_config(
            config.tensor_logging,
            trainer_config=config,
            is_core_trainer=True,
        )


def test_job_config_requires_one_sink_and_core_trainer() -> None:
    config = _enabled_config()
    config.metrics.enable_tensorboard = False

    with pytest.raises(ValueError, match="enable_tensorboard or"):
        TensorLogging.validate_job_config(
            config.tensor_logging,
            trainer_config=config,
            is_core_trainer=True,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: setattr(
                config,
                "activation_checkpoint",
                SelectiveAC.Config(),
            ),
            "does not yet support activation checkpointing",
        ),
        (
            lambda config: setattr(
                config,
                "compile",
                replace(config.compile, enable=True, backend="inductor"),
            ),
            "does not yet support model compilation",
        ),
        (
            lambda config: setattr(config.validator, "enable", True),
            "does not yet support validation-enabled jobs",
        ),
    ],
)
def test_boundary_support_row_fails_closed(mutate, message: str) -> None:
    config = _enabled_config()
    config.tensor_logging.families = (TensorMetricFamily.BOUNDARY_OUTPUT,)
    config.activation_checkpoint = None
    mutate(config)

    with pytest.raises(ValueError, match=message):
        TensorLogging.validate_job_config(
            config.tensor_logging,
            trainer_config=config,
            is_core_trainer=True,
        )


def test_parameter_only_logging_keeps_validation_support() -> None:
    config = _enabled_config()
    config.tensor_logging.families = (TensorMetricFamily.PARAMETER,)
    config.validator.enable = True

    TensorLogging.validate_job_config(
        config.tensor_logging,
        trainer_config=config,
        is_core_trainer=True,
    )


def test_data_only_logging_admits_data_parallel_modes() -> None:
    config = _enabled_config()
    config.tensor_logging.families = (
        TensorMetricFamily.DATASET_LOSS,
        TensorMetricFamily.DOCUMENT_SEGMENTS,
        TensorMetricFamily.BLOCK_CAUSAL_MOMENTS,
    )
    config.parallelism.context_parallel_degree = 2
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.spmd_backend = "spmd_types"
    config.activation_checkpoint = SelectiveAC.Config()
    config.compile = replace(config.compile, enable=True, backend="inductor")

    TensorLogging.validate_job_config(
        config.tensor_logging,
        trainer_config=config,
        is_core_trainer=True,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: setattr(config.parallelism, "spmd_backend", "full_dtensor"),
            "full_dtensor",
        ),
        (
            lambda config: setattr(config.tensor_logging, "layer_ids", (1,)),
            "applies only to layer-owned families",
        ),
    ],
)
def test_data_only_logging_rejects_unsupported_knobs(mutate, message: str) -> None:
    config = _enabled_config()
    config.tensor_logging.families = (
        TensorMetricFamily.DATASET_LOSS,
        TensorMetricFamily.DOCUMENT_SEGMENTS,
        TensorMetricFamily.BLOCK_CAUSAL_MOMENTS,
    )
    mutate(config)

    with pytest.raises(ValueError, match=message):
        TensorLogging.validate_job_config(
            config.tensor_logging,
            trainer_config=config,
            is_core_trainer=True,
        )


@pytest.mark.parametrize(
    "family",
    [
        TensorMetricFamily.DATASET_LOSS,
        TensorMetricFamily.DOCUMENT_SEGMENTS,
        TensorMetricFamily.BLOCK_CAUSAL_MOMENTS,
    ],
)
def test_data_statistics_reject_non_text_dataloader(
    family: TensorMetricFamily,
) -> None:
    config = _enabled_config()
    config.tensor_logging.families = (family,)
    config.dataloader = ParallelAwareDataloader.Config()

    with pytest.raises(ValueError, match="HuggingFace text dataloader"):
        TensorLogging.validate_job_config(
            config.tensor_logging,
            trainer_config=config,
            is_core_trainer=True,
        )


def test_optimizer_statistics_reject_bf16_state_implementation() -> None:
    config = _enabled_config()
    config.tensor_logging.families = (
        TensorMetricFamily.OPTIMIZER_DISTRIBUTION,
        TensorMetricFamily.MOMENTUM_GRADIENT_COSINE,
    )
    config.optimizer.implementation = "fused_opt_states_bf16"

    with pytest.raises(ValueError, match="FP32 AdamW optimizer states"):
        TensorLogging.validate_job_config(
            config.tensor_logging,
            trainer_config=config,
            is_core_trainer=True,
        )


def test_parameter_and_internal_moe_families_are_ep_compatible() -> None:
    config = _enabled_config()
    config.activation_checkpoint = None
    config.parallelism.expert_parallel_degree = 2
    config.tensor_logging.families = (
        TensorMetricFamily.OFFERED_ASSIGNMENTS,
        TensorMetricFamily.EXPERT_COMPUTE_ROWS,
    )

    TensorLogging.validate_job_config(
        config.tensor_logging,
        trainer_config=config,
        is_core_trainer=True,
    )

    config.tensor_logging.families = (
        TensorMetricFamily.OFFERED_ASSIGNMENTS,
        TensorMetricFamily.PARAMETER,
    )
    TensorLogging.validate_job_config(
        config.tensor_logging,
        trainer_config=config,
        is_core_trainer=True,
    )

    config.tensor_logging.families = (TensorMetricFamily.BOUNDARY_OUTPUT,)
    with pytest.raises(ValueError, match="EP-compatible tensor metric families"):
        TensorLogging.validate_job_config(
            config.tensor_logging,
            trainer_config=config,
            is_core_trainer=True,
        )


def test_expert_compute_rows_require_ep() -> None:
    config = _enabled_config()
    config.activation_checkpoint = None
    config.tensor_logging.families = (TensorMetricFamily.EXPERT_COMPUTE_ROWS,)

    with pytest.raises(ValueError, match="requires expert_parallel_degree"):
        TensorLogging.validate_job_config(
            config.tensor_logging,
            trainer_config=config,
            is_core_trainer=True,
        )


@pytest.mark.parametrize("layer_ids", [(), (0, 0), (-1,), (1.5,)])
def test_job_config_rejects_invalid_layer_ids(layer_ids) -> None:
    config = _enabled_config()
    config.tensor_logging.layer_ids = layer_ids

    with pytest.raises(ValueError, match="layer_ids"):
        TensorLogging.validate_job_config(
            config.tensor_logging,
            trainer_config=config,
            is_core_trainer=True,
        )


def test_disabled_model_compile_ignores_inactive_backend() -> None:
    config = _enabled_config()
    config.compile.enable = False
    config.compile.backend = "aot_eager"

    TensorLogging.validate_job_config(
        config.tensor_logging,
        trainer_config=config,
        is_core_trainer=True,
    )

    config.metrics.enable_tensorboard = True
    with pytest.raises(ValueError, match="only the core Trainer"):
        TensorLogging.validate_job_config(
            config.tensor_logging,
            trainer_config=config,
            is_core_trainer=False,
        )


@pytest.mark.parametrize("config_factory", [llama3_debugmodel, qwen3_debugmodel])
def test_ordinary_llama_and_qwen_model_configs_are_supported(config_factory) -> None:
    config = config_factory()
    config.tensor_logging.enable = True
    config.tensor_logging.families = (
        TensorMetricFamily.BOUNDARY_OUTPUT,
        TensorMetricFamily.BOUNDARY_OUTPUT_COTANGENT,
    )
    assert config.model_spec is not None

    TensorLogging.validate_model_config(
        config.tensor_logging,
        model_spec=config.model_spec,
        model_config=config.model_spec.model,
    )


def test_qwen3_moe_internal_families_are_supported() -> None:
    config = qwen3_moe_debug()
    config.tensor_logging.enable = True
    config.tensor_logging.families = (
        TensorMetricFamily.OFFERED_ASSIGNMENTS,
        TensorMetricFamily.EXPERT_COMPUTE_ROWS,
    )
    config.tensor_logging.layer_ids = (0,)
    assert config.model_spec is not None

    TensorLogging.validate_model_config(
        config.tensor_logging,
        model_spec=config.model_spec,
        model_config=config.model_spec.model,
    )


def test_whole_gradient_and_expert_bias_configs_are_supported() -> None:
    llama_config = llama3_debugmodel()
    llama_config.tensor_logging.enable = True
    llama_config.tensor_logging.families = (TensorMetricFamily.WHOLE_GRADIENT,)
    assert llama_config.model_spec is not None
    TensorLogging.validate_model_config(
        llama_config.tensor_logging,
        model_spec=llama_config.model_spec,
        model_config=llama_config.model_spec.model,
    )

    qwen_config = qwen3_moe_debug()
    qwen_config.tensor_logging.enable = True
    qwen_config.tensor_logging.families = (TensorMetricFamily.EXPERT_BIAS,)
    assert qwen_config.model_spec is not None
    TensorLogging.validate_model_config(
        qwen_config.tensor_logging,
        model_spec=qwen_config.model_spec,
        model_config=qwen_config.model_spec.model,
    )

    qwen_config.model_spec.model.layers[0].moe.load_balance_coeff = None
    with pytest.raises(ValueError, match="auxiliary-loss-free balancing"):
        TensorLogging.validate_model_config(
            qwen_config.tensor_logging,
            model_spec=qwen_config.model_spec,
            model_config=qwen_config.model_spec.model,
        )


def test_internal_moe_families_reject_dense_qwen3() -> None:
    config = qwen3_debugmodel()
    config.tensor_logging.enable = True
    config.tensor_logging.families = (TensorMetricFamily.EXPERT_COMPUTE_ROWS,)
    assert config.model_spec is not None

    with pytest.raises(ValueError, match="ordinary MoE.Config"):
        TensorLogging.validate_model_config(
            config.tensor_logging,
            model_spec=config.model_spec,
            model_config=config.model_spec.model,
        )


def test_router_families_reject_converted_router_config() -> None:
    config = qwen3_moe_debug()
    config.tensor_logging.enable = True
    config.tensor_logging.families = (TensorMetricFamily.PER_SEQUENCE_ROUTING,)
    assert config.model_spec is not None
    model_config = config.model_spec.model
    original = model_config.layers[0].moe.router

    @dataclass(kw_only=True, slots=True)
    class ConvertedRouterConfig(TokenChoiceTopKRouter.Config):
        pass

    kwargs = {field.name: getattr(original, field.name) for field in fields(original)}
    model_config.layers[0].moe.router = ConvertedRouterConfig(**kwargs)
    with pytest.raises(ValueError, match="ordinary token-choice router"):
        TensorLogging.validate_model_config(
            config.tensor_logging,
            model_spec=config.model_spec,
            model_config=model_config,
        )


def test_model_config_rejects_invalid_layer_and_converted_projection() -> None:
    config = _enabled_config()
    assert config.model_spec is not None
    model_config = config.model_spec.model
    config.tensor_logging.layer_ids = (len(model_config.layers),)
    with pytest.raises(ValueError, match="but the model has"):
        TensorLogging.validate_model_config(
            config.tensor_logging,
            model_spec=config.model_spec,
            model_config=model_config,
        )

    config.tensor_logging.layer_ids = (0,)
    config.tensor_logging.families = (TensorMetricFamily.BOUNDARY_OUTPUT,)
    original = model_config.layers[0].attention.wo

    @dataclass(kw_only=True, slots=True)
    class ConvertedLinearConfig(Linear.Config):
        pass

    kwargs = {field.name: getattr(original, field.name) for field in fields(original)}
    model_config.layers[0].attention.wo = ConvertedLinearConfig(**kwargs)
    with pytest.raises(ValueError, match="ordinary Linear.Config"):
        TensorLogging.validate_model_config(
            config.tensor_logging,
            model_spec=config.model_spec,
            model_config=model_config,
        )


def test_model_config_rejects_quantization_anywhere() -> None:
    config = _enabled_config()
    assert config.model_spec is not None

    with (
        patch(
            "torchtitan.observability.tensor_logging.component.has_quantization",
            return_value=True,
        ),
        pytest.raises(ValueError, match="quantized models"),
    ):
        TensorLogging.validate_model_config(
            config.tensor_logging,
            model_spec=config.model_spec,
            model_config=config.model_spec.model,
        )


def test_override_can_enable_tensor_logging_before_validation(monkeypatch) -> None:
    config = llama3_debugmodel()
    config.metrics.enable_tensorboard = True

    @override(target=TensorLogging.Config, fqns=["tensor_logging"])
    def enable_tensor_logging(
        current: TensorLogging.Config,
    ) -> TensorLogging.Config:
        return replace(current, enable=True)

    config.override.imports = [f"{__name__}.{enable_tensor_logging.__name__}"]
    monkeypatch.setenv("LOCAL_RANK", "0")
    with (
        patch("torchtitan.trainer.utils.device_module.set_device"),
        patch.object(
            Trainer,
            "init_distributed",
            return_value=Mock(tp=1, cp=1, dp_enabled=False),
        ),
        patch("torchtitan.trainer.dist_utils.set_determinism"),
        patch.object(
            TensorLogging,
            "validate_job_config",
            side_effect=RuntimeError("final enabled config validated"),
        ),
        pytest.raises(RuntimeError, match="final enabled config validated"),
    ):
        Trainer(config)


def test_override_can_disable_tensor_logging_before_validation(monkeypatch) -> None:
    config = _enabled_config()
    config.tensor_logging.layer_ids = ()

    @override(target=TensorLogging.Config, fqns=["tensor_logging"])
    def disable_tensor_logging(
        current: TensorLogging.Config,
    ) -> TensorLogging.Config:
        return replace(current, enable=False)

    config.override.imports = [f"{__name__}.{disable_tensor_logging.__name__}"]
    monkeypatch.setenv("LOCAL_RANK", "0")
    with (
        patch("torchtitan.trainer.utils.device_module.set_device"),
        patch.object(
            Trainer,
            "init_distributed",
            side_effect=RuntimeError("entered distributed setup"),
        ),
        pytest.raises(RuntimeError, match="entered distributed setup"),
    ):
        Trainer(config)


def test_override_selection_is_validated_after_replacement(monkeypatch) -> None:
    config = _enabled_config()

    @override(target=TensorLogging.Config, fqns=["tensor_logging"])
    def duplicate_tensor_logging_layer(
        current: TensorLogging.Config,
    ) -> TensorLogging.Config:
        return replace(current, layer_ids=(0, 0))

    config.override.imports = [f"{__name__}.{duplicate_tensor_logging_layer.__name__}"]
    monkeypatch.setenv("LOCAL_RANK", "0")
    with (
        patch("torchtitan.trainer.utils.device_module.set_device"),
        patch.object(
            Trainer,
            "init_distributed",
            return_value=Mock(tp=1, cp=1, dp_enabled=False),
        ),
        patch("torchtitan.trainer.dist_utils.set_determinism"),
        pytest.raises(ValueError, match="must not contain duplicates"),
    ):
        Trainer(config)


def test_torchft_rejects_tensor_logging() -> None:
    pytest.importorskip("torchft")
    from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer

    config = _enabled_config()
    with pytest.raises(ValueError, match="does not yet support TorchFT"):
        FaultTolerantTrainer(config)


def test_disabled_config_bypasses_tensor_logging_validation() -> None:
    config = llama3_debugmodel()
    config.tensor_logging.layer_ids = ()
    config.tensor_logging.families = ()

    TensorLogging.validate_job_config(
        config.tensor_logging,
        trainer_config=config,
        is_core_trainer=False,
    )


def test_metrics_processor_reports_only_nonempty_writer() -> None:
    processor = object.__new__(MetricsProcessor)
    processor.logger = BaseLogger()
    assert not processor.has_active_logger

    processor.logger = LoggerContainer()
    assert not processor.has_active_logger

    processor.logger.add_logger(BaseLogger())
    assert processor.has_active_logger


def test_metrics_processor_forwards_completed_tensor_metrics_to_sink() -> None:
    processor = object.__new__(MetricsProcessor)
    processor.num_flops_per_token = 1
    processor.ntokens_since_last_log = 1
    processor.parallel_dims = SimpleNamespace(non_data_parallel_size=1)
    processor.time_last_log = time.perf_counter() - 1
    processor.config = SimpleNamespace(log_freq=1)
    processor.has_quantization = False
    processor.gpu_peak_flops = 1
    processor.data_loading_times = [0.0]
    processor.device_memory_monitor = Mock()
    processor.device_memory_monitor.get_peak_stats.return_value = SimpleNamespace(
        max_active_gib=0.0,
        max_active_pct=0.0,
        max_reserved_gib=0.0,
        max_reserved_pct=0.0,
        num_alloc_retries=0,
        num_ooms=0,
    )
    processor.color = SimpleNamespace(
        red="",
        green="",
        orange="",
        turquoise="",
        blue="",
        cyan="",
        magenta="",
        reset="",
    )
    processor.logger = Mock()
    tensor_key = "tensor_metrics/layers.0.attention.wo.weight.w.abs_mean"

    processor.log(
        step=1,
        global_avg_loss=1.0,
        global_max_loss=1.0,
        grad_norm=1.0,
        extra_metrics={tensor_key: 2.5},
    )

    metrics, step = processor.logger.log.call_args.args
    assert step == 1
    assert metrics[tensor_key] == 2.5


def test_component_requires_one_constructed_metrics_writer() -> None:
    dist.init_process_group("fake", rank=0, world_size=1)
    try:
        processor = object.__new__(MetricsProcessor)
        processor.logger = LoggerContainer()
        config = TensorLogging.Config(enable=True)
        with (
            patch(
                "torchtitan.observability.tensor_logging.component.ParameterStatisticsBatch"
            ),
            pytest.raises(RuntimeError, match="exactly one active metrics writer"),
        ):
            TensorLogging(
                config,
                model=Mock(),
                parallel_dims=Mock(),
                metrics_processor=processor,
                local_batch_size=1,
                dataloader_config=Mock(),
                device=torch.device("cpu"),
            )

        processor.logger.add_logger(BaseLogger())
        with patch(
            "torchtitan.observability.tensor_logging.component.ParameterStatisticsBatch"
        ):
            tensor_logging = TensorLogging(
                config,
                model=Mock(),
                parallel_dims=Mock(),
                metrics_processor=processor,
                local_batch_size=1,
                dataloader_config=Mock(),
                device=torch.device("cpu"),
            )
        assert tensor_logging._is_writer
    finally:
        dist.destroy_process_group()


def test_singleton_publication_outcome_advances_only_on_success() -> None:
    dist.init_process_group("fake", rank=0, world_size=1)
    try:
        tensor_logging = object.__new__(TensorLogging)
        tensor_logging._outcome_template = torch.empty((), dtype=torch.int32)
        tensor_logging._last_successful_publication_step = 0

        tensor_logging.complete_publication(step=1, local_error=None)
        assert tensor_logging._last_successful_publication_step == 1

        error = RuntimeError("sink failed")
        with pytest.raises(RuntimeError, match="sink failed"):
            tensor_logging.complete_publication(step=3, local_error=error)
        assert tensor_logging._last_successful_publication_step == 1
    finally:
        dist.destroy_process_group()


def test_nonwriter_derivation_does_not_touch_the_metric_batch() -> None:
    tensor_logging = object.__new__(TensorLogging)
    tensor_logging._is_writer = False
    tensor_logging._parameter_batch = Mock()
    tensor_logging._output_batch = Mock()
    tensor_logging._expert_count_recorder = Mock()
    tensor_logging._router_statistics_recorder = Mock()

    assert tensor_logging.derive_metrics(Mock(), step=1) == {}
    tensor_logging._parameter_batch.derive_metrics.assert_not_called()
    tensor_logging._output_batch.derive_metrics.assert_not_called()
    tensor_logging._expert_count_recorder.derive_metrics.assert_not_called()
    tensor_logging._router_statistics_recorder.derive_metrics.assert_not_called()
