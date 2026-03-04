# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchTitan Observability Library

System metrics: init_observability, set_step, record_span, record_event, EventType
Tensor metrics: record_tensor_metric, TensorMetricContext, child_context, replicate_to_host
Backends: EveryNSteps, CompositeSummaryWriter, TensorBoard, WandB
CPU metrics: record_metric, log_reduced_metrics, DefaultAggregator, MeanMetric
Profiling: Profiler, ProfilerConfig, profile_annotation
RL: RolloutLogger, filter_top_bottom
"""

from torchtitan.observability.backends import (
    CompositeSummaryWriter, InMemorySummaryWriter, LoggingSummaryWriter,
    SummaryWriter, TensorBoardSummaryWriter, WandbSummaryWriter,
)
from torchtitan.observability.common import add_step_tag, clear_step_tags, set_step
from torchtitan.observability.logging_boundary import EveryNSteps
from torchtitan.observability.metrics import (
    DefaultAggregator, log_reduced_metrics, MaxMetric, MeanMetric,
    MinMetric, record_metric, SumMetric,
)
from torchtitan.observability.profiling import Profiler, ProfilerConfig, profile_annotation
from torchtitan.observability.rollout_logger import filter_top_bottom, RolloutLogger
from torchtitan.observability.structured_logging import (
    EventType, init_observability, record_event, record_span,
)
from torchtitan.observability.tensor_metric_context import (
    child_context, current_tensor_metric_context, record_tensor_metric, TensorMetricContext,
)
from torchtitan.observability.tensor_metrics import (
    MaxTMetric, MeanTMetric, MinTMetric, replicate_to_host, SumTMetric, TMetricValue,
)

__all__ = [
    "init_observability", "set_step", "add_step_tag", "clear_step_tags",
    "record_span", "record_event", "EventType",
    "record_tensor_metric", "TensorMetricContext", "current_tensor_metric_context",
    "child_context", "replicate_to_host", "TMetricValue",
    "MeanTMetric", "SumTMetric", "MaxTMetric", "MinTMetric",
    "EveryNSteps", "SummaryWriter", "CompositeSummaryWriter",
    "TensorBoardSummaryWriter", "WandbSummaryWriter",
    "LoggingSummaryWriter", "InMemorySummaryWriter",
    "record_metric", "log_reduced_metrics", "DefaultAggregator",
    "MeanMetric", "MaxMetric", "MinMetric", "SumMetric",
    "Profiler", "ProfilerConfig", "profile_annotation",
    "RolloutLogger", "filter_top_bottom",
]
