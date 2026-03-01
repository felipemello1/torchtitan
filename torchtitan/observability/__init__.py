# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchTitan Observability Library

PR1 — System metrics (phase timing, step context):
    init_observability, set_step, add_step_tag, clear_step_tags,
    record_span, record_event, EventType

PR2 — Tensor experiment metrics (compile-safe):
    record_tensor_metric, InvocationContext, replicate_to_host,
    MeanTMetric, SumTMetric, MaxTMetric, MinTMetric, DerivedTMetric
"""

from torchtitan.observability.invocation_context import (
    child_context,
    current_context,
    InvocationContext,
    record_tensor_metric,
)
from torchtitan.observability.structured_logging import (
    add_step_tag,
    clear_step_tags,
    EventType,
    init_observability,
    record_event,
    record_span,
    set_step,
)

from torchtitan.observability.tensor_metrics import (
    DerivedTMetric,
    MaxTMetric,
    MeanTMetric,
    MinTMetric,
    replicate_to_host,
    SumTMetric,
    TMetricValue,
)

__all__ = [
    # PR1
    "init_observability",
    "set_step",
    "add_step_tag",
    "clear_step_tags",
    "record_span",
    "record_event",
    "EventType",
    # PR2
    "record_tensor_metric",
    "InvocationContext",
    "current_context",
    "child_context",
    "replicate_to_host",
    "TMetricValue",
    "MeanTMetric",
    "SumTMetric",
    "MaxTMetric",
    "MinTMetric",
    "DerivedTMetric",
]
