# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class TensorMetricFamily(str, Enum):
    """Semantic tensor-metric families available to recipes.

    Example:

        families = (
            TensorMetricFamily.PARAMETER,
            TensorMetricFamily.BOUNDARY_OUTPUT,
        )
    """

    PARAMETER = "parameter"
    PRECLIP_GRADIENT = "preclip_gradient"
    BOUNDARY_OUTPUT = "boundary_output"
    BOUNDARY_OUTPUT_COTANGENT = "boundary_output_cotangent"
    ROUTER_DISTRIBUTION = "router_distribution"
    OFFERED_ASSIGNMENTS = "offered_assignments"
    PER_SEQUENCE_ROUTING = "per_sequence_routing"
    EXPERT_COMPUTE_ROWS = "expert_compute_rows"
    WHOLE_GRADIENT = "whole_gradient"
    EXPERT_BIAS = "expert_bias"


PARAMETER_FAMILIES = (
    TensorMetricFamily.PARAMETER,
    TensorMetricFamily.PRECLIP_GRADIENT,
)

BOUNDARY_FAMILIES = (
    TensorMetricFamily.BOUNDARY_OUTPUT,
    TensorMetricFamily.BOUNDARY_OUTPUT_COTANGENT,
)

ROUTER_FAMILIES = (
    TensorMetricFamily.ROUTER_DISTRIBUTION,
    TensorMetricFamily.PER_SEQUENCE_ROUTING,
)

EXPERT_COUNT_FAMILIES = (
    TensorMetricFamily.OFFERED_ASSIGNMENTS,
    TensorMetricFamily.EXPERT_COMPUTE_ROWS,
)

INTERNAL_FAMILIES = ROUTER_FAMILIES + EXPERT_COUNT_FAMILIES

JOB_FAMILIES = (TensorMetricFamily.WHOLE_GRADIENT,)

OPTIMIZER_FAMILIES = (TensorMetricFamily.EXPERT_BIAS,)


def resolve_families(
    requested: tuple[TensorMetricFamily, ...] | None,
) -> tuple[TensorMetricFamily, ...]:
    """Resolve and validate the tensor-metric family selection."""
    selected = PARAMETER_FAMILIES if requested is None else requested
    if not selected:
        raise ValueError("tensor_logging.families must not be empty")
    if any(not isinstance(family, TensorMetricFamily) for family in selected):
        raise ValueError(
            "tensor_logging.families must contain TensorMetricFamily values"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("tensor_logging.families must not contain duplicates")
    return selected
