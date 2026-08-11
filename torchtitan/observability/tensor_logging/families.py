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
    OFFERED_ASSIGNMENTS = "offered_assignments"


PARAMETER_FAMILIES = (
    TensorMetricFamily.PARAMETER,
    TensorMetricFamily.PRECLIP_GRADIENT,
)

BOUNDARY_FAMILIES = (
    TensorMetricFamily.BOUNDARY_OUTPUT,
    TensorMetricFamily.BOUNDARY_OUTPUT_COTANGENT,
)

INTERNAL_FAMILIES = (TensorMetricFamily.OFFERED_ASSIGNMENTS,)


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
