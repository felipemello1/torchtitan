# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class TensorMetricFamily(str, Enum):
    """Semantic tensor-metric families available to recipes."""

    PARAMETER = "parameter"
    PRECLIP_GRADIENT = "preclip_gradient"


PARAMETER_FAMILIES = (
    TensorMetricFamily.PARAMETER,
    TensorMetricFamily.PRECLIP_GRADIENT,
)


def resolve_parameter_families(
    requested: tuple[TensorMetricFamily, ...] | None,
) -> tuple[TensorMetricFamily, ...]:
    """Resolve and validate the parameter-family selection."""
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
