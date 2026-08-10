# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class TensorMetricSite(str, Enum):
    ATTENTION_INPUT = "attention_input"
    ATTENTION_INPUT_GRAD = "attention_input_grad"
    ATTENTION_OUTPUT_WEIGHT = "attention_output_weight"
    ATTENTION_OUTPUT_WEIGHT_GRAD = "attention_output_weight_grad"
    MOE_OFFERED_ASSIGNMENTS = "moe_offered_assignments"
    MOE_COMPUTE_ROWS = "moe_compute_rows"


PARAMETER_SITES = (
    TensorMetricSite.ATTENTION_OUTPUT_WEIGHT,
    TensorMetricSite.ATTENTION_OUTPUT_WEIGHT_GRAD,
)


def resolve_parameter_sites(
    requested: tuple[TensorMetricSite, ...] | None,
) -> tuple[tuple[TensorMetricSite, ...], dict[TensorMetricSite, str]]:
    """Resolve the parameter-first selection and explain every omitted site."""
    selected = PARAMETER_SITES if requested is None else requested
    if not selected:
        raise ValueError("tensor_logging.sites must not be empty")
    if any(not isinstance(site, TensorMetricSite) for site in selected):
        raise ValueError("tensor_logging.sites must contain TensorMetricSite values")
    if len(set(selected)) != len(selected):
        raise ValueError("tensor_logging.sites must not contain duplicates")

    unsupported = [site for site in selected if site not in PARAMETER_SITES]
    if unsupported:
        names = ", ".join(site.name for site in unsupported)
        raise ValueError(
            f"tensor logging sites are not supported in the parameter-first slice: {names}"
        )

    omitted = {
        site: (
            "not requested"
            if site in PARAMETER_SITES
            else "not in parameter-first slice"
        )
        for site in TensorMetricSite
        if site not in selected
    }
    return selected, omitted
