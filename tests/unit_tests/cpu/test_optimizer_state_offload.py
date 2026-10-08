# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.components.optim.offload import (
    _pack_params_by_state_bytes,
    OptimizerStateOffloader,
)


def _params(*numels: int) -> list[torch.Tensor]:
    return [torch.empty(numel) for numel in numels]


def test_packer_keeps_whole_params_and_cuts_before_exceeding_target() -> None:
    # fp32 moments: 8 B per element; target 1 MiB = 131072 elements
    p = _params(60_000, 60_000, 60_000, 10)
    chunks = _pack_params_by_state_bytes(
        p, chunk_size_bytes=1024 * 1024, state_dtype=torch.float32
    )
    assert [[id(t) for t in chunk] for chunk in chunks] == [
        [id(p[0]), id(p[1])],
        [id(p[2]), id(p[3])],
    ]


def test_packer_bf16_moments_pack_twice_as_many() -> None:
    p = _params(60_000, 60_000, 60_000, 60_000)
    chunks = _pack_params_by_state_bytes(
        p, chunk_size_bytes=1024 * 1024, state_dtype=torch.bfloat16
    )
    assert [len(chunk) for chunk in chunks] == [4]


def test_packer_oversized_param_forms_its_own_chunk() -> None:
    p = _params(10, 500_000, 10)
    chunks = _pack_params_by_state_bytes(
        p, chunk_size_bytes=1024 * 1024, state_dtype=torch.float32
    )
    assert [len(chunk) for chunk in chunks] == [1, 1, 1]


def test_config_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError):
        OptimizerStateOffloader.Config(chunk_size_mb=0)


def test_container_rejects_optim_cuda_graph() -> None:
    config = OptimizersContainer.Config(
        optimizers=[
            AdamW.Config(pattern=r".*", state_offload=OptimizerStateOffloader.Config())
        ],
    )
    with pytest.raises(ValueError, match="CUDA graphs"):
        config.build(model_parts=[nn.Linear(2, 2)], enable_cuda_graph=True)


def test_container_rejects_mixed_param_dtypes() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2).to(torch.bfloat16))
    config = OptimizersContainer.Config(
        optimizers=[
            AdamW.Config(pattern=r".*", state_offload=OptimizerStateOffloader.Config())
        ],
    )
    with pytest.raises(ValueError, match="one parameter dtype"):
        config.build(model_parts=[model])
