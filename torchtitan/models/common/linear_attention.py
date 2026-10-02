# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kernel backend selection for Attention Gym's chunked linear-attention kernels."""

from typing import Literal

import torch

ChunkBackend = Literal["auto", "fused", "cudnn"]


def resolve_chunk_backend(
    chunk_backend: ChunkBackend, q: torch.Tensor
) -> Literal["fused", "cudnn"]:
    """Resolve "auto" for the query the kernel gets.

    Attention Gym's cuDNN chunk kernels take only fp16/bf16 inputs and run only on
    SM100 and SM103, so "auto" picks "cudnn" there and "fused" otherwise.
    """
    if chunk_backend != "auto":
        return chunk_backend
    if q.dtype not in (torch.float16, torch.bfloat16):
        return "fused"
    capability = torch.cuda.get_device_capability(q.device)
    return "cudnn" if capability in ((10, 0), (10, 3)) else "fused"
