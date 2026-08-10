# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh


def reduce_sum(value: torch.Tensor, mesh: DeviceMesh | None) -> torch.Tensor:
    """Sum a device tensor over one site-resolved mesh."""
    if mesh is None:
        return value
    return funcol.wait_tensor(funcol.all_reduce(value, reduceOp="sum", group=mesh))


def reduce_max(value: torch.Tensor, mesh: DeviceMesh | None) -> torch.Tensor:
    """Take a device-tensor maximum over one site-resolved mesh."""
    if mesh is None:
        return value
    return funcol.wait_tensor(funcol.all_reduce(value, reduceOp="max", group=mesh))
