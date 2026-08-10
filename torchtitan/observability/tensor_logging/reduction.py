# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.observability.tensor_logging.recorders import FiniteStatistics


def reduce_sum(value: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
    """Sum a device tensor over one site-resolved mesh."""
    return funcol.wait_tensor(funcol.all_reduce(value, reduceOp="sum", group=mesh))


def reduce_max(value: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
    """Take a device-tensor maximum over one site-resolved mesh."""
    return funcol.wait_tensor(funcol.all_reduce(value, reduceOp="max", group=mesh))


def reduce_finite_statistics(
    statistics: FiniteStatistics,
    owner_meshes: tuple[DeviceMesh, ...],
) -> FiniteStatistics:
    """Reduce one owned snapshot over an ordered sequence of owner meshes."""
    counts = statistics.counts.clone()
    sums = statistics.sums.clone()
    abs_max = statistics.abs_max.clone()
    for mesh in owner_meshes:
        counts = reduce_sum(counts, mesh)
        sums = reduce_sum(sums, mesh)
        abs_max = reduce_max(abs_max, mesh)
    return FiniteStatistics(counts=counts, sums=sums, abs_max=abs_max)
