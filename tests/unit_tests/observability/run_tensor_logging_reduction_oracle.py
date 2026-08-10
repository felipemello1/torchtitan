# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Four-rank DP/TP reduction oracle for manual GPU validation."""

import os

import torch
import torch.distributed as dist
import torch.testing._internal.distributed.fake_pg  # noqa: F401
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.observability.tensor_logging.reduction import reduce_sum


def main() -> None:
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise RuntimeError("This oracle requires exactly four ranks.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=2,
        cp=1,
        tp=2,
        pp=1,
        ep=1,
        world_size=dist.get_world_size(),
    )
    parallel_dims.build_mesh()
    dp_mesh = parallel_dims.get_mesh("batch")
    tp_mesh = parallel_dims.get_mesh("tp")
    world_mesh = parallel_dims.world_mesh
    dp_coordinate = dp_mesh.get_local_rank()
    tp_coordinate = tp_mesh.get_local_rank()
    one = torch.ones((), dtype=torch.int64, device=device)
    value = torch.tensor(
        100 * dp_coordinate + 10 * tp_coordinate + 1,
        dtype=torch.int64,
        device=device,
    )

    assert reduce_sum(one, dp_mesh).item() == 2
    assert reduce_sum(one, tp_mesh).item() == 2
    assert reduce_sum(one, world_mesh).item() == 4
    assert reduce_sum(value, dp_mesh).item() == 102 + 20 * tp_coordinate
    assert reduce_sum(value, tp_mesh).item() == 12 + 200 * dp_coordinate
    assert reduce_sum(value, world_mesh).item() == 224

    dist.barrier()
    if dist.get_rank() == 0:
        print("PASS: DP cohorts 102/122; TP cohorts 12/212; WORLD cohort 224")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
