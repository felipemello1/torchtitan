# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.distributed as dist
from torchtitan.components.optimizer import _global_moe_expert_counts
from torchtitan.distributed import ParallelDims
from torchtitan.observability.tensor_logging import (
    init,
    log_stats,
    register,
    set_enabled,
)
from torchtitan.observability.tensor_logging.router import log_router_statistics

from tests.unit_tests.observability.run_tensor_logging_router_topology import (
    _assert_metric_close,
    RouterRoot,
)


def _local_sequence_counts(
    dp_rank: int,
    cp_rank: int,
    tp_rank: int,
    device: torch.device,
) -> torch.Tensor:
    counts = torch.zeros((2, 1, 4), dtype=torch.int64, device=device)
    if cp_rank != 0 or tp_rank != 0:
        return counts
    if dp_rank == 0:
        counts[:, 0] = torch.tensor(
            [[8, 0, 0, 0], [0, 4, 4, 0]],
            dtype=torch.int64,
            device=device,
        )
    else:
        counts[:, 0] = torch.tensor(
            [[0, 0, 3, 3], [9, 0, 0, 0]],
            dtype=torch.int64,
            device=device,
        )
    return counts


def main() -> None:
    dist.init_process_group("nccl")
    if dist.get_world_size() != 8:
        raise ValueError("compound router topology oracle requires eight ranks")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dp_rank, remainder = divmod(rank, 4)
    cp_rank, tp_rank = divmod(remainder, 2)
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=2,
        cp=2,
        tp=2,
        pp=1,
        ep=2,
        world_size=dist.get_world_size(),
        spmd_backend="spmd_types",
    )
    parallel_dims.build_mesh()

    root = RouterRoot(layer_count=2, expert_count=4).to(device)
    register(root, ["rank_oracle"])
    runtime = init(root, device=device)
    try:
        rank_coded_counts = torch.full(
            (1, 4),
            2**24 + rank + 1,
            dtype=torch.int64,
            device=device,
        )
        torch.testing.assert_close(
            _global_moe_expert_counts(rank_coded_counts, parallel_dims),
            torch.full(
                (1, 4),
                8 * 2**24 + 36,
                dtype=torch.int64,
                device=device,
            ),
        )

        local_sequence_counts = _local_sequence_counts(
            dp_rank,
            cp_rank,
            tp_rank,
            device,
        )
        local_tokens_by_layer = list(local_sequence_counts.sum(dim=1))
        global_tokens_by_layer = _global_moe_expert_counts(
            torch.stack(local_tokens_by_layer),
            parallel_dims,
        )
        torch.testing.assert_close(
            global_tokens_by_layer,
            torch.tensor(
                [[8, 0, 3, 3], [9, 4, 4, 0]],
                dtype=torch.int64,
                device=device,
            ),
        )

        moe_layers = []
        for block, sequence_counts_BE in zip(
            root.layers.values(),
            local_sequence_counts,
            strict=True,
        ):
            block.moe._sequence_expert_counts_BE = sequence_counts_BE
            moe_layers.append((block, block.moe))

        with set_enabled(True):
            log_stats(
                root,
                rank_oracle=torch.tensor(float(rank), device=device),
            )
            log_router_statistics(
                moe_layers,
                local_tokens_per_expert_by_layer=local_tokens_by_layer,
                global_tokens_per_expert_by_layer=global_tokens_by_layer,
                parallel_dims=parallel_dims,
            )

        metrics = runtime.buffers_to_metrics(runtime.reduce_buffers())
        first_router = "layers.0.moe.router"
        second_router = "layers.1.moe.router"

        assert metrics["rank_oracle.observation_count"] == 8
        _assert_metric_close(metrics, "rank_oracle.abs_mean", 3.5)
        _assert_metric_close(metrics, "rank_oracle.abs_max", 7.0)
        assert metrics[first_router + ".expert_load.observation_count"] == 8
        _assert_metric_close(
            metrics,
            first_router + ".entropy.abs_mean",
            torch.log(torch.tensor(4.0)),
        )
        _assert_metric_close(metrics, first_router + ".expert_load.abs_mean", 1.0)
        _assert_metric_close(metrics, first_router + ".expert_load.abs_max", 16 / 7)
        _assert_metric_close(
            metrics,
            first_router + ".experts_max_violation.abs_mean",
            9 / 7,
        )
        _assert_metric_close(
            metrics,
            first_router + ".ep_shard_imbalance.abs_mean",
            8 / 7,
        )
        _assert_metric_close(
            metrics,
            first_router + ".seq_expert_imbalance_mean.abs_mean",
            0.75,
        )

        _assert_metric_close(metrics, second_router + ".expert_load.abs_mean", 1.0)
        _assert_metric_close(
            metrics,
            second_router + ".expert_load.abs_max",
            36 / 17,
        )
        _assert_metric_close(
            metrics,
            second_router + ".experts_max_violation.abs_mean",
            19 / 17,
        )
        _assert_metric_close(
            metrics,
            second_router + ".ep_shard_imbalance.abs_mean",
            26 / 17,
        )
        _assert_metric_close(
            metrics,
            second_router + ".seq_expert_imbalance_mean.abs_mean",
            1.5,
        )

        if rank == 0:
            print("topology PASS: DP2 x CP2 x TP2 / EP2 router metrics")
    finally:
        runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
