# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

from torchtitan.components.optimizer import _momentum_gradient_cosine
from torchtitan.observability.tensor_logging import (
    DataStatistics,
    init,
    log_stats,
    register,
    set_enabled,
)


class RouterOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        register(
            self,
            ["seq_expert_imbalance_mean", "seq_expert_imbalance_max"],
        )


class RouterRoot(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {str(layer_id): RouterOwner() for layer_id in range(layer_count)}
        )


def _local_counts(dp_rank: int, cp_rank: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros((2, 1, 3), dtype=torch.int64, device=device)
    if dp_rank == 0:
        counts[0, 0] = torch.tensor(
            [6, 0, 0] if cp_rank == 0 else [0, 0, 0],
            device=device,
        )
        counts[1, 0] = torch.tensor(
            [0, 4, 0] if cp_rank == 0 else [0, 0, 4],
            device=device,
        )
    else:
        counts[0, 0] = torch.tensor(
            [0, 3, 0] if cp_rank == 0 else [0, 0, 3],
            device=device,
        )
        counts[1, 0] = torch.tensor(
            [9, 0, 0] if cp_rank == 0 else [0, 0, 0],
            device=device,
        )
    return counts


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    dp_rank, cp_rank = divmod(rank, 2)
    cp_group = cp_groups[dp_rank]

    shard_mesh = init_device_mesh(
        "cuda",
        (dist.get_world_size(),),
        mesh_dim_names=("dp_shard",),
    )
    momentum = distribute_tensor(
        torch.tensor([1.0, 2.0, 3.0, 4.0], device=device),
        shard_mesh,
        [Shard(0)],
    )
    gradient = distribute_tensor(
        torch.tensor([2.0, 1.0, 4.0, 3.0], device=device),
        shard_mesh,
        [Shard(0)],
    )
    torch.testing.assert_close(
        _momentum_gradient_cosine(momentum, gradient),
        torch.tensor([28.0 / 30.0], device=device),
    )

    tp_rank = cp_rank
    dp_data = DataStatistics(
        dataset_id="dp_oracle",
        data_contributor=tp_rank == 0,
        loss_contributor=tp_rank == 0,
        step_contributor=rank == 0,
        device=device,
    )
    dp_positions = torch.tensor(
        [[0, 1, 0, 1] if dp_rank == 0 else [0, 1, 2, 3]],
        device=device,
    )
    dp_labels = torch.tensor(
        [[1, 2, 3, 4] if dp_rank == 0 else [5, 6, 7, -100]],
        device=device,
    )
    dp_data.record_batch(dp_labels, dp_positions)
    dp_data.record_step_loss(
        torch.tensor((2.0 if dp_rank == 0 else 5.0) / 7, device=device),
        global_valid_tokens=7,
    )
    dp_metrics = dp_data.collect()
    assert dp_metrics["data/datasets.dp_oracle.valid_token_count"] == 7
    assert dp_metrics["data/datasets.dp_oracle.loss_mean"] == 1.0
    assert dp_metrics["data/datasets.dp_oracle.window_steps"] == 1
    assert dp_metrics["data/block_causal.sum_length_squared"] == 24

    cp_data = DataStatistics(
        dataset_id="cp_oracle",
        data_contributor=dp_rank == 0 and tp_rank == 0,
        loss_contributor=tp_rank == 0,
        step_contributor=rank == 0,
        device=device,
    )
    cp_positions = torch.tensor([[0, 1, 0, 1]], device=device)
    cp_labels = torch.tensor([[1, 2, 3, -100]], device=device)
    cp_data.record_batch(cp_labels, cp_positions)
    cp_data.record_step_loss(
        torch.tensor((1.0 if dp_rank == 0 else 3.0) / 3, device=device),
        global_valid_tokens=3,
    )
    cp_metrics = cp_data.collect()
    assert cp_metrics["data/datasets.cp_oracle.valid_token_count"] == 3
    assert cp_metrics["data/datasets.cp_oracle.loss_mean"] == 4 / 3
    assert cp_metrics["data/datasets.cp_oracle.window_steps"] == 1
    assert cp_metrics["data/block_causal.sum_length_squared"] == 8

    root = RouterRoot(layer_count=2).to(device)
    runtime = init(root, device=device)
    try:
        per_sequence_counts = _local_counts(dp_rank, cp_rank, device)
        dist.all_reduce(per_sequence_counts, group=cp_group)

        expected = (
            torch.tensor([[6, 0, 0], [0, 4, 4]], device=device)
            if dp_rank == 0
            else torch.tensor([[0, 3, 3], [9, 0, 0]], device=device)
        )
        torch.testing.assert_close(per_sequence_counts[:, 0], expected)

        with set_enabled(True):
            for router, layer_counts in zip(
                root.layers.values(),
                per_sequence_counts,
                strict=True,
            ):
                average = layer_counts.float().mean(dim=-1).clamp_min(1)
                imbalance = (layer_counts / average.unsqueeze(-1)).max(dim=-1).values
                log_stats(
                    router,
                    seq_expert_imbalance_mean=imbalance.mean().view(1),
                    seq_expert_imbalance_max=imbalance.max().view(1),
                )

        metrics = runtime.buffers_to_metrics(runtime.reduce_buffers())
        for layer_id in range(2):
            prefix = f"layers.{layer_id}.seq_expert_imbalance"
            assert metrics[prefix + "_mean.observation_count"] == 4
            assert metrics[prefix + "_mean.abs_mean"] == 2.25
            assert metrics[prefix + "_max.abs_mean"] == 2.25

        runtime.clear()
        if rank == 0:
            print("topology PASS: CP router + sharded cosine + DP/CP data populations")
    finally:
        runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
