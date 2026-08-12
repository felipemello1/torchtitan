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
from torchtitan.distributed import ParallelDims
from torchtitan.observability.tensor_logging import init, register, set_enabled
from torchtitan.observability.tensor_logging.data import DataStatistics
from torchtitan.observability.tensor_logging.router import log_router_statistics


class RouterOwner(nn.Module):
    def __init__(self, expert_count: int) -> None:
        super().__init__()
        self.score_func = "softmax"
        self.register_buffer(
            "_router_logits_mean_E",
            torch.zeros(expert_count),
            persistent=False,
        )
        register(
            self,
            [
                "entropy",
                "local_expert_imbalance",
                "seq_expert_imbalance_mean",
                "seq_expert_imbalance_max",
                "expert_load",
                "experts_max_violation",
                "ep_shard_imbalance",
            ],
        )


class MoESource(nn.Module):
    def __init__(self, expert_count: int) -> None:
        super().__init__()
        self.router = RouterOwner(expert_count)
        self.expert_bias_E = None
        self._sequence_expert_counts_BE = None


class RouterBlock(nn.Module):
    def __init__(self, expert_count: int) -> None:
        super().__init__()
        self.moe = MoESource(expert_count)


class RouterRoot(nn.Module):
    def __init__(self, layer_count: int, expert_count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                str(layer_id): RouterBlock(expert_count)
                for layer_id in range(layer_count)
            }
        )


def _local_counts(dp_rank: int, cp_rank: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros((2, 1, 3), dtype=torch.float32, device=device)
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


def _assert_metric_close(metrics: dict[str, int | float], key: str, expected) -> None:
    torch.testing.assert_close(
        torch.tensor(metrics[key]),
        torch.as_tensor(expected),
    )


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dp_rank, cp_rank = divmod(rank, 2)
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=2,
        cp=2,
        tp=1,
        pp=1,
        ep=1,
        world_size=dist.get_world_size(),
    )
    parallel_dims.build_mesh()

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
        ignore_index=-100,
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
        ignore_index=-100,
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

    root = RouterRoot(layer_count=2, expert_count=3).to(device)
    runtime = init(root, device=device)
    try:
        local_sequence_counts = _local_counts(dp_rank, cp_rank, device)
        local_tokens_by_layer = list(local_sequence_counts.sum(dim=1))
        global_tokens_by_layer = torch.stack(local_tokens_by_layer)
        dist.all_reduce(
            global_tokens_by_layer,
            group=parallel_dims.get_mesh("loss").get_group(),
        )

        moe_layers = []
        for block, sequence_counts_BE in zip(
            root.layers.values(),
            local_sequence_counts,
            strict=True,
        ):
            block.moe._sequence_expert_counts_BE = sequence_counts_BE
            moe_layers.append((block, block.moe))

        expected_global = torch.tensor(
            [[6, 3, 3], [9, 4, 4]],
            dtype=torch.float32,
            device=device,
        )
        torch.testing.assert_close(global_tokens_by_layer, expected_global)

        with set_enabled(True):
            log_router_statistics(
                moe_layers,
                local_tokens_per_expert_by_layer=local_tokens_by_layer,
                global_tokens_per_expert_by_layer=global_tokens_by_layer,
                parallel_dims=parallel_dims,
            )

        metrics = runtime.buffers_to_metrics(runtime.reduce_buffers())
        for layer_id in range(2):
            prefix = f"layers.{layer_id}.moe.router.seq_expert_imbalance"
            assert metrics[prefix + "_mean.observation_count"] == 4
            assert metrics[prefix + "_mean.abs_mean"] == 2.25
            assert metrics[prefix + "_max.abs_mean"] == 2.25

        first_router = "layers.0.moe.router"
        _assert_metric_close(
            metrics,
            first_router + ".entropy.abs_mean",
            torch.log(torch.tensor(3.0)),
        )
        _assert_metric_close(
            metrics,
            first_router + ".local_expert_imbalance.abs_mean",
            2.25,
        )
        _assert_metric_close(metrics, first_router + ".expert_load.abs_mean", 1.0)
        _assert_metric_close(metrics, first_router + ".expert_load.abs_max", 1.5)
        _assert_metric_close(
            metrics,
            first_router + ".experts_max_violation.abs_mean",
            0.5,
        )
        _assert_metric_close(
            metrics,
            first_router + ".ep_shard_imbalance.abs_mean",
            1.0,
        )

        second_router = "layers.1.moe.router"
        _assert_metric_close(metrics, second_router + ".expert_load.abs_mean", 1.0)
        _assert_metric_close(
            metrics,
            second_router + ".expert_load.abs_max",
            27 / 17,
        )
        _assert_metric_close(
            metrics,
            second_router + ".experts_max_violation.abs_mean",
            10 / 17,
        )

        runtime.clear()
        if rank == 0:
            print("topology PASS: CP router + sharded cosine + DP/CP data populations")
    finally:
        runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
