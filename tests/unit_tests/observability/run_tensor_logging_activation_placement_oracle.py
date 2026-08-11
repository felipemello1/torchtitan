# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import AsyncCollectiveTensor, wait_tensor
from torch.distributed.tensor import DTensor

from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.sharding import set_llama3_sharding_config


def _describe_tensor(value: torch.Tensor) -> dict[str, object]:
    if not isinstance(value, DTensor):
        return {
            "kind": type(value).__name__,
            "shape": list(value.shape),
        }
    local = value._local_tensor
    return {
        "kind": "DTensor",
        "placements": [str(placement) for placement in value.placements],
        "global_shape": list(value.shape),
        "local_shape": list(local.shape),
        "local_kind": type(local).__name__,
    }


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    device = torch.device("cuda", local_rank)
    enable_sp = os.environ["PROBE_ENABLE_SP"] == "1"

    dist_utils.set_spmd_backend("default")
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=2,
        pp=1,
        ep=1,
        world_size=2,
    )
    parallel_dims.build_mesh()

    model_config = llama3_configs["debugmodel"](attn_backend="flex")
    attention_config = model_config.layers[0].attention
    attention_config.inner_attention = ScaledDotProductAttention.Config()
    set_llama3_sharding_config(model_config, enable_sp=enable_sp)
    feed_forward_config = model_config.layers[0].feed_forward
    assert feed_forward_config is not None

    with torch.device("meta"):
        attention = attention_config.build()
        feed_forward = feed_forward_config.build()

    observations: dict[str, dict[str, object]] = {}

    def observe(name: str, value: torch.Tensor) -> torch.Tensor:
        observations[f"{name}.forward"] = _describe_tensor(value)

        def record_gradient(gradient: torch.Tensor) -> torch.Tensor:
            observations[f"{name}.cotangent"] = _describe_tensor(gradient)
            return gradient

        value.register_hook(record_gradient)
        return value

    original_attention_forward = attention.forward

    def observed_attention_forward(
        x_BLD: torch.Tensor,
        attention_masks,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return original_attention_forward(
            observe("attention.input_dst", x_BLD),
            attention_masks,
            positions,
        )

    attention.forward = observed_attention_forward

    original_qkv_forward = attention.qkv_linear.forward

    def observed_qkv_forward(x: torch.Tensor):
        xq, xk, xv = original_qkv_forward(x)
        return (
            observe("attention.xq", xq),
            observe("attention.xk", xk),
            observe("attention.xv", xv),
        )

    attention.qkv_linear.forward = observed_qkv_forward

    original_wo_forward = attention.wo.forward

    def observed_wo_forward(x: torch.Tensor) -> torch.Tensor:
        return original_wo_forward(observe("attention.head_out", x))

    attention.wo.forward = observed_wo_forward

    original_feed_forward = feed_forward.forward

    def observed_feed_forward(x: torch.Tensor) -> torch.Tensor:
        return original_feed_forward(observe("feed_forward.input_dst", x))

    feed_forward.forward = observed_feed_forward

    original_w2_forward = feed_forward.w2.forward

    def observed_w2_forward(x: torch.Tensor) -> torch.Tensor:
        return original_w2_forward(observe("feed_forward.gated_activation", x))

    feed_forward.w2.forward = observed_w2_forward

    attention.parallelize(parallel_dims)
    feed_forward.parallelize(parallel_dims)
    attention.to_empty(device=device)
    feed_forward.to_empty(device=device)
    with torch.no_grad():
        attention.init_states(buffer_device=device)
        feed_forward.init_states(buffer_device=device)
    attention.train()
    feed_forward.train()

    local_sequence = 2 if enable_sp else 4
    torch.manual_seed(100 + dist.get_rank())
    x = torch.randn(
        1,
        local_sequence,
        model_config.dim,
        device=device,
        requires_grad=True,
    )
    output = observe("attention.output_dst", attention(x, None, None))
    local_output = output.to_local() if isinstance(output, DTensor) else output
    if isinstance(local_output, AsyncCollectiveTensor):
        local_output = wait_tensor(local_output)
    local_output.float().sum().backward()

    feed_forward_input = torch.randn(
        1,
        local_sequence,
        model_config.dim,
        device=device,
        requires_grad=True,
    )
    feed_forward_output = observe(
        "feed_forward.output_dst",
        feed_forward(feed_forward_input),
    )
    local_feed_forward_output = (
        feed_forward_output.to_local()
        if isinstance(feed_forward_output, DTensor)
        else feed_forward_output
    )
    if isinstance(local_feed_forward_output, AsyncCollectiveTensor):
        local_feed_forward_output = wait_tensor(local_feed_forward_output)
    local_feed_forward_output.float().sum().backward()

    missing = [
        name
        for name in (
            "attention.input_dst.cotangent",
            "attention.xq.cotangent",
            "attention.xk.cotangent",
            "attention.xv.cotangent",
            "attention.head_out.cotangent",
            "attention.output_dst.cotangent",
            "feed_forward.input_dst.cotangent",
            "feed_forward.gated_activation.cotangent",
            "feed_forward.output_dst.cotangent",
        )
        if name not in observations
    ]
    assert not missing, missing

    replicate = ["R"]
    sequence_shard = ["S(1)"]
    feature_shard = ["S(2)"]
    partial = ["P(sum)"]
    expected_placements = {
        "attention.input_dst.forward": replicate,
        "attention.input_dst.cotangent": partial,
        "attention.xq.forward": feature_shard,
        "attention.xq.cotangent": feature_shard,
        "attention.xk.forward": feature_shard,
        "attention.xk.cotangent": feature_shard,
        "attention.xv.forward": feature_shard,
        "attention.xv.cotangent": feature_shard,
        "attention.head_out.forward": feature_shard,
        "attention.head_out.cotangent": feature_shard,
        "attention.output_dst.forward": sequence_shard if enable_sp else replicate,
        "attention.output_dst.cotangent": (sequence_shard if enable_sp else replicate),
        "feed_forward.input_dst.forward": replicate,
        "feed_forward.input_dst.cotangent": partial,
        "feed_forward.gated_activation.forward": feature_shard,
        "feed_forward.gated_activation.cotangent": feature_shard,
        "feed_forward.output_dst.forward": (sequence_shard if enable_sp else replicate),
        "feed_forward.output_dst.cotangent": (
            sequence_shard if enable_sp else replicate
        ),
    }
    actual_placements = {
        name: observation["placements"] for name, observation in observations.items()
    }
    assert actual_placements == expected_placements, (
        actual_placements,
        expected_placements,
    )

    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "enable_sp": enable_sp,
                    "placements": actual_placements,
                    "shapes": {
                        name: {
                            "global": observation["global_shape"],
                            "local": observation["local_shape"],
                            "local_kind": observation["local_kind"],
                        }
                        for name, observation in observations.items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
