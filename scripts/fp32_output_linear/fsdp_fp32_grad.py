# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Does FSDP2 keep FP32OutputLinear's fp32 grad_weight end to end, eager and compiled? (2 GPUs)

FSDP2 with param_dtype=bf16 and reduce_dtype=fp32 (torchtitan's default), two backward calls with
gradient sync off for the first, like ChunkedLossWrapper's chunks. Toy shapes, the same data on every
rank, so FSDP's average equals the single-rank gradient. Prints grad_weight's error vs fp64 next to
the bf16 rounding of the exact gradient: ~1e-7 means fp32 was kept, ~1.6e-3 means it was rounded to
bf16 somewhere (torch without pytorch/pytorch#194434, or a compiled region: pytorch/pytorch#197381).

Example:
    torchrun --nproc-per-node=2 scripts/fp32_output_linear/fsdp_fp32_grad.py
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.models.common.linear import FP32OutputLinear

# name -> (tokens, out_features, higher_precision_bwd)
LAYERS = {
    "LM head (wide), 2 pieces": (64, 4096, False),
    "LM head (wide), 3 pieces": (64, 4096, True),
    "router (narrow), 3 pieces": (512, 16, True),
}


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    if rank == 0:
        print(
            f"# FSDP2 fp32 grad_weight check, {dist.get_world_size()} GPUs, "
            f"{torch.cuda.get_device_name()}, torch {torch.__version__}"
        )
    for mode in ("eager", "compiled"):
        for name, (num_tokens, out_features, hp) in LAYERS.items():
            torch.manual_seed(0)
            layer = FP32OutputLinear.Config(
                in_features=256, out_features=out_features, higher_precision_bwd=hp
            ).build()
            layer = layer.cuda()
            torch.nn.init.normal_(layer.weight, std=0.02)
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
            forward = torch.compile(layer) if mode == "compiled" else layer
            x = torch.randn(num_tokens, 256, device="cuda").bfloat16()
            grad_output = torch.randn(num_tokens, out_features, device="cuda")
            half = num_tokens // 2
            layer.set_requires_gradient_sync(False)
            forward(x[:half]).backward(grad_output[:half])
            layer.set_requires_gradient_sync(True)
            forward(x[half:]).backward(grad_output[half:])
            exact = grad_output.double().T @ x.double()
            grad = layer.weight.grad.full_tensor()
            if rank == 0:
                error = ((grad.double() - exact).norm() / exact.norm()).item()
                floor = (
                    (exact.bfloat16().double() - exact).norm() / exact.norm()
                ).item()
                print(
                    f"{mode:8s} {name:26s} .grad {grad.dtype}: error {error:.2e} "
                    f"(bf16 rounding of the exact gradient: {floor:.2e})",
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
