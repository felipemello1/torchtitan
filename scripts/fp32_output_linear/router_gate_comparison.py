# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The router gate before this PR (RouterGateLinear) vs FP32OutputLinear, at Qwen3.5-35B-A3B's shape.

RouterGateLinear was deleted by this PR, so ``RouterGateLinearFunction`` below is a local copy of its
autograd Function: bf16 GEMM with an fp32 output, then fp32 backward GEMMs, both gradients rounded to
bf16. The trainer ran fp32 matmuls in BF16x9 on sm100+ (IEEE before), and so does this script.

Router 2048 -> 256 experts, 40 MoE layers, random data, 4k / 16k / 64k tokens. Per Function: GPU time
of fwd + bwd (profiler kernel sum) per layer and x40; grad_input and grad_weight error vs fp64; and
the % of gradient values bitwise equal to RouterGateLinear's (grad_weight compared after rounding to
bf16). FP32OutputLinear returns grad_weight in fp32 (weight.grad_dtype = fp32 here); routers use 3
pieces (higher_precision_bwd=True).

Example:
    python scripts/fp32_output_linear/router_gate_comparison.py
"""

import torch
from common import gpu_kernels, header, relative_error
from torch.autograd.function import once_differentiable

from torchtitan.models.common import linear

NUM_LAYERS = 40


class RouterGateLinearFunction(torch.autograd.Function):
    """Local copy of the deleted ``_RouterGateLinearFunction`` (bf16 operands, CUDA)."""

    @staticmethod
    def forward(ctx, input_TD, weight_ED):
        ctx.save_for_backward(input_TD, weight_ED)
        return torch.mm(input_TD, weight_ED.T, out_dtype=torch.float32)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TE):
        input_TD, weight_ED = ctx.saved_tensors
        grad_output_TE = grad_output_TE.float()
        grad_input_TD = torch.mm(grad_output_TE, weight_ED.float()).to(input_TD.dtype)
        grad_weight_ED = torch.mm(grad_output_TE.T, input_TD.float()).to(
            weight_ED.dtype
        )
        return grad_input_TD, grad_weight_ED


def main():
    print(header("Router gate: RouterGateLinear (before) vs FP32OutputLinear"))
    blackwell = torch.cuda.get_device_capability() >= (10, 0)
    # The trainer's global fp32 matmul mode, which RouterGateLinear's backward ran in.
    torch.backends.cuda.matmul.fp32_precision = "bfx9" if blackwell else "ieee"
    print(f"fp32 matmuls: {torch.backends.cuda.matmul.fp32_precision}")
    functions = {
        "RouterGateLinear (before)": lambda x, w: RouterGateLinearFunction.apply(x, w),
        "FP32OutputLinear, 2 pieces": lambda x, w: linear._FP32OutputLinearFunction.apply(
            x, w, False
        ),
        "FP32OutputLinear, 3 pieces": lambda x, w: linear._FP32OutputLinearFunction.apply(
            x, w, True
        ),
    }
    for num_tokens in (4096, 16384, 65536):
        torch.manual_seed(0)
        x = torch.randn(num_tokens, 2048, device="cuda").bfloat16()
        weight = (torch.randn(256, 2048, device="cuda") * 2048**-0.5).bfloat16()
        grad_output = torch.randn(num_tokens, 256, device="cuda")
        exact_dx = grad_output.double() @ weight.double()
        exact_dw = grad_output.double().T @ x.double()
        print(
            f"\n{num_tokens} tokens: x {tuple(x.shape)}, weight {tuple(weight.shape)}"
        )
        grads = {}
        for name, function in functions.items():
            x_leaf = x.clone().requires_grad_()
            weight_leaf = weight.clone().requires_grad_()
            if name != "RouterGateLinear (before)":
                weight_leaf.grad_dtype = torch.float32  # keeps an fp32 grad_weight

            def step(function=function, x_leaf=x_leaf, weight_leaf=weight_leaf):
                x_leaf.grad = weight_leaf.grad = None
                function(x_leaf, weight_leaf).backward(grad_output)

            _, gpu_ms, _ = gpu_kernels(step, calls=20)
            step()
            grads[name] = (x_leaf.grad, weight_leaf.grad)
            print(
                f"  {name:28s} GPU {gpu_ms:6.3f} ms/layer x{NUM_LAYERS} = "
                f"{gpu_ms * NUM_LAYERS:5.1f} ms   grad_input err "
                f"{relative_error(x_leaf.grad, exact_dx):.2e}  grad_weight "
                f"({x_leaf.grad.dtype}, {weight_leaf.grad.dtype}) err "
                f"{relative_error(weight_leaf.grad, exact_dw):.2e}",
                flush=True,
            )
        print(
            f"  {'bf16 rounding of the exact gradients':28s} {'':30s}grad_input err "
            f"{relative_error(exact_dx.bfloat16(), exact_dx):.2e}  grad_weight err "
            f"{relative_error(exact_dw.bfloat16(), exact_dw):.2e}"
        )
        old_dx, old_dw = grads["RouterGateLinear (before)"]
        for name in list(functions)[1:]:
            new_dx, new_dw = grads[name]
            print(
                f"  {name} vs RouterGateLinear, bitwise equal: grad_input "
                f"{(new_dx == old_dx).double().mean().item():.1%}, grad_weight (as bf16) "
                f"{(new_dw.bfloat16() == old_dw).double().mean().item():.1%}"
            )
        del x, weight, grad_output, exact_dx, exact_dw, grads
        torch.cuda.empty_cache()
    torch.backends.cuda.matmul.fp32_precision = "ieee"


if __name__ == "__main__":
    main()
