# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Memory of FP32OutputLinear's backward temporaries, and the cost of a bf16 weight.grad.

1. Size of each tensor the backward allocates, from its shape, for the docstring's shapes:
   an LM head chunk (Qwen3-8B, 2048 tokens) and a router (64k tokens, 2048 -> 128).
2. Measured peak above what was allocated before the backward, shipped (compiled split) vs eager
   split, 2 and 3 pieces, with an fp32 or a bf16 weight.grad (a bf16 .grad makes autograd cast the
   fp32 grad_weight: torch without pytorch/pytorch#194434, or FSDP reducing in bf16).
3. Backward time with an fp32 vs a bf16 weight.grad (interleaved rounds, median).

Example:
    python scripts/fp32_output_linear/backward_memory.py
"""

import torch
from common import (
    eager_split,
    header,
    interleaved_median_ms,
    patched_split,
    peak_extra_gib,
)

from torchtitan.models.common import fp32_output_linear

# (label, tokens, in_features, out_features)
SHAPES = [
    ("Qwen3-8B head chunk 2048 tok", 2048, 4096, 151936),
    ("router 64k tok, 2048->128", 65536, 2048, 128),
]
BYTES = {torch.float32: 4, torch.bfloat16: 2}


def main():
    print(header("FP32OutputLinear backward: memory of each temporary"))
    print_sizes()
    print_peaks()
    print_grad_dtype_times()


def print_sizes():
    print(
        "\nsize of each tensor, from its shape (T tokens, O out_features, D in_features, P pieces)"
    )
    for label, num_tokens, in_features, out_features in SHAPES:
        wide = out_features > num_tokens
        print(f"{label} ({'wide' if wide else 'narrow'} layout):")
        for num_pieces in (2, 3):
            tensors = [
                (
                    "grad_output [T, O] fp32 (the dlogits)",
                    (num_tokens, out_features),
                    torch.float32,
                ),
                (
                    "eager split: each fp32 temporary [T, O]",
                    (num_tokens, out_features),
                    torch.float32,
                ),
                (
                    "eager split: each bf16 piece [T, O]",
                    (num_tokens, out_features),
                    torch.bfloat16,
                ),
                (
                    "stacked pieces [P*T, O] or [T, P*O] bf16",
                    (num_pieces * num_tokens, out_features),
                    torch.bfloat16,
                ),
                (
                    "cat([x] * P) [P*T, D] bf16",
                    (num_pieces * num_tokens, in_features),
                    torch.bfloat16,
                ),
                (
                    "cat([W] * P) [P*O, D] bf16",
                    (num_pieces * out_features, in_features),
                    torch.bfloat16,
                ),
                (
                    "grad_input per piece [P*T, D] fp32",
                    (num_pieces * num_tokens, in_features),
                    torch.float32,
                ),
                ("grad_input [T, D] bf16", (num_tokens, in_features), torch.bfloat16),
                ("grad_weight [O, D] fp32", (out_features, in_features), torch.float32),
                (
                    "autograd's cast to a bf16 .grad [O, D]",
                    (out_features, in_features),
                    torch.bfloat16,
                ),
            ]
            if num_pieces == 3:
                tensors = [t for t in tensors if "P" in t[0]]
            print(f"  {num_pieces} pieces:")
            for name, shape, dtype in tensors:
                if not wide and "grad_input per piece" in name:
                    continue  # the narrow layout's grad_input GEMM writes bf16 directly
                size = shape[0] * shape[1] * BYTES[dtype]
                print(f"    {name:44s} {format_bytes(size):>10s}")
        copied = "cat([x] * P)" if wide else "cat([W] * P)"
        print(f"  ships {copied}: the smaller copy")


def format_bytes(size: int) -> str:
    return f"{size / 2**30:.2f} GiB" if size >= 2**30 else f"{size / 2**20:.1f} MiB"


def print_peaks():
    print(
        "\nmeasured peak during one backward, above what was allocated before it (GiB)"
    )
    for label, num_tokens, in_features, out_features in SHAPES:
        x, weight, grad_output = random_inputs(num_tokens, in_features, out_features)
        for num_pieces in (2, 3):
            cells = []
            for split_name, split in (
                ("compiled split", fp32_output_linear._split_into_bf16_pieces),
                ("eager split", eager_split),
            ):
                for grad_dtype in (torch.float32, torch.bfloat16):
                    peak = peak_of_backward(
                        x, weight, grad_output, num_pieces, split, grad_dtype
                    )
                    dtype_name = "fp32" if grad_dtype == torch.float32 else "bf16"
                    cells.append(f"{split_name}, {dtype_name} .grad {peak:5.2f}")
            print(f"  {label:30s} {num_pieces} pieces: " + "  ".join(cells), flush=True)
        del x, weight, grad_output
        torch.cuda.empty_cache()


def random_inputs(num_tokens, in_features, out_features):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, in_features, device="cuda").bfloat16()
    weight = torch.randn(out_features, in_features, device="cuda")
    weight = (weight * in_features**-0.5).bfloat16()
    grad_output = torch.randn(num_tokens, out_features, device="cuda") * 1e-3
    return x, weight, grad_output


def peak_of_backward(x, weight, grad_output, num_pieces, split, grad_dtype):
    """Peak GiB of one backward (no .grad yet), past the forward. grad_output is freed by its owner."""
    x = x.clone().requires_grad_()
    weight = weight.clone().requires_grad_()
    weight.grad_dtype = grad_dtype
    with patched_split(split):
        for _ in range(2):  # the first call compiles
            output = fp32_output_linear._FP32OutputLinearFunction.apply(
                x, weight, num_pieces
            )
            x.grad = weight.grad = None
            peak = peak_extra_gib(lambda output=output: output.backward(grad_output))
    return peak


def print_grad_dtype_times():
    print(
        "\nfwd + bwd ms with an fp32 vs a bf16 weight.grad (median of 10 interleaved rounds of 10 calls)"
    )
    for label, num_tokens, in_features, out_features in SHAPES:
        x, weight, grad_output = random_inputs(num_tokens, in_features, out_features)
        for num_pieces in (2, 3):
            fns = {}
            for grad_dtype in (torch.float32, torch.bfloat16):
                x_leaf = x.clone().requires_grad_()
                weight_leaf = weight.clone().requires_grad_()
                weight_leaf.grad_dtype = grad_dtype
                fns[grad_dtype] = backward_fn(
                    x_leaf, weight_leaf, grad_output, num_pieces
                )
            times = interleaved_median_ms(
                {"fp32 .grad": fns[torch.float32], "bf16 .grad": fns[torch.bfloat16]},
                rounds=10,
                calls=10,
            )
            delta = times["bf16 .grad"] - times["fp32 .grad"]
            print(
                f"  {label:30s} {num_pieces} pieces: fp32 .grad {times['fp32 .grad']:7.2f}  "
                f"bf16 .grad {times['bf16 .grad']:7.2f} ms  (bf16 .grad {delta:+.2f} ms)",
                flush=True,
            )
        del x, weight, grad_output
        torch.cuda.empty_cache()


def backward_fn(x, weight, grad_output, num_pieces):
    """fwd + bwd into a fresh .grad. The forward is the same for both .grad dtypes."""

    def step():
        output = fp32_output_linear._FP32OutputLinearFunction.apply(
            x, weight, num_pieces
        )
        output.backward(grad_output)
        x.grad = weight.grad = None

    return step


if __name__ == "__main__":
    main()
