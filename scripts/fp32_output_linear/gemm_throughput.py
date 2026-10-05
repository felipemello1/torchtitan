# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GEMM throughput by precision: why bf16 pieces beat fp32 matmuls (the backward docstring's ~25x).

Per shape: TFLOPS of a bf16 GEMM (bf16 or fp32 output, fp32 accumulation), an fp32 GEMM (IEEE, TF32,
and BF16x9 on sm100+), and how many times faster bf16 is than fp32 IEEE. Operands are ready in their
dtype (no upcast in the timing). Random data.

Example:
    python scripts/fp32_output_linear/gemm_throughput.py
"""

import torch
from common import header, mean_ms

# (label, M, K, N): output [M, N] = [M, K] @ [K, N]
SHAPES = [
    ("square 8192", 8192, 8192, 8192),
    ("Qwen3-8B head grad_input", 2048, 151936, 4096),
    ("Qwen3-8B head grad_weight", 151936, 2048, 4096),
    ("router grad_input", 65536, 128, 2048),
]


def main():
    print(header("GEMM throughput by precision"))
    blackwell = torch.cuda.get_device_capability() >= (10, 0)
    for label, m, k, n in SHAPES:
        torch.manual_seed(0)
        a_bf16 = torch.randn(m, k, device="cuda").bfloat16()
        b_bf16 = torch.randn(k, n, device="cuda").bfloat16()
        a_fp32, b_fp32 = a_bf16.float(), b_bf16.float()
        variants = {
            "bf16, bf16 out": ("ieee", lambda: torch.mm(a_bf16, b_bf16)),
            "bf16, fp32 out": (
                "ieee",
                lambda: torch.mm(a_bf16, b_bf16, out_dtype=torch.float32),
            ),
            "fp32 IEEE": ("ieee", lambda: torch.mm(a_fp32, b_fp32)),
            "fp32 TF32": ("tf32", lambda: torch.mm(a_fp32, b_fp32)),
        }
        if blackwell:
            variants["fp32 BF16x9"] = ("bfx9", lambda: torch.mm(a_fp32, b_fp32))
        tflops = {}
        for name, (precision, fn) in variants.items():
            torch.backends.cuda.matmul.fp32_precision = precision
            tflops[name] = 2 * m * k * n / mean_ms(fn, iters=20) / 1e9
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        cells = "  ".join(f"{name} {value:7.1f}" for name, value in tflops.items())
        ratio = tflops["bf16, fp32 out"] / tflops["fp32 IEEE"]
        print(
            f"{label:26s} [{m}, {k}] @ [{k}, {n}]  TFLOPS: {cells}  "
            f"bf16 (fp32 out) / fp32 IEEE = {ratio:.1f}x",
            flush=True,
        )
        del a_bf16, b_bf16, a_fp32, b_fp32
        torch.cuda.empty_cache()
    if not blackwell:
        print("fp32 BF16x9 skipped: cuBLAS runs it on sm100+ only")


if __name__ == "__main__":
    main()
