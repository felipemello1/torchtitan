# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Split-K for the LM head's grad_input: the TODO in ``_wide_backward``, implemented locally.

grad_input sums over out_features (152k for a Qwen3 LM head), and the GEMM's error grows with the
length of its sum. Split-K sums out_features in L-wide chunks, one bf16 GEMM each, added in fp32
with addmm(out=). Not in torchtitan: ``mm_fp32_split_k`` below is the local implementation.

1. Errors on real data (Qwen3-8B head, 2048 tokens, prepare_data.py), per L: grad_input's relative
   error vs fp64 before its bf16 rounding, and the % of the bf16 grad_input correctly rounded.
2. Backward time (eager, random data): the shipped Function vs the same backward with a split-K
   grad_input, interleaved rounds, median.

Example:
    python scripts/fp32_output_linear/split_k.py --cache-dir ~/.cache/fp32_output_linear
"""

import argparse

import torch
from common import (
    correctly_rounded_pct,
    DEFAULT_CACHE_DIR,
    head_grads,
    header,
    interleaved_median_ms,
    relative_error,
)
from torch.autograd.function import once_differentiable

from torchtitan.models.common import linear

CHUNK_WIDTHS = [
    None,
    16384,
    8192,
    4096,
]  # None: one GEMM over all of out_features (ships)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    print(header("Split-K for the LM head's grad_input"))
    print_errors(args.cache_dir)
    print_times(args.rounds)


def mm_fp32_split_k(
    a_MK: torch.Tensor, b_KN: torch.Tensor, chunk_width
) -> torch.Tensor:
    """``a @ b`` in fp32: one bf16 GEMM per ``chunk_width`` slice of K, added in fp32.

    Example: K = 151936, chunk_width = 8192 -> 18 GEMMs over 8192 and one over 4480.
    """
    if chunk_width is None:
        return torch.mm(a_MK, b_KN, out_dtype=torch.float32)
    a_slices = a_MK.split(chunk_width, dim=1)
    b_slices = b_KN.split(chunk_width)
    out_MN = torch.mm(a_slices[0], b_slices[0], out_dtype=torch.float32)
    for a_slice, b_slice in zip(a_slices[1:], b_slices[1:]):
        # out= adds inside the GEMM; a functional addmm would copy out_MN first.
        torch.addmm(out_MN, a_slice, b_slice, out_dtype=torch.float32, out=out_MN)
    return out_MN


def grad_input_fp32(grad_output_TO, weight_OD, higher_precision_bwd, chunk_width):
    """``_wide_backward``'s grad_input before its bf16 rounding, with split-K."""
    num_pieces = 3 if higher_precision_bwd else 2
    stacked_PTO = linear._split_into_bf16_pieces(
        grad_output_TO, higher_precision_bwd, dim=0
    )
    grad_input_PTD = mm_fp32_split_k(stacked_PTO, weight_OD, chunk_width)
    return grad_input_PTD.unflatten(0, (num_pieces, -1)).sum(dim=0)


def print_errors(cache_dir: str):
    x, weight, grad_output = head_grads(cache_dir)
    exact_dx = grad_output.double() @ weight.double()
    print(f"\nQwen3-8B head (real): x {tuple(x.shape)}, weight {tuple(weight.shape)}")
    print(f"  {'':22s} {'grad_input':>11s} {'rounded ok':>11s}")
    for hp in (False, True):
        for chunk_width in CHUNK_WIDTHS:
            dx = grad_input_fp32(grad_output, weight, hp, chunk_width)
            label = f"L={chunk_width}" if chunk_width else "one GEMM (ships)"
            print(
                f"  {2 + hp} pieces {label:18s} {relative_error(dx, exact_dx):9.2e} "
                f"{correctly_rounded_pct(dx.bfloat16(), exact_dx):10.2f}%",
                flush=True,
            )
    del x, weight, grad_output, exact_dx
    torch.cuda.empty_cache()


class SplitKFunction(torch.autograd.Function):
    """``_FP32OutputLinearFunction`` for a wide layer, with a split-K grad_input."""

    @staticmethod
    def forward(ctx, input_TD, weight_OD, higher_precision_bwd, chunk_width):
        ctx.save_for_backward(input_TD, weight_OD)
        ctx.higher_precision_bwd = higher_precision_bwd
        ctx.chunk_width = chunk_width
        return torch.mm(input_TD, weight_OD.T, out_dtype=torch.float32)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TO):
        input_TD, weight_OD = ctx.saved_tensors
        hp = ctx.higher_precision_bwd
        num_pieces = 3 if hp else 2
        # As _wide_backward, with mm_fp32_split_k for grad_input.
        stacked_PTO = linear._split_into_bf16_pieces(grad_output_TO, hp, dim=0)
        grad_input_PTD = mm_fp32_split_k(stacked_PTO, weight_OD, ctx.chunk_width)
        grad_input_PTD = grad_input_PTD.unflatten(0, (num_pieces, -1))
        grad_input_TD = grad_input_PTD.sum(dim=0).to(input_TD.dtype)
        grad_weight_OD = torch.mm(
            stacked_PTO.T, torch.cat([input_TD] * num_pieces), out_dtype=torch.float32
        )
        return grad_input_TD, grad_weight_OD, None, None


def print_times(rounds: int):
    print(
        f"\nQwen3-8B head chunk (2048 tokens, random data), eager backward ms, median of "
        f"{rounds} interleaved rounds of 10 calls; weight.grad_dtype = fp32"
    )
    torch.manual_seed(0)
    x = torch.randn(2048, 4096, device="cuda").bfloat16().requires_grad_()
    weight = (torch.randn(151936, 4096, device="cuda") * 4096**-0.5).bfloat16()
    weight.requires_grad_()
    weight.grad_dtype = torch.float32
    grad_output = torch.randn(2048, 151936, device="cuda") * 1e-3
    for hp in (False, True):
        variants = {
            "shipped": backward_fn(
                linear._FP32OutputLinearFunction, x, weight, grad_output, hp
            )
        }
        for chunk_width in CHUNK_WIDTHS:
            label = f"L={chunk_width}" if chunk_width else "local, one GEMM"
            variants[label] = backward_fn(
                SplitKFunction, x, weight, grad_output, hp, chunk_width
            )
        # The local copy of the shipped backward must match it bitwise.
        same = torch.equal(variants["shipped"]()[0], variants["local, one GEMM"]()[0])
        times = interleaved_median_ms(variants, rounds=rounds, calls=10)
        base = times["shipped"]
        print(
            f"  {2 + hp} pieces (local one-GEMM grad_input == shipped, bitwise: {same})"
        )
        for label, ms in times.items():
            print(
                f"    {label:16s} {ms:7.2f} ms  {ms / base:5.3f}x shipped", flush=True
            )


def backward_fn(function, x, weight, grad_output, *args):
    """One backward through ``function``'s graph, via torch.autograd.grad (no .grad accumulation)."""
    output = function.apply(x, weight, *args)
    return lambda: torch.autograd.grad(
        output, (x, weight), grad_output, retain_graph=True
    )


if __name__ == "__main__":
    main()
