# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The table in FP32OutputLinear's backward docstring: errors on real data, then backward times.

Errors are relative to fp64 on the same fp32 grad_output (data from prepare_data.py):
    LM head: Qwen3-8B lm_head and 2048 hidden states; grad_output = CE gradient on sampled tokens.
    router:  16k Qwen3-1.7B hidden states, a random [128, 2048] gate, softmax top-8.
    grad_input:  before its bf16 rounding (the backward's GEMMs, with an fp32 output).
    rounded ok:  % of the returned bf16 grad_input equal to the exact value rounded to bf16.
    grad_weight: as returned, in fp32 (weight.grad_dtype = fp32).

Times are fwd + bwd minus fwd, random data, as multiples of a bf16 Linear's, median of --runs:
    eager:       the Function as it ships (its split is compiled, behind a custom op).
    eager split: the same, with ``_split_into_bf16_pieces_eager`` (what TORCH_COMPILE_DISABLE=1 runs).
    compiled:    the Function inside torch.compile.

Example:
    python scripts/fp32_output_linear/docstring_table.py --cache-dir ~/.cache/fp32_output_linear
    python scripts/fp32_output_linear/docstring_table.py --runs 1 --skip-errors
    python scripts/fp32_output_linear/docstring_table.py --runs 0  # errors only
"""

import argparse
import os
import statistics
from collections import defaultdict

# Compile from scratch in every run, so the cache doesn't hide compile choices between runs.
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from common import (  # noqa: E402
    correctly_rounded_pct,
    DEFAULT_CACHE_DIR,
    head_grads,
    header,
    mean_ms,
    patched_split,
    relative_error,
    router_grads,
)

from torchtitan.models.common import linear  # noqa: E402

# (label, tokens, in_features, out_features)
SHAPES = [
    ("router 64k tok, 2048->128", 65536, 2048, 128),
    ("Qwen3-8B head chunk 2048 tok", 2048, 4096, 151936),
]
MODES = ["eager", "eager split", "compiled"]
VARIANTS = ["bf16 Linear", "2 pieces", "3 pieces", "fp32 IEEE"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    print(header("FP32OutputLinear backward docstring table"))
    if not args.skip_errors:
        print_errors(args.cache_dir)
    if args.runs > 0:
        print_times(args.runs)


# ======================================== Errors ========================================


def print_errors(cache_dir: str):
    print("\nrelative error vs fp64 (real data)")
    print(f"{'':20s} {'grad_input':>11s} {'rounded ok':>11s} {'grad_weight':>12s}")
    dx_errors = {}
    for name, (x, weight, grad_output) in [
        ("LM head", head_grads(cache_dir)),
        ("router", router_grads(cache_dir)),
    ]:
        exact_dx = grad_output.double() @ weight.double()
        exact_dw = grad_output.double().T @ x.double()
        print(
            f"{name}: x {tuple(x.shape)}, weight {tuple(weight.shape)}; bf16 rounding of the "
            f"exact grad_input alone: {relative_error(exact_dx.bfloat16(), exact_dx):.2e}"
        )
        for variant in VARIANTS:
            dx_fp32, dx_returned, dw = gradients(variant, x, weight, grad_output)
            dx_errors[(name, variant)] = relative_error(dx_fp32, exact_dx)
            print(
                f"  {variant:18s} {dx_errors[(name, variant)]:11.2e} "
                f"{correctly_rounded_pct(dx_returned, exact_dx):10.2f}% "
                f"{relative_error(dw, exact_dw):12.2e}",
                flush=True,
            )
        del x, weight, grad_output, exact_dx, exact_dw
        torch.cuda.empty_cache()
    for name in ("LM head", "router"):
        ratio = dx_errors[(name, "2 pieces")] / dx_errors[(name, "3 pieces")]
        print(f"{name}: grad_input error, 2 pieces / 3 pieces = {ratio:.1f}x")


def gradients(variant: str, x: torch.Tensor, weight: torch.Tensor, grad_output):
    """(grad_input before its bf16 rounding, grad_input as returned, grad_weight) of ``variant``."""
    if variant == "bf16 Linear":
        grad_output = grad_output.bfloat16()
        dx_fp32 = torch.mm(grad_output, weight, out_dtype=torch.float32)
        dw = torch.mm(grad_output.T, x, out_dtype=torch.float32)
        return dx_fp32, torch.mm(grad_output, weight), dw
    if variant == "fp32 IEEE":
        dx_fp32 = grad_output @ weight.float()
        return dx_fp32, dx_fp32.bfloat16(), grad_output.T @ x.float()

    # The shipped Function, then its grad_input GEMMs again with an fp32 output.
    higher_precision_bwd = variant == "3 pieces"
    x_leaf = x.clone().requires_grad_()
    weight_leaf = weight.clone().requires_grad_()
    weight_leaf.grad_dtype = torch.float32
    output = linear._FP32OutputLinearFunction.apply(
        x_leaf, weight_leaf, higher_precision_bwd
    )
    output.backward(grad_output)
    num_pieces = 3 if higher_precision_bwd else 2
    if weight.shape[0] > x.shape[0]:  # _wide_backward
        stacked_PTO = linear._split_into_bf16_pieces(
            grad_output, higher_precision_bwd, dim=0
        )
        dx_PTD = torch.mm(stacked_PTO, weight, out_dtype=torch.float32)
        dx_fp32 = dx_PTD.unflatten(0, (num_pieces, x.shape[0])).sum(dim=0)
    else:  # _narrow_backward
        stacked_TPO = linear._split_into_bf16_pieces(
            grad_output, higher_precision_bwd, dim=1
        )
        weights = torch.cat([weight] * num_pieces)
        dx_fp32 = torch.mm(stacked_TPO, weights, out_dtype=torch.float32)
    return dx_fp32, x_leaf.grad, weight_leaf.grad


# ======================================== Times ========================================


def print_times(num_runs: int):
    runs = defaultdict(list)
    for run in range(num_runs):
        print(f"\nrun {run + 1}: backward ms (multiple of the bf16 Linear's)")
        for key, ms in one_run().items():
            runs[key].append(ms)

    print(f"\nmedian of {num_runs} runs, as multiples of the bf16 Linear's backward:")
    print(f"{'':30s} {'':12s}" + "".join(f"{name:>13s}" for name in VARIANTS))
    for label, *_ in SHAPES:
        for mode in MODES:
            base = statistics.median(runs[(label, mode, "bf16 Linear")])
            cells = "".join(
                f"{statistics.median(runs[(label, mode, name)]) / base:12.1f}x"
                for name in VARIANTS
            )
            print(f"{label:30s} {mode:12s}{cells}   (bf16 Linear {base:.2f} ms)")
    print("\n3 pieces / 2 pieces backward time (medians):")
    for label, *_ in SHAPES:
        ratios = "  ".join(
            f"{mode} {statistics.median(runs[(label, mode, '3 pieces')]) / statistics.median(runs[(label, mode, '2 pieces')]):.2f}x"
            for mode in MODES
        )
        print(f"  {label:30s} {ratios}")


def one_run() -> dict:
    """{(shape label, mode, variant): backward ms} for every shape, mode and variant."""
    times = {}
    for label, num_tokens, in_features, out_features in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(num_tokens, in_features, device="cuda").bfloat16()
        weight = torch.randn(out_features, in_features, device="cuda")
        weight = (weight * in_features**-0.5).bfloat16()
        grad_output = torch.randn(num_tokens, out_features, device="cuda") * 1e-3
        variants = {
            "bf16 Linear": (F.linear, x, weight, grad_output.bfloat16()),
            "2 pieces": (two_pieces, x, weight, grad_output),
            "3 pieces": (three_pieces, x, weight, grad_output),
            "fp32 IEEE": (F.linear, x.float(), weight.float(), grad_output),
        }
        for mode in MODES:
            split = (
                linear._split_into_bf16_pieces_eager
                if mode == "eager split"
                else linear._split_into_bf16_pieces
            )
            with patched_split(split):
                for name, (fn, a, b, g) in variants.items():
                    times[(label, mode, name)] = backward_ms(
                        fn, a, b, g, compiled=mode == "compiled"
                    )
            base = times[(label, mode, "bf16 Linear")]
            print(
                f"  {label:30s} {mode:12s} "
                + "  ".join(
                    f"{name} {times[(label, mode, name)]:6.2f} "
                    f"({times[(label, mode, name)] / base:.1f}x)"
                    for name in VARIANTS
                ),
                flush=True,
            )
        del x, weight, grad_output
        torch.cuda.empty_cache()
    return times


def two_pieces(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return linear._FP32OutputLinearFunction.apply(x, weight, False)


def three_pieces(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return linear._FP32OutputLinearFunction.apply(x, weight, True)


def backward_ms(fn, x, weight, grad_output, compiled: bool) -> float:
    """fwd + bwd minus fwd. Leaves keep the default grad_dtype: a bf16 weight gets a bf16 .grad."""
    # Compile from a clean state: no shape seen before, so every dim stays static.
    torch._dynamo.reset()
    fn = torch.compile(fn) if compiled else fn
    x = x.clone().requires_grad_()
    weight = weight.clone().requires_grad_()

    def forward():
        with torch.no_grad():
            fn(x, weight)

    def forward_backward():
        fn(x, weight).backward(grad_output)
        x.grad = weight.grad = None

    return mean_ms(forward_backward) - mean_ms(forward)


if __name__ == "__main__":
    main()
