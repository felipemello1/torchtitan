# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Round or truncate each bf16 piece of grad_output: error on real data, and backward time.

Three ways to pick each piece. The pieces always sum to grad_output; only which bf16 is picked changes:
    truncate:   bits & 0xFFFF0000                                    (before review 3)
    half-away:  (bits + 0x8000) & 0xFFFF0000                         (round, ties away from 0: ships)
    RNE:        (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000    (round, ties to even)
Errors as in docstring_table.py. Times: fwd + bwd minus fwd (medians of interleaved rounds) of the
shipped Function with its split swapped, run eagerly and compiled (the shipped split is compiled).
Also prints the backward docstring's 0.1 example and where 3 pieces stop being exact.

Example:
    python scripts/fp32_output_linear/rounding.py --cache-dir ~/.cache/fp32_output_linear
"""

import argparse

import torch
from common import (
    correctly_rounded_pct,
    DEFAULT_CACHE_DIR,
    head_grads,
    header,
    interleaved_median_ms,
    patched_split,
    relative_error,
    router_grads,
)

from torchtitan.models.common import linear

BF16_BITS = -65536  # 0xFFFF0000: sign, exponent and the top 7 mantissa bits


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    print(header("Round vs truncate the bf16 pieces of grad_output"))
    check_half_away_is_shipped()
    print_split_examples()
    print_errors(args.cache_dir)
    print_times()


# ======================================== The three splits ========================================


def truncate(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.view(torch.int32) & BF16_BITS).view(torch.float32)


def half_away(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.view(torch.int32) + 0x8000) & BF16_BITS).view(torch.float32)


def round_to_nearest_even(tensor: torch.Tensor) -> torch.Tensor:
    bits = tensor.view(torch.int32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & BF16_BITS).view(torch.float32)


def split_with(pick, grad_output_TO, higher_precision_bwd, dim):
    """``_split_into_bf16_pieces_eager`` with ``pick`` choosing each piece's bf16 value."""
    hi = pick(grad_output_TO)
    rest = grad_output_TO - hi
    if higher_precision_bwd:
        mid = pick(rest)
        pieces = [hi, mid, rest - mid]
    else:
        pieces = [hi, rest]
    return torch.cat([piece.to(torch.bfloat16) for piece in pieces], dim=dim)


# One function per rounding, so each compiles into its own Dynamo cache entry.
def split_truncate(grad_output_TO, higher_precision_bwd, dim):
    return split_with(truncate, grad_output_TO, higher_precision_bwd, dim)


def split_half_away(grad_output_TO, higher_precision_bwd, dim):
    return split_with(half_away, grad_output_TO, higher_precision_bwd, dim)


def split_round_to_nearest_even(grad_output_TO, higher_precision_bwd, dim):
    return split_with(round_to_nearest_even, grad_output_TO, higher_precision_bwd, dim)


SPLITS = {
    "truncate": split_truncate,
    "half-away": split_half_away,
    "RNE": split_round_to_nearest_even,
}


def check_half_away_is_shipped():
    """The local half-away split must be the shipped one, bitwise (eager and compiled)."""
    magnitudes = torch.logspace(-30, 30, 4096, device="cuda")
    grad_output = torch.randn(256, 4096, device="cuda") * magnitudes
    same = all(
        torch.equal(
            split_half_away(grad_output, hp, dim).view(torch.int16),
            shipped(grad_output, hp, dim=dim).view(torch.int16),
        )
        for hp in (False, True)
        for dim in (0, 1)
        for shipped in (
            linear._split_into_bf16_pieces_eager,
            linear._split_into_bf16_pieces,
        )
    )
    print(
        f"half-away split == the shipped split (eager and custom op), bitwise: {same}"
    )


def print_split_examples():
    # The eager split: bitwise equal to the custom op (checked above), without recompiling it for
    # every tiny shape below.
    split = linear._split_into_bf16_pieces_eager
    print("\nthe shipped split of 0.1 (hi, mid, lo as fp64):")
    x = torch.tensor([[0.1]], device="cuda")
    for hp in (False, True):
        pieces = split(x, hp, dim=0).double().flatten()
        values = "  ".join(f"{v:+.9f}" for v in pieces.tolist())
        print(
            f"  {len(pieces)} pieces: {values}   sum - 0.1f = "
            f"{pieces.sum().item() - x.double().item():+.1e}"
        )

    # 4096 random fp32 values per exponent, from fp32's smallest up to 2^127 (larger ones round
    # up past bf16's largest value, 3.39e38, to inf).
    torch.manual_seed(0)
    exponents = torch.arange(-149, 127, device="cuda").repeat_interleave(4096)
    mantissas = 1 + torch.rand(exponents.numel(), device="cuda", dtype=torch.float64)
    x = (mantissas * torch.exp2(exponents.double())).float()
    x = x[x != 0]
    for hp in (False, True):
        pieces = split(x.view(-1, 1), hp, dim=1).double()
        relative = (pieces.sum(dim=1) - x.double()).abs() / x.double()
        normal = x >= 2**-110
        inexact = x[relative > 0]
        print(
            f"  {2 + hp} pieces, |x| >= 2^-110: max relative error "
            f"{relative[normal].max().item():.1e}; largest |x| not held exactly: "
            f"{inexact.max().item():.2e} (2^-110 = {2**-110:.2e})"
        )

    # A sum the 2-piece split gets wrong: [1 + 2^-8 + 2^-20, 1 + 2^-8] . [1, -1] = 2^-20.
    grad_output = torch.tensor([[1 + 2**-8 + 2**-20, 1 + 2**-8]], device="cuda")
    weight = torch.tensor([[1.0], [-1.0]], device="cuda", dtype=torch.bfloat16)
    for hp in (False, True):
        pieces = split(grad_output, hp, dim=0)
        total = torch.mm(pieces, weight, out_dtype=torch.float32).sum().item()
        print(
            f"  {2 + hp} pieces: [1+2^-8+2^-20, 1+2^-8] . [1, -1] = {total:.3e} "
            f"(exact: 2^-20 = {2**-20:.3e})"
        )


# ======================================== Errors ========================================


def print_errors(cache_dir: str):
    print("\nrelative error vs fp64 (real data), grad_input before its bf16 rounding")
    for name, (x, weight, grad_output) in [
        ("LM head", head_grads(cache_dir)),
        ("router", router_grads(cache_dir)),
    ]:
        exact_dx = grad_output.double() @ weight.double()
        exact_dw = grad_output.double().T @ x.double()
        print(f"{name}: x {tuple(x.shape)}, weight {tuple(weight.shape)}")
        print(
            f"  {'':22s} {'grad_input':>11s} {'rounded ok':>11s} {'grad_weight':>12s}"
        )
        for hp in (False, True):
            for split_name, split in SPLITS.items():
                with patched_split(split):
                    dx_fp32, dx, dw = shipped_gradients(x, weight, grad_output, hp)
                print(
                    f"  {2 + hp} pieces {split_name:12s} "
                    f"{relative_error(dx_fp32, exact_dx):11.2e} "
                    f"{correctly_rounded_pct(dx, exact_dx):10.3f}% "
                    f"{relative_error(dw, exact_dw):12.2e}",
                    flush=True,
                )
        del x, weight, grad_output, exact_dx, exact_dw
        torch.cuda.empty_cache()


def shipped_gradients(x, weight, grad_output, higher_precision_bwd):
    """(grad_input before its bf16 rounding, grad_input, fp32 grad_weight) of the shipped backward
    with whatever split is patched in."""
    x_leaf = x.clone().requires_grad_()
    weight_leaf = weight.clone().requires_grad_()
    weight_leaf.grad_dtype = torch.float32
    linear._FP32OutputLinearFunction.apply(
        x_leaf, weight_leaf, higher_precision_bwd
    ).backward(grad_output)
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


def print_times():
    print(
        "\nbackward ms (fwd + bwd minus fwd), random data, default grad_dtype; % vs truncate"
    )
    shapes = [
        ("router 64k tok, 2048->128", 65536, 2048, 128),
        ("Qwen3-8B head chunk 2048 tok", 2048, 4096, 151936),
    ]
    for label, num_tokens, in_features, out_features in shapes:
        torch.manual_seed(0)
        x = torch.randn(num_tokens, in_features, device="cuda").bfloat16()
        weight = torch.randn(out_features, in_features, device="cuda")
        weight = (weight * in_features**-0.5).bfloat16()
        grad_output = torch.randn(num_tokens, out_features, device="cuda") * 1e-3
        for hp in (False, True):
            for mode in ("split eager", "split compiled"):
                torch._dynamo.reset()
                splits = {
                    name: compiled_like_shipped(split)
                    if mode == "split compiled"
                    else split
                    for name, split in SPLITS.items()
                }
                times = backward_times(splits, x, weight, grad_output, hp)
                base = times["truncate"]
                cells = "  ".join(
                    f"{name} {ms:6.2f} ({ms / base - 1:+5.1%})"
                    for name, ms in times.items()
                )
                print(f"{label:30s} {2 + hp} pieces {mode:15s} {cells}", flush=True)
        del x, weight, grad_output
        torch.cuda.empty_cache()


def compiled_like_shipped(split):
    """``split`` compiled the way the shipped custom op compiles it (out_features static)."""
    compiled = torch.compile(split)

    def run(grad_output_TO, higher_precision_bwd, dim):
        torch._dynamo.mark_static(grad_output_TO, 1)
        return compiled(grad_output_TO, higher_precision_bwd, dim)

    return run


def backward_times(splits, x, weight, grad_output, higher_precision_bwd) -> dict:
    """{split name: backward ms}: median fwd + bwd with that split, minus the median fwd, over
    interleaved rounds."""
    x = x.clone().requires_grad_()
    weight = weight.clone().requires_grad_()

    def forward():
        with torch.no_grad():
            linear._FP32OutputLinearFunction.apply(x, weight, higher_precision_bwd)

    def forward_backward(split):
        with patched_split(split):
            output = linear._FP32OutputLinearFunction.apply(
                x, weight, higher_precision_bwd
            )
            output.backward(grad_output)
        x.grad = weight.grad = None

    fns = {
        name: lambda split=split: forward_backward(split)
        for name, split in splits.items()
    }
    times = interleaved_median_ms({"fwd": forward, **fns}, rounds=10, calls=10)
    return {name: times[name] - times["fwd"] for name in splits}


if __name__ == "__main__":
    main()
