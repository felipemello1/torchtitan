# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The compiled split of grad_output vs the eager one: kernels, time, dynamic shapes, and one chunk.

1. Kernels and GPU time (profiler) of the split alone: the shipped custom op (compiled) vs
   ``_split_into_bf16_pieces_eager``, for an LM head chunk and a router, 2 and 3 pieces.
2. Split time with static or symbolic dims, compiled fresh each time:
       static T, static O      the first call (what ships, before the token count changes)
       symbolic T, static O    what ships once the token count changes (mark_static on O)
       static T, symbolic O    mark_dynamic on out_features
       symbolic T and O        torch.compile(dynamic=True)
3. One Qwen3-8B LM head + loss chunk (FP32OutputLinear forward, CE, backward into an fp32
   weight.grad, 2 pieces), compiled vs eager split, with the CE eager or compiled. Interleaved
   rounds, median. TORCH_COMPILE_DISABLE=1 runs the shipped code with the eager split, like the
   "eager split" rows here.

Example:
    python scripts/fp32_output_linear/split_compile.py
    TORCH_COMPILE_DISABLE=1 python scripts/fp32_output_linear/split_compile.py --sections chunk
"""

import argparse
import os
import statistics
from collections import defaultdict

import torch
from common import gpu_kernels, header, interleaved_median_ms, mean_ms, patched_split

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.models.common import linear
from torchtitan.models.common.linear import FP32OutputLinear

VOCAB, HIDDEN = 151936, 4096
# (label, tokens, out_features, dim the pieces stack along)
SPLIT_SHAPES = [
    ("LM head [2048, 151936]", 2048, VOCAB, 0),
    ("router [65536, 128]", 65536, 128, 1),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["kernels", "dynamic", "chunk"],
        choices=["kernels", "dynamic", "chunk"],
    )
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()

    print(header("Compiled vs eager split of grad_output"))
    if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        print("TORCH_COMPILE_DISABLE=1: every split and CE below runs eagerly")
    if "kernels" in args.sections:
        print_kernels()
    if "dynamic" in args.sections:
        print_dynamic()
    if "chunk" in args.sections:
        print_chunk(args.rounds)


def print_kernels():
    print("\nsplit + cat alone: CUDA kernels per call, GPU ms per call (profiler)")
    for label, num_tokens, out_features, dim in SPLIT_SHAPES:
        torch.manual_seed(0)
        grad_output = torch.randn(num_tokens, out_features, device="cuda") * 1e-3
        for hp in (False, True):
            torch._dynamo.reset()
            rows = {}
            for name, split in (
                ("eager", linear._split_into_bf16_pieces_eager),
                ("compiled", linear._split_into_bf16_pieces),
            ):
                rows[name] = gpu_kernels(
                    lambda split=split: split(grad_output, hp, dim=dim)
                )
            same = torch.equal(
                linear._split_into_bf16_pieces_eager(grad_output, hp, dim=dim).view(
                    torch.int16
                ),
                linear._split_into_bf16_pieces(grad_output, hp, dim=dim).view(
                    torch.int16
                ),
            )
            (eager_n, eager_ms, eager_names), (
                compiled_n,
                compiled_ms,
                _,
            ) = rows.values()
            print(
                f"  {label:24s} {2 + hp} pieces: eager {eager_n:2d} kernels {eager_ms:6.3f} ms -> "
                f"compiled {compiled_n} kernel(s) {compiled_ms:6.3f} ms "
                f"({compiled_ms / eager_ms - 1:+.0%}); bitwise equal: {same}"
            )
            for name, count in eager_names.items():
                print(f"      eager: {count} x {name[:80]}")
        del grad_output
        torch.cuda.empty_cache()


def print_dynamic(num_passes: int = 3):
    print(
        "\nsplit time (ms) with static or symbolic dims: each setting compiled fresh and timed "
        f"alone, median of {num_passes} passes"
    )
    for label, num_tokens, out_features, dim in SPLIT_SHAPES:
        for hp in (False, True):
            torch.manual_seed(0)
            grad_output = torch.randn(num_tokens, out_features, device="cuda") * 1e-3
            expected = linear._split_into_bf16_pieces_eager(grad_output, hp, dim=dim)
            times = defaultdict(list)
            for _ in range(num_passes):
                for setting in DYNAMIC_SETTINGS:
                    split = compiled_split(setting, grad_output, hp, dim, expected)
                    times[setting].append(mean_ms(split))
            medians = {k: statistics.median(v) for k, v in times.items()}
            base = medians["static T, static O"]
            cells = "  ".join(
                f"{k} {v:6.3f} ({v / base - 1:+4.0%})" for k, v in medians.items()
            )
            print(f"  {label:24s} {2 + hp} pieces: {cells}", flush=True)
            del grad_output, expected
            torch.cuda.empty_cache()


DYNAMIC_SETTINGS = [
    "static T, static O",
    "symbolic T, static O",
    "static T, symbolic O",
    "symbolic T and O",
]


def compiled_split(setting, grad_output, hp, dim, expected):
    """A split of ``grad_output``, compiled from a clean Dynamo state with the dims ``setting``
    names symbolic. mark_dynamic raises if compiling specializes such a dim."""
    torch._dynamo.reset()
    dynamic = True if setting == "symbolic T and O" else None
    compiled = torch.compile(linear._split_into_bf16_pieces_eager, dynamic=dynamic)
    tensor = (
        grad_output.clone()
    )  # marks live on the tensor object: one object per setting
    if setting == "symbolic T, static O":
        torch._dynamo.mark_dynamic(tensor, 0)
    elif setting == "static T, symbolic O":
        torch._dynamo.mark_dynamic(tensor, 1)
    if setting != "static T, symbolic O" and setting != "symbolic T and O":
        torch._dynamo.mark_static(tensor, 1)
    assert torch.equal(
        compiled(tensor, hp, dim).view(torch.int16), expected.view(torch.int16)
    )
    return lambda: compiled(tensor, hp, dim)


def print_chunk(rounds: int):
    print(
        "\nQwen3-8B LM head + loss chunk: FP32OutputLinear forward, CE, backward into an fp32 "
        f"weight.grad, 2 pieces; ms per chunk, median of {rounds} interleaved rounds"
    )
    torch.manual_seed(0)
    lm_head = FP32OutputLinear.Config(in_features=HIDDEN, out_features=VOCAB).build()
    lm_head = lm_head.to(device="cuda", dtype=torch.bfloat16)
    torch.nn.init.normal_(lm_head.weight, std=0.02)
    lm_head.weight.grad_dtype = torch.float32  # what FSDP sets with an fp32 reduce
    compiled_ce = torch.compile(cross_entropy_loss)
    for num_tokens in (2048, 8192):
        x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        x.requires_grad_()
        labels = torch.randint(0, VOCAB, (num_tokens,), device="cuda")

        def chunk(ce, split):
            with patched_split(split):
                loss = ce(lm_head(x), labels) / num_tokens
                loss.backward()

        variants = {}
        for ce_name, ce in (
            ("CE eager", cross_entropy_loss),
            ("CE compiled", compiled_ce),
        ):
            for split_name, split in (
                ("compiled split", linear._split_into_bf16_pieces),
                ("eager split", linear._split_into_bf16_pieces_eager),
            ):
                variants[f"{ce_name}, {split_name}"] = lambda ce=ce, split=split: chunk(
                    ce, split
                )
        same = all(
            bitwise_same_grads(
                [
                    variants[f"{ce_name}, compiled split"],
                    variants[f"{ce_name}, eager split"],
                ],
                x,
                lm_head,
            )
            for ce_name in ("CE eager", "CE compiled")
        )
        times = interleaved_median_ms(variants, rounds=rounds)
        print(
            f"  T={num_tokens}: compiled and eager split give bitwise equal gradients: {same}"
        )
        for ce_name in ("CE eager", "CE compiled"):
            eager = times[f"{ce_name}, eager split"]
            compiled = times[f"{ce_name}, compiled split"]
            print(
                f"    {ce_name:12s} eager split {eager:7.2f} -> compiled split {compiled:7.2f} ms "
                f"({compiled / eager - 1:+.1%}, saves {eager - compiled:.2f} ms)",
                flush=True,
            )
        del x, labels
        lm_head.weight.grad = None
        torch.cuda.empty_cache()


def bitwise_same_grads(fns, x, lm_head) -> bool:
    """Do all ``fns`` give bitwise equal grad_input and grad_weight (from a fresh .grad)?"""
    grads = []
    for fn in fns:
        x.grad = lm_head.weight.grad = None
        fn()
        grads.append((x.grad.clone(), lm_head.weight.grad.clone()))
    x.grad = lm_head.weight.grad = None
    return all(
        torch.equal(dx, grads[0][0]) and torch.equal(dw, grads[0][1])
        for dx, dw in grads
    )


if __name__ == "__main__":
    main()
