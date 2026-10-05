# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Four ways to compute an LM head's grad_weight from the P bf16 pieces of grad_output.

grad_weight = sum over pieces of piece.T @ x. Stacking the pieces along tokens runs one GEMM but
copies x P times (what ships); splitting runs one GEMM per piece. With ChunkedLossWrapper, every
chunk after the first adds its grad_weight into an fp32 weight.grad, which addmm(out=) can do inside
the GEMM:

    stack          mm(cat(pieces).T, cat([x] * P))                 one GEMM, copies x (ships)
    stack_addmm    addmm(weight.grad, cat(pieces).T, cat([x] * P), out=weight.grad)
    split          mm(p0.T, x) + mm(p1.T, x) (+ ...)                one GEMM per piece
    split_addmm    addmm(weight.grad, p.T, x, out=weight.grad), once per piece

"first" chunk: no weight.grad yet, the variant returns grad_weight (split_addmm adds the later pieces
into the first piece's result). "later" chunk: the variant adds into an existing fp32 weight.grad;
stack and split add with a separate kernel, as autograd does. Per variant: median time over CUDA-event
runs, memory allocated during the call (including a returned grad_weight), and the relative error vs
fp64 on 16 sampled rows. Random data.

Example:
    python scripts/fp32_output_linear/grad_weight_layouts.py
    python scripts/fp32_output_linear/grad_weight_layouts.py --models qwen3_8b --tokens 8192
"""

import argparse
import statistics

import torch
from common import header, peak_extra_gib

# (vocab, hidden) of real LM heads.
MODELS = {"qwen3_8b": (151936, 4096), "qwen3_5_27b": (248320, 5120)}


def stack(weight_grad, pieces_PTO, x_TD, num_pieces):
    return torch.mm(
        pieces_PTO.T, torch.cat([x_TD] * num_pieces), out_dtype=torch.float32
    )


def stack_addmm(weight_grad, pieces_PTO, x_TD, num_pieces):
    torch.addmm(
        weight_grad,
        pieces_PTO.T,
        torch.cat([x_TD] * num_pieces),
        out_dtype=torch.float32,
        out=weight_grad,
    )
    return weight_grad


def split(weight_grad, pieces_PTO, x_TD, num_pieces):
    pieces = pieces_PTO.chunk(num_pieces)
    chunk_grad = torch.mm(pieces[0].T, x_TD, out_dtype=torch.float32)
    for piece in pieces[1:]:
        chunk_grad += torch.mm(piece.T, x_TD, out_dtype=torch.float32)
    return chunk_grad


def split_addmm(weight_grad, pieces_PTO, x_TD, num_pieces):
    for piece in pieces_PTO.chunk(num_pieces):
        torch.addmm(
            weight_grad, piece.T, x_TD, out_dtype=torch.float32, out=weight_grad
        )
    return weight_grad


def split_mm_addmm(weight_grad, pieces_PTO, x_TD, num_pieces):
    """First chunk: mm for the first piece, then addmm(out=) adds the others into its result."""
    pieces = pieces_PTO.chunk(num_pieces)
    chunk_grad = torch.mm(pieces[0].T, x_TD, out_dtype=torch.float32)
    for piece in pieces[1:]:
        torch.addmm(chunk_grad, piece.T, x_TD, out_dtype=torch.float32, out=chunk_grad)
    return chunk_grad


FIRST_CHUNK = {"stack": stack, "split": split, "split_addmm": split_mm_addmm}
LATER_CHUNK = {
    "stack": stack,
    "stack_addmm": stack_addmm,
    "split": split,
    "split_addmm": split_addmm,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[2048, 4352, 8192, 16384, 32768, 65536]
    )
    parser.add_argument("--pieces", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    print(header("LM-head grad_weight: stack vs split, with and without addmm(out=)"))
    print(
        "model         tokens P  chunk   variant          time                 extra memory    error"
    )
    for model in args.models:
        for num_tokens in args.tokens:
            for num_pieces in args.pieces:
                try:
                    run(model, num_tokens, num_pieces, args)
                except torch.OutOfMemoryError:
                    print(
                        f"{model:12s} {num_tokens:7d} {num_pieces}  out of memory, skipped"
                    )
                torch.cuda.empty_cache()


def run(model, num_tokens, num_pieces, args):
    vocab, hidden = MODELS[model]
    torch.manual_seed(0)
    pieces_PTO = torch.empty(
        num_pieces * num_tokens, vocab, device="cuda", dtype=torch.bfloat16
    ).normal_()
    x_TD = torch.randn(num_tokens, hidden, device="cuda").bfloat16()
    rows = torch.randint(0, vocab, (16,)).tolist()
    for chunk, variants in (("first", FIRST_CHUNK), ("later", LATER_CHUNK)):
        # weight.grad holding the earlier chunks; the first chunk's variants never read it.
        weight_grad = torch.randn(vocab, hidden, device="cuda")
        results = {}
        for name, variant in variants.items():

            def call(variant=variant, name=name, chunk=chunk):
                result = variant(weight_grad, pieces_PTO, x_TD, num_pieces)
                if chunk == "later" and not name.endswith("addmm"):
                    weight_grad.add_(result)  # autograd's separate accumulation kernel

            results[name] = (
                median_ms(call, args.warmup, args.iters),
                peak_extra_gib(call),
                relative_error(variant, pieces_PTO, x_TD, num_pieces, rows),
            )
        stack_ms = results["stack"][0]
        for name, (ms, gib, error) in results.items():
            print(
                f"{model:12s} {num_tokens:7d} {num_pieces}  {chunk:6s}  {name:12s} "
                f"{ms:9.2f} ms ({ms / stack_ms:4.2f}x stack)  extra {gib:5.2f} GiB  "
                f"rel err {error:.1e}",
                flush=True,
            )
        del weight_grad


def median_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def relative_error(variant, pieces_PTO, x_TD, num_pieces, rows):
    """Error of ``variant``'s grad_weight (added into zeros) vs fp64, on ``rows`` of it."""
    weight_grad = torch.zeros(
        pieces_PTO.shape[1], x_TD.shape[1], device="cuda", dtype=torch.float32
    )
    result = variant(weight_grad, pieces_PTO, x_TD, num_pieces)
    x_cat = torch.cat([x_TD] * num_pieces).double()
    exact = pieces_PTO[:, rows].double().T @ x_cat
    return ((result[rows].double() - exact).norm() / exact.norm()).item()


if __name__ == "__main__":
    main()
