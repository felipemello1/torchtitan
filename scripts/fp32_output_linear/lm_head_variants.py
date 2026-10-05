# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The PR's "Controlled results": ways to get fp32 logits from a real LM head, inference and training.

Real lm_head weight and 2048 hidden states (prepare_data.py; Qwen3.5-27B for the PR's tables).
Tokens are sampled from the fp64 softmax, as a generator samples them. "logprob |err|" is the mean
|error| of the sampled tokens' logprobs vs fp64; gradient errors are relative to exact fp64 gradients.

1. Inference, LM head alone: us per call for 1 and 16 sequences (one token each).
2. Training, one 2048-token ChunkedLossWrapper chunk (forward + CE(sum) + backward): ms per chunk,
   eager, then compiled (torch.compile(fullgraph=True) over the head and the CE) for the
   fp32-output rows.

Rows that no longer exist in torchtitan are local copies, labeled as such:
    upcast + {IEEE, BF16x9, TF32} GEMM:  CastLinear (deleted), F.linear(x.float(), w.float())
    fp32 bwd (router):  RouterGateLinear's backward (deleted), see router_gate_comparison.py;
                        BF16x9 on sm100+ like the trainer, IEEE before
    bf16 bwd (megatron):  grad_output rounded to bf16, then bf16 GEMMs
BF16x9 rows need sm100+ and are skipped elsewhere.

Example:
    python scripts/fp32_output_linear/lm_head_variants.py --model qwen3_5_27b
    python scripts/fp32_output_linear/lm_head_variants.py --model qwen3_8b
"""

import argparse

import torch
import torch.nn.functional as F
from common import DEFAULT_CACHE_DIR, header, load_cache, relative_error, sample_tokens
from router_gate_comparison import RouterGateLinearFunction
from torch.autograd.function import once_differentiable

from torchtitan.models.common import fp32_output_linear

CACHES = {"qwen3_5_27b": "qwen3_5_27b_head.pt", "qwen3_8b": "qwen3_8b_head.pt"}


class Bf16BackwardFunction(torch.autograd.Function):
    """bf16 GEMM with an fp32 output; backward rounds grad_output to bf16 (as Megatron does)."""

    @staticmethod
    def forward(ctx, input_TD, weight_VD):
        ctx.save_for_backward(input_TD, weight_VD)
        return torch.mm(input_TD, weight_VD.T, out_dtype=torch.float32)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TV):
        input_TD, weight_VD = ctx.saved_tensors
        grad_output_TV = grad_output_TV.to(input_TD.dtype)
        return grad_output_TV @ weight_VD, grad_output_TV.T @ input_TD


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", default="qwen3_5_27b", choices=CACHES)
    args = parser.parse_args()

    print(header(f"LM head variants, real {args.model} weight and hidden states"))
    hidden, weight = load_cache(args.cache_dir, CACHES[args.model])
    tokens, exact_logprobs = sample_tokens(hidden, weight)
    print(f"hidden {tuple(hidden.shape)}, weight {tuple(weight.shape)}")
    blackwell = torch.cuda.get_device_capability() >= (10, 0)
    # The precision fp32 matmuls ran in for the router rows: the trainer's BF16x9 on sm100+.
    router_precision = "bfx9" if blackwell else "ieee"
    print(f"fp32 matmuls in the 'fp32 bwd (router)' rows: {router_precision}")

    upcast = lambda h, w: F.linear(h.float(), w.float())  # noqa: E731
    shipped = lambda h, w: fp32_output_linear._FP32OutputLinearFunction.apply(
        h, w, 2
    )  # noqa: E731
    # name -> (fp32 matmul precision, function of (hidden, weight) -> logits)
    variants = {
        "upcast + IEEE GEMM (CastLinear)": ("ieee", upcast),
        "upcast + BF16x9 GEMM (CastLinear)": ("bfx9", upcast),
        "upcast + TF32 GEMM (CastLinear)": ("tf32", upcast),
        "bf16 GEMM, fp32 out, fp32 bwd (router)": (
            router_precision,
            RouterGateLinearFunction.apply,
        ),
        "bf16 GEMM, fp32 out, bf16 bwd (megatron)": (
            "ieee",
            Bf16BackwardFunction.apply,
        ),
        "bf16 GEMM, fp32 out, hi+lo bwd (this PR)": ("ieee", shipped),
        "bf16 GEMM, bf16 out (bf16 head)": ("ieee", F.linear),
    }
    if not blackwell:
        del variants["upcast + BF16x9 GEMM (CastLinear)"]
        print("upcast + BF16x9 GEMM skipped: cuBLAS runs BF16x9 on sm100+ only")

    print_inference(variants, hidden, weight, tokens, exact_logprobs)
    print_training(variants, hidden, weight, tokens, exact_logprobs)
    torch.backends.cuda.matmul.fp32_precision = "ieee"


def logprob_error(logits_TV, tokens_T, exact_logprobs_T) -> float:
    logprobs_T = torch.log_softmax(logits_TV.float(), dim=-1).gather(
        1, tokens_T[:, None]
    )
    return (logprobs_T.squeeze(1).double() - exact_logprobs_T).abs().mean().item()


def time_ms(fn, iters=10):
    """Mean ms per call after 2 warmup calls (the method of the PR's tables)."""
    for _ in range(2):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def print_inference(variants, hidden, weight, tokens, exact_logprobs):
    print("\ninference, LM head alone, per call (logprob |err| over all 2048 tokens)")
    print(f"{'':42s} {'1 seq (us)':>11s} {'16 seq (us)':>12s} {'logprob |err|':>14s}")
    for name, (precision, fn) in variants.items():
        if "router" in name or "megatron" in name:
            continue  # same forward as "this PR"
        torch.backends.cuda.matmul.fp32_precision = precision
        with torch.no_grad():
            micros = [
                1000 * time_ms(lambda m=m: fn(hidden[:m], weight), iters=50)
                for m in (1, 16)
            ]
            error = logprob_error(fn(hidden, weight), tokens, exact_logprobs)
        label = name.replace(" (CastLinear)", "")
        print(
            f"{label:42s} {micros[0]:11.0f} {micros[1]:12.0f} {error:14.1e}", flush=True
        )


def print_training(variants, hidden, weight, tokens, exact_logprobs):
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    hidden_ref = hidden.double().requires_grad_()
    weight_ref = weight.double().requires_grad_()
    F.cross_entropy(
        F.linear(hidden_ref, weight_ref), tokens, reduction="sum"
    ).backward()
    exact_dh, exact_dw = hidden_ref.grad, weight_ref.grad
    del hidden_ref, weight_ref
    torch.cuda.empty_cache()

    print(
        "\ntraining, one 2048-token chunk (forward + CE + backward); bf16 rounding of the exact "
        f"gradients alone: dh {relative_error(exact_dh.bfloat16(), exact_dh):.2e}, "
        f"dW {relative_error(exact_dw.bfloat16(), exact_dw):.2e}"
    )
    print(f"{'':62s} {'ms':>6s} {'logprob |err|':>14s} {'dh err':>9s} {'dW err':>9s}")
    rows = list(variants.items()) + [
        (
            "bf16 GEMM, fp32 out, hi+lo bwd (this PR), fp32 .grad",
            variants["bf16 GEMM, fp32 out, hi+lo bwd (this PR)"],
        )
    ]
    for name, (precision, fn) in rows:
        fp32_grad = name.endswith("fp32 .grad")
        for compiled in (False, True):
            fp32_out_row = "fp32 out" in name and "fp32 .grad" not in name
            if compiled and not fp32_out_row:
                continue
            torch.backends.cuda.matmul.fp32_precision = precision
            step = training_step(fn, hidden, weight, tokens, compiled, fp32_grad)
            logits, dh, dw = step()
            ms = time_ms(step)
            label = f"(compile) {name}" if compiled else name
            print(
                f"{label:62s} {ms:6.1f} {logprob_error(logits, tokens, exact_logprobs):14.1e} "
                f"{relative_error(dh, exact_dh):9.2e} {relative_error(dw, exact_dw):9.2e}",
                flush=True,
            )
            del logits, dh, dw
            torch.cuda.empty_cache()


def training_step(fn, hidden, weight, tokens, compiled, fp32_grad):
    """One chunk: fresh leaves (as the PR's tables timed it), forward, CE(sum), backward.

    Returns a function that runs the chunk and returns (logits, dh, dW). A bf16 weight gets a bf16
    .grad unless ``fp32_grad`` (what FSDP sets with an fp32 reduce, torch with
    pytorch/pytorch#194434). Compiled code rounds an fp32 grad_weight to bf16 regardless
    (pytorch/pytorch#197381).
    """

    def loss_fn(h, w):
        logits = fn(h, w)
        return F.cross_entropy(logits.float(), tokens, reduction="sum"), logits.detach()

    if compiled:
        torch._dynamo.reset()
        loss_fn = torch.compile(loss_fn, fullgraph=True)

    def step():
        h = hidden.clone().requires_grad_()
        w = weight.clone().requires_grad_()
        if fp32_grad:
            w.grad_dtype = torch.float32
        loss, logits = loss_fn(h, w)
        loss.backward()
        return logits, h.grad, w.grad

    return step


if __name__ == "__main__":
    main()
