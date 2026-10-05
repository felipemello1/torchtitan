# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers shared by the scripts in this folder: real tensors from prepare_data.py, errors, timing.

Example:
    from common import head_grads, relative_error
    x, weight, grad_output = head_grads("~/.cache/fp32_output_linear")
"""

import contextlib
import datetime
import os
import random
import statistics
import subprocess
from collections import defaultdict

import torch
import torch.nn.functional as F

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common import fp32_output_linear

# Compile the split, as the models with FP32OutputLinear do (their local_compile_regions list it).
apply_local_compile(["fp32_output_split"])
# The split without its local_compile wrapper: always eager.
eager_split = fp32_output_linear._split_into_bf16_pieces_impl.__wrapped__

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/fp32_output_linear")

# File in the cache dir -> (Hugging Face repo, 1024-token sequences, also save the lm_head weight).
CACHE_FILES = {
    "qwen3_8b_head.pt": ("Qwen/Qwen3-8B", 2, True),
    "qwen3_1p7b_hidden.pt": ("Qwen/Qwen3-1.7B", 16, False),
    "qwen3_5_27b_head.pt": ("Qwen/Qwen3.5-27B", 2, True),
}


def header(title: str) -> str:
    """First line of every output: what ran, on which GPU, torch and torchtitan commit."""
    commit = subprocess.run(
        [
            "git",
            "-C",
            os.path.dirname(fp32_output_linear.__file__),
            "rev-parse",
            "--short",
            "HEAD",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return (
        f"# {title}\n# {torch.cuda.get_device_name()}, torch {torch.__version__}, "
        f"torchtitan {commit or 'not a git checkout'}, {datetime.datetime.now():%Y-%m-%d %H:%M}"
    )


# ======================================== Real data ========================================


def load_cache(cache_dir: str, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(hidden states [T, D] bf16, lm_head weight [V, D] bf16 or None) saved by prepare_data.py."""
    saved = torch.load(
        os.path.join(os.path.expanduser(cache_dir), name), weights_only=True
    )
    weight = saved["weight"]
    return saved["hidden"].cuda(), None if weight is None else weight.cuda()


def sample_tokens(
    hidden_TD: torch.Tensor, weight_VD: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens sampled from the fp64 softmax of the logits (seed 0), as a generator samples them,
    and their exact fp64 logprobs."""
    torch.manual_seed(0)
    with torch.no_grad():
        logprobs_TV = torch.log_softmax(
            hidden_TD.double() @ weight_VD.double().T, dim=-1
        )
        tokens_T = torch.multinomial(logprobs_TV.exp().float(), 1).squeeze(1)
        return tokens_T, logprobs_TV.gather(1, tokens_T[:, None]).squeeze(1)


def head_grads(
    cache_dir: str, name: str = "qwen3_8b_head.pt"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LM head (input, weight, fp32 grad_output): grad_output is the gradient of the summed
    cross-entropy of the sampled tokens w.r.t. the fp32 logits."""
    hidden_TD, weight_VD = load_cache(cache_dir, name)
    tokens_T, _ = sample_tokens(hidden_TD, weight_VD)
    logits_TV = torch.mm(hidden_TD, weight_VD.T, out_dtype=torch.float32)
    logits_TV.requires_grad_()
    F.cross_entropy(logits_TV, tokens_T, reduction="sum").backward()
    return hidden_TD, weight_VD, logits_TV.grad


def router_grads(
    cache_dir: str, num_experts: int = 128, top_k: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Router (input, gate weight, fp32 grad_output): real Qwen3-1.7B hidden states, a random
    [num_experts, 2048] gate, softmax top-k renormalized, times random expert outputs."""
    hidden_TD, _ = load_cache(cache_dir, "qwen3_1p7b_hidden.pt")
    torch.manual_seed(0)
    weight_ED = torch.randn(num_experts, hidden_TD.shape[1], device="cuda") * 0.02
    weight_ED = weight_ED.bfloat16()
    logits_TE = torch.mm(hidden_TD, weight_ED.T, out_dtype=torch.float32)
    logits_TE.requires_grad_()
    top_scores_TK, _ = logits_TE.softmax(-1).topk(top_k, dim=-1)
    top_scores_TK = top_scores_TK / top_scores_TK.sum(-1, keepdim=True)
    (top_scores_TK * torch.randn_like(top_scores_TK)).sum().backward()
    return hidden_TD, weight_ED, logits_TE.grad


# ======================================== Errors ========================================


def relative_error(actual: torch.Tensor, exact: torch.Tensor) -> float:
    """||actual - exact|| / ||exact||, with ``exact`` in fp64."""
    return ((actual.double() - exact).norm() / exact.norm()).item()


def correctly_rounded_pct(actual_bf16: torch.Tensor, exact: torch.Tensor) -> float:
    """% of ``actual_bf16``'s values equal to the exact value rounded to bf16."""
    return 100 * (actual_bf16 == exact.bfloat16()).double().mean().item()


# ======================================== Timing and memory ========================================


def mean_ms(fn, warmup: int = 5, iters: int = 30) -> float:
    """Mean ms per call, from CUDA events around ``iters`` back-to-back calls."""
    for _ in range(warmup):
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


def interleaved_median_ms(
    fns: dict, rounds: int = 30, warmup: int = 3, calls: int = 1
) -> dict:
    """Median ms per call of each fn. Each round times ``calls`` back-to-back calls of every fn, in
    a shuffled order, so clock drift and the previous fn's load hit all fns alike.

    Example:
        interleaved_median_ms({"a": f, "b": g}, rounds=10, calls=10) -> {"a": 7.1, "b": 10.6}
    """
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    order = list(fns)
    shuffle = random.Random(0).shuffle
    times = defaultdict(list)
    for _ in range(rounds):
        shuffle(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(calls):
                fns[name]()
            end.record()
            torch.cuda.synchronize()
            times[name].append(start.elapsed_time(end) / calls)
    return {name: statistics.median(times[name]) for name in fns}


def gpu_kernels(fn, calls: int = 5) -> tuple[int, float, dict]:
    """CUDA kernels one call of ``fn`` launches, from the profiler.

    Returns (kernels per call, GPU ms per call, {kernel name: launches per call}).
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    events = [
        e for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    launches = defaultdict(int)
    for event in events:
        launches[event.name] += 1
    gpu_ms = sum(e.device_time for e in events) / calls / 1e3
    per_call = {name: count // calls for name, count in launches.items()}
    return len(events) // calls, gpu_ms, per_call


def peak_extra_gib(fn) -> float:
    """Peak memory allocated during ``fn()``, above what was allocated before it, in GiB."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 2**30


@contextlib.contextmanager
def patched_split(split):
    """Make FP32OutputLinear's backward call ``split`` instead of the custom op (compiled split).

    Example:
        with patched_split(eager_split):  # 5-8 eager kernels
            out.backward(grad_output)
    """
    shipped = fp32_output_linear._split_into_bf16_pieces
    fp32_output_linear._split_into_bf16_pieces = split
    try:
        yield
    finally:
        fp32_output_linear._split_into_bf16_pieces = shipped
