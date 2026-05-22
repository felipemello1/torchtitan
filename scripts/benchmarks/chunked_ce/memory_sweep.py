# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Peak-memory sweep for chunked CE across L.

Standalone shape-sensitivity microbench. NOT a current_titan comparator
(per RFC.md §8.3 caveat). Use to sanity-check that the chunked path
holds bounded peak memory as L grows.

Usage:
    python memory_sweep.py --vocab 128000 --hidden 2048 --num-chunks 8 \\
        --seq-lens 8192 16384 32768 65536 131072 \\
        --json > /tmp/memory_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Row:
    path: str  # "chunked_sft" | "chunked_grpo" | "chunked_gspo" | "dense"
    seq_len: int
    peak_mib: float
    elapsed_ms: float
    loss: float


def make_inputs(B, L, H, V, device, dtype):
    torch.manual_seed(42)
    hidden = torch.randn(B, L, H, device=device, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, V, (B, L), device=device)
    lm_head = nn.Linear(H, V, bias=False, device=device, dtype=dtype)
    return hidden, labels, lm_head


def selected_token_logprobs(logits, labels, ignore_index=-100):
    losses = F.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(labels)
    return -losses


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_mib(device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def bench_dense(hidden, labels, lm_head, device):
    reset_peak(device)
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if start:
        start.record()

    logits = lm_head(hidden)  # [B, L, V]
    loss = F.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=-100,
    )
    loss.backward()

    if end:
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
    else:
        ms = 0.0
    return float(loss.item()), peak_mib(device), ms


def bench_chunked_sft(hidden, labels, lm_head, num_chunks, device):
    reset_peak(device)
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if start:
        start.record()

    B, L, H = hidden.shape
    chunk_size = (L + num_chunks - 1) // num_chunks
    total_loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
    grad_buffer = torch.zeros_like(hidden)
    next_start = 0
    for ci in range(num_chunks):
        cs = ci * chunk_size
        ce = min(cs + chunk_size, L)
        if cs >= ce:
            break
        h_chunk = hidden[:, cs:ce, :].contiguous().detach().requires_grad_(True)
        logits = lm_head(h_chunk)
        chunk_loss = F.cross_entropy(
            logits.flatten(0, 1).float(),
            labels[:, cs:ce].flatten(0, 1),
            reduction="sum",
            ignore_index=-100,
        )
        chunk_loss.backward()
        total_loss = total_loss + chunk_loss.detach()
        grad_buffer[:, cs:ce, :] = h_chunk.grad
        next_start = ce
    # Outer backward (would normally go through _ChunkedLossWithParamGrads):
    hidden.backward(grad_buffer)

    if end:
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
    else:
        ms = 0.0
    return float(total_loss.item()), peak_mib(device), ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=128000)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[8192, 16384, 32768, 65536, 131072]
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)

    rows = []
    for L in args.seq_lens:
        # Dense reference. Skip if it OOMs.
        try:
            hidden, labels, lm_head = make_inputs(
                args.batch, L, args.hidden, args.vocab, device, dtype
            )
            loss, mib, ms = bench_dense(hidden, labels, lm_head, device)
            rows.append(Row("dense", L, mib, ms, loss))
        except torch.cuda.OutOfMemoryError:
            rows.append(Row("dense_OOM", L, -1, -1, -1))

        # Chunked SFT.
        hidden, labels, lm_head = make_inputs(
            args.batch, L, args.hidden, args.vocab, device, dtype
        )
        loss, mib, ms = bench_chunked_sft(
            hidden, labels, lm_head, args.num_chunks, device
        )
        rows.append(Row("chunked_sft", L, mib, ms, loss))

    if args.json:
        json.dump([asdict(r) for r in rows], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"{'path':<16} {'L':>8} {'peak_MiB':>10} {'ms':>8} {'loss':>14}")
        for r in rows:
            print(
                f"{r.path:<16} {r.seq_len:>8} {r.peak_mib:>10.1f} {r.elapsed_ms:>8.1f} {r.loss:>14.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
