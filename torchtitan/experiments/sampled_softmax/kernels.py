# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernels for cross-entropy over bf16 logits ``[T, P]`` held in place.

The CE runs in two passes so a vocab-parallel all-gather of the per-row
statistics can sit between them:

1. ``row_stats``: per-row max, sum(exp(z - max)) and target logit, in fp32.
2. ``softmax_grad_``: overwrites the logits with
   ``(softmax(z + b) - onehot(label)) * valid`` in bf16, the gradient of the
   summed NLL. The logits buffer is reused, so the op never holds an fp32
   ``[T, P]`` tensor.

``b`` is an optional per-column fp32 bias (log importance weights), not added
to a row's own target column. Labels are column indices into the local logits, ``-1`` when the
target is not a local column. ``valid`` marks rows whose global label is not
``IGNORE_INDEX``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _row_stats_kernel(
    logits_ptr,
    bias_ptr,
    labels_ptr,
    max_ptr,
    sumexp_ptr,
    target_ptr,
    num_cols,
    stride_row,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride_row
    label = tl.load(labels_ptr + row)
    running_max = tl.full((BLOCK,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, num_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < num_cols
        z = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
            z += tl.where(cols == label, 0.0, bias)
        new_max = tl.maximum(running_max, z)
        # Lanes that have only seen -inf keep a zero sum (avoid inf - inf).
        scale = tl.where(new_max == float("-inf"), 0.0, tl.exp(running_max - new_max))
        running_sum = running_sum * scale + tl.where(mask, tl.exp(z - new_max), 0.0)
        running_max = new_max
    row_max = tl.max(running_max, axis=0)
    row_sum = tl.sum(running_sum * tl.exp(running_max - row_max), axis=0)
    target = tl.load(row_ptr + tl.maximum(label, 0)).to(tl.float32)
    target = tl.where(label >= 0, target, 0.0)
    tl.store(max_ptr + row, row_max)
    tl.store(sumexp_ptr + row, row_sum)
    tl.store(target_ptr + row, target)


@triton.jit
def _softmax_grad_kernel(
    logits_ptr,
    out_ptr,
    bias_ptr,
    labels_ptr,
    lse_ptr,
    valid_ptr,
    num_cols,
    stride_row,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride_row
    out_row_ptr = out_ptr + row * stride_row
    lse = tl.load(lse_ptr + row)
    label = tl.load(labels_ptr + row)
    valid = tl.load(valid_ptr + row)
    for start in range(0, num_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < num_cols
        z = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
            z += tl.where(cols == label, 0.0, bias)
        g = tl.exp(z - lse) - tl.where(cols == label, 1.0, 0.0)
        g = tl.where(valid, g, 0.0)
        tl.store(out_row_ptr + cols, g.to(out_ptr.dtype.element_ty), mask=mask)


def _block_and_warps(num_cols: int) -> tuple[int, int]:
    block = min(4096, triton.next_power_of_2(num_cols))
    return block, 8 if block >= 2048 else 4


def row_stats(
    logits: torch.Tensor, bias: torch.Tensor | None, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fp32 ``(row_max, row_sumexp, target_logit)``, each ``[T]``."""
    num_rows, num_cols = logits.shape
    assert logits.stride(1) == 1
    out = torch.empty(3, num_rows, device=logits.device, dtype=torch.float32)
    block, num_warps = _block_and_warps(num_cols)
    _row_stats_kernel[(num_rows,)](
        logits,
        bias if bias is not None else logits,
        labels,
        out[0],
        out[1],
        out[2],
        num_cols,
        logits.stride(0),
        HAS_BIAS=bias is not None,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out[0], out[1], out[2]


def softmax_grad_(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    labels: torch.Tensor,
    lse: torch.Tensor,
    valid: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write the summed-NLL gradient into ``out`` (default: over ``logits``) and return it."""
    num_rows, num_cols = logits.shape
    out = logits if out is None else out
    assert out.stride() == logits.stride()
    block, num_warps = _block_and_warps(num_cols)
    _softmax_grad_kernel[(num_rows,)](
        logits,
        out,
        bias if bias is not None else logits,
        labels,
        lse,
        valid,
        num_cols,
        logits.stride(0),
        HAS_BIAS=bias is not None,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out
