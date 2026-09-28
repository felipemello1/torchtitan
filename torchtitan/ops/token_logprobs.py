# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LM head fused with the sampled-token logprob and entropy of its fp32 logits.

Used by ``ChunkedLossWrapper`` for losses that only read ``log p(label)`` (and entropy
as a metric), e.g. the RL policy-gradient losses. Compared with ``Fp32OutputLinear``
followed by ``compute_logprobs``, it reads the ``[T, V]`` logits twice instead of making ~15
passes over ``[T, V]`` tensors, and its backward uses 2 GEMMs instead of hi + lo's 4 at about
the same gradient error.
"""

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import triton
import triton.language as tl

# Shape suffix legend:
#   T = num tokens, D = model dimension, V = local vocab size

IGNORE_INDEX = -100

# dlogits = g * (one_hot - softmax) is written as fp16 (one_hot - softmax) * 2^14, where g is
# the per-token gradient of the logprob. Factoring g out keeps each row's values in [-2^14, 2^14]
# whatever the loss scale; the 2^14 keeps probabilities down to ~4e-9 as fp16 normals.
_DLOGITS_SCALE = 2.0**14


@triton.jit
def _partial_softmax_stats_kernel(
    logits_ptr,
    labels_ptr,
    max_ptr,
    sum_exp_ptr,
    sum_exp_logit_ptr,
    label_logit_ptr,
    vocab_size,
    vocab_start,
    stride_row,
    BLOCK: tl.constexpr,
):
    """One pass over a row: max, sum(exp(x - max)), sum(exp(x - max) * x), and x[label]."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride_row
    # Per-lane online softmax; lanes are combined once after the loop.
    max_lanes = tl.full([BLOCK], -1e30, tl.float32)
    sum_exp_lanes = tl.zeros([BLOCK], tl.float32)
    sum_exp_logit_lanes = tl.zeros([BLOCK], tl.float32)
    for start in range(0, vocab_size, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        logits = tl.load(
            row_ptr + offsets, mask=offsets < vocab_size, other=float("-inf")
        )
        new_max = tl.maximum(max_lanes, logits)
        rescale = tl.exp(max_lanes - new_max)
        exp_logits = tl.exp(logits - new_max)
        sum_exp_lanes = sum_exp_lanes * rescale + exp_logits
        # exp(-inf) * -inf is NaN; masked and padded logits contribute 0.
        sum_exp_logit_lanes = sum_exp_logit_lanes * rescale + tl.where(
            exp_logits > 0, exp_logits * logits, 0.0
        )
        max_lanes = new_max
    row_max = tl.max(max_lanes, 0)
    lane_rescale = tl.exp(max_lanes - row_max)
    tl.store(max_ptr + row, row_max)
    tl.store(sum_exp_ptr + row, tl.sum(sum_exp_lanes * lane_rescale, 0))
    tl.store(sum_exp_logit_ptr + row, tl.sum(sum_exp_logit_lanes * lane_rescale, 0))
    # 0 when the label is ignored (-100) or lives on another vocab shard, so shards can be summed.
    local_label = tl.load(labels_ptr + row) - vocab_start
    is_local = (local_label >= 0) & (local_label < vocab_size)
    label_logit = tl.load(row_ptr + tl.where(is_local, local_label, 0))
    tl.store(label_logit_ptr + row, tl.where(is_local, label_logit, 0.0))


@triton.jit
def _dlogits_kernel(
    logits_ptr,
    labels_ptr,
    logsumexp_ptr,
    dlogits_ptr,
    vocab_size,
    vocab_start,
    stride_row,
    scale,
    IGNORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """dlogits[row] = (one_hot(label) - softmax(logits[row])) * scale, 0 for ignored labels."""
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    logits = tl.load(
        logits_ptr + row * stride_row + offsets, mask=mask, other=float("-inf")
    )
    logsumexp = tl.load(logsumexp_ptr + row)
    label = tl.load(labels_ptr + row)
    one_hot = (offsets + vocab_start == label).to(tl.float32)
    dlogits = tl.where(
        label != IGNORE, (one_hot - tl.exp(logits - logsumexp)) * scale, 0.0
    )
    tl.store(
        dlogits_ptr + row * vocab_size + offsets,
        dlogits.to(dlogits_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _scaled_add_kernel(
    out_ptr,
    src_ptr,
    acc_ptr,
    scale_ptr,
    numel,
    ACCUMULATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """out = src * scale (+ acc if ACCUMULATE), cast to out's dtype; scale is read on device."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    value = tl.load(src_ptr + offsets, mask=mask) * tl.load(scale_ptr)
    if ACCUMULATE:
        value += tl.load(acc_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, value.to(out_ptr.dtype.element_ty), mask=mask)


def _softmax_stats(
    logits_TV: torch.Tensor,
    labels_T: torch.Tensor,
    vocab_start: int,
    vocab_parallel_group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(logsumexp, label logprob, entropy)``, each ``[T]``, over the full vocab."""
    num_tokens, vocab_size = logits_TV.shape
    row_max, sum_exp, sum_exp_logit, label_logit = torch.empty(
        4, num_tokens, device=logits_TV.device, dtype=torch.float32
    )
    _partial_softmax_stats_kernel[(num_tokens,)](
        logits_TV,
        labels_T,
        row_max,
        sum_exp,
        sum_exp_logit,
        label_logit,
        vocab_size,
        vocab_start,
        logits_TV.stride(0),
        BLOCK=2048,
        num_warps=8,
    )
    if vocab_parallel_group is not None:
        # Combine the shards: rescale each shard's sums to the global max, then add.
        global_max = funcol.all_reduce(row_max, "max", group=vocab_parallel_group)
        rescale = torch.exp(row_max - global_max)
        sums = torch.stack([sum_exp * rescale, sum_exp_logit * rescale, label_logit])
        sum_exp, sum_exp_logit, label_logit = funcol.all_reduce(
            sums, "sum", group=vocab_parallel_group
        )
        row_max = global_max
    logsumexp = row_max + torch.log(sum_exp)
    logprobs = torch.where(labels_T != IGNORE_INDEX, label_logit - logsumexp, 0.0)
    # H(p) = logsumexp - sum(p * logits)
    entropy = logsumexp - sum_exp_logit / sum_exp
    return logsumexp, logprobs, entropy


class TokenLogprobsGradState:
    """Per-microbatch state shared by the chunks' backwards.

    The first chunk's forward fills ``weight_fp16`` from the weight it receives (unsharded
    under FSDP), cast once for the fp16 backward GEMMs. ``grad_weight`` accumulates the chunks'
    weight gradients in fp32; the last chunk returns the sum to autograd in the weight's gradient
    dtype (bf16, or fp32 with ``Tensor.grad_dtype``), so it is rounded at most once per microbatch.

    Not a dataclass: FSDP's forward-input cast copies dataclass arguments (casting their
    tensors to the param dtype), which would give each chunk its own accumulator.
    """

    def __init__(self) -> None:
        self.weight_fp16: torch.Tensor | None = None
        self.grad_weight: torch.Tensor | None = None


class TokenLogprobs(torch.autograd.Function):
    """``(hidden [T, D], weight [V, D], labels [T]) -> (logprobs [T], entropy [T])``.

    Forward: ``logits = hidden @ weight.T`` as a bf16 GEMM with fp32 output (the same op as
    ``Fp32OutputLinear``), then one Triton pass for logsumexp, the label logit and entropy.

    Backward, with ``g = dloss/dlogprobs`` and ``M = (one_hot - softmax)``:

        dlogits     = g * M                                 never materialized in fp32
        grad_hidden = g * (M @ W)                           one fp16 GEMM, fp32 output
        grad_weight = M.T @ (g * hidden)                    one fp16 GEMM, fp32 output

    ``M`` is written once as fp16, which keeps 10 mantissa bits to bf16's 7. Rounding it costs
    ~8x less gradient error than rounding dlogits to bf16, at the same GEMM speed; with 2^-11
    relative error per element, the gradients land within ~1% of exact fp64 gradients rounded to
    bf16 (hi + lo, which ``Fp32OutputLinear`` uses, lands within ~0.5% with twice the GEMMs).
    ``g`` must stay out of the fp16 operands: a loss normalized by 1e5 tokens makes ``g * M``
    fp16 subnormal. So ``g`` scales the fp32 output rows of ``grad_hidden``, and scales
    ``hidden`` for ``grad_weight`` after normalizing it by ``max|g|``.

    Entropy is a metric: it is non-differentiable and must not receive a gradient.
    With ``vocab_parallel_group``, ``weight`` is this rank's vocab shard starting at
    ``vocab_start``; the forward all-reduces ``[T]`` statistics, and ``grad_hidden`` is this
    shard's partial sum (reduced by the hidden states' TP redistribute, as for the lm_head).
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_TD: torch.Tensor,
        weight_VD: torch.Tensor,
        labels_T: torch.Tensor,
        grad_state: TokenLogprobsGradState | None,
        return_grad_weight: bool,
        vocab_start: int,
        vocab_parallel_group: dist.ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if grad_state is not None and grad_state.weight_fp16 is None:
            grad_state.weight_fp16 = weight_VD.detach().to(torch.float16)
        logits_TV = torch.mm(hidden_TD, weight_VD.T, out_dtype=torch.float32)
        logsumexp_T, logprobs_T, entropy_T = _softmax_stats(
            logits_TV, labels_T, vocab_start, vocab_parallel_group
        )
        ctx.save_for_backward(hidden_TD, labels_T, logits_TV, logsumexp_T)
        ctx.grad_state = grad_state
        ctx.return_grad_weight = return_grad_weight
        ctx.vocab_start = vocab_start
        # fp32 when FSDP's unsharded weight accumulates fp32 gradients (Tensor.grad_dtype, which
        # only leaf tensors have).
        grad_dtype = weight_VD.grad_dtype if weight_VD.is_leaf else None
        ctx.grad_weight_dtype = grad_dtype or weight_VD.dtype
        ctx.mark_non_differentiable(entropy_T)
        ctx.set_materialize_grads(False)
        return logprobs_T, entropy_T

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_logprobs_T: torch.Tensor, grad_entropy_T: None):
        assert grad_entropy_T is None, "entropy is a metric and has no gradient"
        hidden_TD, labels_T, logits_TV, logsumexp_T = ctx.saved_tensors
        grad_state = ctx.grad_state
        assert grad_state is not None, "backward needs a TokenLogprobsGradState"
        num_tokens, vocab_size = logits_TV.shape

        dlogits_TV = torch.empty(
            num_tokens, vocab_size, device=logits_TV.device, dtype=torch.float16
        )
        block = 4096
        _dlogits_kernel[(num_tokens, triton.cdiv(vocab_size, block))](
            logits_TV,
            labels_T,
            logsumexp_T,
            dlogits_TV,
            vocab_size,
            ctx.vocab_start,
            logits_TV.stride(0),
            _DLOGITS_SCALE,
            IGNORE=IGNORE_INDEX,
            BLOCK=block,
            num_warps=8,
        )
        del logits_TV
        grad_T = grad_logprobs_T.float()

        grad_hidden_TD = torch.mm(
            dlogits_TV, grad_state.weight_fp16, out_dtype=torch.float32
        )
        grad_hidden_TD = grad_hidden_TD * (grad_T / _DLOGITS_SCALE)[:, None]
        grad_hidden_TD = grad_hidden_TD.to(hidden_TD.dtype)
        # A frozen lm_head (e.g. LoRA on the decoder) skips the dW GEMM and its accumulator.
        if not ctx.needs_input_grad[1]:
            return grad_hidden_TD, None, None, None, None, None, None

        # Normalize g so that g * hidden stays in fp16's range; undo it on the fp32 output.
        grad_max = grad_T.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        scaled_hidden_TD = (hidden_TD.float() * (grad_T / grad_max)[:, None]).to(
            torch.float16
        )
        grad_weight_VD = torch.mm(
            dlogits_TV.T, scaled_hidden_TD, out_dtype=torch.float32
        )
        # The first chunk's scaled gradient becomes the fp32 accumulator. The last chunk adds its
        # own and, unless the weight takes fp32 gradients, casts the sum in the same pass.
        accumulated = grad_state.grad_weight
        if ctx.return_grad_weight and ctx.grad_weight_dtype != torch.float32:
            out = torch.empty_like(grad_weight_VD, dtype=ctx.grad_weight_dtype)
        else:
            out = grad_state.grad_weight = (
                grad_weight_VD if accumulated is None else accumulated
            )
        numel = out.numel()
        add_block = 8192
        _scaled_add_kernel[(triton.cdiv(numel, add_block),)](
            out,
            grad_weight_VD,
            grad_weight_VD if accumulated is None else accumulated,
            grad_max / _DLOGITS_SCALE,
            numel,
            ACCUMULATE=accumulated is not None,
            BLOCK=add_block,
            num_warps=8,
        )
        return (
            grad_hidden_TD,
            out if ctx.return_grad_weight else None,
            None,
            None,
            None,
            None,
            None,
        )


def vocab_shard_start(
    global_vocab_size: int, vocab_parallel_group: dist.ProcessGroup
) -> int:
    """First vocab id of this rank's shard, matching ``_LossParallelCrossEntropy``."""
    tp_size = dist.get_world_size(vocab_parallel_group)
    shard_size = (global_vocab_size + tp_size - 1) // tp_size
    return min(global_vocab_size, shard_size * dist.get_rank(vocab_parallel_group))
