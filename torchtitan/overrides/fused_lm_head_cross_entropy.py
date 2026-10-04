# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused FP32OutputLinear lm_head + cross-entropy override for `ChunkedLossWrapper`.

The lm_head's grad_output is the cross-entropy gradient `softmax(logits) - one_hot`. Computing
it inside the lm_head's backward lets one kernel write its bf16 pieces straight from the fp32
logits: no fp32 gradient [T, V], no separate softmax and split kernels.

Activate with an FP32OutputLinear lm_head (`LMHeadFP32OutputConverter`) and:
    --override.imports torchtitan.overrides.fused_lm_head_cross_entropy.fused_lm_head_cross_entropy
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import spmd_types as spmd
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from torch.distributed.fsdp import register_fsdp_forward_method
from triton.language.extra.cuda import libdevice

from torchtitan.components.loss import (
    ChunkedLossWrapper,
    CrossEntropyLoss,
    IGNORE_INDEX,
)
from torchtitan.config import CompileConfig, derive, override
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.linear import FP32OutputLinear, weight_to_accumulate_into

__all__ = [
    "FusedLMHeadCrossEntropyLoss",
    "fp32_linear_cross_entropy",
    "fused_lm_head_cross_entropy",
]

# Shape suffix legend:
#   T = num tokens, D = model dimension, V = vocab (head out_features), P = grad pieces (2 or 3)

# Each grad_weight GEMM sums at most this many stacked rows, adding the GEMMs in fp32 (a non-fp32
# .grad is added into only when one GEMM covers the chunk). The pieces are stacked piece-major
# ([hi; lo]), so in one long call each GEMM sums 4096 tokens of one piece: one GEMM over 16k tokens
# (32k rows) has 1.39x its dW error (Qwen3-8B). TODO: stacking [hi; lo] per 2048-token block would
# match 8 token chunks' dW (6.43e-5 -> 6.01e-5) at the same GEMM count.
_GRAD_WEIGHT_ROWS_PER_GEMM = 4096


class FusedLMHeadCrossEntropyLoss(ChunkedLossWrapper):
    """`ChunkedLossWrapper` that runs each chunk's lm_head and cross-entropy as one fused function.

    Same loss as the unfused wrapper. Qwen3-8B head, H100, 8 vocab chunks, vs FP32OutputLinear +
    cross_entropy_loss with the CE compiled:
    - 16k tokens in 8 chunks: 184 -> 175 ms, peak 5.0 -> 4.0 GiB;
    - hidden-state gradients correctly rounded to bf16: 94.7% -> 99.1% of the time, from confident
      tokens; the bf16 L2 error stays 1.66e-3.
    Like FP32OutputLinear, chunks after the first add into weight.grad inside the GEMM under
    `accumulate_into_weight_grad`.

    Runs the unfused path where FP32OutputLinear falls back (off CUDA, non-bf16, batch-invariant
    mode) and on ROCm.

    The fused path bypasses `loss_fn`, so `compile.components=["loss"]` no longer compiles the
    cross-entropy. It runs eager; see `fp32_linear_cross_entropy`.

    Raises at build, in `set_lm_head`, or on the first chunk for:
    - a `loss_fn` other than `CrossEntropyLoss` (e.g. MTPLoss);
    - an lm_head that is not a bias-free FP32OutputLinear;
    - multi-output (MTP) chunks or extra loss inputs;
    - tensor parallelism > 1 (vocab-parallel lm_head).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        num_vocab_chunks: int = 8
        """See `fp32_linear_cross_entropy`: more chunks, more accurate grad_input, smaller
        buffers, slightly slower. 8 sum 19k Qwen3-8B vocab entries per grad_input GEMM
        (FP32OutputLinear: 8192); 16 take bf16-correct hidden-state gradients from 99.15% to
        99.34% for ~0.4-0.8 ms per 2048-token chunk."""

        recompute_logits: bool = False
        """See `fp32_linear_cross_entropy`: True drops the fp32 logits between forward and
        backward for one more forward GEMM. 16k Qwen3-8B tokens in 8 chunks peak at 3.0 GiB
        (211 ms) vs 4.0 GiB (175 ms) with the logits kept; keeping them needs 32 chunks for
        3.0 GiB (225 ms)."""

        num_dim_chunks: int = 1
        """See `fp32_linear_cross_entropy`: 1 runs FP32OutputLinear's logits GEMM. 2 halves
        grad_weight's error vs fp64 (2.3e-5 -> 1.2e-5, Qwen3-8B) for ~1 ms per chunk."""

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        super().__init__(config, compile_config=compile_config)
        # Exact type: a CrossEntropyLoss subclass (MTPLoss) computes a different loss.
        if type(self.loss_fn) is not CrossEntropyLoss:
            raise ValueError(
                f"FusedLMHeadCrossEntropyLoss fuses CrossEntropyLoss, got {type(self.loss_fn)}"
            )
        self.num_vocab_chunks = config.num_vocab_chunks
        self.recompute_logits = config.recompute_logits
        self.num_dim_chunks = config.num_dim_chunks

    def set_lm_head(self, lm_head: torch.nn.Module) -> None:
        if not isinstance(lm_head, FP32OutputLinear) or lm_head.bias is not None:
            raise ValueError(
                "FusedLMHeadCrossEntropyLoss needs a bias-free FP32OutputLinear lm_head "
                "(LMHeadFP32OutputConverter)."
            )
        super().set_lm_head(lm_head)
        # A registered FSDP forward method, so lm_head's gradient is reduce-scattered when its
        # last chunk's backward ends, overlapping the decoder backward. Unregistered, it waits
        # for the end of backward. No-op without FSDP.
        # pyrefly: ignore[bad-assignment]
        lm_head.fused_cross_entropy = functools.partial(_lm_head_cross_entropy, lm_head)
        register_fsdp_forward_method(lm_head, "fused_cross_entropy")

    def _lm_head_and_loss(
        self,
        h_chunks: tuple[torch.Tensor, ...],
        label_chunks: tuple[torch.Tensor, ...],
        global_valid_tokens: torch.Tensor | None,
        *,
        is_multi_output: bool,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert self.lm_head is not None
        if is_multi_output or loss_inputs:
            raise ValueError(
                "FusedLMHeadCrossEntropyLoss supports one prediction and no extra loss inputs."
            )
        if spmd_mesh_size("tp") > 1:
            raise NotImplementedError(
                "FusedLMHeadCrossEntropyLoss does not support a vocab-parallel lm_head yet."
            )
        hidden_TD, weight_VD = h_chunks[0], self.lm_head.weight
        # FP32OutputLinear's own fallbacks (its cases (a)-(c)) take the unfused path, and so
        # does ROCm: the kernels use CUDA libdevice (mul_rn).
        if not (
            hidden_TD.is_cuda
            and torch.version.hip is None
            and hidden_TD.dtype == weight_VD.dtype == torch.bfloat16
            and not is_in_batch_invariant_mode()
        ):
            return super()._lm_head_and_loss(
                h_chunks, label_chunks, global_valid_tokens, is_multi_output=False
            )
        # Set by set_lm_head; FSDP2 wraps it with its forward hooks.
        loss = self.lm_head.fused_cross_entropy(  # pyrefly: ignore[not-callable]
            hidden_TD,
            label_chunks[0],
            num_vocab_chunks=self.num_vocab_chunks,
            recompute_logits=self.recompute_logits,
            num_dim_chunks=self.num_dim_chunks,
        )
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        return loss, {}


@override(
    target=ChunkedLossWrapper.Config,
    exact=True,
    description="Fuse the FP32OutputLinear lm_head with cross-entropy (Triton).",
)
def fused_lm_head_cross_entropy(
    cfg: ChunkedLossWrapper.Config, **fields: Any
) -> FusedLMHeadCrossEntropyLoss.Config:
    """`fields` set the fused-only knobs, e.g. `{"recompute_logits": true}` in the import entry."""
    return derive(cfg, FusedLMHeadCrossEntropyLoss.Config, **fields)


def _lm_head_cross_entropy(
    lm_head: FP32OutputLinear,
    hidden_TD: torch.Tensor,
    labels_T: torch.Tensor,
    *,
    num_vocab_chunks: int,
    recompute_logits: bool,
    num_dim_chunks: int,
) -> torch.Tensor:
    """`fp32_linear_cross_entropy` with the lm_head's weight, read after FSDP all-gathers it."""
    return fp32_linear_cross_entropy(
        hidden_TD,
        lm_head.weight,
        labels_T,
        num_pieces=3 if lm_head.exact_grad_output_split else 2,
        num_vocab_chunks=num_vocab_chunks,
        recompute_logits=recompute_logits,
        num_dim_chunks=num_dim_chunks,
    )


def fp32_linear_cross_entropy(
    hidden_TD: torch.Tensor,
    weight_VD: torch.Tensor,
    labels_T: torch.Tensor,
    *,
    num_pieces: int,
    num_vocab_chunks: int = 8,
    recompute_logits: bool = False,
    num_dim_chunks: int = 1,
) -> torch.Tensor:
    """Summed cross-entropy of `hidden @ weight.T` against `labels`, with fp32 logits.

    Same math as `F.cross_entropy(FP32OutputLinear(hidden), labels, reduction="sum")`, but the
    backward never writes the fp32 gradient [T, V]: one kernel turns logits into its bf16 pieces.

    Args:
        hidden_TD: bf16 CUDA hidden states.
        weight_VD: bf16 LM head weight.
        labels_T: target ids; `IGNORE_INDEX` tokens add nothing to the loss or the gradients.
        num_pieces: bf16 pieces of the fp32 gradient, as FP32OutputLinear's split: 2 keep 16 of
            its 24 bits, 3 are exact.
        num_vocab_chunks: the backward runs one vocab chunk at a time. grad_input adds the chunks'
            GEMMs in fp32, which is more accurate than one GEMM over the whole vocab.
        recompute_logits: False keeps the fp32 logits [T, V] from forward to backward. True keeps
            none and recomputes each chunk's logits in the backward: one more forward GEMM.
        num_dim_chunks: the logits GEMM sums the model dim in this many chunks, added in fp32.

    Under `accumulate_into_weight_grad`, adds into an existing weight.grad in place and returns no
    grad_weight, as FP32OutputLinear does.

    Example:
        hidden [2048, 4096] bf16, weight [151936, 4096] bf16, labels [2048]
        loss = fp32_linear_cross_entropy(hidden, weight, labels, num_pieces=2) / num_valid_tokens
        loss.backward()   # hidden.grad bf16, weight.grad fp32 (if weight.grad_dtype is fp32)
    """
    return _eager_apply(
        hidden_TD,
        weight_VD,
        labels_T,
        num_pieces,
        num_vocab_chunks,
        recompute_logits,
        num_dim_chunks,
    )


class _FP32LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def spmd_typecheck(loss: torch.Tensor) -> None:
        # This rank's loss summed over its tokens, partial over dp/cp like CrossEntropyLoss's.
        spmd.assert_type(loss, {"dp": spmd.P, "cp": spmd.P})

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        hidden_TD: torch.Tensor,
        weight_VD: torch.Tensor,
        labels_T: torch.Tensor,
        num_pieces: int,
        num_vocab_chunks: int,
        recompute_logits: bool,
        num_dim_chunks: int,
    ) -> torch.Tensor:
        num_tokens, vocab_size = hidden_TD.shape[0], weight_VD.shape[0]
        torch._assert_async(
            torch.all(
                (labels_T == IGNORE_INDEX) | ((labels_T >= 0) & (labels_T < vocab_size))
            ),
            f"labels must be {IGNORE_INDEX} or in [0, {vocab_size})",
        )
        chunk_size = _vocab_chunk_size(vocab_size, num_vocab_chunks)
        max_T = hidden_TD.new_full((num_tokens,), float("-inf"), dtype=torch.float32)
        others_T = hidden_TD.new_zeros((num_tokens,), dtype=torch.float32)
        label_logit_T = hidden_TD.new_zeros((num_tokens,), dtype=torch.float32)
        logits_TV = (
            None if recompute_logits else _logits(hidden_TD, weight_VD, num_dim_chunks)
        )
        for start in range(0, vocab_size, chunk_size):
            if logits_TV is None:
                logits_chunk_TV = _logits(
                    hidden_TD, weight_VD[start : start + chunk_size], num_dim_chunks
                )
            else:
                logits_chunk_TV = logits_TV[:, start : start + chunk_size]
            _online_logsumexp(
                logits_TV=logits_chunk_TV,
                labels_T=labels_T,
                vocab_start=start,
                max_T=max_T,
                others_T=others_T,
                label_logit_T=label_logit_T,
            )
        # logsumexp = max + log(1 + others), others = sum of exp(logit - max) over all but the max.
        # log1p keeps the loss and gradient of a confident token (others << 1) accurate.
        log_sum_T = others_T.log1p()
        valid_T = labels_T != IGNORE_INDEX
        loss = torch.where(valid_T, (max_T - label_logit_T) + log_sum_T, 0.0).sum()

        ctx.save_for_backward(hidden_TD, weight_VD, labels_T, max_T, log_sum_T)
        # As FP32OutputLinear: under accumulate_into_weight_grad, the backward adds into .grad.
        ctx.weight_param = weight_to_accumulate_into(weight_VD)
        ctx.num_pieces = num_pieces
        ctx.chunk_size = chunk_size
        ctx.num_dim_chunks = num_dim_chunks
        # A ctx attribute, not save_for_backward, so the backward can free it early.
        ctx.logits_TV = logits_TV
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss: torch.Tensor):  # pyrefly: ignore[bad-override]
        hidden_TD, weight_VD, labels_T, max_T, log_sum_T = ctx.saved_tensors
        needs_grad_input, needs_grad_weight = ctx.needs_input_grad[:2]
        num_pieces, chunk_size = ctx.num_pieces, ctx.chunk_size
        logits_TV, ctx.logits_TV = ctx.logits_TV, None
        num_tokens, vocab_size = hidden_TD.shape[0], weight_VD.shape[0]
        grad_loss = grad_loss.float().reshape(1)

        # ======== Per vocab chunk: CE gradient pieces, then the grad_input and grad_weight GEMMs ========
        grad_input_PTD = grad_weight_VD = None
        hidden_PTD = torch.cat([hidden_TD] * num_pieces) if needs_grad_weight else None
        running_grad_VD = None if ctx.weight_param is None else ctx.weight_param.grad
        # max(.., 1): zero tokens still run one (empty) GEMM, which writes zeros.
        num_rows = max(num_pieces * num_tokens, 1)
        # A non-fp32 .grad would round after every row GEMM: add into it only when one GEMM does.
        if running_grad_VD is not None and running_grad_VD.dtype != torch.float32:
            if num_rows > _GRAD_WEIGHT_ROWS_PER_GEMM:
                running_grad_VD = None
        for start in range(0, vocab_size, chunk_size):
            weight_chunk_VD = weight_VD[start : start + chunk_size]
            if logits_TV is None:
                logits_chunk_TV = _logits(
                    hidden_TD, weight_chunk_VD, ctx.num_dim_chunks
                )
            else:
                logits_chunk_TV = logits_TV[:, start : start + chunk_size]
            pieces_PTV = _cross_entropy_grad_pieces(
                logits_TV=logits_chunk_TV,
                max_T=max_T,
                log_sum_T=log_sum_T,
                labels_T=labels_T,
                grad_loss=grad_loss,
                vocab_start=start,
                num_pieces=num_pieces,
            )
            del logits_chunk_TV
            if start + chunk_size >= vocab_size:
                logits_TV = None  # free the kept logits before the last chunk's GEMMs
            if needs_grad_input:
                # Each chunk's GEMM is added in fp32: split-K over the vocab.
                if grad_input_PTD is None:
                    grad_input_PTD = torch.mm(
                        pieces_PTV, weight_chunk_VD, out_dtype=torch.float32
                    )
                else:
                    torch.addmm(
                        grad_input_PTD,
                        pieces_PTV,
                        weight_chunk_VD,
                        out_dtype=torch.float32,
                        out=grad_input_PTD,
                    )
            if needs_grad_weight:
                assert hidden_PTD is not None
                # A later chunk adds into weight.grad inside the GEMM (as FP32OutputLinear under
                # accumulate_into_weight_grad) and returns no grad_weight.
                if running_grad_VD is not None:
                    target_VD = running_grad_VD
                else:
                    if grad_weight_VD is None:
                        grad_weight_VD = weight_VD.new_empty(
                            weight_VD.shape, dtype=torch.float32
                        )
                    target_VD = grad_weight_VD
                target_chunk_VD = target_VD[start : start + chunk_size]
                for row in range(0, num_rows, _GRAD_WEIGHT_ROWS_PER_GEMM):
                    rows = slice(row, row + _GRAD_WEIGHT_ROWS_PER_GEMM)
                    if row == 0 and running_grad_VD is None:
                        torch.mm(
                            pieces_PTV[rows].T,
                            hidden_PTD[rows],
                            out_dtype=torch.float32,
                            out=target_chunk_VD,
                        )
                    else:
                        torch.addmm(
                            target_chunk_VD,
                            pieces_PTV[rows].T,
                            hidden_PTD[rows],
                            out_dtype=target_chunk_VD.dtype,
                            out=target_chunk_VD,
                        )
            del pieces_PTV

        # ======== Sum the pieces' grad_input, round once ========
        grad_input_TD = None
        if needs_grad_input:
            assert grad_input_PTD is not None
            grad_input_TD = grad_input_PTD.unflatten(0, (num_pieces, num_tokens)).sum(
                dim=0
            )
            grad_input_TD = grad_input_TD.to(hidden_TD.dtype)
        return grad_input_TD, grad_weight_VD, None, None, None, None, None


def _apply(*args: Any) -> torch.Tensor:
    """`_FP32LinearCrossEntropyFunction.apply`, looked up per call: SPMD type checking patches it."""
    return _FP32LinearCrossEntropyFunction.apply(*args)


# A compiled caller graph-breaks here and runs the Function eager. TODO: compile it; today more
# than one vocab chunk fails to lower addmm(out_dtype=) (pytorch/pytorch#190936), and tracing
# rounds grad_weight to bf16 (pytorch/pytorch#197381). Each stage is already one kernel.
_eager_apply = cast(Callable[..., torch.Tensor], torch.compiler.disable(_apply))


def _logits(
    hidden_TD: torch.Tensor, weight_VD: torch.Tensor, num_dim_chunks: int
) -> torch.Tensor:
    """fp32 `hidden @ weight.T`, summing the model dim in `num_dim_chunks` fp32-added GEMMs."""
    chunk = triton.cdiv(hidden_TD.shape[1], num_dim_chunks)
    logits_TV = torch.mm(
        hidden_TD[:, :chunk], weight_VD[:, :chunk].T, out_dtype=torch.float32
    )
    for start in range(chunk, hidden_TD.shape[1], chunk):
        torch.addmm(
            logits_TV,
            hidden_TD[:, start : start + chunk],
            weight_VD[:, start : start + chunk].T,
            out_dtype=torch.float32,
            out=logits_TV,
        )
    return logits_TV


def _vocab_chunk_size(vocab_size: int, num_vocab_chunks: int) -> int:
    """Chunk length, a multiple of 128 so every chunk's rows stay 16-byte aligned."""
    return triton.cdiv(triton.cdiv(vocab_size, num_vocab_chunks), 128) * 128


def _online_logsumexp(
    *,
    logits_TV: torch.Tensor,
    labels_T: torch.Tensor,
    vocab_start: int,
    max_T: torch.Tensor,
    others_T: torch.Tensor,
    label_logit_T: torch.Tensor,
) -> None:
    """Fold one vocab chunk of logits into the running max, sum of the others' exp, and label
    logit (`max_T`, `others_T`, `label_logit_T`, updated in place)."""
    num_tokens, chunk_vocab = logits_TV.shape
    # A plain launch, not torch.library.wrap_triton: the function runs eager, and wrap_triton adds
    # ~0.1 ms of host time per launch. pyrefly can't type Triton's launch arguments.
    _online_logsumexp_kernel[(num_tokens,)](
        logits_TV,
        labels_T,
        max_T,
        others_T,
        label_logit_T,
        vocab_start,
        chunk_vocab,
        logits_TV.stride(0),
        BLOCK_V=4096,  # pyrefly: ignore[bad-argument-type]
        num_warps=8,  # pyrefly: ignore[unexpected-keyword]
    )


@triton.jit
def _online_logsumexp_kernel(
    logits_ptr,
    labels_ptr,
    max_ptr,
    others_ptr,
    label_logit_ptr,
    vocab_start,
    chunk_vocab,
    stride_t,
    BLOCK_V: tl.constexpr,
):
    """Per token: running max, and the sum of exp(logit - max) over all logits but one max."""
    token = tl.program_id(0)
    row_ptr = logits_ptr + token.to(tl.int64) * stride_t
    label = tl.load(labels_ptr + token) - vocab_start
    running_max = tl.load(max_ptr + token)
    running_others = tl.load(others_ptr + token)
    for block_start in range(0, chunk_vocab, BLOCK_V):
        offs = block_start + tl.arange(0, BLOCK_V)
        logits = tl.load(row_ptr + offs, mask=offs < chunk_vocab, other=float("-inf"))
        block_max = tl.max(logits, axis=0)
        is_block_max = tl.arange(0, BLOCK_V) == tl.argmax(logits, axis=0)
        block_others = tl.sum(
            tl.where(is_block_max, 0.0, libdevice.exp(logits - block_max)), axis=0
        )
        # The smaller of the two maxima joins the others; adding 1 for it never cancels.
        merged_others = tl.where(
            block_max > running_max,
            (running_others + 1.0) * libdevice.exp(running_max - block_max)
            + block_others,
            running_others
            # pyrefly reads tl.sum's result as a tuple.
            + libdevice.exp(block_max - running_max)
            * (block_others + 1.0),  # pyrefly: ignore[unsupported-operation]
        )
        # A block of only -inf logits adds nothing (exp(-inf - -inf) would be NaN).
        running_others = tl.where(
            block_max == float("-inf"), running_others, merged_others
        )
        running_max = tl.maximum(running_max, block_max)
    tl.store(max_ptr + token, running_max)
    tl.store(others_ptr + token, running_others)
    if (label >= 0) & (label < chunk_vocab):
        tl.store(label_logit_ptr + token, tl.load(row_ptr + label))


def _cross_entropy_grad_pieces(
    *,
    logits_TV: torch.Tensor,
    max_T: torch.Tensor,
    log_sum_T: torch.Tensor,
    labels_T: torch.Tensor,
    grad_loss: torch.Tensor,
    vocab_start: int,
    num_pieces: int,
) -> torch.Tensor:
    """bf16 pieces of `grad_loss * (softmax - one_hot)`, stacked along tokens: [P * T, V]."""
    num_tokens, chunk_vocab = logits_TV.shape
    pieces_PTV = logits_TV.new_empty(
        (num_pieces * num_tokens, chunk_vocab), dtype=torch.bfloat16
    )
    block_v = 2048
    grid = (num_tokens, triton.cdiv(chunk_vocab, block_v))
    # A plain launch, as in _online_logsumexp.
    _cross_entropy_grad_pieces_kernel[grid](
        logits_TV,
        max_T,
        log_sum_T,
        labels_T,
        grad_loss,
        pieces_PTV,
        num_tokens * chunk_vocab,
        vocab_start,
        chunk_vocab,
        logits_TV.stride(0),
        IGNORE_INDEX,
        NUM_PIECES=num_pieces,  # pyrefly: ignore[bad-argument-type]
        BLOCK_V=block_v,  # pyrefly: ignore[bad-argument-type]
        num_warps=8,  # pyrefly: ignore[unexpected-keyword]
    )
    return pieces_PTV


@triton.jit
def _round_to_bf16(x):
    """Nearest bf16 value, ties away from zero, kept in fp32 (FP32OutputLinear's rounding)."""
    bits = x.to(tl.int32, bitcast=True)
    return ((bits + 0x8000) & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _cross_entropy_grad_pieces_kernel(
    logits_ptr,
    max_ptr,
    log_sum_ptr,
    labels_ptr,
    grad_loss_ptr,
    pieces_ptr,
    piece_stride,
    vocab_start,
    chunk_vocab,
    stride_t,
    ignore_index,
    NUM_PIECES: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    token = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < chunk_vocab
    logits = tl.load(
        logits_ptr + token.to(tl.int64) * stride_t + offs, mask=mask, other=0.0
    )
    label = tl.load(labels_ptr + token)
    scale = tl.where(label == ignore_index, 0.0, tl.load(grad_loss_ptr))
    # log(softmax) = (logit - max) - log_sum. At the label, softmax - 1 = expm1(log(softmax)) stays
    # accurate when softmax is close to 1, where probs - 1.0 would cancel.
    log_probs = (logits - tl.load(max_ptr + token)) - tl.load(log_sum_ptr + token)
    is_label = offs == label - vocab_start
    softmax_minus_one_hot = tl.where(
        is_label, libdevice.expm1(log_probs), libdevice.exp(log_probs)
    )
    # mul_rn: an FMA contracted into `grad - hi` below would split the unrounded product, so
    # the pieces would no longer be those of the fp32 gradient.
    grad = libdevice.mul_rn(softmax_minus_one_hot, scale)

    out_ptrs = pieces_ptr + token.to(tl.int64) * chunk_vocab + offs
    hi = _round_to_bf16(grad)
    rest = grad - hi
    tl.store(out_ptrs, hi.to(tl.bfloat16), mask=mask)
    # Step one piece at a time: 2 * piece_stride can overflow int32 when piece_stride fits.
    out_ptrs += piece_stride
    if NUM_PIECES == 2:
        tl.store(out_ptrs, rest.to(tl.bfloat16), mask=mask)
    else:
        mid = _round_to_bf16(rest)
        tl.store(out_ptrs, mid.to(tl.bfloat16), mask=mask)
        tl.store(out_ptrs + piece_stride, (rest - mid).to(tl.bfloat16), mask=mask)
