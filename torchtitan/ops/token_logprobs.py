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

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # CPU-only installs (e.g. torch's CPU wheels); the fused path is then off.
    _HAS_TRITON = False

# Shape suffix legend:
#   T = num tokens, D = model dimension, V = local vocab size

IGNORE_INDEX = -100

# M = one_hot - softmax is written as fp16 with one scale per row (g = dloss/dlogprob stays out,
# so any loss scale fits). A row's max |M| is 1 - p(label), so 2^15 / (1 - p(label)) makes the
# label's entry exactly 2^15, under fp16's 65504. The floor caps the scale for confident tokens
# (1 - p < 2^-6), whose label entry keeps fp32's rounding of 1 - p, as in log_softmax.
_M_LABEL_ENTRY = 2.0**15
_ONE_MINUS_P_FLOOR = 2.0**-6


if _HAS_TRITON:

    @triton.jit  # pyrefly: ignore[unbound-name]
    def _partial_softmax_stats_kernel(
        logits_ptr,
        labels_ptr,
        max_ptr,
        sum_exp_ptr,
        sum_exp_shifted_ptr,
        label_logit_ptr,
        vocab_size,
        vocab_start,
        stride_row,
        BLOCK: tl.constexpr,
    ):
        """One pass over a row: max, sum(exp(x - max)), sum(exp(x - max) * (x - max)), x[label]."""
        row = tl.program_id(0).to(tl.int64)
        row_ptr = logits_ptr + row * stride_row
        # Per-lane online softmax; lanes are combined once after the loop.
        max_lanes = tl.full([BLOCK], -1e30, tl.float32)
        sum_exp_lanes = tl.zeros([BLOCK], tl.float32)
        sum_exp_shifted_lanes = tl.zeros([BLOCK], tl.float32)
        for start in range(0, vocab_size, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            logits = tl.load(
                row_ptr + offsets, mask=offsets < vocab_size, other=float("-inf")
            ).to(tl.float32)
            new_max = tl.maximum(max_lanes, logits)
            rescale = tl.exp(max_lanes - new_max)
            exp_logits = tl.exp(logits - new_max)
            # Re-center the old sum on the new max: x - new = (x - old) + (old - new). exp(-inf) * -inf
            # is NaN, so masked and padded logits contribute 0.
            sum_exp_shifted_lanes = rescale * (
                sum_exp_shifted_lanes + (max_lanes - new_max) * sum_exp_lanes
            ) + tl.where(exp_logits > 0, exp_logits * (logits - new_max), 0.0)
            sum_exp_lanes = sum_exp_lanes * rescale + exp_logits
            max_lanes = new_max
        row_max = tl.max(max_lanes, 0)
        lane_rescale = tl.exp(max_lanes - row_max)
        tl.store(max_ptr + row, row_max)
        tl.store(sum_exp_ptr + row, tl.sum(sum_exp_lanes * lane_rescale, 0))
        lane_shift = (max_lanes - row_max) * sum_exp_lanes
        tl.store(
            sum_exp_shifted_ptr + row,
            tl.sum((sum_exp_shifted_lanes + lane_shift) * lane_rescale, 0),
        )
        # 0 when the label is ignored (-100) or lives on another vocab shard, so shards can be summed.
        local_label = tl.load(labels_ptr + row) - vocab_start
        is_local = (local_label >= 0) & (local_label < vocab_size)
        label_logit = tl.load(row_ptr + tl.where(is_local, local_label, 0)).to(
            tl.float32
        )
        tl.store(label_logit_ptr + row, tl.where(is_local, label_logit, 0.0))

    @triton.jit  # pyrefly: ignore[unbound-name]
    def _dlogits_kernel(
        logits_ptr,
        labels_ptr,
        row_max_ptr,
        sum_exp_ptr,
        row_scale_ptr,
        dlogits_ptr,
        vocab_size,
        vocab_start,
        stride_row,
        stride_out,
        IGNORE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """dlogits[row] = (one_hot(label) - softmax(logits[row])) * row_scale[row].

        0 for ignored labels.
        """
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < vocab_size
        logits = tl.load(
            logits_ptr + row * stride_row + offsets, mask=mask, other=float("-inf")
        ).to(tl.float32)
        row_max = tl.load(row_max_ptr + row)
        inv_sum_exp = 1.0 / tl.load(sum_exp_ptr + row)
        label = tl.load(labels_ptr + row)
        scale = tl.load(row_scale_ptr + row)
        one_hot = (offsets + vocab_start == label).to(tl.float32)
        dlogits = tl.where(
            label != IGNORE,
            (one_hot - tl.exp(logits - row_max) * inv_sum_exp) * scale,
            0.0,
        )
        tl.store(
            dlogits_ptr + row * stride_out + offsets,
            dlogits.to(dlogits_ptr.dtype.element_ty),
            mask=mask,
        )


def _softmax_stats(
    logits_TV: torch.Tensor,
    labels_T: torch.Tensor,
    vocab_start: int,
    vocab_parallel_group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(row_max, sum_exp, label logprob, entropy)``, each ``[T]``, over the full vocab.

    ``softmax = exp(logits - row_max) / sum_exp``. The two stay apart because a single
    logsumexp rounds to ulp(row_max) (~2e-6 at logit 23), which swamps ``1 - p`` for confident
    labels; the logprob is ``(label_logit - row_max) - log(sum_exp)`` and the entropy
    ``log(sum_exp) - sum(exp(x - row_max) * (x - row_max)) / sum_exp`` for the same reason.
    """
    num_tokens, vocab_size = logits_TV.shape
    labels_T = labels_T.contiguous()
    row_max, sum_exp, sum_exp_shifted, label_logit = torch.empty(
        4, num_tokens, device=logits_TV.device, dtype=torch.float32
    )
    torch.library.wrap_triton(_partial_softmax_stats_kernel)[(num_tokens,)](
        logits_TV,
        labels_T,
        row_max,
        sum_exp,
        sum_exp_shifted,
        label_logit,
        vocab_size,
        vocab_start,
        logits_TV.stride(0),
        BLOCK=2048,
        num_warps=8,
    )
    if vocab_parallel_group is not None:
        # Combine the shards: rescale each shard's sums to the global max, then add. Wait
        # explicitly: row_max and sum_exp go to a Triton kernel, which would not wait on an
        # AsyncCollectiveTensor.
        global_max = funcol.wait_tensor(
            funcol.all_reduce(row_max, "max", group=vocab_parallel_group)
        )
        rescale = torch.exp(row_max - global_max)
        shifted = sum_exp_shifted + (row_max - global_max) * sum_exp
        sums = torch.stack([sum_exp * rescale, shifted * rescale, label_logit])
        sum_exp, sum_exp_shifted, label_logit = funcol.wait_tensor(
            funcol.all_reduce(sums, "sum", group=vocab_parallel_group)
        )
        row_max = global_max
    log_sum_exp = torch.log(sum_exp)
    logprobs = torch.where(
        labels_T != IGNORE_INDEX, (label_logit - row_max) - log_sum_exp, 0.0
    )
    # H(p) = -sum(p * (x - row_max - log_sum_exp))
    entropy = log_sum_exp - sum_exp_shifted / sum_exp
    return row_max, sum_exp, logprobs, entropy


def _fp16_scale(absmax: torch.Tensor) -> torch.Tensor:
    """Power-of-two scale that moves ``absmax`` into [2^14, 2^15), the top of fp16's range.

    Exact to undo, and values down to ~2^-29 of ``absmax`` stay fp16 normals. Zero gives 2^15; the
    clamp keeps a tiny nonzero ``absmax`` (below ~2^-85) from overflowing the scale.

    Example:
        absmax 139.0 (frexp exponent 8) -> scale 2^7, and 139 * 2^7 = 17792
    """
    _, exponent = torch.frexp(absmax.float())
    return torch.exp2((15 - exponent).clamp(max=100).float())


def _start_hidden_scale(
    hidden_TD: torch.Tensor,
    grad_scale_T: torch.Tensor,
    grad_state: "TokenLogprobsGradState",
) -> tuple[torch.Tensor, bool]:
    """Power-of-two scale for ``hidden * grad_scale``; starts copying it to the host.

    Called before the grad_hidden GEMM is queued, so waiting on the copy later does not stall the
    GPU. Returns ``(scale, copied)``; under CUDA graph capture nothing is copied.
    """
    hidden_absmax_T = torch.linalg.vector_norm(hidden_TD, ord=float("inf"), dim=1)
    hidden_scale = _fp16_scale((hidden_absmax_T.float() * grad_scale_T.abs()).amax())
    if torch.cuda.is_current_stream_capturing():
        return hidden_scale, False
    grad_state.hidden_scale_host.copy_(hidden_scale, non_blocking=True)
    grad_state.hidden_scale_copied.record()
    return hidden_scale, True


class TokenLogprobsGradState:
    """Per-microbatch state shared by the chunks' backwards.

    The first chunk's forward fills ``weight_fp16`` from the weight it receives (unsharded
    under FSDP), scaled by ``weight_scale`` and cast once for the fp16 backward GEMMs.
    ``grad_weight`` accumulates the chunks' weight gradients in fp32; the last chunk returns the
    sum to autograd in the weight's gradient dtype (bf16, or fp32 with ``Tensor.grad_dtype``), so
    it is rounded at most once per microbatch. Each chunk's grad_weight unscale reaches the host
    through the pinned ``hidden_scale_host``.

    Not a dataclass: FSDP's forward-input cast copies dataclass arguments (casting their
    tensors to the param dtype), which would give each chunk its own accumulator.
    """

    def __init__(self) -> None:
        self.weight_fp16: torch.Tensor | None = None
        self.weight_scale: torch.Tensor | None = None
        self.grad_weight: torch.Tensor | None = None
        self.hidden_scale_host = torch.empty((), dtype=torch.float32, pin_memory=True)
        self.hidden_scale_copied = torch.cuda.Event()


class TokenLogprobs(torch.autograd.Function):
    """``(hidden [T, D], weight [V, D], labels [T]) -> (logprobs [T], entropy [T])``.

    Forward: ``logits = hidden @ weight.T`` as a bf16 GEMM with fp32 output (the same op as
    ``Fp32OutputLinear``), then one Triton pass for the softmax statistics (row max, sum of
    exps), the label logit and entropy.

    Backward, with ``g = dloss/dlogprobs`` and ``M = (one_hot - softmax)``:

        dlogits     = g * M                                 never materialized in fp32
        grad_hidden = g * (M @ W)                           one fp16 GEMM, fp32 output
        grad_weight = M.T @ (g * hidden)                    one fp16 GEMM, fp32 output

    ``M`` is written once as fp16, which keeps 10 mantissa bits to bf16's 7. Rounding it costs
    ~8x less gradient error than rounding dlogits to bf16, at the same GEMM speed; with 2^-11
    relative error per element, the gradients land within ~1% of exact fp64 gradients rounded to
    bf16 (hi + lo, which ``Fp32OutputLinear`` uses, lands within ~0.5% with twice the GEMMs).
    ``g`` must stay out of the fp16 operands: a loss normalized by 1e5 tokens makes ``g * M``
    fp16 subnormal. So each row of ``M`` gets its own scale (its max, the label's entry, becomes
    exactly 2^15 unless 1 - p < 2^-6), ``g`` scales the fp32 output rows of ``grad_hidden``, and ``g * hidden``
    gets a power-of-two scale that puts its largest element at the top of fp16's range, which
    keeps tokens down to ~1e-9 of the largest ``|g|`` as fp16 normals. ``W`` is scaled the same
    way once per microbatch, so small-init heads do not go subnormal.

    Entropy is a metric: it is non-differentiable and must not receive a gradient.
    With ``vocab_parallel_group``, ``weight`` is this rank's vocab shard starting at
    ``vocab_start``; the forward all-reduces ``[T]`` statistics, and ``grad_hidden`` is this
    shard's partial sum (reduced by the hidden states' TP redistribute, as for the lm_head).
    """

    @staticmethod
    def spmd_typecheck(
        result: tuple[torch.Tensor, torch.Tensor],
        *,
        weight_VD: torch.Tensor,
        labels_T: torch.Tensor,
        vocab_parallel_group: dist.ProcessGroup | None,
    ) -> None:
        """SPMD type: weight S(0)@TP, labels I@TP -> logprobs and entropy I@TP; local without TP."""
        overrides = {}
        if vocab_parallel_group is not None:
            spmd.assert_type(weight_VD, {vocab_parallel_group: spmd.S(0)})
            spmd.assert_type(labels_T, {vocab_parallel_group: spmd.I})
            overrides = {vocab_parallel_group: spmd.I}
        for output_T in result:
            spmd.assert_local_type_like(
                output_T, labels_T, overrides  # pyrefly: ignore [bad-argument-type]
            )

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
        # Slow fallback for non-bf16 operands (e.g. an fp32 weight under autocast): fp32 matmuls,
        # as in Fp32OutputLinear.
        ctx.use_bf16_gemm = hidden_TD.dtype == weight_VD.dtype == torch.bfloat16
        if ctx.use_bf16_gemm:
            if grad_state is not None and grad_state.weight_fp16 is None:
                grad_state.weight_scale = _fp16_scale(
                    torch.linalg.vector_norm(weight_VD.detach(), ord=float("inf"))
                )
                # Scale and cast in one kernel, without a [V, D] bf16 temporary.
                grad_state.weight_fp16 = torch.mul(
                    weight_VD.detach(),
                    grad_state.weight_scale,
                    out=torch.empty_like(weight_VD, dtype=torch.float16),
                )
            logits_TV = torch.mm(hidden_TD, weight_VD.T, out_dtype=torch.float32)
        else:
            logits_TV = torch.mm(hidden_TD.float(), weight_VD.float().T)
        labels_T = labels_T.contiguous()
        row_max_T, sum_exp_T, logprobs_T, entropy_T = _softmax_stats(
            logits_TV, labels_T, vocab_start, vocab_parallel_group
        )
        one_minus_p_T = -torch.expm1(logprobs_T)
        m_row_scale_T = _M_LABEL_ENTRY / one_minus_p_T.clamp_min(_ONE_MINUS_P_FLOOR)
        ctx.save_for_backward(
            hidden_TD, weight_VD, labels_T, row_max_T, sum_exp_T, m_row_scale_T
        )
        # Kept off save_for_backward so backward can free the [T, V] fp32 logits before its GEMMs.
        ctx.logits_TV = logits_TV
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
        (
            hidden_TD,
            weight_VD,
            labels_T,
            row_max_T,
            sum_exp_T,
            m_row_scale_T,
        ) = ctx.saved_tensors
        logits_TV, ctx.logits_TV = ctx.logits_TV, None
        grad_state = ctx.grad_state
        assert grad_state is not None, "backward needs a TokenLogprobsGradState"
        num_tokens, vocab_size = logits_TV.shape
        grad_T = grad_logprobs_T.float().contiguous()

        # bf16 path: M = (one_hot - softmax) * m_row_scale in fp16, with g applied outside the
        # GEMMs. Fallback: dlogits = g * (one_hot - softmax) in fp32.
        dlogits_TV = torch.empty(
            num_tokens,
            vocab_size,
            device=logits_TV.device,
            dtype=torch.float16 if ctx.use_bf16_gemm else torch.float32,
        )
        block = 4096
        torch.library.wrap_triton(_dlogits_kernel)[
            (num_tokens, triton.cdiv(vocab_size, block))
        ](
            logits_TV,
            labels_T,
            row_max_T,
            sum_exp_T,
            m_row_scale_T if ctx.use_bf16_gemm else grad_T,
            dlogits_TV,
            vocab_size,
            ctx.vocab_start,
            logits_TV.stride(0),
            dlogits_TV.stride(0),
            IGNORE=IGNORE_INDEX,
            BLOCK=block,
            num_warps=8,
        )
        del logits_TV

        if not ctx.use_bf16_gemm:
            grad_hidden_TD = torch.mm(dlogits_TV, weight_VD.float()).to(hidden_TD.dtype)
            if not ctx.needs_input_grad[1]:
                return grad_hidden_TD, None, None, None, None, None, None
            grad_weight_VD = torch.mm(dlogits_TV.T, hidden_TD.float())
            if grad_state.grad_weight is None:
                grad_state.grad_weight = grad_weight_VD
            else:
                grad_state.grad_weight += grad_weight_VD
            out = None
            if ctx.return_grad_weight:
                out = grad_state.grad_weight.to(ctx.grad_weight_dtype)
                grad_state.grad_weight = None
            return grad_hidden_TD, out, None, None, None, None, None

        # Per-token factor that undoes M's row scale and applies g.
        grad_scale_T = grad_T / m_row_scale_T
        hidden_scale, scale_copied = _start_hidden_scale(
            hidden_TD, grad_scale_T, grad_state
        )
        grad_hidden_TD = torch.mm(
            dlogits_TV, grad_state.weight_fp16, out_dtype=torch.float32
        )
        grad_hidden_TD = (
            grad_hidden_TD * (grad_scale_T / grad_state.weight_scale)[:, None]
        )
        grad_hidden_TD = grad_hidden_TD.to(hidden_TD.dtype)
        # A frozen lm_head (e.g. LoRA on the decoder) skips the dW GEMM and its accumulator.
        if not ctx.needs_input_grad[1]:
            return grad_hidden_TD, None, None, None, None, None, None

        scaled_hidden_TD = torch.mul(
            hidden_TD,
            (grad_scale_T * hidden_scale)[:, None],
            out=torch.empty_like(hidden_TD, dtype=torch.float16),
        )
        # Accumulate this chunk's gradient into the fp32 accumulator. The unscale 1 / hidden_scale
        # is a host alpha, so cuBLAS computes alpha * M.T @ hidden + acc in one pass.
        accumulated = grad_state.grad_weight
        if not scale_copied:
            # Graph capture cannot wait on the host: unscale on the device instead.
            grad_weight_VD = torch.mm(
                dlogits_TV.T, scaled_hidden_TD, out_dtype=torch.float32
            ).mul_(hidden_scale.reciprocal())
            if accumulated is None:
                accumulated = grad_weight_VD
            else:
                accumulated += grad_weight_VD
        else:
            beta = 1.0
            if accumulated is None:
                # beta=0: cuBLAS does not read the empty buffer.
                accumulated, beta = (
                    torch.empty_like(weight_VD, dtype=torch.float32),
                    0.0,
                )
            grad_state.hidden_scale_copied.synchronize()
            torch.addmm(
                accumulated,
                dlogits_TV.T,
                scaled_hidden_TD,
                beta=beta,
                alpha=1.0 / grad_state.hidden_scale_host.item(),
                out_dtype=torch.float32,
                out=accumulated,
            )
        grad_state.grad_weight = accumulated
        if not ctx.return_grad_weight:
            return grad_hidden_TD, None, None, None, None, None, None
        # A no-op for fp32 gradients. Dropping the state's references lets autograd take the
        # buffer without a copy.
        out = accumulated.to(ctx.grad_weight_dtype)
        grad_state.grad_weight = grad_state.weight_fp16 = grad_state.weight_scale = None
        return grad_hidden_TD, out, None, None, None, None, None


class TokenLogprobsFromLogits(torch.autograd.Function):
    """``(logits [T, V], labels [T]) -> (logprobs [T], entropy [T])``, reading the logits once.

    The logits-level counterpart of ``TokenLogprobs``, for heads it does not fuse with (e.g. bf16,
    soft-capped or LoRA heads). Forward is the same stats pass; backward writes
    ``g * (one_hot - softmax)`` in the logits' dtype in one pass. It replaces the ~10 passes of
    log_softmax forward/backward plus a separate entropy softmax. Logits may be bf16 or fp32;
    statistics are computed in fp32. TP handling matches ``TokenLogprobs``.
    """

    @staticmethod
    def spmd_typecheck(
        result: tuple[torch.Tensor, torch.Tensor],
        *,
        logits_TV: torch.Tensor,
        labels_T: torch.Tensor,
        vocab_parallel_group: dist.ProcessGroup | None,
    ) -> None:
        """SPMD type: logits S(-1)@TP, labels I@TP -> logprobs and entropy I@TP; local without TP."""
        overrides = {}
        if vocab_parallel_group is not None:
            spmd.assert_type(logits_TV, {vocab_parallel_group: spmd.S(1)})
            spmd.assert_type(labels_T, {vocab_parallel_group: spmd.I})
            overrides = {vocab_parallel_group: spmd.I}
        for output_T in result:
            spmd.assert_local_type_like(
                output_T, labels_T, overrides  # pyrefly: ignore [bad-argument-type]
            )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        logits_TV: torch.Tensor,
        labels_T: torch.Tensor,
        vocab_start: int,
        vocab_parallel_group: dist.ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_TV, labels_T = logits_TV.contiguous(), labels_T.contiguous()
        row_max_T, sum_exp_T, logprobs_T, entropy_T = _softmax_stats(
            logits_TV, labels_T, vocab_start, vocab_parallel_group
        )
        ctx.save_for_backward(logits_TV, labels_T, row_max_T, sum_exp_T)
        ctx.vocab_start = vocab_start
        ctx.mark_non_differentiable(entropy_T)
        ctx.set_materialize_grads(False)
        return logprobs_T, entropy_T

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_logprobs_T: torch.Tensor, grad_entropy_T: None):
        assert grad_entropy_T is None, "entropy is a metric and has no gradient"
        logits_TV, labels_T, row_max_T, sum_exp_T = ctx.saved_tensors
        num_tokens, vocab_size = logits_TV.shape
        dlogits_TV = torch.empty_like(logits_TV)
        block = 4096
        torch.library.wrap_triton(_dlogits_kernel)[
            (num_tokens, triton.cdiv(vocab_size, block))
        ](
            logits_TV,
            labels_T,
            row_max_T,
            sum_exp_T,
            grad_logprobs_T.float().contiguous(),
            dlogits_TV,
            vocab_size,
            ctx.vocab_start,
            logits_TV.stride(0),
            dlogits_TV.stride(0),
            IGNORE=IGNORE_INDEX,
            BLOCK=block,
            num_warps=8,
        )
        return dlogits_TV, None, None, None


def can_use_token_logprobs_kernels(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` can go to the Triton kernels: an eager, plain CUDA tensor.

    Excludes DTensors, fake and functional tensors, and tracing (torch.compile, make_fx), where a
    raw kernel launch fails or is not recorded.
    """
    if not _HAS_TRITON:
        return False
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import FunctionalTensor
    from torch.distributed.tensor import DTensor
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return (
        tensor.is_cuda
        and not isinstance(tensor, (DTensor, FakeTensor, FunctionalTensor))
        and not torch.compiler.is_compiling()
        and get_proxy_mode() is None
    )


def vocab_shard_start(
    labels_T: torch.Tensor,
    local_vocab_size: int,
    vocab_parallel_group: dist.ProcessGroup | None,
    global_vocab_size: int | None,
) -> int:
    """Return this rank's first vocab id, after checking the shard layout and the labels.

    Shards follow ``_LossParallelCrossEntropy``: ``ceil(V / tp)`` rows each, the last possibly
    fewer. The kernels would read an out-of-range label as "on another vocab shard", so labels
    are device-asserted (no host sync) to be ``IGNORE_INDEX`` or in ``[0, V)``.

    Example:
        V = 151936, tp = 2, rank 1, local_vocab_size = 75968 -> 75968
        global_vocab_size = 151669 with the same shard -> ValueError (rank 1 expects 75701)
    """
    vocab_start, num_classes = 0, local_vocab_size
    if vocab_parallel_group is not None:
        if global_vocab_size is None:
            raise ValueError(
                "global_vocab_size is required for vocab-parallel policy statistics"
            )
        tp_size = dist.get_world_size(vocab_parallel_group)
        shard_size = (global_vocab_size + tp_size - 1) // tp_size
        vocab_start = min(
            global_vocab_size, shard_size * dist.get_rank(vocab_parallel_group)
        )
        expected = min(global_vocab_size, vocab_start + shard_size) - vocab_start
        if local_vocab_size != expected or expected == 0:
            raise ValueError(
                f"expected a non-empty local vocab shard of {expected} for global vocab "
                f"size {global_vocab_size}, got {local_vocab_size}"
            )
        num_classes = global_vocab_size
    torch._assert_async(
        torch.all(
            (labels_T == IGNORE_INDEX) | ((labels_T >= 0) & (labels_T < num_classes))
        ),
        f"labels must be {IGNORE_INDEX} or in [0, {num_classes})",
    )
    return vocab_start
