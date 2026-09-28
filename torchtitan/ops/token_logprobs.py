# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sampled-token logprobs and entropy from ``[T, V]`` logits, reading the logits once.

``compute_logprobs`` and eager ``cross_entropy_loss`` use it: one Triton pass for the softmax
statistics forward and one for the gradient backward, instead of ~10 passes over ``[T, V]``
(log_softmax forward and backward plus a separate entropy softmax).
"""

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # CPU-only installs (e.g. torch's CPU wheels); callers keep the torch path.
    _HAS_TRITON = False

# Shape suffix legend:
#   T = num tokens, V = local vocab size

IGNORE_INDEX = -100

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


class TokenLogprobsFromLogits(torch.autograd.Function):
    """``(logits [T, V], labels [T]) -> (logprobs [T], entropy [T])``, reading the logits once.

    One stats pass forward, and a backward that writes ``g * (one_hot - softmax)`` in one pass,
    instead of log_softmax's forward and backward plus an entropy softmax (~10 passes).

    Example:
        logits [8192, 124160] bf16 -> logprobs, entropy [8192] fp32; backward: dlogits bf16
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
