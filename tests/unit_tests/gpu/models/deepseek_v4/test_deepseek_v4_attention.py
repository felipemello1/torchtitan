# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks the DeepSeek-V4 attention cores are wired to ``gather_attn``
correctly (layouts, window, compressed positions, sink, scale) against dense
masked attention, on CPU (reference impl) and CUDA (fused impl). Kernel
numerics are tested in Attention Gym.
"""

from unittest.mock import patch

import pytest
import torch

from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4.attention import (
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    SlidingWindowAttention,
)
from torchtitan.models.deepseek_v4.compressor import Indexer, SparseIndexerLoss

SEQLEN, N_HEADS, HEAD_DIM, WINDOW, SCALE, TOPK = 32, 4, 8, 6, 0.1234, 5


def dense_attention(q, swa_k, cmp_k, cmp_allowed, attn_sink):
    """Softmax over [swa_k; cmp_k; sink] with a causal window over swa_k and
    ``cmp_allowed`` [T, S] over cmp_k."""
    t = torch.arange(SEQLEN).unsqueeze(1)
    window = (torch.arange(SEQLEN) <= t) & (t - torch.arange(SEQLEN) < WINDOW)
    sink = torch.ones(SEQLEN, 1, dtype=torch.bool)
    allowed = torch.cat([window, cmp_allowed, sink], dim=-1)

    kv = torch.cat([swa_k, cmp_k, swa_k.new_zeros(1, HEAD_DIM)])
    scores_HTS = torch.einsum("thd,sd->hts", q, kv) * SCALE
    scores_HTS[:, :, -1] = attn_sink.unsqueeze(-1)
    scores_HTS = scores_HTS.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hts,sd->thd", scores_HTS.softmax(dim=-1), kv)


def causal_cmp_allowed(ratio):
    """Compressed block s summarizes tokens [s * ratio, (s + 1) * ratio)."""
    t = torch.arange(SEQLEN).unsqueeze(1)
    block_end = (torch.arange(SEQLEN // ratio) + 1) * ratio - 1
    return block_end <= t


# HCA runs at ratio 8 instead of 128 so SEQLEN 32 has compressed blocks; the
# all-causal-blocks pattern is the same.
@pytest.mark.parametrize(
    "cls, ratio",
    [
        (SlidingWindowAttention, 1),
        (HeavilyCompressedAttention, 8),
        (CompressedSparseAttention, 4),
    ],
    ids=["swa", "hca", "csa"],
)
@pytest.mark.parametrize(
    "device, dtype, tol",
    [
        pytest.param("cpu", torch.float64, {}, id="cpu"),
        pytest.param(
            "cuda",
            torch.float32,
            {"atol": 1e-3, "rtol": 1e-2},
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_matches_dense_masked_attention(cls, ratio, device, dtype, tol):
    torch.manual_seed(0)
    module = cls(
        cls.Config(
            window_size=WINDOW,
            compress_ratio=ratio,
            softmax_scale=SCALE,
            index_topk=TOPK,
        )
    )
    n_cmp = 0 if ratio == 1 else SEQLEN // ratio
    # fp64 CPU copies feed the oracle; the module runs on `device` in `dtype`.
    ref_inputs = [
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in (
            (SEQLEN, N_HEADS, HEAD_DIM),
            (SEQLEN, HEAD_DIM),
            (n_cmp, HEAD_DIM),
            (N_HEADS,),
        )
    ]
    if ratio == 1:
        ref_inputs[2].requires_grad_(False)
    inputs = [
        t.detach().to(device, dtype).requires_grad_(t.requires_grad) for t in ref_inputs
    ]
    q, swa_k, cmp_k, attn_sink = inputs

    match module:
        case SlidingWindowAttention():
            out = module(q, swa_k, attn_sink)
            cmp_allowed = torch.zeros(SEQLEN, 0, dtype=torch.bool)
        case HeavilyCompressedAttention():
            out = module(q, swa_k, cmp_k, attn_sink)
            cmp_allowed = causal_cmp_allowed(ratio)
        case CompressedSparseAttention():
            idx_q, idx_k, idx_w = (
                torch.randn(*shape)
                for shape in ((SEQLEN, 3, 6), (n_cmp, 6), (SEQLEN, 3))
            )
            idx = [t.to(device) for t in (idx_q, idx_k, idx_w)]
            out = module(q, swa_k, cmp_k, *idx, attn_sink)
            cmp_topk, _ = Indexer.select(
                idx_q,
                idx_k,
                idx_w,
                max_seqlen=SEQLEN,
                ratio=ratio,
                topk=TOPK,
                return_scores=False,
            )
            cmp_topk = cmp_topk.long()
            cmp_allowed = torch.zeros(SEQLEN, n_cmp, dtype=torch.bool)
            rows = torch.arange(SEQLEN).unsqueeze(1).expand_as(cmp_topk)
            cmp_allowed[rows[cmp_topk >= 0], cmp_topk[cmp_topk >= 0]] = True
            cmp_allowed &= causal_cmp_allowed(ratio)

    ref = dense_attention(*ref_inputs[:3], cmp_allowed, ref_inputs[3])
    diff_inputs = [t for t in inputs if t.requires_grad]
    diff_ref_inputs = [t for t in ref_inputs if t.requires_grad]
    grads = torch.autograd.grad(out.sum(), diff_inputs)
    ref_grads = torch.autograd.grad(ref.sum(), diff_ref_inputs)

    torch.testing.assert_close(out.cpu().double(), ref, **tol)
    for grad, ref_grad in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(grad.cpu().double(), ref_grad, **tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_indexer_select_score_grads_cuda():
    """The indexer loss's score grads on CUDA (bf16) vs fp32 on the same selection.
    One 1024-token document at top-k 256: each key is picked ~500 times."""
    seqlen, ratio, topk = 1024, 4, 256
    g = torch.Generator().manual_seed(0)
    idx_q = torch.randn(seqlen, 64, 128, generator=g)
    idx_k = torch.randn(seqlen // ratio, 128, generator=g)
    idx_w = 0.01 * torch.randn(seqlen, 64, generator=g)
    inputs = [t.cuda().bfloat16().requires_grad_() for t in (idx_q, idx_k, idx_w)]
    selected, scores = Indexer.select(
        *inputs, max_seqlen=seqlen, ratio=ratio, topk=topk
    )
    grad = torch.randn(selected.shape, generator=g).cuda()
    grads = torch.autograd.grad(scores, inputs, grad.bfloat16())

    ref_inputs = [t.detach().float().requires_grad_() for t in inputs]
    q, k, w = ref_inputs
    ref_scores = (torch.einsum("thd,sd->tsh", q, k).relu() * w.unsqueeze(1)).sum(-1)
    expected = ref_scores.gather(1, selected.long().clamp_min(0))
    valid = selected >= 0
    ref_grads = torch.autograd.grad(expected[valid], ref_inputs, grad[valid])
    # GB300: 2.9e-3 / 3.3e-3 / 3.3e-3. Indexing's backward, which rounds each
    # pick to bf16, gives 3.1e-2 for idx_k. This 256-row pool runs embedding's
    # atomic backward, 1x4096's 1024 rows the two-pass one: both sum in fp32.
    for actual, ref in zip(grads, ref_grads):
        assert (actual.float() - ref).norm() / ref.norm() < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_indexer_loss_teacher_bf16_matches_fp32_cuda():
    """The teacher on bf16 CUDA inputs (bf16 GEMM, fp32 output) vs the same values
    in fp32, on a causal top-k selection with -1 slots."""
    seqlen, heads, dim, ratio, topk = 512, 8, 64, 4, 32
    g = torch.Generator().manual_seed(0)
    q = torch.randn(seqlen, heads, dim, generator=g).cuda().bfloat16()
    cmp_k = torch.randn(seqlen // ratio, dim, generator=g).cuda().bfloat16()
    lse = torch.randn(seqlen, heads, generator=g).cuda() + 4.0
    slots = torch.arange(topk, device="cuda")
    limit = (torch.arange(seqlen, device="cuda") + 1) // ratio
    indices = torch.where(slots < limit[:, None], slots, -1).int()
    with patch.dict(AuxLoss._group_counts, clear=True):
        loss = SparseIndexerLoss.Config(
            coeff=0.01, softmax_scale=dim**-0.5, num_heads=heads
        ).build()
    with patch("torch.bmm", wraps=torch.bmm) as bmm:
        actual = loss._teacher(q, cmp_k, indices, lse, None)
    assert bmm.called  # the bf16 GEMM path ran
    expected = loss._teacher(q.float(), cmp_k.float(), indices, lse, None)
    # GB300: max abs diff 4.8e-7. Logits rounded to bf16 give 1.4e-2 (CPU check).
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
