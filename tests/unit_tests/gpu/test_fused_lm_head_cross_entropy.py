# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from dataclasses import dataclass, field

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchtitan.distributed.utils as dist_utils
from spmd_types._checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import (
    ChunkedLossWrapper,
    CrossEntropyLoss,
    IGNORE_INDEX,
    MSELoss,
)
from torchtitan.config import Configurable
from torchtitan.config.override import (
    apply_overrides,
    OverrideConfig,
    parse_cli_imports,
)
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common import linear as linear_module
from torchtitan.models.common.linear import (
    _FP32OutputLinearFunction,
    _split_into_bf16_pieces,
    FP32OutputLinear,
)
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan.overrides import fused_lm_head_cross_entropy as fused_module
from torchtitan.overrides.fused_lm_head_cross_entropy import (
    _cross_entropy_grad_pieces,
    _online_logsumexp,
    fp32_linear_cross_entropy,
    fused_lm_head_cross_entropy,
    FusedLMHeadCrossEntropyLoss,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# V = 1000 is not a multiple of the 128-entry vocab chunk alignment or of the kernels' blocks.
NUM_TOKENS, DIM, VOCAB = 96, 256, 1000
OVERRIDE = (
    "torchtitan.overrides.fused_lm_head_cross_entropy.fused_lm_head_cross_entropy"
)


def _inputs(seed=0, ignore_every=7):
    torch.manual_seed(seed)
    hidden = torch.randn(NUM_TOKENS, DIM, device="cuda").bfloat16()
    weight = (torch.randn(VOCAB, DIM, device="cuda") * 0.1).bfloat16()
    labels = torch.randint(0, VOCAB, (NUM_TOKENS,), device="cuda")
    labels[::ignore_every] = IGNORE_INDEX
    return hidden, weight, labels


def _lm_head(weight, *, bias=False, exact_grad_output_split=True):
    lm_head = FP32OutputLinear.Config(
        in_features=DIM,
        out_features=VOCAB,
        bias=bias,
        exact_grad_output_split=exact_grad_output_split,
    ).build()
    lm_head = lm_head.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        lm_head.weight.copy_(weight)
    lm_head.weight.grad_dtype = torch.float32
    return lm_head


def _grads(loss_fn, hidden, weight, labels):
    """(loss, dX, dW) of loss_fn / 37 (a scale that rounds); dW stays fp32 through grad_dtype."""
    hidden = hidden.detach().requires_grad_()
    weight = weight.detach().requires_grad_()
    weight.grad_dtype = torch.float32
    loss = loss_fn(hidden, weight, labels) / 37.0
    loss.backward()
    return loss.detach(), hidden.grad, weight.grad


def _fused(**kwargs):
    def loss_fn(hidden, weight, labels):
        return fp32_linear_cross_entropy(
            hidden, weight, labels, **({"num_pieces": 2} | kwargs)
        )

    return loss_fn


def _exact(hidden, weight, labels):
    return _grads(
        lambda hidden, weight, labels: F.cross_entropy(
            hidden @ weight.T, labels, reduction="sum"
        ),
        hidden.double(),
        weight.double(),
        labels,
    )


def _unfused(exact_split):
    def loss_fn(hidden, weight, labels):
        logits = _FP32OutputLinearFunction.apply(hidden, weight, exact_split)
        return F.cross_entropy(logits, labels, reduction="sum")

    return loss_fn


def _relative_error(actual, exact):
    return ((actual.double() - exact.double()).norm() / exact.double().norm()).item()


def _logsumexp_state(logits, labels, vocab_start=0, state=None):
    """Run _online_logsumexp on one vocab chunk; returns (max, others, label_logit)."""
    if state is None:
        num_tokens = logits.shape[0]
        state = (
            torch.full((num_tokens,), float("-inf"), device="cuda"),
            torch.zeros(num_tokens, device="cuda"),
            torch.zeros(num_tokens, device="cuda"),
        )
    max_T, others_T, label_logit_T = state
    _online_logsumexp(
        logits_TV=logits,
        labels_T=labels,
        vocab_start=vocab_start,
        max_T=max_T,
        others_T=others_T,
        label_logit_T=label_logit_T,
    )
    return state


def _pieces(logits, labels, scale, num_pieces):
    max_T, others_T, _ = _logsumexp_state(logits, labels)
    return _cross_entropy_grad_pieces(
        logits_TV=logits,
        max_T=max_T,
        log_sum_T=others_T.log1p(),
        labels_T=labels,
        grad_loss=torch.full((1,), scale, device="cuda"),
        vocab_start=0,
        num_pieces=num_pieces,
    ).view(num_pieces, *logits.shape)


# ======== The function: matches the unfused path, and is at least as accurate ========


@pytest.mark.parametrize("num_dim_chunks", [1, 2])
@pytest.mark.parametrize("recompute_logits", [False, True])
@pytest.mark.parametrize("num_vocab_chunks", [1, 3])
@pytest.mark.parametrize("num_pieces", [2, 3])
def test_matches_unfused_and_fp64(
    num_pieces, num_vocab_chunks, recompute_logits, num_dim_chunks
):
    hidden, weight, labels = _inputs()
    fused = _fused(
        num_pieces=num_pieces,
        num_vocab_chunks=num_vocab_chunks,
        recompute_logits=recompute_logits,
        num_dim_chunks=num_dim_chunks,
    )

    loss, dx, dw = _grads(fused, hidden, weight, labels)
    ref_loss, ref_dx, ref_dw = _grads(_unfused(num_pieces == 3), hidden, weight, labels)
    exact_loss, exact_dx, exact_dw = _exact(hidden, weight, labels)

    assert dx.dtype == torch.bfloat16 and dw.dtype == torch.float32
    torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=0)
    torch.testing.assert_close(dx, ref_dx, rtol=1.6e-2, atol=1e-5)
    torch.testing.assert_close(dw, ref_dw, rtol=1e-4, atol=1e-7)
    # At least as accurate as the unfused path (fp32 dW before any bf16 rounding).
    assert _relative_error(loss, exact_loss) < 1e-6
    assert _relative_error(dw, exact_dw) <= 1.1 * _relative_error(ref_dw, exact_dw)
    assert _relative_error(dx, exact_dx) <= 1.1 * _relative_error(ref_dx, exact_dx)


@pytest.mark.parametrize("num_dim_chunks", [1, 4])
def test_recomputed_logits_match_kept_logits(num_dim_chunks):
    # The backward's recomputed logits must be the forward's, or the softmax drifts (a backward
    # without the forward's num_dim_chunks was off by 5e-4 in dW on the Qwen3-8B head).
    hidden, weight, labels = _inputs()
    results = [
        _grads(
            _fused(
                num_vocab_chunks=3,
                recompute_logits=recompute,
                num_dim_chunks=num_dim_chunks,
            ),
            hidden,
            weight,
            labels,
        )
        for recompute in (False, True)
    ]
    # Same logsumexp, and logits that differ at most in cuBLAS's choice of kernel for a chunk.
    for kept, recomputed in zip(*results, strict=True):
        assert _relative_error(recomputed, kept) < 1e-5


def test_confident_token_gradient_stays_accurate():
    # p(label) = 1 - 3.4e-7: softmax - 1 cancels in fp32, and PyTorch's CE gradient is off by over
    # 1%. expm1/log1p keep it to fp32 precision.
    logits = torch.zeros(1, 4, device="cuda")
    logits[0, 0] = 16.0
    labels = torch.tensor([0], device="cuda")
    fused = _pieces(logits, labels, 1.0, num_pieces=3).float().sum(0)

    exact = torch.softmax(logits.double(), -1)
    exact[0, 0] -= 1.0
    torch_ce = logits.clone().requires_grad_()
    F.cross_entropy(torch_ce, labels, reduction="sum").backward()

    assert _relative_error(fused, exact) < 1e-6
    assert _relative_error(torch_ce.grad, exact) > 1e-2


@pytest.mark.parametrize("num_pieces", [2, 3])
def test_pieces_follow_fp32_output_linear_split(num_pieces):
    # The kernel's pieces are exactly _split_into_bf16_pieces of its fp32 gradient (the sum of the
    # 3 exact pieces). These gradients are all above 2^-110, the split's exact range.
    hidden, weight, labels = _inputs()
    logits = torch.mm(hidden, weight.T, out_dtype=torch.float32)
    grad = _pieces(logits, labels, 0.3, num_pieces=3).float().sum(0)

    pieces = _pieces(logits, labels, 0.3, num_pieces=num_pieces)

    expected = _split_into_bf16_pieces(grad, exact=num_pieces == 3)
    for piece, expected_piece in zip(pieces, expected, strict=True):
        assert torch.equal(piece, expected_piece)
    ignored = labels == IGNORE_INDEX
    assert torch.equal(grad[ignored], torch.zeros_like(grad[ignored]))


def test_logsumexp_skips_blocks_of_only_minus_inf():
    # Row 0: its first 4096-entry block is all -inf. Row 1: the whole first vocab chunk is -inf.
    torch.manual_seed(0)
    logits = torch.randn(2, 3 * 4096 + 5, device="cuda") * 3
    logits[0, :4096] = float("-inf")
    logits[1, :6000] = float("-inf")
    labels = torch.tensor([4100, 7000], device="cuda")
    state = _logsumexp_state(logits[:, :6000], labels)
    max_T, others_T, label_logit_T = _logsumexp_state(
        logits[:, 6000:], labels, vocab_start=6000, state=state
    )
    expected = torch.logsumexp(logits.double(), -1)
    torch.testing.assert_close(
        (max_T + others_T.log1p()).double(), expected, rtol=1e-6, atol=0
    )
    assert torch.equal(label_logit_T, logits[[0, 1], labels])


def test_third_piece_offset_past_int32():
    # T * V = 2^30: the third piece starts 2 * 2^30 past the first, which overflows int32.
    # Stride-0 logits, so only the 6 GiB pieces buffer is allocated.
    num_tokens, vocab = 8192, 131072
    required_bytes = 3 * num_tokens * vocab * 2 + 2**30
    if torch.cuda.mem_get_info()[0] < required_bytes:
        pytest.skip(f"needs {required_bytes} free CUDA bytes")
    torch.manual_seed(0)
    logits = torch.randn(1, vocab, device="cuda").expand(num_tokens, vocab)
    labels = torch.full((num_tokens,), 7, device="cuda")

    pieces = _pieces(logits, labels, 1.0, num_pieces=3)

    # Every token has the same logits and label, so its pieces equal a 1-token call's.
    expected = _pieces(logits[:1], labels[:1], 1.0, num_pieces=3)[:, 0]
    assert expected[2].abs().max() > 0
    for piece, expected_piece in zip(pieces, expected, strict=True):
        assert torch.equal(piece[0], expected_piece)
        assert torch.equal(piece[-1], expected_piece)


def test_grad_weight_split_over_tokens(monkeypatch):
    # 2 pieces x 96 tokens = 192 stacked rows; 64 per GEMM exercises the fp32-added split.
    hidden, weight, labels = _inputs()
    _, _, ref_dw = _grads(_fused(), hidden, weight, labels)
    monkeypatch.setattr(fused_module, "_GRAD_WEIGHT_ROWS_PER_GEMM", 64)
    _, _, dw = _grads(_fused(), hidden, weight, labels)
    exact_dw = _exact(hidden, weight, labels)[2]
    assert _relative_error(dw, ref_dw) < 1e-5
    assert _relative_error(dw, exact_dw) <= 1.1 * _relative_error(ref_dw, exact_dw)


def test_frozen_weight_skips_grad_weight():
    hidden, weight, labels = _inputs()
    hidden = hidden.detach().requires_grad_()
    _fused()(hidden, weight, labels).backward()
    assert hidden.grad is not None and weight.grad is None


def test_zero_tokens():
    _, weight, _ = _inputs()
    hidden = torch.empty(0, DIM, device="cuda", dtype=torch.bfloat16)
    labels = torch.empty(0, device="cuda", dtype=torch.long)
    loss, dx, dw = _grads(_fused(), hidden, weight, labels)
    assert loss.item() == 0.0 and dx.shape == (0, DIM)
    assert torch.equal(dw, torch.zeros_like(dw))


def test_compiled_caller_runs_it_eager():
    # torch.compile would fail to lower addmm(out_dtype=) at the default 8 vocab chunks
    # (pytorch/pytorch#190936); the compiler.disable makes a compiled caller graph-break instead.
    hidden, weight, labels = _inputs()

    def caller(hidden, weight, labels):
        return fp32_linear_cross_entropy(hidden, weight, labels, num_pieces=2) * 2.0

    expected = _grads(caller, hidden, weight, labels)
    torch._dynamo.reset()
    try:
        compiled = _grads(torch.compile(caller), hidden, weight, labels)
    finally:
        torch._dynamo.reset()
    for actual, eager in zip(compiled, expected, strict=True):
        assert torch.equal(actual, eager)


# ======== The ChunkedLossWrapper override ========


def test_wrapper_matches_chunked_loss_wrapper():
    hidden, weight, labels = _inputs()
    global_valid_tokens = (labels != IGNORE_INDEX).sum().float()
    results = []
    for config, expected_forward_calls in (
        (ChunkedLossWrapper.Config(num_chunks=4), 4),
        (FusedLMHeadCrossEntropyLoss.Config(num_chunks=4, num_vocab_chunks=3), 0),
    ):
        lm_head = _lm_head(weight)
        forward_calls = []
        lm_head.register_forward_hook(lambda *_: forward_calls.append(1))
        loss_fn = config.build()
        loss_fn.set_lm_head(lm_head)
        hidden_leaf = hidden.detach().requires_grad_()
        loss, _ = loss_fn(hidden_leaf, labels, global_valid_tokens)
        loss.backward()
        # The fused path never runs lm_head.forward, so it can't silently fall back.
        assert len(forward_calls) == expected_forward_calls
        results.append((loss.detach(), hidden_leaf.grad, lm_head.weight.grad))

    (ref_loss, ref_dx, ref_dw), (loss, dx, dw) = results
    torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=0)
    torch.testing.assert_close(dx, ref_dx, rtol=1.6e-2, atol=1e-5)
    torch.testing.assert_close(dw, ref_dw, rtol=1e-4, atol=1e-7)


def test_wrapper_rejects_unsupported_inputs():
    hidden, weight, labels = _inputs()
    for loss_config in (MSELoss.Config(), MTPLoss.Config()):
        with pytest.raises(ValueError, match="fuses CrossEntropyLoss"):
            FusedLMHeadCrossEntropyLoss.Config(
                num_chunks=2, loss_fn=loss_config
            ).build()
    loss_fn = FusedLMHeadCrossEntropyLoss.Config(num_chunks=2).build()
    for lm_head in (_lm_head(weight, bias=True), nn.Linear(DIM, VOCAB, bias=False)):
        with pytest.raises(ValueError, match="bias-free FP32OutputLinear"):
            loss_fn.set_lm_head(lm_head)
    loss_fn.set_lm_head(_lm_head(weight))
    hidden_leaf = hidden.detach().requires_grad_()
    with pytest.raises(ValueError, match="one prediction"):
        loss_fn((hidden_leaf, hidden_leaf), (labels, labels))
    with pytest.raises(ValueError, match="one prediction"):
        loss_fn(hidden_leaf, labels, loss_mask=torch.ones_like(labels))


def test_wrapper_falls_back_off_bf16():
    # FP32OutputLinear's fp32 fallback (here: fp32 weights) takes the unfused path.
    hidden, weight, labels = _inputs()
    global_valid_tokens = (labels != IGNORE_INDEX).sum().float()
    lm_head = (
        FP32OutputLinear.Config(in_features=DIM, out_features=VOCAB).build().cuda()
    )
    with torch.no_grad():
        lm_head.weight.copy_(weight)
    loss_fn = FusedLMHeadCrossEntropyLoss.Config(num_chunks=2).build()
    loss_fn.set_lm_head(lm_head)
    hidden_leaf = hidden.float().requires_grad_()
    loss, _ = loss_fn(hidden_leaf, labels, global_valid_tokens)
    loss.backward()
    expected = F.cross_entropy(
        hidden.double() @ weight.double().T, labels, reduction="sum"
    )
    torch.testing.assert_close(
        loss.double(), expected / global_valid_tokens.double(), rtol=1e-5, atol=0
    )


@pytest.mark.parametrize("exact_grad_output_split", [False, True])
def test_wrapper_passes_knobs_and_split(monkeypatch, exact_grad_output_split):
    # The config's knobs and the lm_head's split reach the function, for both piece counts
    # (LMHeadFP32OutputConverter turns exact_grad_output_split off: the 2-piece branch).
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return fp32_linear_cross_entropy(*args, **kwargs)

    monkeypatch.setattr(fused_module, "fp32_linear_cross_entropy", spy)
    hidden, weight, labels = _inputs()
    knobs = {"num_vocab_chunks": 3, "recompute_logits": True, "num_dim_chunks": 2}
    loss_fn = FusedLMHeadCrossEntropyLoss.Config(num_chunks=2, **knobs).build()
    loss_fn.set_lm_head(
        _lm_head(weight, exact_grad_output_split=exact_grad_output_split)
    )
    loss_fn(hidden.detach().requires_grad_(), labels)
    num_pieces = 3 if exact_grad_output_split else 2
    assert calls == [{"num_pieces": num_pieces, **knobs}] * 2


@pytest.mark.parametrize("unsupported", ["rocm", "batch_invariant"])
def test_wrapper_falls_back_on_rocm_and_batch_invariant(monkeypatch, unsupported):
    if unsupported == "rocm":
        monkeypatch.setattr(torch.version, "hip", "7.0")
    else:
        monkeypatch.setattr(dist_utils, "_batch_invariant_enabled", True)
    hidden, weight, labels = _inputs()
    lm_head = _lm_head(weight)
    forward_calls = []
    lm_head.register_forward_hook(lambda *_: forward_calls.append(1))
    loss_fn = FusedLMHeadCrossEntropyLoss.Config(num_chunks=2).build()
    loss_fn.set_lm_head(lm_head)
    # Forward only: the unfused backward would compile the split for ROCm.
    loss_fn(hidden.detach(), labels)
    assert len(forward_calls) == 2


def test_wrapper_raises_under_tp(monkeypatch):
    monkeypatch.setattr(
        fused_module, "spmd_mesh_size", lambda axis: 2 if axis == "tp" else 1
    )
    hidden, weight, labels = _inputs()
    loss_fn = FusedLMHeadCrossEntropyLoss.Config(num_chunks=2).build()
    loss_fn.set_lm_head(_lm_head(weight))
    with pytest.raises(NotImplementedError, match="vocab-parallel"):
        loss_fn(hidden.detach().requires_grad_(), labels)


def _num_addmm_calls(fn) -> int:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        fn()
    return sum(e.count for e in prof.key_averages() if e.key == "aten::addmm")


def _expected_addmm_calls(num_chunks, vocab, num_vocab_chunks, *, accumulate):
    """grad_input's split-K adds per chunk, plus one in-GEMM .grad add per vocab chunk for every
    chunk after the first (one grad_weight GEMM per vocab chunk at these sizes)."""
    chunk_size = fused_module._vocab_chunk_size(vocab, num_vocab_chunks)
    vocab_chunks = -(-vocab // chunk_size)
    grad_weight_adds = (num_chunks - 1) * vocab_chunks if accumulate else 0
    return num_chunks * (vocab_chunks - 1) + grad_weight_adds


@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
def test_wrapper_adds_into_weight_grad_inside_the_gemm(monkeypatch, grad_dtype):
    # As FP32OutputLinear under accumulate_into_weight_grad: every chunk after the first adds its
    # grad_weight into weight.grad with addmm(out=), and .grad equals autograd's separate add
    # bitwise, at one row GEMM per vocab chunk (as 2048-token chunks with 2 pieces).
    hidden, weight, labels = _inputs()
    lm_head = _lm_head(weight)
    lm_head.weight.grad_dtype = grad_dtype
    loss_fn = FusedLMHeadCrossEntropyLoss.Config(
        num_chunks=4, num_vocab_chunks=3
    ).build()
    loss_fn.set_lm_head(lm_head)

    def step():
        loss, _ = loss_fn(hidden.detach().requires_grad_(), labels)
        loss.backward()

    num_addmm = _num_addmm_calls(step)
    in_gemm, lm_head.weight.grad = lm_head.weight.grad, None
    monkeypatch.setattr(
        linear_module, "accumulate_into_weight_grad", contextlib.nullcontext
    )
    num_addmm_off = _num_addmm_calls(step)

    assert num_addmm == _expected_addmm_calls(4, VOCAB, 3, accumulate=True)
    assert num_addmm_off == _expected_addmm_calls(4, VOCAB, 3, accumulate=False)
    assert torch.equal(in_gemm, lm_head.weight.grad)


@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
def test_weight_grad_with_several_row_gemms(monkeypatch, grad_dtype):
    # 48 stacked rows per chunk in 3 row GEMMs: a bf16 .grad must be rounded once per chunk, as
    # autograd's add does, not after every row GEMM.
    monkeypatch.setattr(fused_module, "_GRAD_WEIGHT_ROWS_PER_GEMM", 16)
    hidden, weight, labels = _inputs()

    def weight_grad():
        lm_head = _lm_head(weight, exact_grad_output_split=False)
        lm_head.weight.grad_dtype = grad_dtype
        loss_fn = FusedLMHeadCrossEntropyLoss.Config(
            num_chunks=4, num_vocab_chunks=3
        ).build()
        loss_fn.set_lm_head(lm_head)
        loss, _ = loss_fn(hidden.detach().requires_grad_(), labels)
        loss.backward()
        return lm_head.weight.grad

    in_gemm = weight_grad()
    monkeypatch.setattr(
        linear_module, "accumulate_into_weight_grad", contextlib.nullcontext
    )
    separate = weight_grad()
    if grad_dtype == torch.float32:
        torch.testing.assert_close(in_gemm, separate, rtol=1e-6, atol=1e-6)
    else:
        assert torch.equal(in_gemm, separate)


@pytest.mark.parametrize("accumulate", [False, True])
def test_weight_grad_accumulation_replays_in_a_cuda_graph(monkeypatch, accumulate):
    # torchtitan captures forward + backward in a CUDA graph and zeroes .grad in place between
    # steps. Replays must keep adding into that buffer, even while something else holds it.
    if not accumulate:
        monkeypatch.setattr(
            linear_module, "accumulate_into_weight_grad", contextlib.nullcontext
        )
    hidden, weight, labels = _inputs()
    lm_head = _lm_head(weight)
    loss_fn = FusedLMHeadCrossEntropyLoss.Config(num_chunks=3).build()
    loss_fn.set_lm_head(lm_head)
    hidden_leaf = hidden.detach().requires_grad_()

    def step():
        loss, _ = loss_fn(hidden_leaf, labels)
        loss.backward()

    step()
    expected = lm_head.weight.grad.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        lm_head.weight.grad.zero_()
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    held_grad = lm_head.weight.grad
    held_grad.zero_()
    with torch.cuda.graph(graph):
        step()

    for _ in range(3):
        lm_head.weight.grad.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert lm_head.weight.grad is held_grad
        torch.testing.assert_close(held_grad, expected, rtol=0, atol=0)


class _Root(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        loss: ChunkedLossWrapper.Config = field(
            default_factory=lambda: ChunkedLossWrapper.Config(num_chunks=1)
        )

    def __init__(self, config: Config):
        self.config = config


def test_override_keeps_config_and_takes_knobs():
    cfg = ChunkedLossWrapper.Config(
        num_chunks=2, loss_fn=CrossEntropyLoss.Config(global_vocab_size=VOCAB)
    )
    replacement = fused_lm_head_cross_entropy(cfg, num_vocab_chunks=16)
    assert isinstance(replacement, FusedLMHeadCrossEntropyLoss.Config)
    assert replacement.num_chunks == 2 and replacement.loss_fn is cfg.loss_fn
    assert replacement.num_vocab_chunks == 16

    # The CLI route: --override.imports '<target>={"recompute_logits": true}'.
    root = _Root.Config()
    imports = parse_cli_imports([OVERRIDE + '={"recompute_logits": true}'])
    apply_overrides(OverrideConfig(imports=imports), root)
    assert isinstance(root.loss, FusedLMHeadCrossEntropyLoss.Config)
    assert root.loss.recompute_logits and root.loss.num_chunks == 1


# ======== Distributed: SPMD types, FSDP2 ========


class TestFusedLossSPMD(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    @with_comms
    def test_spmd_matches_eager_and_types(self):
        # As TestChunkedLossWrapperSPMD (CPU), on a (1, 1, 1) mesh: strict typechecking accepts the
        # fused path, and its loss and hidden-state gradients match the unfused wrapper's.
        hidden, weight, labels = _inputs()
        mesh = init_device_mesh("cuda", (1, 1, 1), mesh_dim_names=("dp", "cp", "tp"))
        tp_group = mesh.get_group("tp")
        # Count the Function's typecheck rule: 0 if its apply bypassed spmd_types' patched one.
        function = fused_module._FP32LinearCrossEntropyFunction
        rule = function.spmd_typecheck
        typecheck_calls = []

        def counted_rule(loss):
            typecheck_calls.append(1)
            rule(loss)

        self.addCleanup(setattr, function, "spmd_typecheck", staticmethod(rule))
        function.spmd_typecheck = staticmethod(counted_rule)
        results = []
        for config_cls in (
            ChunkedLossWrapper.Config,
            FusedLMHeadCrossEntropyLoss.Config,
        ):
            lm_head = _lm_head(weight)
            loss_fn = config_cls(
                num_chunks=2, loss_fn=CrossEntropyLoss.Config(global_vocab_size=VOCAB)
            ).build()
            loss_fn.set_lm_head(lm_head)
            hidden_leaf = hidden.detach().requires_grad_()
            with set_current_spmd_mesh(mesh):
                spmd.assert_type(hidden_leaf, {tp_group: spmd.R})
                spmd.assert_type(labels, {tp_group: spmd.I})
                spmd.assert_type(lm_head.weight, {tp_group: spmd.S(0)})
                with typecheck(strict_mode="strict", local=False):
                    loss, _ = loss_fn(hidden_leaf, labels)
            loss.backward()
            results.append((loss.detach(), hidden_leaf.grad))
        (ref_loss, ref_dx), (loss, dx) = results
        torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=0)
        torch.testing.assert_close(dx, ref_dx, rtol=1.6e-2, atol=1e-5)
        assert len(typecheck_calls) == 2


class _Decoder(nn.Module):
    """A body Linear and an FP32OutputLinear lm_head, sharded as the trainer does."""

    def __init__(self, dim, vocab):
        super().__init__()
        self.body = nn.Linear(dim, dim, bias=False)
        self.lm_head = FP32OutputLinear.Config(
            in_features=dim, out_features=vocab
        ).build()

    def forward(self, x):
        return self.body(x)


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
class TestFusedLossFSDP(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_fsdp2_matches_unsharded_and_overlaps(self):
        # Different tokens per rank. vs the unsharded fused loss, all-reduced:
        # - the lm_head gradient matches;
        # - lm_head.forward never runs;
        # - lm_head's gradient is reduce-scattered by the end of the loss call, before the
        #   decoder backward (what register_fsdp_forward_method buys);
        # - chunks after the first add into the unsharded weight.grad inside the GEMM.
        num_tokens, dim, vocab = 1024, 256, 5000
        torch.manual_seed(0)
        state = _Decoder(dim, vocab).state_dict()
        torch.manual_seed(100 + self.rank)
        x = torch.randn(num_tokens, dim, device="cuda").bfloat16()
        labels = torch.randint(0, vocab, (num_tokens,), device="cuda")
        labels[:: 5 + self.rank] = IGNORE_INDEX
        global_valid_tokens = (labels != IGNORE_INDEX).sum().float()
        dist.all_reduce(global_valid_tokens)
        grads = []
        for use_fsdp in (True, False):
            model = _Decoder(dim, vocab).cuda()
            model.load_state_dict(state)
            if use_fsdp:
                policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                )
                fully_shard(model.lm_head, mp_policy=policy)
                fully_shard(model, mp_policy=policy)
            else:
                model = model.bfloat16()
                model.lm_head.weight.grad_dtype = torch.float32
            forward_calls = []
            model.lm_head.register_forward_hook(lambda *_: forward_calls.append(1))
            loss_fn = FusedLMHeadCrossEntropyLoss.Config(num_chunks=4).build()
            loss_fn.set_lm_head(model.lm_head)
            hidden = model(x)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as prof:
                loss, _ = loss_fn(hidden, labels, global_valid_tokens)
            num_addmm = sum(
                e.count for e in prof.key_averages() if e.key == "aten::addmm"
            )
            assert num_addmm == _expected_addmm_calls(4, vocab, 8, accumulate=True)
            if use_fsdp:
                assert model.lm_head.weight.grad is not None
            loss.backward()
            assert not forward_calls
            grad = model.lm_head.weight.grad
            if use_fsdp:
                grad = grad.full_tensor()
            else:
                dist.all_reduce(grad, op=dist.ReduceOp.AVG)
            grads.append(grad.float())
        torch.testing.assert_close(grads[0], grads[1], rtol=1e-6, atol=0)
