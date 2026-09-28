# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import textwrap
import unittest

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.linear import Fp32OutputLinear, Linear
from torchtitan.ops.token_logprobs import TokenLogprobs, TokenLogprobsGradState
from torchtitan.rl.losses.dapo import DAPOLoss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        (actual.double() - expected.double()).norm() / expected.double().norm()
    ).item()


def _inputs(num_tokens, dim, vocab, device="cuda", seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(num_tokens, dim, device=device, generator=generator)
    weight = (
        torch.randn(vocab, dim, device=device, generator=generator) * 3 / dim**0.5
    )
    labels = torch.randint(0, vocab, (num_tokens,), device=device, generator=generator)
    labels[::7] = -100
    loss_inputs = {
        "generator_logprobs": -torch.rand(
            num_tokens, device=device, generator=generator
        ),
        "advantages": torch.randn(num_tokens, device=device, generator=generator),
        "loss_mask": torch.rand(num_tokens, device=device, generator=generator) < 0.8,
    }
    return hidden.bfloat16(), weight.bfloat16(), labels, loss_inputs


def test_token_logprobs_matches_fp64():
    hidden, weight, labels, _ = _inputs(num_tokens=256, dim=128, vocab=5003)
    grad_logprobs = torch.randn(256, device="cuda") * 1e-5
    hidden_ref = hidden.double().requires_grad_()
    weight_ref = weight.double().requires_grad_()
    logprobs_ref = -F.cross_entropy(
        hidden_ref @ weight_ref.T, labels, reduction="none", ignore_index=-100
    )
    (logprobs_ref * grad_logprobs).sum().backward()
    log_softmax_ref = (hidden.double() @ weight.double().T).log_softmax(-1)
    entropy_ref = -(log_softmax_ref.exp() * log_softmax_ref).sum(-1)

    hidden = hidden.clone().requires_grad_()
    weight = weight.clone().requires_grad_()
    logprobs, entropy = TokenLogprobs.apply(
        hidden, weight, labels, TokenLogprobsGradState(), True, 0, None
    )
    (logprobs * grad_logprobs).sum().backward()

    torch.testing.assert_close(logprobs.double(), logprobs_ref, atol=1e-5, rtol=0)
    torch.testing.assert_close(entropy.double(), entropy_ref, atol=1e-4, rtol=0)
    assert not entropy.requires_grad
    # bf16 rounding of exact gradients alone gives ~2e-3.
    assert _relative_error(hidden.grad, hidden_ref.grad) < 4e-3
    assert _relative_error(weight.grad, weight_ref.grad) < 4e-3


@pytest.mark.parametrize("loss_config", [DAPOLoss.Config(), CrossEntropyLoss.Config()])
def test_chunked_loss_token_logprobs_matches_logits_path(loss_config):
    num_tokens, dim, vocab = 512, 128, 5003
    hidden, weight, labels, loss_inputs = _inputs(num_tokens, dim, vocab)
    if isinstance(loss_config, CrossEntropyLoss.Config):
        loss_inputs = {}
    global_valid_tokens = torch.tensor(float(num_tokens), device="cuda")
    lm_head = Fp32OutputLinear.Config(in_features=dim, out_features=vocab).build()
    lm_head = lm_head.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        lm_head.weight.copy_(weight)

    results = []
    for use_token_logprobs in (False, True):
        wrapper = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(num_chunks=4, loss_fn=loss_config)
        )
        wrapper.set_lm_head(lm_head)
        assert wrapper._uses_token_logprobs(hidden, is_multi_output=False)
        if not use_token_logprobs:
            wrapper._uses_token_logprobs = lambda *args: False
        lm_head.weight.grad = None
        hidden_input = hidden.clone().requires_grad_()
        loss, metrics = wrapper(
            hidden_input, labels, global_valid_tokens, **loss_inputs
        )
        loss.backward()
        results.append((loss, metrics, hidden_input.grad, lm_head.weight.grad))

    (loss, metrics, grad_hidden, grad_weight), fused = results
    torch.testing.assert_close(fused[0], loss, atol=0, rtol=1e-6)
    assert fused[1].keys() == metrics.keys()
    for key, value in metrics.items():
        torch.testing.assert_close(fused[1][key], value, atol=1e-6, rtol=1e-5)
    assert _relative_error(fused[2], grad_hidden) < 4e-3
    # The logits path adds four bf16 chunk gradients in bf16; token_logprobs rounds once.
    assert _relative_error(fused[3], grad_weight) < 6e-3


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestTokenLogprobsDistributed(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_fsdp_and_vocab_parallel_match_logits_path(self):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=4,
            enable_sequence_parallel=False,
        )
        dp_mesh = parallel_dims.get_mesh("dp_shard")
        tp_mesh = parallel_dims.get_mesh("tp")
        num_tokens, dim, vocab = 512, 128, 1001  # uneven vocab shards: 501 + 500
        _, weight, _, _ = _inputs(num_tokens, dim, vocab, seed=0)
        hidden, _, labels, loss_inputs = _inputs(
            num_tokens, dim, vocab, seed=1 + dp_mesh.get_local_rank()
        )
        shard_size = (vocab + 1) // 2
        vocab_start = shard_size * tp_mesh.get_local_rank()
        local_weight = weight[vocab_start : vocab_start + shard_size]
        global_valid_tokens = torch.tensor(float(2 * num_tokens), device="cuda")

        results = []
        for use_token_logprobs in (False, True):
            lm_head = Fp32OutputLinear.Config(
                in_features=dim, out_features=local_weight.shape[0]
            ).build()
            lm_head = lm_head.to(device="cuda")
            with torch.no_grad():
                lm_head.weight.copy_(local_weight)
            fully_shard(
                lm_head,
                mesh=dp_mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
            )
            wrapper = ChunkedLossWrapper(
                ChunkedLossWrapper.Config(
                    num_chunks=4, loss_fn=DAPOLoss.Config(global_vocab_size=vocab)
                )
            )
            wrapper.set_lm_head(lm_head)
            if not use_token_logprobs:
                wrapper._uses_token_logprobs = lambda *args: False
            # Two microbatches: gradients accumulate across wrapper calls.
            for _ in range(2):
                hidden_input = hidden.clone().requires_grad_()
                with set_current_spmd_mesh(parallel_dims.spmd_dense_mesh()):
                    loss, metrics = wrapper(
                        hidden_input, labels, global_valid_tokens, **loss_inputs
                    )
                loss.backward()
            results.append(
                (loss, metrics, hidden_input.grad, lm_head.weight.grad.full_tensor())
            )

        (loss, metrics, grad_hidden, grad_weight), fused = results
        torch.testing.assert_close(fused[0], loss, atol=0, rtol=1e-6)
        for key, value in metrics.items():
            torch.testing.assert_close(fused[1][key], value, atol=1e-6, rtol=1e-5)
        assert _relative_error(fused[2], grad_hidden) < 4e-3
        assert _relative_error(fused[3], grad_weight) < 4e-3

    @with_comms
    def test_vocab_shard_mismatch_raises(self):
        from torchtitan.components.loss import compute_logprobs

        tp_mesh = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=4,
            pp=1,
            ep=1,
            world_size=4,
            enable_sequence_parallel=False,
        ).get_mesh("tp")
        # 1000 classes over 4 ranks: 250 rows each. A wrong global size shifts the shards.
        logits = torch.randn(8, 250, device="cuda")
        labels = torch.zeros(8, dtype=torch.long, device="cuda")
        with self.assertRaisesRegex(ValueError, "local vocab shard"):
            compute_logprobs(
                logits,
                labels,
                vocab_parallel_group=tp_mesh.get_group(),
                global_vocab_size=990,
            )


# The fused path (Fp32OutputLinear) and the logits path (a plain bf16 Linear) both skip.
@pytest.mark.parametrize("head_cls", [Fp32OutputLinear, Linear])
@pytest.mark.parametrize("loss_token_frac", [0.1, 0.0])
def test_chunked_loss_skips_non_loss_tokens(loss_token_frac, head_cls):
    num_tokens, dim, vocab = 4096, 128, 5003
    hidden, weight, labels, loss_inputs = _inputs(num_tokens, dim, vocab)
    loss_inputs["loss_mask"] = (
        torch.arange(num_tokens, device="cuda") % 100 < 100 * loss_token_frac
    )
    global_valid_tokens = torch.tensor(float(num_tokens), device="cuda")
    lm_head = head_cls.Config(in_features=dim, out_features=vocab).build()
    lm_head = lm_head.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        lm_head.weight.copy_(weight)

    results = []
    for skip in (False, True):
        wrapper = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(num_chunks=4, loss_fn=DAPOLoss.Config())
        )
        wrapper.set_lm_head(lm_head)
        if skip:
            assert wrapper._loss_token_indices(labels, loss_inputs) is not None
        else:
            wrapper._loss_token_indices = lambda *args: None
        lm_head.weight.grad = None
        hidden_input = hidden.clone().requires_grad_()
        loss, metrics = wrapper(
            hidden_input, labels, global_valid_tokens, **loss_inputs
        )
        loss.backward()
        results.append((loss, metrics, hidden_input.grad, lm_head.weight.grad))

    (loss, metrics, grad_hidden, grad_weight), skipped = results
    torch.testing.assert_close(skipped[0], loss, atol=1e-8, rtol=1e-5)
    for key, value in metrics.items():
        torch.testing.assert_close(skipped[1][key], value, atol=1e-7, rtol=1e-5)
    assert torch.all(skipped[2][~loss_inputs["loss_mask"]] == 0)
    torch.testing.assert_close(skipped[2], grad_hidden, atol=1e-6, rtol=1e-2)
    torch.testing.assert_close(skipped[3], grad_weight, atol=1e-6, rtol=1e-2)


@pytest.mark.parametrize("head_cls", [Fp32OutputLinear, Linear])
def test_cross_entropy_skip_ignored_tokens(head_cls):
    # SFT: prompt tokens labeled IGNORE_INDEX skip the lm_head with skip_ignored_tokens.
    num_tokens, dim, vocab = 4096, 128, 5003
    hidden, weight, labels, _ = _inputs(num_tokens, dim, vocab)
    labels[torch.arange(num_tokens, device="cuda") % 512 < 300] = -100
    global_valid_tokens = (labels != -100).sum().float()
    lm_head = head_cls.Config(in_features=dim, out_features=vocab).build()
    lm_head = lm_head.to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        lm_head.weight.copy_(weight)

    results = []
    for skip in (False, True):
        loss_config = CrossEntropyLoss.Config(skip_ignored_tokens=skip)
        wrapper = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(num_chunks=4, loss_fn=loss_config)
        )
        wrapper.set_lm_head(lm_head)
        assert (wrapper._loss_token_indices(labels, {}) is not None) == skip
        lm_head.weight.grad = None
        hidden_input = hidden.clone().requires_grad_()
        loss, _ = wrapper(hidden_input, labels, global_valid_tokens)
        loss.backward()
        results.append((loss, hidden_input.grad, lm_head.weight.grad))

    (loss, grad_hidden, grad_weight), skipped = results
    torch.testing.assert_close(skipped[0], loss, atol=1e-8, rtol=1e-5)
    assert torch.all(skipped[1][labels == -100] == 0)
    # A bf16 head accumulates dW in bf16 chunk by chunk; fewer chunks round differently.
    assert _relative_error(skipped[1], grad_hidden) < 5e-3
    assert _relative_error(skipped[2], grad_weight) < 5e-3


def test_token_logprobs_returns_fp32_weight_grad_for_fp32_grad_dtype():
    hidden, weight, labels, _ = _inputs(num_tokens=512, dim=128, vocab=5003)
    grad_logprobs = torch.randn(512, device="cuda") * 1e-5
    weight_ref = weight.double().requires_grad_()
    logprobs_ref = -F.cross_entropy(
        hidden.double() @ weight_ref.T, labels, reduction="none", ignore_index=-100
    )
    (logprobs_ref * grad_logprobs).sum().backward()

    errors = {}
    for grad_dtype in (torch.bfloat16, torch.float32):
        weight_param = weight.clone().requires_grad_()
        weight_param.grad_dtype = grad_dtype
        grad_state = TokenLogprobsGradState()
        for chunk, is_last in ((slice(0, 256), False), (slice(256, 512), True)):
            logprobs, _ = TokenLogprobs.apply(
                hidden[chunk], weight_param, labels[chunk], grad_state, is_last, 0, None
            )
            (logprobs * grad_logprobs[chunk]).sum().backward()
        assert weight_param.grad.dtype is grad_dtype
        errors[grad_dtype] = _relative_error(weight_param.grad, weight_ref.grad)
    # Skipping the final bf16 rounding leaves only the fp16 dlogits error (~3e-4).
    assert errors[torch.float32] < 1e-3 < errors[torch.bfloat16]


def test_chunked_loss_rejects_out_of_range_labels():
    # A device-side assert poisons the CUDA context, so run it in a subprocess.
    code = textwrap.dedent(
        """
        import torch
        from torchtitan.components.loss import ChunkedLossWrapper
        from torchtitan.models.common.linear import Fp32OutputLinear

        head = Fp32OutputLinear.Config(in_features=64, out_features=100).build()
        head = head.to(device="cuda", dtype=torch.bfloat16)
        wrapper = ChunkedLossWrapper(ChunkedLossWrapper.Config(num_chunks=2))
        wrapper.set_lm_head(head)
        labels = torch.full((256,), 100, device="cuda")
        hidden = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
        loss, _ = wrapper(hidden.requires_grad_(), labels)
        torch.cuda.synchronize()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "labels must be" in result.stderr


def test_token_logprobs_frozen_weight_skips_grad_weight():
    hidden, weight, labels, _ = _inputs(num_tokens=256, dim=128, vocab=5003)
    grad_logprobs = torch.randn(256, device="cuda") * 1e-5
    grads = []
    for weight_requires_grad in (True, False):
        weight_param = weight.clone().requires_grad_(weight_requires_grad)
        hidden_input = hidden.clone().requires_grad_()
        grad_state = TokenLogprobsGradState()
        logprobs, _ = TokenLogprobs.apply(
            hidden_input, weight_param, labels, grad_state, True, 0, None
        )
        (logprobs * grad_logprobs).sum().backward()
        grads.append(hidden_input.grad)
        assert (weight_param.grad is None) != weight_requires_grad
        if not weight_requires_grad:
            assert grad_state.grad_weight is None
    torch.testing.assert_close(grads[1], grads[0], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compute_logprobs_from_logits_matches_fp64(dtype):
    from torchtitan.components.loss import compute_logprobs

    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(512, 5003, device="cuda", generator=generator) * 3
    logits = logits.to(dtype).requires_grad_()
    labels = torch.randint(0, 5003, (512,), device="cuda", generator=generator)
    labels[::7] = -100
    grad_logprobs = torch.randn(512, device="cuda", generator=generator)

    logprobs, entropy = compute_logprobs(
        logits, labels, vocab_parallel_group=None, return_entropy=True
    )
    (logprobs * grad_logprobs).sum().backward()

    logits_ref = logits.detach().double().requires_grad_()
    logprobs_ref = -F.cross_entropy(
        logits_ref, labels, reduction="none", ignore_index=-100
    )
    (logprobs_ref * grad_logprobs).sum().backward()
    log_softmax_ref = logits_ref.detach().log_softmax(-1)
    entropy_ref = -(log_softmax_ref.exp() * log_softmax_ref).sum(-1)

    assert logprobs.dtype is torch.float32 and not entropy.requires_grad
    torch.testing.assert_close(logprobs.double(), logprobs_ref, atol=1e-5, rtol=0)
    torch.testing.assert_close(entropy.double(), entropy_ref, atol=1e-4, rtol=0)
    assert logits.grad.dtype is dtype
    # bf16 logits get a bf16 gradient, like F.cross_entropy's backward.
    tolerance = 5e-3 if dtype is torch.bfloat16 else 1e-6
    assert _relative_error(logits.grad, logits_ref.grad) < tolerance


def test_confident_tokens_keep_accurate_logprob_and_gradient():
    # Logits near 100 with the label 18.4 above the rest: p(label) = 1 - ~1e-5. A single
    # logsumexp would round to ulp(100) ~ 8e-6 and swamp 1 - p.
    from torchtitan.ops.token_logprobs import TokenLogprobsFromLogits

    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = 100 + torch.randn(64, 1000, device="cuda", generator=generator) * 0.1
    labels = torch.randint(0, 1000, (64,), device="cuda", generator=generator)
    logits[torch.arange(64), labels] += 18.4
    logits.requires_grad_()
    logprobs, entropy = TokenLogprobsFromLogits.apply(logits, labels, 0, None)
    logprobs.sum().backward()

    logits_ref = logits.detach().double().requires_grad_()
    log_softmax_ref = logits_ref.log_softmax(-1)
    logprobs_ref = log_softmax_ref.gather(1, labels[:, None]).squeeze(1)
    logprobs_ref.sum().backward()
    entropy_ref = -(log_softmax_ref.exp() * log_softmax_ref).sum(-1).detach()
    torch.testing.assert_close(logprobs.double(), logprobs_ref, atol=1e-6, rtol=0)
    # Entropy ~2e-4 here; logsumexp - E[logits] would cancel at ulp(100).
    torch.testing.assert_close(entropy.double(), entropy_ref, atol=1e-6, rtol=0)
    label_grad = logits.grad.gather(1, labels[:, None]).double()
    label_grad_ref = logits_ref.grad.gather(1, labels[:, None])
    torch.testing.assert_close(label_grad, label_grad_ref, atol=0, rtol=5e-2)


@pytest.mark.parametrize("case", ["wide_grad_spread", "small_init_weight"])
def test_token_logprobs_gradient_dynamic_range(case):
    # The fp16 backward operands get power-of-two scales, so neither per-token gradients far
    # below the largest one nor a small-init weight fall into fp16 subnormals.
    hidden, weight, labels, _ = _inputs(num_tokens=1024, dim=128, vocab=5003)
    grad_logprobs = torch.full((1024,), 1e-6, device="cuda")
    grad_logprobs[0] = 1.0
    if case == "small_init_weight":
        grad_logprobs = torch.randn(1024, device="cuda") * 1e-5
        weight = (weight.float() * 1e-5).bfloat16()
    hidden_ref = hidden.double().requires_grad_()
    weight_ref = weight.double().requires_grad_()
    logprobs_ref = -F.cross_entropy(
        hidden_ref @ weight_ref.T, labels, reduction="none", ignore_index=-100
    )
    (logprobs_ref * grad_logprobs).sum().backward()

    hidden = hidden.clone().requires_grad_()
    weight = weight.clone().requires_grad_()
    weight.grad_dtype = torch.float32
    logprobs, _ = TokenLogprobs.apply(
        hidden, weight, labels, TokenLogprobsGradState(), True, 0, None
    )
    (logprobs * grad_logprobs).sum().backward()

    assert _relative_error(hidden.grad, hidden_ref.grad) < 4e-3
    # Per weight row, over rows holding at least 1% of the median row norm.
    row_norm = weight_ref.grad.norm(dim=1)
    rows = row_norm >= 1e-2 * row_norm.median()
    row_error = (weight.grad.double() - weight_ref.grad).norm(dim=1) / row_norm
    assert row_error[rows].max() < 1e-2


def test_token_logprobs_cuda_graph_matches_eager():
    # Under graph capture, grad_weight is unscaled on the device instead of by a host alpha.
    hidden, weight, labels, _ = _inputs(num_tokens=256, dim=128, vocab=5003)
    grad_logprobs = torch.randn(256, device="cuda") * 1e-5
    hidden = hidden.clone().requires_grad_()
    weight = weight.clone().requires_grad_()

    def step():
        grad_state = TokenLogprobsGradState()
        for chunk, is_last in ((slice(0, 128), False), (slice(128, 256), True)):
            logprobs, _ = TokenLogprobs.apply(
                hidden[chunk], weight, labels[chunk], grad_state, is_last, 0, None
            )
            (logprobs * grad_logprobs[chunk]).sum().backward()

    step()
    eager_grads = hidden.grad.clone(), weight.grad.clone()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        hidden.grad = weight.grad = None
        step()
    torch.cuda.current_stream().wait_stream(side_stream)
    hidden.grad = weight.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph.replay()
    torch.cuda.synchronize()

    assert _relative_error(hidden.grad, eager_grads[0]) < 1e-3
    assert _relative_error(weight.grad, eager_grads[1]) < 1e-3


def test_token_logprobs_fp32_weight_falls_back_to_fp32_matmuls():
    # e.g. an fp32 weight with bf16 activations under autocast: no dtype error, fp32 accuracy.
    hidden, weight, labels, _ = _inputs(num_tokens=256, dim=128, vocab=5003)
    grad_logprobs = torch.randn(256, device="cuda") * 1e-5
    weight_param = weight.float().requires_grad_()
    hidden_input = hidden.clone().requires_grad_()
    logprobs, _ = TokenLogprobs.apply(
        hidden_input, weight_param, labels, TokenLogprobsGradState(), True, 0, None
    )
    (logprobs * grad_logprobs).sum().backward()
    assert hidden_input.grad.dtype is torch.bfloat16
    assert weight_param.grad.dtype is torch.float32

    hidden_ref = hidden.double().requires_grad_()
    weight_ref = weight_param.detach().double().requires_grad_()
    logprobs_ref = -F.cross_entropy(
        hidden_ref @ weight_ref.T, labels, reduction="none", ignore_index=-100
    )
    (logprobs_ref * grad_logprobs).sum().backward()
    torch.testing.assert_close(logprobs.double(), logprobs_ref, atol=1e-5, rtol=0)
    assert _relative_error(weight_param.grad, weight_ref.grad) < 1e-5
    assert _relative_error(hidden_input.grad, hidden_ref.grad) < 4e-3


def test_compute_logprobs_strided_labels():
    from torchtitan.components.loss import compute_logprobs

    logits = torch.randn(256, 1000, device="cuda")
    token_ids = torch.randint(0, 1000, (256, 3), device="cuda")
    strided = compute_logprobs(logits, token_ids[:, 1], vocab_parallel_group=None)
    contiguous = compute_logprobs(
        logits, token_ids[:, 1].contiguous(), vocab_parallel_group=None
    )
    torch.testing.assert_close(strided, contiguous, atol=0, rtol=0)


def test_chunked_loss_keeps_logits_path_for_overridden_call():
    class _WeightedCrossEntropy(CrossEntropyLoss):
        def __call__(self, pred, labels, global_valid_tokens=None, **kwargs):
            loss, metrics = super().__call__(pred, labels, global_valid_tokens)
            return 2 * loss, metrics

    hidden, weight, _, _ = _inputs(num_tokens=64, dim=128, vocab=512)
    lm_head = Fp32OutputLinear.Config(in_features=128, out_features=512).build()
    lm_head = lm_head.to(device="cuda", dtype=torch.bfloat16)
    wrapper = ChunkedLossWrapper(
        ChunkedLossWrapper.Config(num_chunks=2, loss_fn=CrossEntropyLoss.Config())
    )
    wrapper.set_lm_head(lm_head)
    assert wrapper._uses_token_logprobs(hidden, is_multi_output=False)
    wrapper.loss_fn = _WeightedCrossEntropy(CrossEntropyLoss.Config())
    assert not wrapper._uses_token_logprobs(hidden, is_multi_output=False)


def test_cross_entropy_loss_soft_targets_keep_f_cross_entropy():
    from torchtitan.components.loss import cross_entropy_loss

    # [T, V] class-probability targets are an F.cross_entropy feature, not a label index.
    logits = torch.randn(8, 8, device="cuda")
    targets = torch.randn(8, 8, device="cuda").softmax(-1)
    torch.testing.assert_close(
        cross_entropy_loss(logits, targets),
        F.cross_entropy(logits, targets, reduction="sum"),
    )
