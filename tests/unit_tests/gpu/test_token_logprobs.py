# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from torchtitan.models.common.linear import Fp32OutputLinear
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


@pytest.mark.parametrize("loss_token_frac", [0.1, 0.0])
def test_chunked_loss_skips_non_loss_tokens(loss_token_frac):
    num_tokens, dim, vocab = 4096, 128, 5003
    hidden, weight, labels, loss_inputs = _inputs(num_tokens, dim, vocab)
    loss_inputs["loss_mask"] = (
        torch.arange(num_tokens, device="cuda") % 100 < 100 * loss_token_frac
    )
    global_valid_tokens = torch.tensor(float(num_tokens), device="cuda")
    lm_head = Fp32OutputLinear.Config(in_features=dim, out_features=vocab).build()
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
