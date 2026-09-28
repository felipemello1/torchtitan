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
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        (actual.double() - expected.double()).norm() / expected.double().norm()
    ).item()


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestTokenLogprobsDistributed(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

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


def test_compute_logprobs_strided_labels():
    from torchtitan.components.loss import compute_logprobs

    logits = torch.randn(256, 1000, device="cuda")
    token_ids = torch.randint(0, 1000, (256, 3), device="cuda")
    strided = compute_logprobs(logits, token_ids[:, 1], vocab_parallel_group=None)
    contiguous = compute_logprobs(
        logits, token_ids[:, 1].contiguous(), vocab_parallel_group=None
    )
    torch.testing.assert_close(strided, contiguous, atol=0, rtol=0)


def test_cross_entropy_loss_soft_targets_keep_f_cross_entropy():
    from torchtitan.components.loss import cross_entropy_loss

    # [T, V] class-probability targets are an F.cross_entropy feature, not a label index.
    logits = torch.randn(8, 8, device="cuda")
    targets = torch.randn(8, 8, device="cuda").softmax(-1)
    torch.testing.assert_close(
        cross_entropy_loss(logits, targets),
        F.cross_entropy(logits, targets, reduction="sum"),
    )
