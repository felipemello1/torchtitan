# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU tests for ``Linear(matmul_mode="bf16_matmul_fp32_out")``."""

import pytest
import torch
import torch.nn.functional as F

from torchtitan.models.common import linear as linear_module
from torchtitan.models.common.linear import Linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _lm_head(in_features: int = 256, out_features: int = 1024) -> Linear:
    lm_head = Linear.Config(
        in_features=in_features,
        out_features=out_features,
        matmul_mode="bf16_matmul_fp32_out",
    ).build()
    torch.nn.init.normal_(lm_head.weight, std=0.02)
    return lm_head.to(device="cuda", dtype=torch.bfloat16)


def test_forward_matches_fp64_reference():
    lm_head = _lm_head()
    x = torch.randn(3, 5, 256, device="cuda", dtype=torch.bfloat16)
    reference = F.linear(x.double(), lm_head.weight.double())

    out = lm_head(x)

    assert out.dtype == torch.float32
    assert out.shape == reference.shape
    torch.testing.assert_close(out.double(), reference, rtol=1e-4, atol=1e-4)


def test_backward_error_stays_at_bf16_rounding_floor():
    lm_head = _lm_head()
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(64, 1024, device="cuda")

    x_exact = x.double().requires_grad_()
    weight_exact = lm_head.weight.detach().double().requires_grad_()
    F.linear(x_exact, weight_exact).backward(grad_output.double())

    x_leaf = x.clone().requires_grad_()
    weight = lm_head.weight.detach().clone().requires_grad_()
    linear_module._Fp32OutputLinearFunction.apply(x_leaf, weight).backward(grad_output)

    def relative_error(actual, exact):
        return ((actual.double() - exact).norm() / exact.norm()).item()

    # Gradients return in bf16, so rounding the exact gradients to bf16 is the floor.
    # Rounding grad_output to bf16 before the GEMM adds only a little on top of it.
    for grad, exact in ((x_leaf.grad, x_exact.grad), (weight.grad, weight_exact.grad)):
        assert grad.dtype == torch.bfloat16
        floor = relative_error(exact.bfloat16(), exact)
        assert relative_error(grad, exact) < 1.5 * floor


def test_batch_invariant_mode_upcasts(monkeypatch):
    lm_head = _lm_head()
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(linear_module, "is_in_batch_invariant_mode", lambda: True)

    def fail(*args):
        raise AssertionError("batch-invariant mode must not use the out_dtype GEMM")

    monkeypatch.setattr(linear_module._Fp32OutputLinearFunction, "apply", fail)

    out = lm_head(x)

    assert torch.equal(out, F.linear(x.float(), lm_head.weight.float()))
