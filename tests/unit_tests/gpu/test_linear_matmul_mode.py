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


def _relative_error(actual, exact):
    return ((actual.double() - exact).norm() / exact.norm()).item()


def _backward_errors_vs_bf16_floor(function, num_tokens, in_features, out_features):
    """Backward of ``function`` vs exact fp64 gradients, as multiples of the bf16 floor."""
    x = torch.randn(num_tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(out_features, in_features, device="cuda") * 0.02).bfloat16()
    grad_output = torch.randn(num_tokens, out_features, device="cuda")

    x_exact = x.double().requires_grad_()
    weight_exact = weight.double().requires_grad_()
    F.linear(x_exact, weight_exact).backward(grad_output.double())

    x_leaf = x.clone().requires_grad_()
    weight_leaf = weight.clone().requires_grad_()
    function(x_leaf, weight_leaf).backward(grad_output)

    # Gradients return in bf16, so rounding the exact gradients to bf16 is the floor.
    ratios = []
    for grad, exact in (
        (x_leaf.grad, x_exact.grad),
        (weight_leaf.grad, weight_exact.grad),
    ):
        assert grad.dtype == torch.bfloat16
        ratios.append(
            _relative_error(grad, exact) / _relative_error(exact.bfloat16(), exact)
        )
    return ratios


# out_features > num_tokens takes the LM-head layout, out_features < num_tokens the router one.
@pytest.mark.parametrize(
    "num_tokens,in_features,out_features", [(64, 256, 1024), (512, 256, 16)]
)
def test_backward_error_stays_at_bf16_rounding_floor(
    num_tokens, in_features, out_features
):
    ratios = _backward_errors_vs_bf16_floor(
        linear_module._Fp32OutputLinearFunction.apply,
        num_tokens,
        in_features,
        out_features,
    )
    assert max(ratios) < 1.05, ratios


def test_compiled_backward_keeps_lo_half():
    # torch.compile folds a bf16 round trip away inside fused kernels; if the split used one,
    # lo would compile to zero and the error would rise to that of a bf16 grad_output.
    ratios = _backward_errors_vs_bf16_floor(
        torch.compile(linear_module._Fp32OutputLinearFunction.apply), 64, 256, 1024
    )
    assert max(ratios) < 1.05, ratios


def test_batch_invariant_mode_upcasts(monkeypatch):
    lm_head = _lm_head()
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(linear_module, "is_in_batch_invariant_mode", lambda: True)

    def fail(*args):
        raise AssertionError("batch-invariant mode must not use the out_dtype GEMM")

    monkeypatch.setattr(linear_module._Fp32OutputLinearFunction, "apply", fail)

    out = lm_head(x)

    assert torch.equal(out, F.linear(x.float(), lm_head.weight.float()))
