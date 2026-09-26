# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.config import CompileConfig
from torchtitan.models.common.linear import Linear, RouterGateLinear


def _relative_error(actual: torch.Tensor, exact: torch.Tensor) -> float:
    return ((actual.double() - exact).norm() / exact.norm()).item()


def _bf16_layer(config, fp32_weight_grad: bool):
    torch.manual_seed(0)
    layer = config.build().bfloat16()
    # FSDP sets this on its unsharded parameters (enable_fp32_weight_grads).
    for param in layer.parameters():
        param.grad_dtype = torch.float32
    layer.fp32_weight_grad = fp32_weight_grad
    return layer


@pytest.mark.parametrize("num_linears,bias", [(1, False), (1, True), (2, False)])
def test_linear_fp32_weight_grad_is_exact_and_keeps_the_input_grad(num_linears, bias):
    config = Linear.Config(
        in_features=64, out_features=32, num_linears=num_linears, bias=bias
    )
    torch.manual_seed(1)
    x = torch.randn(3, 16, 64).bfloat16()
    grad_output = torch.randn(3, 16, num_linears, 32).bfloat16().squeeze(2)

    grads = {}
    for fp32_weight_grad in (False, True):
        layer = _bf16_layer(config, fp32_weight_grad)
        x_leaf = x.clone().requires_grad_()
        layer(x_leaf).backward(grad_output)
        grads[fp32_weight_grad] = (x_leaf.grad, layer.weight.grad)

    exact = (
        grad_output.reshape(48, -1).double().T @ x.reshape(48, 64).double()
    ).reshape(grads[True][1].shape)
    (bf16_dx, bf16_dw), (fp32_dx, fp32_dw) = grads[False], grads[True]
    assert fp32_dw.dtype == torch.float32
    assert torch.equal(fp32_dx, bf16_dx)
    assert _relative_error(fp32_dw, exact) < 1e-5
    # Without the flag, the weight gradient is rounded to bf16 before the upcast.
    assert _relative_error(bf16_dw, exact) > 1e-4


def test_router_gate_linear_keeps_fp32_weight_grad():
    config = RouterGateLinear.Config(in_features=64, out_features=8)
    torch.manual_seed(1)
    x = torch.randn(16, 64).bfloat16()
    grad_output = torch.randn(16, 8)

    errors = {}
    for fp32_weight_grad in (False, True):
        layer = _bf16_layer(config, fp32_weight_grad)
        layer(x).backward(grad_output)
        exact = grad_output.double().T @ x.double()
        errors[fp32_weight_grad] = _relative_error(layer.weight.grad, exact)
    assert errors[True] < 1e-5 < errors[False]


def test_mixed_precision_grad_float32_rejects_model_compile():
    from torchtitan.models.llama3.config_registry import llama3_debugmodel

    config = llama3_debugmodel()
    config.training.mixed_precision_grad = "float32"
    config.compile = CompileConfig(components=["model"])
    with pytest.raises(ValueError, match="mixed_precision_grad"):
        config.__post_init__()

    config.compile = CompileConfig(components=["loss"])
    config.__post_init__()
