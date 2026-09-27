# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.config import CompileConfig
from torchtitan.models.common.linear import GroupedLinear, Linear, RouterGateLinear


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


@pytest.mark.parametrize("num_linears", [1, 2])
def test_grouped_linear_fp32_weight_grad_is_exact_and_keeps_the_input_grad(
    num_linears, monkeypatch
):
    # CPU torch._grouped_mm can't write fp32 from bf16 inputs; emulate it with fp32 inputs,
    # which is exact since bf16 products are exact in fp32.
    grouped_mm = torch._grouped_mm

    def grouped_mm_with_fp32_output(mat_a, mat_b, offs=None, bias=None, out_dtype=None):
        if out_dtype == torch.float32 and mat_a.dtype == torch.bfloat16:
            return grouped_mm(mat_a.float(), mat_b.float(), offs=offs)
        return grouped_mm(mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype)

    monkeypatch.setattr(torch, "_grouped_mm", grouped_mm_with_fp32_output)
    config = GroupedLinear.Config(
        group_size=4, in_features=32, out_features=16, num_linears=num_linears
    )
    torch.manual_seed(1)
    # 4 experts over 40 routed rows, one of them empty
    offsets_E = torch.tensor([8, 8, 24, 40], dtype=torch.int32)
    x = torch.randn(40, 32).bfloat16()
    grad_output = torch.randn(40, num_linears, 16).bfloat16().squeeze(1)

    grads = {}
    for fp32_weight_grad in (False, True):
        torch.manual_seed(0)
        layer = config.build().bfloat16()
        torch.nn.init.normal_(layer.weight, std=0.02)
        layer.weight.grad_dtype = torch.float32
        layer.fp32_weight_grad = fp32_weight_grad
        x_leaf = x.clone().requires_grad_()
        layer(x_leaf, offsets_E).backward(grad_output)
        grads[fp32_weight_grad] = (x_leaf.grad, layer.weight.grad)

    grad_output_RO = grad_output.reshape(40, -1)
    starts = [0, *offsets_E.tolist()[:-1]]
    exact = torch.stack(
        [
            grad_output_RO[start:end].double().T @ x[start:end].double()
            for start, end in zip(starts, offsets_E.tolist())
        ]
    ).reshape(grads[True][1].shape)
    (bf16_dx, bf16_dw), (fp32_dx, fp32_dw) = grads[False], grads[True]
    assert fp32_dw.dtype == torch.float32
    assert torch.equal(fp32_dx, bf16_dx)
    assert _relative_error(fp32_dw, exact) < 1e-5
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
