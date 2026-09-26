# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for ``Linear.Config.matmul_mode``."""

import pytest
import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear, RouterGateLinear


def _bf16_linear(**kwargs) -> Linear:
    return (
        Linear.Config(in_features=8, out_features=16, **kwargs)
        .build()
        .to(torch.bfloat16)
    )


def test_default_mode_keeps_input_dtype():
    linear = _bf16_linear()
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    out = linear(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, F.linear(x, linear.weight))


def test_upcast_mode_returns_fp32():
    linear = _bf16_linear(matmul_mode="upcast_fp32_matmul")
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    out = linear(x)

    assert out.dtype == torch.float32
    assert torch.equal(out, F.linear(x.float(), linear.weight.float()))
    # The stored weight keeps its dtype, so a tied embedding is unaffected.
    assert linear.weight.dtype == torch.bfloat16


def test_upcast_mode_preserves_stacked_shape():
    linear = _bf16_linear(matmul_mode="upcast_fp32_matmul", num_linears=2, bias=True)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    out = linear(x)

    expected = F.linear(
        x.float(), linear.weight.float().flatten(0, -2), linear.bias.float().flatten()
    ).unflatten(-1, linear.weight.shape[:-1])
    assert out.shape == (2, 4, 2, 16)
    assert torch.equal(out, expected)


def test_bf16_matmul_fp32_out_rejects_cpu_operands():
    linear = _bf16_linear(matmul_mode="bf16_matmul_fp32_out")
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="needs bf16 CUDA operands"):
        linear(x)


def test_subclass_overriding_linear_rejects_matmul_mode():
    with pytest.raises(ValueError, match="ignores matmul_mode='upcast_fp32_matmul'"):
        RouterGateLinear.Config(
            in_features=8, out_features=4, matmul_mode="upcast_fp32_matmul"
        ).build()


def test_lm_head_converter_sets_only_lm_head():
    from torchtitan.config.transform import LMHeadFp32OutputConverter
    from torchtitan.models.qwen3 import qwen3_configs

    build_config, max_context_length = qwen3_configs["0.6B"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)

    for matmul_mode in ("bf16_matmul_fp32_out", "upcast_fp32_matmul"):
        LMHeadFp32OutputConverter.Config(matmul_mode=matmul_mode).build().convert(
            config
        )

        changed = [
            (fqn, linear_config.matmul_mode)
            for fqn, linear_config, _, _ in config.traverse(Linear.Config)
            if linear_config.matmul_mode != "default"
        ]
        assert changed == [("lm_head", matmul_mode)]

    config.lm_head = None
    with pytest.raises(ValueError, match="lm_head"):
        LMHeadFp32OutputConverter.Config().build().convert(config)
