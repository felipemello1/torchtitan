# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for ``Linear.Config.output_dtype``. CPU takes the upcast path."""

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


def test_default_output_dtype_is_input_dtype():
    linear = _bf16_linear()
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    out = linear(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, F.linear(x, linear.weight))


def test_float32_output_upcasts_off_cuda():
    linear = _bf16_linear(output_dtype="float32")
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    out = linear(x)

    assert out.dtype == torch.float32
    assert torch.equal(out, F.linear(x.float(), linear.weight.float()))
    # The stored weight keeps its dtype, so a tied embedding is unaffected.
    assert linear.weight.dtype == torch.bfloat16


def test_float32_output_preserves_stacked_shape():
    linear = _bf16_linear(output_dtype="float32", num_linears=2, bias=True)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    out = linear(x)

    expected = F.linear(
        x.float(), linear.weight.float().flatten(0, -2), linear.bias.float().flatten()
    ).unflatten(-1, linear.weight.shape[:-1])
    assert out.shape == (2, 4, 2, 16)
    assert torch.equal(out, expected)


def test_subclass_overriding_linear_rejects_output_dtype():
    with pytest.raises(ValueError, match="ignores output_dtype='float32'"):
        RouterGateLinear.Config(
            in_features=8, out_features=4, output_dtype="float32"
        ).build()


def test_lm_head_converter_sets_only_lm_head():
    from torchtitan.config.transform import LMHeadFp32OutputConverter
    from torchtitan.models.qwen3 import qwen3_configs

    build_config, max_context_length = qwen3_configs["0.6B"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)

    LMHeadFp32OutputConverter.Config().build().convert(config)

    changed = [
        fqn
        for fqn, linear_config, _, _ in config.traverse(Linear.Config)
        if linear_config.output_dtype != "input"
    ]
    assert changed == ["lm_head"]

    config.lm_head = None
    with pytest.raises(ValueError, match="lm_head"):
        LMHeadFp32OutputConverter.Config().build().convert(config)
