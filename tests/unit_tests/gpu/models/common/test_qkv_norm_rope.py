# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _run(attention, qkv, positions, grads):
    """Inference outputs (the generator), then training outputs and the fused output's grad."""
    with torch.no_grad():
        inference = attention._prepare_qkv(qkv, positions)
    qkv = qkv.detach().clone().requires_grad_()
    outputs = attention._prepare_qkv(qkv, positions)
    torch.autograd.backward(outputs, [g.to(o.dtype) for g, o in zip(grads, outputs)])
    return [*inference, *outputs, qkv.grad]


# Qwen3: q/k norms + CosSinRoPE. Llama 3: no norms, ComplexRoPE.
@pytest.mark.parametrize("model", ["qwen3", "llama3"])
def test_qkv_norm_rope_region_matches_eager(model):
    if model == "qwen3":
        from torchtitan.models.qwen3 import build_model_config
    else:
        from torchtitan.models.llama3 import build_model_config
    torch.manual_seed(0)
    config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    attention = config.layers[0].attention.build().cuda()
    qkv_linear = attention.qkv_linear
    num_tokens = 64
    qkv = torch.randn(
        num_tokens,
        qkv_linear.wqkv.out_features // (qkv_linear.r_dim * qkv_linear.head_dim),
        qkv_linear.r_dim,
        qkv_linear.head_dim,
        device="cuda",
    )
    positions = torch.randint(0, 1024, (num_tokens,), device="cuda")
    shapes = [(num_tokens, attention.n_heads), (num_tokens, attention.n_kv_heads)]
    grads = [
        torch.randn(*shape, attention.head_dim, device="cuda")
        for shape in (shapes[0], shapes[1], shapes[1])
    ]

    try:
        LocalCompileConfig().apply()
        reference = _run(attention, qkv, positions, grads)
        attention.bfloat16()
        expected = _run(attention, qkv.bfloat16(), positions, grads)
        LocalCompileConfig(regions=["rope", "qkv_norm_rope"]).apply()
        actual = _run(attention, qkv.bfloat16(), positions, grads)
    finally:
        LocalCompileConfig().apply()

    # The compiled norm reduces in another order than the eager kernel, so a few
    # elements differ by a bf16 ulp. Check the region is as close as eager to an
    # fp32 run of the same math: q/k/v in inference and training, and the grad of
    # the fused output.
    for actual_tensor, expected_tensor, reference_tensor in zip(
        actual, expected, reference, strict=True
    ):
        actual_error = (actual_tensor.float() - reference_tensor).norm()
        eager_error = (expected_tensor.float() - reference_tensor).norm()
        assert actual_error <= 1.05 * eager_error, (actual_error, eager_error)
        assert actual_tensor.is_contiguous()
