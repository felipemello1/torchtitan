# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext


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
    fp32_params = {name: p.detach().clone() for name, p in attention.named_parameters()}
    token_counts = (64, 37, 1)
    inputs = []
    for num_tokens in token_counts:
        qkv = torch.randn(
            num_tokens,
            attention.n_kv_heads,
            qkv_linear.r_dim,
            qkv_linear.head_dim,
            device="cuda",
        )
        positions = torch.randint(0, 1024, (num_tokens,), device="cuda")
        grads = [
            torch.randn(num_tokens, heads, attention.head_dim, device="cuda")
            for heads in (attention.n_heads, attention.n_kv_heads, attention.n_kv_heads)
        ]
        inputs.append((qkv, positions, grads))

    def run(dtype, regions):
        LocalCompileConfig(regions=regions).apply()
        # Parameters only: the RoPE cache stays fp32, as in production.
        for name, param in attention.named_parameters():
            param.data, param.grad = fp32_params[name].to(dtype), None
        return [
            _run(attention, qkv.to(dtype), positions, grads)
            for qkv, positions, grads in inputs
        ]

    try:
        reference = run(torch.float32, [])
        expected = run(torch.bfloat16, [])
        # One binding, as in the generator: the token counts compile a static graph, a
        # dynamic one and a T=1 one, each with and without grad.
        actual = run(torch.bfloat16, ["rope", "qkv_norm_rope"])
    finally:
        LocalCompileConfig().apply()
        torch._dynamo.reset()

    # The compiled norm reduces in another order than the eager kernel, so a few elements
    # differ by a bf16 ulp; without emulate_precision_casts ~31% of q would. Check both:
    # few differing elements vs eager, and as close as eager to an fp32 run of the same
    # math (q/k/v in inference and training, the fused output's grad).
    for num_tokens, *outputs in zip(
        token_counts, actual, expected, reference, strict=True
    ):
        for i, (actual_tensor, expected_tensor, reference_tensor) in enumerate(
            zip(*outputs, strict=True)
        ):
            assert actual_tensor.is_contiguous()
            if i < 6:  # q/k/v outputs; v is a plain copy
                differing = (actual_tensor != expected_tensor).float().mean()
                assert differing < 1e-3, (num_tokens, i, differing)
                if i in (2, 5):
                    assert torch.equal(actual_tensor, expected_tensor)
            actual_error = (actual_tensor.float() - reference_tensor).norm()
            eager_error = (expected_tensor.float() - reference_tensor).norm()
            assert actual_error <= 1.05 * eager_error, (
                num_tokens,
                i,
                actual_error,
                eager_error,
            )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestQKVNormRopeTensorParallel(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_tp2_matches_eager_in_three_graphs(self):
        """TP2 generator setting (no SP, no grad): the region's local q/k/v heads match eager,
        and the decode batch sizes compile 3 graphs: static, dynamic and T=1."""
        from torchtitan.models.qwen3 import build_model_config

        torch.manual_seed(0)
        config = build_model_config(
            "debugmodel",
            parallelism=ParallelismConfig(enable_sequence_parallel=False),
        )
        attention = config.layers[0].attention.build().to(self.device_type)
        # Parameters only: the RoPE cache stays fp32, as in production.
        for param in attention.parameters():
            param.data = param.data.to(torch.bfloat16)
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        with patch(
            "torchtitan.distributed.parallelism_context.device_type", self.device_type
        ):
            parallelism_context.build_mesh()
        attention._parallelize(parallelism_context)

        token_counts = (64, 37, 1, 2, 16, 128)
        inputs = []
        for num_tokens in token_counts:
            x_TD = torch.randn(
                num_tokens, config.dim, device=self.device_type, dtype=torch.bfloat16
            )
            positions = torch.randint(0, 1024, (num_tokens,), device=self.device_type)
            inputs.append((x_TD, positions))

        def run(regions):
            LocalCompileConfig(regions=regions).apply()
            with parallelism_context.activate_spmd(), torch.no_grad():
                return [
                    attention._prepare_qkv(attention.qkv_linear(x_TD), positions)
                    for x_TD, positions in inputs
                ]

        torch._dynamo.reset()
        try:
            expected = run([])
            unique_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            actual = run(["rope", "qkv_norm_rope"])
            unique_graphs = (
                torch._dynamo.utils.counters["stats"]["unique_graphs"] - unique_graphs
            )
        finally:
            LocalCompileConfig().apply()
            torch._dynamo.reset()

        self.assertEqual(unique_graphs, 3)
        for num_tokens, actual_qkv, expected_qkv in zip(
            token_counts, actual, expected, strict=True
        ):
            for i, (actual_tensor, expected_tensor) in enumerate(
                zip(actual_qkv, expected_qkv, strict=True)
            ):
                self.assertEqual(actual_tensor.shape, expected_tensor.shape)
                differing = (actual_tensor != expected_tensor).float().mean()
                self.assertLess(differing, 1e-3, (num_tokens, i))
            torch.testing.assert_close(actual_qkv[2], expected_qkv[2], rtol=0, atol=0)
