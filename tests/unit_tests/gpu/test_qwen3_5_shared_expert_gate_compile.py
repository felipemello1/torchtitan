# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.qwen3_5 import build_model_config

TOKENS = 64


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSharedExpertGateLocalCompile(unittest.TestCase):
    def tearDown(self):
        LocalCompileConfig(regions=[]).apply_local_compile()
        torch._dynamo.reset()

    def test_compiled_matches_eager(self):
        torch.manual_seed(0)
        moe_config = next(
            layer.moe
            for layer in build_model_config("debugmodel_moe").layers
            if layer.moe is not None
        )
        with torch.device("cuda"):
            shared_expert = moe_config.shared_experts.build()
        dim = moe_config.shared_experts.w2.out_features
        gate = torch.randn(TOKENS, 1, device="cuda").bfloat16()
        out = torch.randn(TOKENS, dim, device="cuda").bfloat16()

        def run():
            inputs = [gate.clone().requires_grad_(), out.clone().requires_grad_()]
            result = shared_expert._gate_output(*inputs)
            generator = torch.Generator(device="cuda").manual_seed(1)
            grad = torch.randn(
                result.shape, dtype=result.dtype, device="cuda", generator=generator
            )
            return (result, *torch.autograd.grad(result, inputs, grad))

        LocalCompileConfig(regions=[]).apply_local_compile()
        eager = run()
        LocalCompileConfig(regions=["shared_expert_gate"]).apply_local_compile()
        compiled = run()
        # Compiled keeps the sigmoid in fp32; eager rounds it to bf16 first.
        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
