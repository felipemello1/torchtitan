# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher

TOKENS, TOP_K, NUM_EXPERTS, DIM = 64, 4, 8, 128


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMoEDispatchCombineLocalCompile(unittest.TestCase):
    def tearDown(self):
        LocalCompileConfig(regions=[]).apply_local_compile()
        torch._dynamo.reset()

    def test_compiled_matches_eager(self):
        torch.manual_seed(0)
        dispatcher = LocalTokenDispatcher.Config(
            num_experts=NUM_EXPERTS, top_k=TOP_K
        ).build()
        x = torch.randn(TOKENS, DIM, device="cuda", dtype=torch.bfloat16)
        expert_ids = torch.randint(0, NUM_EXPERTS, (TOKENS, TOP_K), device="cuda")
        scores = torch.rand(TOKENS, TOP_K, device="cuda")
        grad_out = torch.randn_like(x)

        def run():
            x_leaf = x.clone().requires_grad_()
            scores_leaf = scores.clone().requires_grad_()
            routed_input, _, metadata = dispatcher.dispatch(
                x_leaf, scores_leaf, expert_ids, torch.bincount(expert_ids.view(-1))
            )
            # Stands in for the experts: every routed row is transformed.
            out = dispatcher.combine(torch.tanh(routed_input), metadata, x_leaf)
            grads = torch.autograd.grad(out, (x_leaf, scores_leaf), grad_out)
            return (out, *grads)

        LocalCompileConfig(regions=[]).apply_local_compile()
        eager = run()
        LocalCompileConfig(regions=["moe_dispatch_combine"]).apply_local_compile()
        compiled = run()
        # Compiled sums each token's rows in fp32; eager scatter-adds in bf16.
        for actual, expected in zip(compiled, eager, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
