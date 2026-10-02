# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.kimi_k3.model import _attention_residual

TOKENS, DIM, EPS = 64, 256, 1e-6


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestAttentionResidualLocalCompile(unittest.TestCase):
    def tearDown(self):
        LocalCompileConfig(regions=[]).apply_local_compile()
        torch._dynamo.reset()

    def _run(self, partial, stack, projection, norm_weight):
        inputs = [
            x.detach().clone().requires_grad_()
            for x in (partial, stack, projection, norm_weight)
        ]
        output = _attention_residual(*inputs, EPS)
        generator = torch.Generator(device="cuda").manual_seed(1)
        grad_output = torch.randn(
            output.shape, dtype=output.dtype, device="cuda", generator=generator
        )
        grads = torch.autograd.grad(output, inputs, grad_output)
        return (output, *grads)

    def test_compiled_matches_eager(self):
        torch.manual_seed(0)
        projection = torch.randn(DIM, device="cuda").bfloat16()
        norm_weight = torch.randn(DIM, device="cuda").bfloat16()
        cases = [
            (
                torch.randn(TOKENS, DIM, device="cuda").bfloat16(),
                torch.randn(TOKENS, num_entries, DIM, device="cuda").bfloat16(),
            )
            for num_entries in (1, 7, 3)
        ]
        LocalCompileConfig(regions=[]).apply_local_compile()
        eager = [self._run(*case, projection, norm_weight) for case in cases]
        # One binding for all widths, so widths 7 and 3 run the dynamic-width graph.
        LocalCompileConfig(regions=["attention_residual"]).apply_local_compile()
        compiled = [self._run(*case, projection, norm_weight) for case in cases]
        for compiled_tensors, eager_tensors in zip(compiled, eager, strict=True):
            for actual, expected in zip(compiled_tensors, eager_tensors, strict=True):
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
