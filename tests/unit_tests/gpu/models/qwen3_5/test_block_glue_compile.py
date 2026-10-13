# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.qwen3_5 import model as qwen3_5


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBlockGlueLocalCompile(unittest.TestCase):
    def setUp(self):
        # A local generator leaves the global RNG stream unchanged for later tests.
        self.generator = torch.Generator(device="cuda").manual_seed(0)

    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, generator=self.generator
        )

    def test_residual_add_norm_matches_separate_ops(self):
        norm = qwen3_5.OffsetRMSNorm(qwen3_5.OffsetRMSNorm.Config(dim=2048)).cuda()
        with torch.no_grad():
            norm.weight.copy_(self.randn(2048))
        for num_tokens in (1, 34, 4096):
            x, h = self.randn(num_tokens, 2048), self.randn(num_tokens, 2048)
            with torch.no_grad():
                # Before: a separate add, then the compiled offset_rmsnorm region.
                apply_local_compile(["offset_rmsnorm"])
                expected_x = x + h
                expected_norm = norm(expected_x)
                apply_local_compile(["offset_rmsnorm", "residual_norm"])
                actual_x, actual_norm = qwen3_5._residual_add_norm(x, h, norm)
            torch.testing.assert_close(actual_x, expected_x, rtol=0, atol=0)
            torch.testing.assert_close(actual_norm, expected_norm, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
