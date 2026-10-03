# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan_recipes.overrides.quack_rmsnorm import _QUACK_IMPORT_ERROR, QuackRMSNorm


@unittest.skipUnless(
    torch.cuda.is_available() and _QUACK_IMPORT_ERROR is None,
    "requires CUDA and quack",
)
class TestQuackRMSNormNumerics(unittest.TestCase):
    @parametrize("weight_dtype", [torch.bfloat16, torch.float32])
    def test_matches_stock_forward_and_backward(self, weight_dtype):
        """FSDP runs the norm with a bf16 weight; unsharded runs keep it in fp32."""
        torch.manual_seed(0)
        dim = 7168
        stock = RMSNorm.Config(normalized_shape=dim, eps=1e-6).build()
        quack = QuackRMSNorm.Config(normalized_shape=dim, eps=1e-6).build()
        weight = 1 + 0.1 * torch.randn(dim)
        for norm in (stock, quack):
            norm.to(device="cuda", dtype=weight_dtype)
            with torch.no_grad():
                norm.weight.copy_(weight)
        x = torch.randn(2, 512, dim, device="cuda", dtype=torch.bfloat16)
        grad_y = torch.randn_like(x)
        self.assertTrue(quack._runs_quack(x))

        results = []
        for norm in (stock, quack):
            x_leaf = x.clone().requires_grad_()
            y = norm(x_leaf)
            y.backward(grad_y)
            results.append((y, x_leaf.grad, norm.weight.grad))

        for expected, actual in zip(*results):
            self.assertEqual(actual.dtype, expected.dtype)
            torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=2e-2)


instantiate_parametrized_tests(TestQuackRMSNormNumerics)


if __name__ == "__main__":
    unittest.main()
