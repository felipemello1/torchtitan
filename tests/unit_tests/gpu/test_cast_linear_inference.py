# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU tests for ``CastLinear``'s fp32-accumulating inference path."""

import pytest
import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import CastLinear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _lm_head(in_features: int = 256, out_features: int = 1024) -> CastLinear:
    lm_head = CastLinear.Config(in_features=in_features, out_features=out_features).build()
    torch.nn.init.normal_(lm_head.weight, std=0.02)
    return lm_head.to(device="cuda", dtype=torch.bfloat16)


def test_inference_matches_fp32_upcast():
    lm_head = _lm_head()
    x = torch.randn(3, 5, 256, device="cuda", dtype=torch.bfloat16)
    reference = F.linear(x.float(), lm_head.weight.float())

    with torch.no_grad():
        out = lm_head(x)

    assert out.dtype == torch.float32
    assert out.shape == reference.shape
    # bf16 products are exact in fp32; only the accumulation order differs.
    torch.testing.assert_close(out, reference, rtol=1e-5, atol=1e-5)


def test_training_keeps_differentiable_upcast_path():
    lm_head = _lm_head()
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    lm_head(x).sum().backward()

    assert lm_head.weight.grad is not None
    assert x.grad is not None
