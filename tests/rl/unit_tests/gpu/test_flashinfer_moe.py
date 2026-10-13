# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest

import torch

from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.rl.model.flashinfer_moe import (
    enable_flashinfer_routed_experts,
    to_flashinfer_layout_,
)


def _reference(x, w13, w2, ids, scores):
    """fp64 routed experts: each token's top-k SwiGLU experts, weighted by its scores."""
    out = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    x, w13, w2, scores = x.double(), w13.double(), w2.double(), scores.double()
    for k in range(ids.shape[1]):
        e = ids[:, k]
        gate = torch.einsum("td,tfd->tf", x, w13[e, 0])
        up = torch.einsum("td,tfd->tf", x, w13[e, 1])
        hidden = torch.nn.functional.silu(gate) * up
        out += scores[:, k : k + 1] * torch.einsum("tf,tdf->td", hidden, w2[e])
    return out


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
    and importlib.util.find_spec("flashinfer") is not None,
    "FlashInfer's TRT-LLM MoE needs a Blackwell GPU and flashinfer",
)
class TestFlashInferRoutedExperts(unittest.TestCase):
    def test_matches_fp64_reference_and_stays_off_with_grad(self):
        torch.manual_seed(0)
        dim, hidden, num_experts, top_k, tokens = 512, 256, 16, 4, 34
        experts = make_routed_experts_config(
            dim=dim, hidden_dim=hidden, num_experts=num_experts, top_k=top_k,
            param_init={}, enable_ep=False,
        ).build().cuda().bfloat16()
        with torch.no_grad():
            experts.w13.weight.copy_(torch.randn_like(experts.w13.weight) * 0.05)
            experts.w2.weight.copy_(torch.randn_like(experts.w2.weight) * 0.05)
        w13, w2 = experts.w13.weight.clone(), experts.w2.weight.clone()
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        ids = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(tokens)]).cuda()
        scores = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=-1)
        counts = torch.bincount(ids.flatten(), minlength=num_experts)

        enabled = enable_flashinfer_routed_experts(torch.nn.ModuleList([experts]))
        self.assertEqual(enabled, [experts])
        to_flashinfer_layout_(enabled)
        with torch.no_grad():
            out = experts(x, scores, ids, counts)

        ref = _reference(x, w13, w2, ids, scores)
        rel = ((out.double() - ref).norm() / ref.norm()).item()
        self.assertLess(rel, 1e-2)
        # The FlashInfer kernel is inference-only; training keeps the TorchTitan path.
        self.assertFalse(torch.equal(experts.w13.weight, w13))


if __name__ == "__main__":
    unittest.main()
