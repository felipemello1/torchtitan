# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU tests for the Triton routed-experts override.

Token counts cover every kernel config: the decode scan (at most 1024 routes), the
16-row scatter path, the 64- and 128-row blocks, and the grouped-GEMM prefill path.
"""

import unittest
from functools import partial

import torch
from torch.nn import init

from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan_recipes.overrides.triton_routed_experts import triton_routed_experts

_DIM, _HIDDEN, _E, _K = 1024, 256, 16, 4
# scan, 16-row scatter, 64-row and 128-row blocks, grouped GEMMs
_TOKEN_COUNTS = (1, 37, 300, 700, 2500, 4500)


def _build_pair():
    """Return stock and Triton routed experts with the same bf16 weights."""
    param_init = {
        name: partial(init.trunc_normal_, std=0.02)
        for name in ("w1_EFD", "w2_EDF", "w3_EFD")
    }
    cfg = make_routed_experts_config(
        dim=_DIM,
        hidden_dim=_HIDDEN,
        num_experts=_E,
        top_k=_K,
        param_init=param_init,
        enable_ep=False,
    )
    torch.manual_seed(0)
    with torch.device("cuda"):
        stock = cfg.build()
        triton_experts = triton_routed_experts(cfg).build()
    with torch.no_grad():
        for param in stock.parameters():
            param.copy_(torch.randn_like(param) * 0.02)
    triton_experts.load_state_dict(stock.state_dict())
    return stock.bfloat16(), triton_experts.bfloat16()


def _routing(num_tokens: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x_TD = torch.randn(
        num_tokens, _DIM, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    logits_TE = torch.randn(num_tokens, _E, device="cuda", generator=gen)
    scores_TK, ids_TK = logits_TE.softmax(-1).topk(_K, dim=-1)
    return x_TD, scores_TK, ids_TK, torch.bincount(ids_TK.reshape(-1), minlength=_E)


def _fp64_reference(experts, x_TD, scores_TK, ids_TK) -> torch.Tensor:
    w13 = experts.w13.weight.double()
    w2 = experts.w2.weight.double()
    out_TD = torch.zeros(x_TD.shape, dtype=torch.float64, device=x_TD.device)
    for token in range(x_TD.shape[0]):
        x_D = x_TD[token].double()
        for k in range(_K):
            expert = ids_TK[token, k]
            gate, up = w13[expert, 0] @ x_D, w13[expert, 1] @ x_D
            hidden = torch.nn.functional.silu(gate) * up
            out_TD[token] += scores_TK[token, k].double() * (w2[expert] @ hidden)
    return out_TD


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "needs TMA (SM90+)",
)
class TestTritonRoutedExperts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock, cls.triton_experts = _build_pair()

    def test_matches_fp64_at_least_as_well_as_stock(self):
        for num_tokens in _TOKEN_COUNTS:
            with self.subTest(num_tokens=num_tokens):
                routing = _routing(num_tokens, seed=num_tokens)
                # A 300-token reference loop is slow; check a slice of the rows.
                rows = slice(0, min(num_tokens, 64))
                with torch.no_grad():
                    expected = _fp64_reference(
                        self.stock, routing[0][rows], routing[1][rows], routing[2][rows]
                    )
                    stock_out = self.stock(*routing)[rows]
                    triton_out = self.triton_experts(*routing)[rows]
                self.assertLessEqual(
                    _rel(triton_out, expected), _rel(stock_out, expected)
                )
                self.assertLess(_rel(triton_out, expected), 5e-3)

    def test_replayed_expert_ids(self):
        """Any distinct expert ids per token work: the kernels only read ids and counts."""
        num_tokens = 37
        x_TD, scores_TK, _, _ = _routing(num_tokens, seed=1)
        token_T1 = torch.arange(num_tokens, device="cuda")[:, None]
        ids_TK = (token_T1 * 3 + torch.arange(_K, device="cuda")) % _E
        counts_E = torch.bincount(ids_TK.reshape(-1), minlength=_E)
        with torch.no_grad():
            out_TD = self.triton_experts(x_TD, scores_TK, ids_TK, counts_E)
        expected = _fp64_reference(self.stock, x_TD, scores_TK, ids_TK)
        self.assertLess(_rel(out_TD, expected), 5e-3)

    def test_cuda_graph_replay_matches_eager_bitwise(self):
        for num_tokens in (37, 700, 4500):
            with self.subTest(num_tokens=num_tokens):
                routing = _routing(num_tokens, seed=2)
                with torch.no_grad():
                    eager = self.triton_experts(*routing)
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        self.triton_experts(*routing)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graphed = self.triton_experts(*routing)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(graphed, eager))

    def test_deterministic(self):
        routing = _routing(300, seed=3)
        with torch.no_grad():
            first = self.triton_experts(*routing)
            second = self.triton_experts(*routing)
        self.assertTrue(torch.equal(first, second))

    def test_grad_enabled_takes_stock_path(self):
        routing = _routing(37, seed=4)
        self.assertTrue(
            torch.equal(self.triton_experts(*routing), self.stock(*routing))
        )


if __name__ == "__main__":
    unittest.main()
