# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU tests for the Triton routed-experts override.

Token counts cover both tile configs: 16-row blocks below 32 rows per expert, and
persistent 128-row blocks from there.
"""

import unittest
from functools import partial

import torch
from torch.nn import init

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.activation import Softmax
from torchtitan.models.common.config_utils import (
    make_routed_experts_config,
    make_router_config,
)
from torchtitan_recipes.overrides.triton_routed_experts import (
    triton_routed_experts,
    triton_topk_router,
)

_DIM, _HIDDEN, _E, _K = 1024, 256, 16, 4
# 16-row blocks below 128 tokens (32 rows per expert), persistent 64-row blocks to 256, 128-row above
_TOKEN_COUNTS = (1, 37, 100, 200, 300, 1500)


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
        # The trainer and the stock generator run these regions compiled.
        apply_local_compile(["fused_binary_activation", "moe_dispatch", "moe_combine"])
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

    def test_cuda_graph_replay_matches_eager_bitwise(self):
        for num_tokens in (8, 37, 1500):
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

    def test_expert_sorted_rows_match_stock_bitwise(self):
        """Rows as any dispatcher returns them, incl. an expert-parallel rank with none.

        Bitwise equal to the stock path with its compiled SwiGLU region, at decode and
        prefill row counts: the generator matches the trainer. A Triton or Inductor change
        that alters either side's rounding fails here.
        """
        for num_rows in (0, 40, 1500):
            with self.subTest(num_rows=num_rows):
                gen = torch.Generator(device="cuda").manual_seed(num_rows)
                counts_E = torch.bincount(
                    torch.randint(0, _E, (num_rows,), device="cuda", generator=gen),
                    minlength=_E,
                )
                offsets_E = torch.cumsum(counts_E, 0, dtype=torch.int32)
                x_RD = torch.randn(
                    num_rows, _DIM, device="cuda", dtype=torch.bfloat16, generator=gen
                )
                with torch.no_grad():
                    actual = self.triton_experts._expert_mlp(x_RD, offsets_E)
                    if num_rows == 0:
                        self.assertEqual(actual.shape, (0, _DIM))
                        continue
                    expected = self.stock._expert_mlp(x_RD, offsets_E)
                self.assertTrue(torch.equal(actual, expected))

    def test_routed_experts_match_stock_bitwise(self):
        """The EP=1 path equals the stock compiled path bitwise at decode and prefill.

        A Triton or Inductor change that alters either side's rounding fails here.
        """
        for num_tokens in (37, 1500):
            with self.subTest(num_tokens=num_tokens):
                routing = _routing(num_tokens, seed=6)
                with torch.no_grad():
                    actual = self.triton_experts(*routing)
                    expected = self.stock(*routing)
                self.assertTrue(torch.equal(actual, expected))

    def test_counts_left_to_the_kernels(self):
        """On the EP=1 path the kernels count the ids themselves, so MoE skips the count."""
        for num_tokens in (37, 1500):
            with self.subTest(num_tokens=num_tokens):
                x_TD, scores_TK, ids_TK, counts_E = _routing(num_tokens, seed=5)
                routing_map_TE = torch.zeros(
                    num_tokens, _E, dtype=torch.bool, device="cuda"
                )
                routing_map_TE.scatter_(1, ids_TK, True)
                with torch.no_grad():
                    self.assertIsNone(self.triton_experts.count_tokens(routing_map_TE))
                    expected = self.triton_experts(x_TD, scores_TK, ids_TK, counts_E)
                    actual = self.triton_experts(x_TD, scores_TK, ids_TK, None)
                self.assertTrue(torch.equal(actual, expected))

    def test_grad_enabled_takes_stock_path(self):
        routing = _routing(37, seed=4)
        self.assertTrue(
            torch.equal(self.triton_experts(*routing), self.stock(*routing))
        )


class TestTritonTopKRouter(unittest.TestCase):
    def test_matches_stock_router_with_ties(self):
        cfg = make_router_config(
            dim=_DIM,
            num_experts=256,
            gate_param_init={"weight": partial(init.trunc_normal_, std=0.02)},
            score_func=Softmax.Config(),
            top_k=8,
            route_norm=True,
        )
        with torch.device("cuda"):
            stock, fused = cfg.build().eval(), triton_topk_router(cfg).build().eval()
        gen = torch.Generator(device="cuda").manual_seed(0)
        # Integer logits: many exact ties, which must go to the lower expert id.
        gate_TE = torch.randint(0, 6, (300, 256), device="cuda", generator=gen).float()
        with torch.no_grad():
            _, scores, ids, routing_map = stock._score_and_select_experts(
                gate_TE, None, padding_mask_T=None
            )
            _, f_scores, f_ids, f_routing_map = fused._score_and_select_experts(
                gate_TE, None, padding_mask_T=None
            )
        self.assertTrue(torch.equal(f_ids, ids))
        self.assertTrue(torch.equal(f_routing_map, routing_map))
        torch.testing.assert_close(f_scores, scores, rtol=1e-6, atol=0)

    def test_gate_gemv_matches_cublas(self):
        cfg = make_router_config(
            dim=_DIM,
            num_experts=256,
            gate_param_init={"weight": partial(init.trunc_normal_, std=0.02)},
            score_func=Softmax.Config(),
        )
        with torch.device("cuda"):
            gate = triton_topk_router(cfg).build().gate.bfloat16()
        for num_tokens in (1, 8, 9):  # 9: the stock GEMM
            x_TD = torch.randn(num_tokens, _DIM, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                expected = torch.mm(x_TD, gate.weight.T, out_dtype=torch.float32)
                torch.testing.assert_close(gate(x_TD), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
