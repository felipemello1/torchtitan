# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.config_utils import (
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
    make_shared_expert_ffn_config,
)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestSharedExpertsSideStream(unittest.TestCase):
    def test_cuda_graph_matches_eager(self):
        # Under capture the shared experts run on a side stream; same kernels, same output.
        dim, hidden_dim, num_experts, top_k = 256, 512, 8, 2
        init = {"weight": nn.init.normal_}
        moe = make_moe_config(
            num_experts=num_experts,
            router=make_router_config(
                dim=dim,
                num_experts=num_experts,
                gate_param_init=init,
                score_func=Sigmoid.Config(),
                top_k=top_k,
            ),
            routed_experts=make_routed_experts_config(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                param_init={},
                enable_ep=False,
            ),
            shared_experts=make_shared_expert_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w13_param_init=init,
                w2_param_init=init,
                enable_sp=False,
            ),
            load_balance_coeff=1e-3,
            enable_sp=False,
        ).build()
        torch.manual_seed(0)
        with torch.no_grad():
            for p in moe.parameters():
                p.normal_(std=0.02)
        moe = moe.to(device="cuda", dtype=torch.bfloat16).eval()
        moe.expert_bias_E = moe.expert_bias_E.float()
        x_TD = torch.randn(34, dim, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            expected_TD = moe(x_TD)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out_TD = moe(x_TD)
            graph.replay()

        torch.testing.assert_close(out_TD, expected_TD, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
