# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.muse_spark import build_model_config


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMuseSparkPoolerSideStream(unittest.TestCase):
    def test_cuda_graph_matches_eager(self):
        # Under capture the pooler and pre-expert norm run once, on a side stream; same output.
        torch.manual_seed(0)
        moe = build_model_config("debugmodel", seq_len=8).layers[0].feed_forward.build()
        moe.init_states()
        moe = moe.to(device="cuda", dtype=torch.bfloat16).eval()
        x_TD = torch.randn(
            34, moe.pooler.in_features, device="cuda", dtype=torch.bfloat16
        )

        with torch.no_grad():
            expected_TD = moe(x_TD)
            pooler_streams = []
            moe.pooler.register_forward_pre_hook(
                lambda *_: pooler_streams.append(torch.cuda.current_stream())
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                main_stream = torch.cuda.current_stream()
                out_TD = moe(x_TD)
            graph.replay()

        self.assertEqual(len(pooler_streams), 1)
        self.assertNotEqual(pooler_streams[0], main_stream)
        torch.testing.assert_close(out_TD, expected_TD, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
