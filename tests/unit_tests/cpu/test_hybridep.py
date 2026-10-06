# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, ShapeEnv

from torchtitan.distributed.deepep import hybridep  # noqa: F401  (registers the ops)

NUM_TOKENS, TOP_K, NUM_EXPERTS, EP_SIZE = 16, 2, 8, 2


def _fake_dispatch(non_blocking: bool, capacity_factor: float | None):
    with FakeTensorMode(shape_env=ShapeEnv()):
        x = torch.empty(NUM_TOKENS, 4, dtype=torch.bfloat16)
        topk_idx = torch.zeros(NUM_TOKENS, TOP_K, dtype=torch.int64)
        topk_weights = torch.ones(NUM_TOKENS, TOP_K)
        return torch.ops.hybridep.dispatch(
            x,
            topk_idx,
            topk_weights,
            NUM_EXPERTS,
            EP_SIZE,
            "ep",
            non_blocking,
            capacity_factor,
            None,
        )


def test_blocking_dispatch_fake_rows_are_data_dependent():
    hidden, scores, tokens_per_expert, _ = _fake_dispatch(False, None)
    assert free_unbacked_symbols(hidden.shape[0])
    assert scores.shape[0] == hidden.shape[0]
    assert tokens_per_expert.shape == (NUM_EXPERTS // EP_SIZE,)


def test_non_blocking_dispatch_fake_rows_are_the_capacity():
    hidden, _, _, _ = _fake_dispatch(True, 0.5)
    # NUM_TOKENS * EP_SIZE * min(local experts, TOP_K) * capacity_factor
    assert hidden.shape[0] == 16 * 2 * 2 * 0.5
