# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, ShapeEnv

from torchtitan.distributed.deepep import hybridep

NUM_TOKENS, HIDDEN_DIM, TOP_K, NUM_EXPERTS, EP_SIZE = 16, 4, 2, 8, 2
NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE


def _dispatch(x, topk_weights, *, non_blocking, capacity_factor):
    topk_idx = torch.zeros(NUM_TOKENS, TOP_K, dtype=torch.int64)
    return torch.ops.hybridep.dispatch(
        x,
        topk_idx,
        topk_weights,
        num_experts=NUM_EXPERTS,
        ep_size=EP_SIZE,
        group_name="ep",
        non_blocking=non_blocking,
        moe_expert_capacity_factor=capacity_factor,
        pad_multiple=None,
    )


def test_blocking_dispatch_fake_rows_are_data_dependent():
    with FakeTensorMode(shape_env=ShapeEnv()):
        x = torch.empty(NUM_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16)
        topk_weights = torch.ones(NUM_TOKENS, TOP_K)
        hidden, scores, tokens_per_expert, _ = _dispatch(
            x, topk_weights, non_blocking=False, capacity_factor=None
        )
    assert free_unbacked_symbols(hidden.shape[0])
    assert scores.shape[0] == hidden.shape[0]
    assert tokens_per_expert.shape == (NUM_LOCAL_EXPERTS,)


def test_blocking_dispatch_fake_traces_forward_and_backward():
    with FakeTensorMode(shape_env=ShapeEnv()):
        x = torch.empty(NUM_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16)
        x.requires_grad_()
        topk_weights = torch.ones(NUM_TOKENS, TOP_K, requires_grad=True)
        hidden, scores, _, handle = _dispatch(
            x, topk_weights, non_blocking=False, capacity_factor=None
        )
        scaled = hidden * scores.to(hidden.dtype).unsqueeze(-1)
        out = torch.ops.hybridep.combine(scaled, handle, NUM_TOKENS, None)
        out.sum().backward()
    assert x.grad.shape == x.shape
    assert topk_weights.grad.shape == topk_weights.shape


def test_combine_backward_fake_matches_a_rank_with_no_tokens():
    with FakeTensorMode(shape_env=ShapeEnv()):
        grad_hidden = torch.empty(0, HIDDEN_DIM, dtype=torch.bfloat16)
        grad_scores = torch.empty(0)
        grad_x, grad_probs = torch.ops.hybridep.combine_bwd(
            grad_hidden,
            grad_scores,
            hybridep.DispatchHandle(),
            num_tokens=NUM_TOKENS,
            num_experts=NUM_EXPERTS,
        )
    assert grad_x.shape == (NUM_TOKENS, HIDDEN_DIM)
    assert grad_probs.shape == (NUM_TOKENS, NUM_EXPERTS)


def test_non_blocking_dispatch_fake_rows_are_the_capacity():
    capacity_factor = 0.5
    with FakeTensorMode(shape_env=ShapeEnv()):
        x = torch.empty(NUM_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16)
        topk_weights = torch.ones(NUM_TOKENS, TOP_K)
        hidden, _, _, _ = _dispatch(
            x, topk_weights, non_blocking=True, capacity_factor=capacity_factor
        )
    capacity = int(
        NUM_TOKENS * EP_SIZE * min(NUM_LOCAL_EXPERTS, TOP_K) * capacity_factor
    )
    assert hidden.shape[0] == capacity
