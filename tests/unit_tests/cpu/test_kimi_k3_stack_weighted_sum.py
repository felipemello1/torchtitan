# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.models.kimi_k3.model import (
    _attention_residual,
    _MAX_STACK_ENTRIES,
    _StackWeightedSum,
)


@pytest.mark.parametrize("num_entries", [1, 3, _MAX_STACK_ENTRIES])
def test_stack_weighted_sum_gradcheck(num_entries):
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(3, num_entries + 1, dtype=torch.float64), -1)
    stack = torch.randn(3, num_entries, 5, dtype=torch.float64)
    partial = torch.randn(3, 5, dtype=torch.float64)
    inputs = [x.requires_grad_() for x in (probs, stack, partial)]

    expected = (probs[:, :-1, None] * stack).sum(1) + probs[:, -1:] * partial
    torch.testing.assert_close(_StackWeightedSum.apply(*inputs), expected)
    assert torch.autograd.gradcheck(_StackWeightedSum.apply, inputs)


def test_compiled_branch_takes_stack_weighted_sum():
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm

    torch._dynamo.reset()
    compiled = torch.compile(
        _attention_residual.__wrapped__, backend=backend, fullgraph=True
    )
    projection, norm_weight = torch.randn(16), torch.randn(16)
    widths = [1, 3, _MAX_STACK_ENTRIES + 1]
    for num_entries in widths:
        partial = torch.randn(4, 16, requires_grad=True)
        stack = torch.randn(4, num_entries, 16, requires_grad=True)
        compiled(partial, stack, projection, norm_weight, 1e-6)
    torch._dynamo.reset()

    apply_op = torch.ops.higher_order.autograd_function_apply
    uses_function = [
        any(node.target is apply_op for node in gm.graph.nodes) for gm in graphs
    ]
    # One-entry and over-wide stacks keep the plain sum.
    assert uses_function == [False, True, False]
