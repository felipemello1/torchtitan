# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.tools.utils import compile_friendly, trace_compile_friendly


def _step(x: torch.Tensor) -> torch.Tensor:
    # A compile-friendly path that is easy to tell apart from eager.
    return (x * 3 if compile_friendly() else x + x).sum()


def _traced_targets(enabled: bool) -> set:
    with trace_compile_friendly(enabled), torch.compiler._non_strict_tracing_context():
        graph = make_fx(_step)(torch.ones(2)).graph
    return {node.target for node in graph.nodes if node.op == "call_function"}


def test_eager_is_never_compile_friendly() -> None:
    with trace_compile_friendly():
        assert _step(torch.ones(2)).item() == 4.0


def test_torch_compile_is_compile_friendly() -> None:
    compiled = torch.compile(_step, backend="eager", fullgraph=True)
    assert compiled(torch.ones(2)).item() == 6.0


def test_non_strict_trace_follows_the_context() -> None:
    assert torch.ops.aten.mul.Tensor in _traced_targets(True)
    assert torch.ops.aten.add.Tensor in _traced_targets(False)
