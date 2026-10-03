# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.configs import (
    compiles_full_graph,
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.tools.utils import compile_friendly, trace_compile_friendly


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.weight
        # Distinct ops on each path, in forward and in backward.
        return y.exp() if compile_friendly() else y.sigmoid()


def _traced_targets(enabled: bool) -> set:
    model = _Model()

    def step(x: torch.Tensor):
        loss = model(x).sum()
        return loss, torch.autograd.grad(loss, list(model.parameters()))

    with trace_compile_friendly(enabled):
        gm = minimal_fx_tracer(step, module=model)(torch.randn(2, 4)).gm
    return {node.target for node in gm.graph.nodes if node.op == "call_function"}


class TestCompileFriendlyTrace(unittest.TestCase):
    def test_graph_trainer_traces_the_compile_friendly_path(self) -> None:
        targets = _traced_targets(True)
        self.assertIn(torch.ops.aten.exp.default, targets)
        self.assertNotIn(torch.ops.aten.sigmoid_backward.default, targets)

        targets = _traced_targets(False)
        self.assertIn(torch.ops.aten.sigmoid_backward.default, targets)
        self.assertNotIn(torch.ops.aten.exp.default, targets)

    def test_only_full_inductor_traces_the_compile_friendly_path(self) -> None:
        self.assertTrue(
            compiles_full_graph(GraphTrainerCompileConfig(inductor_compilation="full"))
        )
        self.assertFalse(compiles_full_graph(GraphTrainerCompileConfig()))
        self.assertFalse(
            compiles_full_graph(
                GraphTrainerCompileConfig(
                    inductor_compilation="full", enable_passes=False
                )
            )
        )
        self.assertFalse(
            compiles_full_graph(
                GraphTrainerCompileConfig(
                    inductor_compilation="full",
                    disable_passes=["full_inductor_compilation_pass"],
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
