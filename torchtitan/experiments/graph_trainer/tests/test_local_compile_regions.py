# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.local_compile import local_compile, LocalCompileConfig
from torchtitan.experiments.graph_trainer.inductor_passes import (
    strip_inductor_tags_from_collectives_pass,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    FSDP_PARAM_FQNS_META,
)


class _Norm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    @local_compile("test_graph_trainer_norm", batch_invariant=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class _Model(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.norm = _Norm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.lin(x))


def _trace_train_step(model: nn.Module) -> torch.fx.GraphModule:
    def step(x: torch.Tensor):
        loss = model(x).float().pow(2).mean()
        return loss, torch.autograd.grad(loss, list(model.parameters()))

    return minimal_fx_tracer(step, module=model)(torch.randn(8, 16)).gm


def _is_tagged(node: torch.fx.Node) -> bool:
    return "compile_with_inductor" in node.meta.get("custom", {})


def _is_collective(node: torch.fx.Node) -> bool:
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == "_c10d_functional"
    )


class TestLocalCompileRegions(unittest.TestCase):
    def setUp(self) -> None:
        LocalCompileConfig(regions=["test_graph_trainer_norm"]).apply_local_compile(
            tag_regions=True
        )

    def tearDown(self) -> None:
        LocalCompileConfig(regions=[]).apply_local_compile()

    def test_forward_and_backward_nodes_are_tagged(self) -> None:
        gm = _trace_train_step(_Model())
        nodes = [node for node in gm.graph.nodes if node.op == "call_function"]
        tagged = [node for node in nodes if _is_tagged(node)]
        self.assertIn(torch.ops.aten.rsqrt.default, {node.target for node in tagged})
        self.assertTrue(any(node.meta.get("autograd_backward") for node in tagged))
        # The Linear runs outside the region, in forward and backward.
        self.assertFalse(
            any(node.target is torch.ops.aten.mm.default for node in tagged)
        )

    def test_fsdp_collectives_are_not_tagged_after_strip_pass(self) -> None:
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
        try:
            mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("dp",))
            model = data_parallel(_Model(), mesh, mode="fully_shard")
            gm = _trace_train_step(model)
        finally:
            dist.destroy_process_group()
        # norm.weight is all-gathered inside the region, so its all-gather and
        # (through the copied forward metadata) its reduce-scatter are tagged.
        collectives = [node for node in gm.graph.nodes if _is_collective(node)]
        self.assertTrue(any(_is_tagged(node) for node in collectives))

        gm = strip_inductor_tags_from_collectives_pass(gm, ())

        fsdp_nodes = [
            node
            for node in gm.graph.nodes
            if FSDP_PARAM_FQNS_META in node.meta.get("custom", {})
        ]
        self.assertFalse(any(_is_tagged(node) for node in collectives + fsdp_nodes))
        self.assertTrue(
            any(
                node.target is torch.ops.aten.rsqrt.default and _is_tagged(node)
                for node in gm.graph.nodes
            )
        )


if __name__ == "__main__":
    unittest.main()
