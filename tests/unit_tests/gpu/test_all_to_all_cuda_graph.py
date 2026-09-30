# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import spmd_types as spmd
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.protocols.module import Module


pytestmark = pytest.mark.multi_gpu

_NUM_EXPERTS = 8
_TOP_K = 2
_NUM_TOKENS = 16
_MODEL_DIM = 16  # BF16 grouped-MM rows require a 16-byte stride.


def _local_routed_experts() -> RoutedExperts:
    """Build this rank's local expert shard around a global-E all-to-all dispatcher."""
    num_local_experts = _NUM_EXPERTS // torch.distributed.get_world_size()
    routed_experts = RoutedExperts.__new__(RoutedExperts)
    Module.__init__(routed_experts)
    routed_experts.w13 = GroupedLinear.Config(
        group_size=num_local_experts,
        in_features=_MODEL_DIM,
        out_features=_MODEL_DIM,
        num_linears=2,
    ).build()
    routed_experts.w2 = GroupedLinear.Config(
        group_size=num_local_experts,
        in_features=_MODEL_DIM,
        out_features=_MODEL_DIM,
    ).build()
    routed_experts.activation_fn = SwiGLU.Config().build()
    routed_experts.output_postprocess = None
    routed_experts.token_dispatcher = AllToAllTokenDispatcher.Config(
        num_experts=_NUM_EXPERTS,
        top_k=_TOP_K,
    ).build()
    with torch.no_grad():
        for parameter in routed_experts.parameters():
            parameter.normal_()
    return routed_experts


def _routing(
    seed: int, *, expert_bias_E: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return this rank's (topk_scores_TK, topk_expert_ids_TK, num_tokens_per_expert_E)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores_TE = torch.rand(
        _NUM_TOKENS, _NUM_EXPERTS, device="cuda", generator=generator
    )
    if expert_bias_E is not None:
        scores_TE = scores_TE + expert_bias_E
    topk_scores_TK, topk_expert_ids_TK = scores_TE.topk(_TOP_K, dim=-1)
    num_tokens_per_expert_E = torch.bincount(
        topk_expert_ids_TK.flatten(), minlength=_NUM_EXPERTS
    )
    return topk_scores_TK, topk_expert_ids_TK, num_tokens_per_expert_E


def _replicated_tokens(seed: int) -> torch.Tensor:
    """Return the same ``(EP * T, D)`` tokens on every rank, like a TP-replicated MoE input."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        2 * _NUM_TOKENS,
        _MODEL_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestAllToAllCudaGraph(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    @torch.no_grad()
    def test_captured_forward_matches_eager_for_new_routing(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)
        torch.manual_seed(0)
        routed_experts = _local_routed_experts().to(self.device_type)

        def forward(replicated_x_TD, *routing):
            # MoE splits TP-replicated tokens like this before routing them.
            x_TD = spmd.redistribute(replicated_x_TD, "ep", src=spmd.I, dst=spmd.S(0))
            return routed_experts(x_TD, *routing)

        with set_current_spmd_mesh(mesh):
            static_inputs = (_replicated_tokens(seed=0), *_routing(seed=self.rank))
            # Eager warmup initializes NCCL and the grouped-MM kernels before capture.
            forward(*static_inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out_TD = forward(*static_inputs)

            # Replay with tokens and routing the graph never saw, including
            # every token routed to rank 0's experts.
            rank0_bias_E = torch.zeros(_NUM_EXPERTS, device=self.device_type)
            rank0_bias_E[: _NUM_EXPERTS // self.world_size] = 1.0
            for inputs in (
                (_replicated_tokens(seed=1), *_routing(seed=self.rank)),
                (_replicated_tokens(seed=2), *_routing(seed=100 + self.rank)),
                (
                    _replicated_tokens(seed=3),
                    *_routing(seed=200 + self.rank, expert_bias_E=rank0_bias_E),
                ),
            ):
                for static_tensor, tensor in zip(static_inputs, inputs, strict=True):
                    static_tensor.copy_(tensor)
                graph.replay()
                expected_TD = forward(*inputs)
                # Top-2 sums round once in both paths, so they match bitwise.
                torch.testing.assert_close(static_out_TD, expected_TD, rtol=0, atol=0)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
