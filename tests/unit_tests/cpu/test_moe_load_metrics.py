# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.activation import Softmax
from torchtitan.models.common.config_utils import make_router_config
from torchtitan.models.common.moe import (
    collect_moe_load_metrics,
    register_moe_load_balancing_hook,
    TokenChoiceTopKRouter,
)


def _router(step_counts: list[float] | None = None) -> TokenChoiceTopKRouter:
    router = make_router_config(
        dim=8,
        num_experts=4,
        gate_param_init={"weight": nn.init.normal_},
        score_func=Softmax.Config(),
        top_k=2,
    ).build()
    if step_counts is not None:
        router.step_tokens_per_expert_E = torch.tensor(step_counts)
    return router


class _MoEModel(nn.Module):
    """MoE blocks under `layers`, where the hook finds them; no load balancing."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Module() for _ in range(num_layers))
        for block in self.layers:
            block.moe_enabled = True
            block.moe = nn.Module()
            block.moe.load_balance_coeff = None
            block.moe.router = _router()


class _SingleRankContext:
    def get_optional_mesh(self, name):
        return None


class TestCollectMoELoadMetrics(unittest.TestCase):
    def test_stats_of_known_counts(self):
        routers = nn.ModuleList(
            [
                _router([4.0, 4.0, 4.0, 4.0]),  # cv 0, max_vio 0, cold_frac 0
                _router([12.0, 4.0, 0.0, 0.0]),  # cv 1.22, max_vio 2, cold_frac 0.5
            ]
        )
        torch.testing.assert_close(
            collect_moe_load_metrics([routers], _SingleRankContext()),
            {
                "moe_load/cv/mean": 1.5**0.5 / 2,
                "moe_load/cv/max": 1.5**0.5,
                "moe_load/max_vio/mean": 1.0,
                "moe_load/max_vio/max": 2.0,
                "moe_load/cold_frac/mean": 0.25,
                "moe_load/cold_frac/max": 0.5,
            },
            rtol=1e-5,
            atol=1e-6,
        )

    def test_nothing_to_log_without_step_counts(self):
        # A dense model, and a router before its first optimizer step.
        for model in (nn.Linear(2, 2), _router()):
            self.assertEqual(
                collect_moe_load_metrics([model], _SingleRankContext()), {}
            )


class TestCollectMoELoadMetricsDistributed(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cpu"

    def _build_parallelism_context(self, **degrees) -> ParallelismContext:
        degrees = {
            "dp_replicate": 1,
            "dp_shard": 1,
            "cp": 1,
            "tp": 1,
            "pp": 1,
            "ep": 1,
        } | degrees
        with patch("torchtitan.distributed.parallelism_context.device_type", "cpu"):
            parallelism_context = ParallelismContext(
                **degrees, world_size=self.world_size, enable_sequence_parallel=False
            )
            parallelism_context.build_mesh()
        return parallelism_context

    @with_comms
    def test_every_rank_keeps_global_counts_under_ep_and_tp(self):
        """dp_shard 2 x tp 2 with EP 4: each rank routes its own tokens."""
        parallelism_context = self._build_parallelism_context(dp_shard=2, tp=2, ep=4)
        rank = dist.get_rank()
        model = _MoEModel(num_layers=2)
        # Layer 0: rank r sends 4 tokens to expert r. Layer 1: r + 1 tokens to expert 0.
        model.layers[0].moe.router.tokens_per_expert_E.copy_(
            torch.tensor([4.0 if expert == rank else 0.0 for expert in range(4)])
        )
        model.layers[1].moe.router.tokens_per_expert_E.copy_(
            torch.tensor([rank + 1.0, 1.0, 1.0, 1.0])
        )
        optimizers = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", fused=False, lr=0.0)]
        ).build(model_parts=[model])
        register_moe_load_balancing_hook(optimizers, [model], parallelism_context)

        optimizers.step()

        torch.testing.assert_close(
            model.layers[0].moe.router.step_tokens_per_expert_E,
            torch.tensor([4.0, 4.0, 4.0, 4.0]),
        )
        torch.testing.assert_close(
            model.layers[1].moe.router.step_tokens_per_expert_E,
            torch.tensor([10.0, 4.0, 4.0, 4.0]),
        )

    @with_comms
    def test_pp_stages_are_combined(self):
        """pp 4: stages 1 and 2 own one MoE layer each, stages 0 and 3 own none."""
        parallelism_context = self._build_parallelism_context(pp=4)
        step_counts_by_stage = {1: [8.0, 0.0, 0.0, 0.0], 2: [2.0, 2.0, 2.0, 2.0]}
        step_counts = step_counts_by_stage.get(dist.get_rank())
        routers = nn.ModuleList([] if step_counts is None else [_router(step_counts)])
        torch.testing.assert_close(
            collect_moe_load_metrics([routers], parallelism_context),
            {
                "moe_load/cv/mean": 3**0.5 / 2,
                "moe_load/cv/max": 3**0.5,
                "moe_load/max_vio/mean": 1.5,
                "moe_load/max_vio/max": 3.0,
                "moe_load/cold_frac/mean": 0.375,
                "moe_load/cold_frac/max": 0.75,
            },
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
