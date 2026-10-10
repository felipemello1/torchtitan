# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.models.qwen3_5 import build_model_config as build_qwen3_5_model_config
from torchtitan.rl.controller import RLModelDefaults

_RECIPE_MODULES = [
    "torchtitan_recipes.rl.alphabet_sort",
    "torchtitan_recipes.rl.dapo_math",
    "torchtitan_recipes.rl.search_r1",
    "torchtitan_recipes.rl.verifiers_dapo_math",
    "torchtitan_recipes.tests.rl",
]


@pytest.mark.parametrize("module_name", _RECIPE_MODULES)
def test_rl_recipes_get_model_defaults(module_name: str) -> None:
    # dapo_math and verifiers_dapo_math import optional reward packages (math_verify, verifiers).
    module = pytest.importorskip(module_name)
    recipes = [
        fn
        for name, fn in vars(module).items()
        if name.startswith("rl_") and getattr(fn, "__module__", None) == module_name
    ]
    assert recipes

    for recipe in recipes:
        config = recipe()
        regions_before = list(config.model.local_compile_regions)

        model = config.model_defaults.apply_(config.model)

        name = recipe.__name__
        is_fp32_head = isinstance(model.lm_head, HiMidLoLinear.Config)
        assert is_fp32_head == config.model_defaults.fp32_lm_head, name
        moe_configs = [moe for _, moe, _, _ in model.traverse(MoE.Config)]
        assert all(moe.freeze_expert_bias for moe in moe_configs), name
        assert all(moe.router.freeze_gate for moe in moe_configs), name
        assert all(moe.router.aux_loss is None for moe in moe_configs), name
        assert model.local_compile_regions == regions_before, name


def test_model_defaults_disable_router_aux_loss() -> None:
    # DeepSeek-V3 builds a load-balancing aux loss on every MoE router.
    model = build_deepseek_v3_model_config("debugmodel")
    routers = [moe.router for _, moe, _, _ in model.traverse(MoE.Config)]
    assert routers and all(router.aux_loss is not None for router in routers)

    RLModelDefaults(disable_router_aux_loss=False).apply_(model)
    assert all(router.aux_loss is not None for router in routers)

    RLModelDefaults().apply_(model)
    assert all(router.aux_loss is None for router in routers)


@pytest.mark.parametrize("freeze_router_gate", [True, False])
def test_frozen_router_gate_stays_at_loaded_value(freeze_router_gate: bool) -> None:
    model = build_qwen3_5_model_config("debugmodel_moe", seq_len=64)
    model = RLModelDefaults(freeze_router_gate=freeze_router_gate).apply_(model)
    torch.manual_seed(0)
    moe = model.layers[0].moe.build()
    moe.init_states()
    params_before = {
        name: param.detach().clone() for name, param in moe.named_parameters()
    }
    optimizers = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=1e-2, fused=False)]
    ).build(model_parts=[moe])

    moe(torch.randn(32, model.dim)).sum().backward()
    optimizers.step()

    is_changed = {
        name: not torch.equal(param, params_before[name])
        for name, param in moe.named_parameters()
    }
    assert is_changed.pop("router.gate.weight") is not freeze_router_gate
    # Every other parameter trains, including the shared expert's sigmoid gate.
    assert "shared_experts.gate.weight" in is_changed
    assert all(is_changed.values()), is_changed
    gate_has_state = any(
        moe.router.gate.weight in optimizer.state for optimizer in optimizers
    )
    assert gate_has_state is not freeze_router_gate
