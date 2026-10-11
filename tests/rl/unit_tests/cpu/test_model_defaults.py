# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import (
    MoE,
    register_moe_load_balancing_hook,
    register_moe_quantile_balancing_hook,
)
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.rl.controller import RLModelDefaults

_RECIPE_MODULES = [
    "torchtitan_recipes.rl.alphabet_sort",
    "torchtitan_recipes.rl.dapo_math",
    "torchtitan_recipes.rl.search_r1",
    "torchtitan_recipes.rl.verifiers_dapo_math",
    "torchtitan_recipes.rl.verifiers_terminal_bench",
    "torchtitan_recipes.tests.rl.alphabet_sort",
]
# Recipes that need an optional package: torchao nightly (MXFP8) and dist_moe.
_OPTIONAL_PACKAGE_RECIPES = {
    "rl_grpo_qwen3_0_6b_varlen_mxfp8",
    "rl_grpo_moe_debug_dist_moe_tp2_ep4",
}


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
        try:
            config = recipe()
        except ImportError:
            if recipe.__name__ not in _OPTIONAL_PACKAGE_RECIPES:
                raise
            continue
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


# (family, debug flavor, drop_expert_bias): one entry per MoE family in torchtitan/models.
_MOE_DEBUG_MODELS = [
    ("qwen3", "debugmodel_moe", False),
    ("qwen3", "debugmodel_moe", True),  # load_balance_coeff=None: no expert bias
    ("qwen3_5", "debugmodel_moe", False),  # shared-expert sigmoid gate
    ("qwen3_6", "debugmodel_moe", False),
    ("qwen3_8", "debugmodel_moe", False),
    ("kimi_k2_7", "debugmodel", False),
    ("kimi_k3", "debugmodel", False),  # quantile-balanced router and hook
    ("deepseek_v3", "debugmodel", False),  # router aux loss
    ("deepseek_v4", "debugmodel", False),
    ("gpt_oss", "debugmodel", False),  # gate bias
]


@pytest.mark.parametrize("freeze", [True, False])
@pytest.mark.parametrize("family,flavor,drop_expert_bias", _MOE_DEBUG_MODELS)
def test_model_defaults_freeze_every_moe_family(
    family: str, flavor: str, drop_expert_bias: bool, freeze: bool
) -> None:
    model = importlib.import_module(f"torchtitan.models.{family}").build_model_config(
        flavor
    )
    if drop_expert_bias:
        for _, moe_config, _, _ in model.traverse(MoE.Config):
            moe_config.load_balance_coeff = None
    defaults = RLModelDefaults(freeze_router_gate=freeze, freeze_expert_bias=freeze)
    model = defaults.apply_(model)
    # The last MoE layer: DeepSeek-V4's first MoE layers route by token id hash.
    moe_config = [moe for _, moe, _, _ in model.traverse(MoE.Config)][-1]
    torch.manual_seed(0)
    moe = moe_config.build()
    moe.init_states()
    params_before = {
        name: param.detach().clone() for name, param in moe.named_parameters()
    }
    expert_bias_before = (
        None if moe.expert_bias_E is None else moe.expert_bias_E.clone()
    )
    # The balancing hooks look for MoE blocks under `layers`.
    block = nn.Module()
    block.moe, block.moe_enabled = moe, True
    model_part = nn.ModuleDict({"layers": nn.ModuleList([block])})
    optimizers = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=1e-2, fused=False)]
    ).build(model_parts=[model_part])
    parallelism = SimpleNamespace(
        ep_enabled=False, tp=1, get_optional_mesh=lambda name: None
    )
    register_moe_load_balancing_hook(optimizers, [model_part], parallelism)
    register_moe_quantile_balancing_hook(optimizers, [model_part], parallelism)

    moe(torch.randn(64, model.dim)).sum().backward()
    optimizers.step()

    assert moe.router.aux_loss is None
    if drop_expert_bias:
        assert moe.expert_bias_E is None
    else:
        assert torch.equal(moe.expert_bias_E, expert_bias_before) == freeze
    for name, param in moe.named_parameters():
        # router.gate.weight, and router.gate.bias on gpt-oss. Every other parameter
        # trains, including the Qwen3.5 shared-expert gate.
        is_frozen = freeze and name.startswith("router.gate.")
        assert torch.equal(param, params_before[name]) == is_frozen, name
        has_state = any(param in optimizer.state for optimizer in optimizers)
        assert has_state != is_frozen, name
