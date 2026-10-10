# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MoE

_RECIPE_MODULES = [
    "torchtitan_recipes.rl.alphabet_sort",
    "torchtitan_recipes.rl.dapo_math",
    "torchtitan_recipes.rl.search_r1",
    "torchtitan_recipes.rl.verifiers_dapo_math",
    "torchtitan_recipes.tests.rl.alphabet_sort",
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
        assert model.local_compile_regions == regions_before, name
