# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest

from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MoE

_RECIPE_MODULES = [
    "torchtitan_recipes.rl.alphabet_sort",
    "torchtitan_recipes.rl.dapo_math",
    "torchtitan_recipes.rl.search_r1",
    "torchtitan_recipes.rl.verifiers_dapo_math",
    "torchtitan_recipes.rl.verifiers_terminal_bench",
    "torchtitan_recipes.tests.rl.alphabet_sort",
]


@pytest.mark.parametrize("module_name", _RECIPE_MODULES)
def test_rl_recipes_get_model_defaults(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # dapo_math and the verifiers recipes import optional reward packages (math_verify, verifiers).
    module = pytest.importorskip(module_name)
    if module_name == "torchtitan_recipes.rl.verifiers_terminal_bench":
        # Its Sandoq recipes read the sandbox pool at load time and import the Sandoq plugin.
        pytest.importorskip("harbor")
        from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq

        monkeypatch.syspath_prepend(str(Path(terminal_bench_sandoq.__file__).parent))
        monkeypatch.setenv("DOME_SANDOQ_POOL", "920")
        monkeypatch.setenv("VF_SANDBOX_PROVIDER", "oci-runner")
        monkeypatch.setenv("OCI_RUNNER_TASK_NETWORK", "host")
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
