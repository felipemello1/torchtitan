# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch

from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.config.override import _REGISTRY
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.qwen3_6 import build_model_config
from torchtitan_recipes.overrides.dist_moe import dist_moe_routed_experts


_TARGET = f"{dist_moe_routed_experts.__module__}.{dist_moe_routed_experts.__name__}"
_DIST_MOE_OVERRIDE = _REGISTRY[_TARGET]


def _apply_dist_moe_override(model_config) -> list[str]:
    # Other tests clear the global registry; restore this module's override.
    _REGISTRY.setdefault(_TARGET, _DIST_MOE_OVERRIDE)
    return apply_overrides(OverrideConfig(imports=[_TARGET]), model_config)


def test_override_rewrites_only_the_copy_it_is_applied_to() -> None:
    """The trainer's copy gets Dist-MoE experts; the shared config stays stock."""
    shared = build_model_config("debugmodel_moe", attn_backend="varlen")
    num_moe_layers = len(list(shared.traverse(RoutedExperts.Config)))
    trainer_copy = copy.deepcopy(shared)
    trainer_copy.set_sharding_(
        ParallelismConfig(
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            expert_parallel_degree=4,
        )
    )
    stock_w13_shardings = [
        cfg.w13.sharding_config
        for _, cfg, _, _ in trainer_copy.traverse(RoutedExperts.Config)
    ]

    replacements = _apply_dist_moe_override(trainer_copy)

    assert num_moe_layers > 0
    assert len(replacements) == num_moe_layers
    assert len(list(shared.traverse(RoutedExperts.Config))) == num_moe_layers
    assert not list(shared.traverse(DistMoeRoutedExperts.Config))
    dist_moe_configs = [
        cfg for _, cfg, _, _ in trainer_copy.traverse(DistMoeRoutedExperts.Config)
    ]
    assert len(dist_moe_configs) == num_moe_layers
    for cfg, stock_sharding in zip(dist_moe_configs, stock_w13_shardings, strict=True):
        assert stock_sharding is not None
        assert cfg.w13.sharding_config is stock_sharding


def test_override_keeps_state_dict_keys_and_shapes() -> None:
    """Trainer and generator models stay interchangeable for weight sync."""
    stock = build_model_config("debugmodel_moe", attn_backend="varlen")
    overridden = copy.deepcopy(stock)
    _apply_dist_moe_override(overridden)

    with torch.device("meta"):
        stock_state = stock.build().state_dict()
        overridden_state = overridden.build().state_dict()

    assert {key: value.shape for key, value in overridden_state.items()} == {
        key: value.shape for key, value in stock_state.items()
    }
