# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dist-MoE routed experts for one actor's model config, e.g. only the RL trainer.

The RL trainer and generator share one ``model_config``. ``DistMoeTransform``
rewrites that shared config, so the vLLM generator would also receive Dist-MoE
experts. This override applies the same replacement to the copy owned by the
actor that lists it::

    trainer.override = OverrideConfig(
        imports=["torchtitan_recipes.overrides.dist_moe.dist_moe_routed_experts"]
    )
    trainer.dist_moe = DistMoeRuntime.Config(device_scratch_capacity_factor=4.0)

The generator keeps the stock ``RoutedExperts``. Checkpoint and weight-sync keys
do not change.
"""

from torchtitan.config import override
from torchtitan.config.transform import DistMoeTransform
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.protocols.module import Module


__all__ = ["dist_moe_routed_experts"]


@override(
    target=RoutedExperts.Config,
    description="BF16 Dist-MoE routed experts (SM100+); needs trainer.dist_moe.",
)
def dist_moe_routed_experts(cfg: RoutedExperts.Config) -> Module.Config:
    return DistMoeTransform().transform(cfg)
