# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import cast

from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.models.deepseek_v3.model import DeepSeekV3Model
from torchtitan_recipes.overrides.quack_rmsnorm import _QUACK_IMPORT_ERROR, QuackRMSNorm
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel_mtp


@unittest.skipIf(_QUACK_IMPORT_ERROR is not None, "requires quack")
class TestQuackRMSNormOverride(unittest.TestCase):
    def test_override_claims_only_block_norms(self):
        config = deepseek_v3_debugmodel_mtp(seq_len=2048)
        model_config = cast(DeepSeekV3Model.Config, config.model)
        stock_norm_config = copy.deepcopy(model_config.layers[0].attention_norm)

        replacements = apply_overrides(
            OverrideConfig(
                imports=["torchtitan_recipes.overrides.quack_rmsnorm.quack_rmsnorm"]
            ),
            config,
        )

        layers = [*model_config.layers, *model_config.mtp_layers]
        self.assertEqual(len(replacements), 2 * len(layers))
        for layer in layers:
            self.assertIsInstance(layer.attention_norm, QuackRMSNorm.Config)
            self.assertIsInstance(layer.ffn_norm, QuackRMSNorm.Config)
            self.assertNotIsInstance(layer.attention.kv_norm, QuackRMSNorm.Config)
        self.assertNotIsInstance(model_config.norm, QuackRMSNorm.Config)
        self.assertEqual(
            list(stock_norm_config.build().state_dict()),
            list(layers[0].attention_norm.build().state_dict()),
        )


if __name__ == "__main__":
    unittest.main()
