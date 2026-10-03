# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified DeepSeek V3 GraphTrainer recipes."""

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.deepseek_v3 import build_model_config

from torchtitan_recipes.models.deepseek_v3 import deepseek_v3_16b


def graph_trainer_deepseek_v3_16b() -> GraphTrainer.Config:
    base_config = deepseek_v3_16b(seq_len=4096)
    # GraphTrainer is not validated with varlen attention, so it keeps flex.
    base_config.model = build_model_config("16B", seq_len=4096, attn_backend="flex")
    config = to_graph_trainer_config(base_config, GraphTrainerDeepSeekV3Model.Config)
    config.compile = GraphTrainerCompileConfig()
    return config
