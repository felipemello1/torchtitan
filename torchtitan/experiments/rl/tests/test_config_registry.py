# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.rl.config_registry import (
    rl_grpo_qwen3_0_6b_alphabet_sort,
    rl_grpo_qwen3_1_7b,
    rl_grpo_qwen3_1_7b_alphabet_sort,
)


def test_1_7b_configs_disable_compile():
    assert not rl_grpo_qwen3_1_7b().compile.enable
    cfg = rl_grpo_qwen3_1_7b_alphabet_sort()
    assert not cfg.compile.enable
    assert cfg.trainer.max_microbatch_samples == 4


def test_alphabet_sort_0_6b_defaults_keep_compile_and_disable_thinking():
    cfg = rl_grpo_qwen3_0_6b_alphabet_sort()

    assert cfg.compile.enable
    assert cfg.renderer.name == "qwen3"
    assert cfg.renderer.enable_thinking is False
    assert cfg.trainer.max_microbatch_samples == 8
