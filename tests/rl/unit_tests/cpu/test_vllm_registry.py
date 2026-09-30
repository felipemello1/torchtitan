# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import ModuleType, SimpleNamespace

from torchtitan.config import OverrideConfig
from torchtitan.models.qwen3 import model_registry
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.model.vllm_registry import (
    _configure_gdn_hybrid_model,
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
)

from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import try_get_generation_config


def test_missing_generation_config_adds_no_stop_token(tmp_path):
    """Without generation_config.json, vLLM reads eos from our config; it must add no stop id."""
    register_to_vllm(
        model_registry("debugmodel", attn_backend="varlen"),
        parallelism=InferenceParallelismConfig(),
        compile_config=None,
        checkpointer_config=None,
        override=OverrideConfig(),
    )
    # tmp_path has no generation_config.json, like the Qwen/Qwen3.5-9B HF repo.
    generation_config = try_get_generation_config(
        str(tmp_path),
        trust_remote_code=False,
        config_format=TORCHTITAN_CONFIG_FORMAT,
    )
    renderer_stop_ids = [248046, 248044]
    sampling_params = SamplingParams(stop_token_ids=list(renderer_stop_ids))

    sampling_params.update_from_generation_config(
        generation_config.to_diff_dict(), eos_token_id=renderer_stop_ids[0]
    )

    assert sorted(sampling_params.stop_token_ids) == sorted(renderer_stop_ids)


def test_gdn_hybrid_model_registers_state_copy_funcs(monkeypatch):
    copy_funcs = (object(), object())

    class FakeStateCopyFuncCalculator:
        @staticmethod
        def gated_delta_net_state_copy_func():
            return copy_funcs

    mamba_utils = ModuleType("vllm.model_executor.layers.mamba.mamba_utils")
    mamba_utils.MambaStateCopyFuncCalculator = FakeStateCopyFuncCalculator
    mamba_utils.MambaStateDtypeCalculator = object()
    mamba_utils.MambaStateShapeCalculator = object()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.mamba.mamba_utils",
        mamba_utils,
    )

    gdn_config = SimpleNamespace(
        in_proj_q=SimpleNamespace(out_features=8),
        in_proj_v=SimpleNamespace(out_features=12),
        key_head_dim=4,
        value_head_dim=6,
        conv_kernel_size=4,
    )
    model_config = SimpleNamespace(layers=[SimpleNamespace(delta_net=gdn_config)])

    class Model:
        pass

    _configure_gdn_hybrid_model(Model, model_config)

    gdn_type = object()
    short_conv_type = object()
    assert Model.get_mamba_state_copy_func() is copy_funcs
    assert Model.get_mamba_state_copy_funcs({gdn_type, short_conv_type}) == {
        gdn_type: copy_funcs,
        short_conv_type: copy_funcs,
    }
