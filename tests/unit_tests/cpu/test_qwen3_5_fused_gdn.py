# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("attn_gym")

from torchtitan.config import ParallelismConfig
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.gdn import FusedGatedDeltaNet, GatedDeltaNet
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter


def test_registry_fuses_every_deltanet_input_projection() -> None:
    separate_config = model_registry("debugmodel")
    fused_config = model_registry("debugmodel", fuse_gdn_input_projections=True)

    num_deltanet_layers = 0
    for separate_layer, fused_layer in zip(
        separate_config.layers, fused_config.layers, strict=True
    ):
        if separate_layer.delta_net is None:
            assert fused_layer.delta_net is None
            continue
        num_deltanet_layers += 1
        separate = separate_layer.delta_net
        fused = fused_layer.delta_net
        assert isinstance(separate, GatedDeltaNet.Config)
        assert isinstance(fused, FusedGatedDeltaNet.Config)

        assert fused.num_key_heads == separate.num_key_heads
        assert fused.num_value_heads == separate.num_value_heads
        key_dim = separate.in_proj_q.out_features
        value_dim = separate.in_proj_v.out_features
        assert fused.in_proj_qkv.out_features == 2 * key_dim + value_dim
        assert (
            fused.in_proj_zab.out_features == value_dim + 2 * separate.num_value_heads
        )
        assert fused.conv1d.groups == 2 * key_dim + value_dim
    assert num_deltanet_layers > 0


def test_fused_deltanet_state_dict_round_trips_through_hf() -> None:
    config = model_registry("debugmodel", fuse_gdn_input_projections=True)
    adapter = Qwen35StateDictAdapter(config, hf_assets_path=None)
    delta_net = config.layers[0].delta_net
    assert isinstance(delta_net, FusedGatedDeltaNet.Config)
    key_dim = delta_net.num_key_heads * delta_net.key_head_dim
    value_dim = delta_net.num_value_heads * delta_net.value_head_dim
    num_value_heads = delta_net.num_value_heads
    dim = delta_net.in_proj_qkv.in_features

    prefix = "model.language_model.layers.0.linear_attn"
    hf_state_dict = {
        f"{prefix}.in_proj_qkv.weight": torch.randn(2 * key_dim + value_dim, dim),
        f"{prefix}.conv1d.weight": torch.randn(2 * key_dim + value_dim, 1, 4),
        f"{prefix}.in_proj_z.weight": torch.randn(value_dim, dim),
        f"{prefix}.in_proj_a.weight": torch.randn(num_value_heads, dim),
        f"{prefix}.in_proj_b.weight": torch.randn(num_value_heads, dim),
        "lm_head.weight": torch.randn(2, 3),
    }

    state_dict = adapter.from_hf(dict(hf_state_dict))
    assert set(state_dict) == {
        "layers.0.attn.in_proj_qkv.weight",
        "layers.0.attn.conv1d.weight",
        "layers.0.attn.in_proj_zab.weight",
        "lm_head.weight",
    }
    torch.testing.assert_close(
        state_dict["layers.0.attn.in_proj_qkv.weight"],
        hf_state_dict[f"{prefix}.in_proj_qkv.weight"],
    )
    torch.testing.assert_close(
        state_dict["layers.0.attn.in_proj_zab.weight"],
        torch.cat(
            [
                hf_state_dict[f"{prefix}.in_proj_z.weight"],
                hf_state_dict[f"{prefix}.in_proj_a.weight"],
                hf_state_dict[f"{prefix}.in_proj_b.weight"],
            ]
        ),
    )

    restored = adapter.to_hf(state_dict)
    assert set(restored) == set(hf_state_dict)
    for key, value in hf_state_dict.items():
        torch.testing.assert_close(restored[key], value)


def test_fused_deltanet_rejects_tensor_parallel() -> None:
    config = model_registry("debugmodel", fuse_gdn_input_projections=True)
    runtime_config = SimpleNamespace(
        parallelism=ParallelismConfig(tensor_parallel_degree=2)
    )
    with pytest.raises(ValueError, match="fuse_gdn_input_projections=False"):
        config.update_from_config(config=runtime_config)
