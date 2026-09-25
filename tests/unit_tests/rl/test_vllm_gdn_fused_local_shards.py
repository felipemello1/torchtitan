# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import inspect

import pytest
import torch
import torch.nn as nn

from torchtitan.config.transform.base import convert_config_type
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.rl.model.gdn import (
    fused_inner_sharding_config,
    VLLMGatedDeltaNet,
    VLLMInnerGatedDeltaNet,
)


def _inner_math(
    mixed_qkv_TC, a_TH, b_TH, conv_weight_CW, A_log_H, dt_bias_H, value_head_dim
):
    """Stand-in recurrence that reads every input, so the test covers the projections."""
    conv_TC = mixed_qkv_TC * conv_weight_CW[:, -1]
    num_tokens, num_heads = a_TH.shape
    value_THV = conv_TC[:, -num_heads * value_head_dim :].view(
        num_tokens, num_heads, value_head_dim
    )
    decay_TH = a_TH * A_log_H + dt_bias_H + conv_TC[:, :1]
    return value_THV * torch.sigmoid(b_TH).unsqueeze(-1) + decay_TH.unsqueeze(-1)


class _PerProjectionInner(nn.Module):
    """Stand-in for the model's inner GDN (one input per projection)."""

    def forward(
        self,
        query_TC,
        key_TC,
        value_TC,
        a_TH,
        b_TH,
        conv_q,
        conv_k,
        conv_v,
        A_log_H,
        dt_bias_H,
        cu_seqlens,
        *,
        key_head_dim,
        value_head_dim,
    ):
        mixed_qkv_TC = torch.cat([query_TC, key_TC, value_TC], dim=-1)
        conv_weight_CW = torch.cat([conv_q, conv_k, conv_v]).squeeze(1)
        return _inner_math(
            mixed_qkv_TC, a_TH, b_TH, conv_weight_CW, A_log_H, dt_bias_H, value_head_dim
        )


class _FusedInner(nn.Module):
    """Stand-in for VLLMInnerGatedDeltaNet (fused [q|k|v] and conv weight)."""

    def forward(
        self,
        mixed_qkv_TC,
        a_TH,
        b_TH,
        conv_weight_CW,
        A_log_H,
        dt_bias_H,
        cu_seqlens,
        *,
        key_head_dim,
        value_head_dim,
    ):
        return _inner_math(
            mixed_qkv_TC, a_TH, b_TH, conv_weight_CW, A_log_H, dt_bias_H, value_head_dim
        )


def _local_rows(weight, num_heads, rank, tp):
    """Rows of the head shard that TP rank ``rank`` of ``tp`` holds."""
    return weight.view(num_heads, -1, *weight.shape[1:]).chunk(tp)[rank].flatten(0, 1)


def _shard_like_tp(module, full_state, rank, tp):
    """Load the parameters TP rank ``rank`` holds, as colwise/rowwise sharding lays them out."""
    num_k_heads = module.in_proj_q.weight.shape[0] // module.key_head_dim
    num_v_heads = module.in_proj_v.weight.shape[0] // module.value_head_dim
    heads = {
        "in_proj_q": num_k_heads,
        "in_proj_k": num_k_heads,
        "conv_q": num_k_heads,
        "conv_k": num_k_heads,
        "in_proj_v": num_v_heads,
        "in_proj_z": num_v_heads,
        "in_proj_a": num_v_heads,
        "in_proj_b": num_v_heads,
        "conv_v": num_v_heads,
    }
    for name, num_heads in heads.items():
        getattr(module, name).weight = nn.Parameter(
            _local_rows(full_state[f"{name}.weight"], num_heads, rank, tp),
            requires_grad=False,
        )
    module.A_log = nn.Parameter(
        full_state["A_log"].chunk(tp)[rank], requires_grad=False
    )
    module.dt_bias = nn.Parameter(
        full_state["dt_bias"].chunk(tp)[rank], requires_grad=False
    )
    module.norm.weight = nn.Parameter(full_state["norm.weight"], requires_grad=False)
    # Row-parallel out_proj: each rank holds its heads' input columns (partial output).
    module.out_proj.weight = nn.Parameter(
        full_state["out_proj.weight"].chunk(tp, dim=1)[rank], requires_grad=False
    )


@pytest.mark.parametrize("tp", [1, 2])
def test_fused_local_shards_match_per_projection_path(tp):
    config = model_registry("debugmodel")
    base_config = next(layer.delta_net for layer in config.layers if layer.delta_net)
    fused_config = convert_config_type(
        dataclasses.replace(base_config), VLLMGatedDeltaNet
    )
    torch.manual_seed(0)
    reference = base_config.build()
    with torch.no_grad():
        for param in reference.parameters():
            param.normal_(0, 0.1)
    full_state = {name: value.clone() for name, value in reference.state_dict().items()}
    x_TD = torch.randn(6, reference.in_proj_q.in_features)

    for rank in range(tp):
        per_projection = base_config.build()
        fused = fused_config.build()
        for module in (per_projection, fused):
            _shard_like_tp(module, full_state, rank, tp)
        per_projection.inner_gated_delta_net = _PerProjectionInner()
        fused.inner_gated_delta_net = _FusedInner()
        fused.share_input_storage()

        assert fused.state_dict().keys() == per_projection.state_dict().keys()
        with torch.no_grad():
            torch.testing.assert_close(fused(x_TD), per_projection(x_TD))


def test_loading_writes_through_the_shared_storage():
    config = model_registry("debugmodel")
    base_config = next(layer.delta_net for layer in config.layers if layer.delta_net)
    fused = convert_config_type(
        dataclasses.replace(base_config), VLLMGatedDeltaNet
    ).build()
    fused.share_input_storage()
    new_z = torch.randn_like(fused.in_proj_z.weight)
    new_conv_k = torch.randn_like(fused.conv_k.weight)
    fused.load_state_dict(
        {"in_proj_z.weight": new_z, "conv_k.weight": new_conv_k}, strict=False
    )

    q, k, v, z = fused.in_proj_split_sizes[:4]
    torch.testing.assert_close(fused.in_proj_weight[q + k + v : q + k + v + z], new_z)
    conv_q = fused.conv_q.weight.shape[0]
    torch.testing.assert_close(
        fused.conv_weight[conv_q : conv_q + new_conv_k.shape[0]], new_conv_k.squeeze(1)
    )


def test_fused_inner_sharding_matches_the_inner_signature():
    names = (
        "query_TC",
        "key_TC",
        "value_TC",
        "a_TH",
        "b_TH",
        "conv_q_weight_C1W",
        "conv_k_weight_C1W",
        "conv_v_weight_C1W",
        "A_log_H",
        "dt_bias_H",
        "cu_seqlens",
    )
    per_projection = ShardingConfig(
        in_src_shardings=dict.fromkeys(names), in_dst_shardings=dict.fromkeys(names)
    )
    fused = fused_inner_sharding_config(per_projection)
    positional = [
        name
        for name, parameter in inspect.signature(
            VLLMInnerGatedDeltaNet.forward
        ).parameters.items()
        if name != "self" and parameter.kind is parameter.POSITIONAL_OR_KEYWORD
    ]
    assert list(fused.in_src_shardings) == list(fused.in_dst_shardings) == positional
