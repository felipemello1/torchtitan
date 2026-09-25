# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import torch

from torchtitan.config.transform import convert_config_type
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.rl.model.gdn import VLLMFusedGatedDeltaNet


class _ValueKernel(torch.nn.Module):
    """Stand-in recurrence that returns v, so the test covers only the projections."""

    def forward(self, xq_THK, xk_THK, xv_THV, g_TH, beta_TH, *, cu_seqlens=None):
        return xv_THV * torch.sigmoid(beta_TH).unsqueeze(-1) + g_TH.unsqueeze(-1)


def _build_pair():
    config = model_registry("debugmodel", fuse_gdn_input_projections=True)
    fused_config = next(layer.delta_net for layer in config.layers if layer.delta_net)
    single_gemm_config = convert_config_type(
        dataclasses.replace(fused_config), VLLMFusedGatedDeltaNet
    )
    torch.manual_seed(0)
    fused = fused_config.build()
    single_gemm = single_gemm_config.build()
    with torch.no_grad():
        for param in fused.parameters():
            param.normal_(0, 0.1)
    single_gemm.load_state_dict(fused.state_dict())
    for module in (fused, single_gemm):
        module.inner_gated_delta_net.kernel = _ValueKernel()
    single_gemm.share_input_projection_storage()
    return fused, single_gemm


def test_single_gemm_matches_two_gemms_and_keeps_state_dict_keys():
    fused, single_gemm = _build_pair()
    assert single_gemm.state_dict().keys() == fused.state_dict().keys()

    x_TD = torch.randn(6, fused.in_proj_qkv.in_features)
    torch.testing.assert_close(single_gemm(x_TD), fused(x_TD))


def test_loading_projection_weights_writes_into_shared_buffer():
    _, single_gemm = _build_pair()
    new_zab = torch.randn_like(single_gemm.in_proj_zab.weight)
    single_gemm.load_state_dict({"in_proj_zab.weight": new_zab}, strict=False)

    num_qkv_rows = single_gemm.in_proj_qkv.weight.shape[0]
    torch.testing.assert_close(single_gemm.in_proj_weight[num_qkv_rows:], new_zab)
