# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.rl.model.attention import TorchTitanVarlenInnerAttentionImpl
from vllm.v1.attention.backend import AttentionType


def test_paged_attention_runs_flash_not_cudnn():
    """The generator calls torch's varlen_attn_out with only the Flash backend enabled."""
    num_tokens, num_heads, head_dim, page_size = 4, 2, 8, 16
    impl = TorchTitanVarlenInnerAttentionImpl.__new__(
        TorchTitanVarlenInnerAttentionImpl
    )
    impl.vllm_flash_attn_version = 3
    impl.attn_type = AttentionType.DECODER
    impl.head_size = head_dim
    impl.kv_cache_dtype = "auto"
    impl.dcp_world_size = 1
    impl.sliding_window = (-1, -1)
    impl.alibi_slopes = None
    impl.enable_gqa = False
    impl.out_transform = None
    impl.scale = head_dim**-0.5
    attn_metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        use_cascade=False,
        causal=True,
        query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
        seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        max_query_len=num_tokens,
        max_seq_len=num_tokens,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    # [num_pages, num_kv_heads, page_size, K and V]
    kv_cache = torch.zeros(1, num_heads, page_size, 2 * head_dim)
    query = torch.zeros(num_tokens, num_heads, head_dim)
    output = torch.empty(num_tokens, num_heads, head_dim)
    enabled_backends = []

    def record_enabled_backends(out, *args, **kwargs):
        enabled_backends.append(
            {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            }
        )
        return out

    with (
        # Importing vLLM disables cuDNN process-wide; turn it on to test the pin itself.
        sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]),
        patch(
            "torchtitan.rl.model.attention.get_attention_context",
            return_value=(attn_metadata, None, kv_cache, None),
        ),
        patch(
            "torchtitan.rl.model.attention.current_flash_attention_impl",
            return_value="FA3",
        ),
        patch(
            "torch.nn.attention.varlen.varlen_attn_out",
            side_effect=record_enabled_backends,
        ),
    ):
        impl.forward(
            SimpleNamespace(layer_name="layers.0.attention"),
            query,
            query,
            query,
            kv_cache,
            attn_metadata,
            output=output,
        )
        # The pin only covers the varlen call.
        assert torch.backends.cuda.cudnn_sdp_enabled()

    assert enabled_backends == [{"flash": True, "cudnn": False}]
