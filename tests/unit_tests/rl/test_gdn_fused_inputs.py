# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.rl.model import gdn
from torchtitan.rl.model.gdn_backend import (
    GDNExecutionPath,
    TorchTitanGDNAttentionMetadata,
)
from vllm.forward_context import ForwardContext, override_forward_context

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM, CONV_KERNEL_SIZE = 2, 4, 128, 4
KEY_DIM, VALUE_DIM = NUM_K_HEADS * HEAD_DIM, NUM_V_HEADS * HEAD_DIM
QKV_CHANNELS = 2 * KEY_DIM + VALUE_DIM


def _layer(cls, kv_cache):
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.prefix = "gdn"
    layer.local_num_k_heads, layer.local_num_v_heads = NUM_K_HEADS, NUM_V_HEADS
    layer.head_k_dim = layer.head_v_dim = HEAD_DIM
    layer.local_key_dim = KEY_DIM
    layer.conv_kernel_size = CONV_KERNEL_SIZE
    layer.kv_cache = kv_cache
    return layer


@pytest.mark.parametrize("decode", [False, True], ids=["prefill", "decode"])
def test_fused_inputs_match_separate_inputs(decode: bool) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    # Two requests: slot 1 continues from its cached state, slot 2 starts fresh.
    query_start_loc = [0, 1, 2] if decode else [0, 5, 12]
    num_tokens = query_start_loc[-1]
    metadata = TorchTitanGDNAttentionMetadata(
        execution_path=(
            GDNExecutionPath.SINGLE_TOKEN if decode else GDNExecutionPath.PACKED
        ),
        num_prefills=0 if decode else 2,
        num_prefill_tokens=0 if decode else num_tokens,
        num_decodes=2 if decode else 0,
        num_decode_tokens=2 if decode else 0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_tokens,
        non_spec_query_start_loc=torch.tensor(
            query_start_loc, device=device, dtype=torch.int32
        ),
        non_spec_state_indices_tensor=torch.tensor(
            [1, 2], device=device, dtype=torch.int32
        ),
        has_initial_state=torch.tensor([True, False], device=device),
    )
    conv_state = torch.randn(
        3, CONV_KERNEL_SIZE - 1, QKV_CHANNELS, device=device, dtype=torch.bfloat16
    )
    ssm_state = torch.randn(3, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device=device)
    separate = _layer(gdn.VLLMInnerGatedDeltaNet, (conv_state, ssm_state))
    fused = _layer(
        gdn.VLLMFusedInnerGatedDeltaNet, (conv_state.clone(), ssm_state.clone())
    )

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=device, dtype=dtype)

    query, key, value = (
        randn(num_tokens, KEY_DIM),
        randn(num_tokens, KEY_DIM),
        randn(num_tokens, VALUE_DIM),
    )
    conv_q, conv_k, conv_v = (
        randn(KEY_DIM, 1, CONV_KERNEL_SIZE),
        randn(KEY_DIM, 1, CONV_KERNEL_SIZE),
        randn(VALUE_DIM, 1, CONV_KERNEL_SIZE),
    )
    # Fused a/b are row slices of one [z|a|b] projection output.
    zab = randn(num_tokens, VALUE_DIM + 2 * NUM_V_HEADS)
    a, b = zab[:, VALUE_DIM:].split(NUM_V_HEADS, dim=-1)
    A_log = randn(NUM_V_HEADS, dtype=torch.float32)
    dt_bias = randn(NUM_V_HEADS, dtype=torch.float32)
    cu_seqlens = metadata.non_spec_query_start_loc

    def context(layer):
        # Each layer runs under its own context, so the name maps to that layer.
        return override_forward_context(
            ForwardContext(
                no_compile_layers={"gdn": layer},
                attn_metadata={"gdn": metadata},
                slot_mapping={},
            )
        )

    with torch.inference_mode(), context(separate):
        expected = separate(
            query,
            key,
            value,
            a.contiguous(),
            b.contiguous(),
            conv_q,
            conv_k,
            conv_v,
            A_log,
            dt_bias,
            cu_seqlens,
            key_head_dim=HEAD_DIM,
            value_head_dim=HEAD_DIM,
        )
    with torch.inference_mode(), context(fused):
        actual = fused(
            torch.cat([query, key, value], dim=-1),
            a,
            b,
            torch.cat([conv_q, conv_k, conv_v]),
            A_log,
            dt_bias,
            cu_seqlens,
            key_head_dim=HEAD_DIM,
            value_head_dim=HEAD_DIM,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for fused_state, separate_state in zip(fused.kv_cache, separate.kv_cache):
        torch.testing.assert_close(fused_state, separate_state, rtol=0, atol=0)
