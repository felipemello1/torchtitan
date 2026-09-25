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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires a Blackwell GPU",
)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def test_flashinfer_prefill_matches_attention_gym() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    # Qwen3.5-4B's local shapes: 16 key heads, 32 value heads.
    num_k_heads, num_v_heads, head_dim = 16, 32, 128
    conv_dim = (2 * num_k_heads + num_v_heads) * head_dim
    # Request 0 continues from slot 1, request 1 starts fresh in slot 3, and 20
    # padded rows follow. Metadata capacity is 4 requests plus the null interval.
    num_real_tokens, num_actual_tokens = 140, 160
    metadata = TorchTitanGDNAttentionMetadata(
        execution_path=GDNExecutionPath.PACKED,
        num_prefills=2,
        num_prefill_tokens=num_real_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_actual_tokens,
        non_spec_query_start_loc=torch.tensor(
            [
                0,
                50,
                num_real_tokens,
                num_actual_tokens,
                num_actual_tokens,
                num_actual_tokens,
            ],
            device=device,
            dtype=torch.int32,
        ),
        non_spec_state_indices_tensor=torch.tensor(
            [1, 3, 0, 0, 0], device=device, dtype=torch.int32
        ),
        has_initial_state=torch.tensor(
            [True, False, False, False, False], device=device
        ),
    )
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={"gdn": metadata}, slot_mapping={}
    )
    layer = gdn.VLLMInnerGatedDeltaNet.__new__(gdn.VLLMInnerGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.prefix = "gdn"
    layer.local_num_k_heads, layer.local_num_v_heads = num_k_heads, num_v_heads
    layer.head_k_dim = layer.head_v_dim = head_dim
    layer.local_key_dim = num_k_heads * head_dim
    conv_pool = torch.randn(6, 3, conv_dim, device=device, dtype=torch.bfloat16)
    ssm_pool = torch.randn(6, num_v_heads, head_dim, head_dim, device=device)
    mixed_qkv = torch.randn(
        num_actual_tokens, conv_dim, device=device, dtype=torch.bfloat16
    )
    conv_weight = torch.randn(conv_dim, 4, device=device, dtype=torch.bfloat16) * 0.5
    a, b = torch.randn(
        2, num_actual_tokens, num_v_heads, device=device, dtype=torch.bfloat16
    )
    A_log = torch.rand(num_v_heads, device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn(num_v_heads, device=device, dtype=torch.bfloat16)

    results = {}
    for use_flashinfer in (False, True):
        layer.use_flashinfer_prefill = use_flashinfer
        layer.kv_cache = (conv_pool.clone(), ssm_pool.clone())
        output = torch.zeros(
            num_actual_tokens,
            num_v_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        with torch.inference_mode(), override_forward_context(context):
            layer._forward(mixed_qkv, a, b, conv_weight, None, A_log, dt_bias, output)
        results[use_flashinfer] = (output, *layer.kv_cache)

    (expected, expected_conv, expected_ssm) = results[False]
    (actual, actual_conv, actual_ssm) = results[True]
    assert _relative_error(actual[:num_real_tokens], expected[:num_real_tokens]) < 1e-2
    assert not actual[num_real_tokens:].count_nonzero()
    torch.testing.assert_close(actual_conv, expected_conv, rtol=0, atol=0)
    for slot in (1, 3):
        assert _relative_error(actual_ssm[slot], expected_ssm[slot]) < 1e-2
    untouched = [0, 2, 4, 5]
    assert torch.equal(actual_ssm[untouched], ssm_pool[untouched])
