# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch

from torchtitan.rl.generator.vllm import kda
from torchtitan.rl.generator.vllm.linear_attention import (
    LinearAttentionExecutionPath,
    TorchTitanLinearAttentionMetadata,
)
from vllm.forward_context import ForwardContext, override_forward_context


@pytest.mark.parametrize("path", ["prefill", "decode", "decode_replay"])
def test_forward_zeroes_rows_no_sequence_owns(monkeypatch, path):
    """Every output row is written or zeroed; decode passes the 16-bit gate parameters as is."""
    decode = path != "prefill"
    layer = kda.VLLMInnerKDA.__new__(kda.VLLMInnerKDA)
    torch.nn.Module.__init__(layer)
    layer.prefix, layer.lower_bound = "kda", -5.0
    layer.local_num_heads = layer.head_dim = 1
    layer.use_chunk_replay = path == "decode_replay"
    layer.kv_cache = (torch.empty(0),) * (8 if layer.use_chunk_replay else 2)
    extent = 4 if decode else 3
    metadata = TorchTitanLinearAttentionMetadata(
        execution_path=(
            LinearAttentionExecutionPath.SINGLE_TOKEN
            if decode
            else LinearAttentionExecutionPath.PACKED
        ),
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=extent,
        num_decode_tokens=extent,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=3,
        non_spec_query_start_loc=(
            torch.arange(5) if decode else torch.tensor([0, 1, 2, 3, 3])
        ),
        non_spec_state_indices_tensor=(
            torch.tensor([1, 2, 0, 0]) if decode else torch.tensor([1, 2, 3, 0])
        ),
        has_initial_state=torch.tensor([False, True, False, False]),
    )
    A_log = torch.ones(1, dtype=torch.bfloat16)
    dt_bias = torch.ones(1, 1, dtype=torch.bfloat16)

    def identity(x, *args, **kwargs):
        return x

    def recurrent(conv, gate, beta, A_log_H, dt_bias_HK, *args, out, **kwargs):
        assert A_log_H is A_log and dt_bias_HK is dt_bias
        out.fill_(1)

    def chunk(q, *args, **kwargs):
        return torch.ones(*q.shape[:3], 1)

    for name, fn in (
        ("causal_conv1d_decode", identity),
        ("paged_causal_conv1d", identity),
        ("recurrent_kda_decode", recurrent),
        ("paged_chunk_kda", chunk),
        ("l2norm", identity),
        ("bound_gate", identity),
        ("chunk_kernel_options", lambda q: None),
    ):
        monkeypatch.setattr(kda, name, fn)
    # NaN catches a row that is neither written nor zeroed.
    monkeypatch.setattr(
        torch.Tensor,
        "new_empty",
        lambda self, *size: self.new_zeros(*size).fill_(float("nan")),
    )
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={"kda": metadata}, slot_mapping={}
    )
    mixed, gate, beta = torch.ones(6, 3), torch.ones(6, 1, 1), torch.ones(6, 1)
    with override_forward_context(context):
        output = layer._forward(mixed, gate, beta, None, A_log, dt_bias)
        # Profiling and empty-batch early returns give zeros.
        for attn_metadata in (None, {"kda": replace(metadata, num_actual_tokens=0)}):
            context.attn_metadata = attn_metadata
            early = layer._forward(mixed, gate, beta, None, A_log, dt_bias)
            assert torch.equal(early, torch.zeros(6, 1, 1))
    assert torch.equal(output.flatten(), (torch.arange(6) < extent).float())
