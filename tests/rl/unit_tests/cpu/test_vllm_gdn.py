# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torchtitan.rl.generator.vllm import gdn
from torchtitan.rl.generator.vllm.linear_attention import (
    TorchTitanLinearAttentionBackend,
)
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend


@pytest.mark.parametrize(
    "num_speculative_tokens, backend, core",
    [
        (0, TorchTitanLinearAttentionBackend, None),
        (3, GDNAttentionBackend, "vllm core"),
    ],
    ids=["spec-off", "spec-on"],
)
def test_gdn_runs_vllm_core_only_with_speculative_decoding(
    monkeypatch, num_speculative_tokens, backend, core
):
    speculative_config = (
        SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
        if num_speculative_tokens
        else None
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        model_config=None,
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
        speculative_config=speculative_config,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(gdn, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(gdn, "is_conv_state_dim_first", lambda: False)
    # vLLM's core builds GPU kernels; only its selection is under test here.
    monkeypatch.setattr(gdn, "_VLLMGDNCore", lambda layer: "vllm core")

    layer = gdn.VLLMInnerGDN(
        gdn.VLLMInnerGDN.Config(
            layer_idx=0, num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128
        )
    )

    assert layer.get_attn_backend() is backend
    assert layer._vllm_core == core


def test_vllm_gdn_core_passes_our_tensors_under_vllm_names(monkeypatch):
    # ChunkGatedDeltaRule picks a GPU prefill kernel; only the name mapping is under test.
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "ChunkGatedDeltaRule",
        lambda: SimpleNamespace(gdn_prefill_backend="triton"),
    )
    # Global head counts at TP2.
    layer = SimpleNamespace(
        prefix="gdn",
        get_state_dtype=None,
        tensor_parallel_size=2,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        kv_cache=None,
    )
    core = gdn._VLLMGDNCore(layer)
    calls = []
    core._forward_core = lambda *args, **kwargs: calls.append((args, kwargs))
    # vLLM binds the cache after the layer is built.
    layer.kv_cache = (torch.zeros(1), torch.zeros(1))
    mixed_qkv, a, b, output = (torch.randn(5, 4) for _ in range(4))
    conv_weight = torch.randn(6144, 4, dtype=torch.bfloat16)
    A_log, dt_bias = torch.randn(2, 16, dtype=torch.bfloat16)

    core.forward(mixed_qkv, a, b, conv_weight, None, A_log, dt_bias, output)

    [(args, kwargs)] = calls
    assert args == (mixed_qkv,)
    assert kwargs["b"] is b and kwargs["a"] is a and kwargs["core_attn_out"] is output
    assert core.kv_cache is layer.kv_cache
    assert core.conv1d.weight.shape == (6144, 1, 4)
    assert core.A_log.dtype == core.dt_bias.dtype == torch.float32
    # vLLM's core divides the global head counts by tp_size.
    assert (core.tp_size, core.num_k_heads, core.num_v_heads) == (2, 16, 32)
    assert (core.key_dim, core.value_dim) == (16 * 128, 32 * 128)
