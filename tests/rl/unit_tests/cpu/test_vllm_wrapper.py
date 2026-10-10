# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.metadata
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torchtitan.config.parallelism import ParallelismConfig

from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.rl.model.attention import _fa4_splits_paged_kv

from torchtitan.rl.model.vllm_wrapper import (
    _replace_vllm_layer_configs,
    PlainToDTensorStateDictAdapter,
    VLLMModelWrapper,
)


def test_state_dict_layouts_include_native_feed_forward_weight():
    """Verify the fused dense FFN layout uses its native w13 state-dict key."""
    colwise = dense_param_placement(tp=spmd.S(1))
    rowwise = dense_param_placement(tp=spmd.S(1))
    config = FeedForward.Config(
        w13=Linear.Config(
            in_features=16,
            out_features=32,
            num_linears=2,
            sharding_config=ShardingConfig(state_shardings={"weight": colwise}),
        ),
        w2=Linear.Config(
            in_features=32,
            out_features=16,
            sharding_config=ShardingConfig(state_shardings={"weight": rowwise}),
        ),
    )
    model = torch.nn.Module()
    model.feed_forward = config.build()
    wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model

    layouts = wrapper.get_state_dict_layouts()

    assert layouts["feed_forward.w13.weight"] is colwise
    assert "feed_forward.w1.weight" not in layouts
    assert "feed_forward.w3.weight" not in layouts
    assert layouts["feed_forward.w2.weight"] is rowwise


def test_state_dict_layouts_include_native_qkv_weight():
    """Verify QKV layout lookup uses the native packed state-dict key."""
    colwise = dense_param_placement(tp=spmd.S(0))
    config = QKVLinear.Config(
        head_dim=8,
        n_heads=4,
        n_kv_heads=2,
        wqkv=Linear.Config(
            in_features=16,
            out_features=64,
            sharding_config=ShardingConfig(state_shardings={"weight": colwise}),
        ),
    )
    model = torch.nn.Module()
    model.qkv_linear = config.build()
    wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model

    layouts = wrapper.get_state_dict_layouts()

    assert layouts["qkv_linear.wqkv.weight"] is colwise
    assert "qkv_linear.wq.weight" not in layouts
    assert "qkv_linear.wk.weight" not in layouts
    assert "qkv_linear.wv.weight" not in layouts


def test_state_dict_layouts_include_native_grouped_linear_weights():
    """Verify routed expert layouts use native grouped-linear state keys."""
    physical_colwise = dense_param_placement(tp=spmd.S(2))
    rowwise = dense_param_placement(tp=spmd.S(1))
    w13_config = GroupedLinear.Config(
        group_size=4,
        in_features=16,
        out_features=32,
        num_linears=2,
        sharding_config=ShardingConfig(state_shardings={"weight": physical_colwise}),
    )
    w2_config = GroupedLinear.Config(
        group_size=4,
        in_features=32,
        out_features=16,
        sharding_config=ShardingConfig(state_shardings={"weight": rowwise}),
    )
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.w13 = w13_config.build()
    model.experts.w2 = w2_config.build()
    wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model

    layouts = wrapper.get_state_dict_layouts()

    assert layouts["experts.w13.weight"] is physical_colwise
    assert layouts["experts.w2.weight"] is rowwise


def test_vllm_replacements_preserve_resolved_sharding():
    model_config = build_model_config("debugmodel", attn_backend="flex")
    model_config.set_sharding_(
        ParallelismConfig(tensor_parallel_degree=2, enable_sequence_parallel=True)
    )
    model_config.layers = [
        layer for layer in model_config.layers if layer.attention is not None
    ]

    vllm_config = _replace_vllm_layer_configs(model_config)

    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        assert model_layer.attention is not None
        assert vllm_layer.attention is not None
        model_sharding = model_layer.attention.inner_attention.sharding_config
        vllm_sharding = vllm_layer.attention.inner_attention.sharding_config
        assert model_sharding is not None
        assert vllm_sharding is not None
        assert vllm_sharding.in_src_shardings is model_sharding.in_src_shardings
        assert vllm_sharding.in_dst_shardings is model_sharding.in_dst_shardings
        assert vllm_sharding.out_src_shardings is model_sharding.out_src_shardings
        assert vllm_sharding.out_dst_shardings is model_sharding.out_dst_shardings
        assert vllm_sharding.local_spmd is model_sharding.local_spmd
        for name, layout in model_sharding.state_shardings.items():
            assert vllm_sharding.state_shardings[name] is layout


def test_routers_expose_routed_experts_to_vllm_capture():
    """vLLM binds ``capture_fn`` by attribute; every MoE router must call it with its ids."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from torchtitan.models.common.activation import Sigmoid
    from torchtitan.models.common.config_utils import make_router_config

    def build_router():
        router = make_router_config(
            dim=4,
            num_experts=4,
            score_func=Sigmoid.Config(),
            gate_param_init={"weight": torch.nn.init.zeros_},
            top_k=2,
        ).build()
        router.init_states()
        return router

    for tp_enabled, gathered_rows in ((False, 3), (True, 6)):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleDict(
            {"0": torch.nn.Module(), "1": torch.nn.Module()}
        )
        model.layers["0"].feed_forward = torch.nn.Linear(4, 4)
        model.layers["1"].router = build_router()
        wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
        torch.nn.Module.__init__(wrapper)
        wrapper.model = model
        wrapper.parallelism_context = SimpleNamespace(
            ep_enabled=True, tp_enabled=tp_enabled
        )
        tp_group = SimpleNamespace(
            all_gather=lambda tensor, dim: torch.cat([tensor, tensor], dim=dim)
        )
        captured = []

        with patch(
            "torchtitan.rl.model.vllm_wrapper.get_tp_group", return_value=tp_group
        ):
            wrapper._expose_routed_experts_to_vllm()
            router = model.layers["1"].router
            assert router.layer_id == 1 and router.capture_fn is None
            router(torch.randn(3, 4))  # before vLLM binds capture_fn: no capture
            router.capture_fn = captured.append
            _, topk_expert_ids_TK, _ = router(torch.randn(3, 4))

        assert not hasattr(model.layers["0"].feed_forward, "layer_id")
        assert len(captured) == 1 and captured[0].shape == (gathered_rows, 2)
        assert torch.equal(captured[0][:3], topk_expert_ids_TK)


@pytest.mark.parametrize(
    ("fa4_version", "splits"), [("4.0.0b33", False), ("4.0.0b34.dev10+g33985c6", True)]
)
def test_fa4_splits_paged_kv_only_from_the_version_that_supports_it(
    fa4_version: str, splits: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: fa4_version)
    _fa4_splits_paged_kv.cache_clear()
    try:
        assert _fa4_splits_paged_kv() is splits
    finally:
        _fa4_splits_paged_kv.cache_clear()


def _check_hf_adapter_restores_local_shards(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        model_config = build_model_config("0.8B", seq_len=256, attn_backend="varlen")
        assert isinstance(model_config, Qwen35Model.Config)
        # This state dict carries lm_head without tok_embeddings; untie so the
        # adapter keeps lm_head instead of expecting it from embed_tokens.
        model_config = replace(model_config, enable_weight_tying=False)
        model_adapter = Qwen35StateDictAdapter(model_config, hf_assets_path=None)
        state_dict, expected, layouts = {}, {}, {}
        for pattern, shape in (
            ("layers.0.attn.in_proj_{}.weight", (2048, 1024)),
            ("layers.0.attn.conv_{}.weight", (2048, 1, 4)),
            ("vision_encoder.layers.0.attn.w{}.weight", (768, 768)),
            ("vision_encoder.layers.0.attn.w{}.bias", (768,)),
        ):
            for index, part in enumerate(("q", "k", "v")):
                key = pattern.format(part)
                full = torch.arange(torch.Size(shape).numel()).reshape(shape).float()
                full += index * 100
                state_dict[key] = distribute_tensor(full, mesh, [Shard(0)])
                expected[key] = full.chunk(2, dim=0)[rank].clone()
                layouts[key] = dense_param_placement(tp=spmd.S(0))
        # Check unchanged row-sharded, replicated, and plain values as well.
        for key, placement, shape in (
            ("layers.3.attn.wo.weight", Shard(1), (8, 8)),
            ("norm.weight", Replicate(), (8,)),
        ):
            full = torch.arange(torch.Size(shape).numel()).reshape(shape).float()
            state_dict[key] = distribute_tensor(full, mesh, [placement])
            expected[key] = (
                full.chunk(2, dim=1)[rank].clone()
                if isinstance(placement, Shard)
                else full
            )
            layouts[key] = dense_param_placement(
                tp=spmd.S(1) if isinstance(placement, Shard) else spmd.R
            )
        state_dict["lm_head.weight"] = expected["lm_head.weight"] = torch.ones(1)
        adapter = PlainToDTensorStateDictAdapter(
            model_adapter,
            layouts,
            ParallelismContext(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=2,
                pp=1,
                ep=1,
                world_size=2,
                enable_sequence_parallel=False,
            ),
        )
        restored = adapter.from_hf(model_adapter.to_hf(state_dict))
        assert restored.keys() == expected.keys()
        for key in expected:
            assert type(restored[key]) is torch.Tensor
            torch.testing.assert_close(restored[key], expected[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_hf_adapter_restores_local_shards(tmp_path: Path) -> None:
    mp.spawn(
        _check_hf_adapter_restores_local_shards,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )
