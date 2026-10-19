# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LoRATransform
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.deepseek_v3 import build_model_config as build_deepseek_config
from torchtitan.models.kimi_k3 import build_model_config as build_kimi_config
from torchtitan.models.qwen3 import build_model_config as build_qwen3_config
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.generator.vllm import model_wrapper
from torchtitan.rl.generator.vllm.attention import VLLMInnerAttention
from torchtitan.rl.generator.vllm.model_wrapper import (
    _replace_vllm_layer_configs,
    VLLMModelWrapper,
)


@pytest.mark.parametrize(
    "model_name, expected_scale",
    [
        ("gpt_oss", 0.22665755425493697),
        ("deepseek_v3", 0.11425195291833387),
        ("qwen3", None),
    ],
)
def test_vllm_replacement_preserves_attention_scale(model_name, expected_scale):
    """Preserve YaRN scaling as well as the backend's default scale."""
    model_config = importlib.import_module(
        f"torchtitan.models.{model_name}"
    ).build_model_config("debugmodel", seq_len=128, parallelism=ParallelismConfig())
    vllm_config = _replace_vllm_layer_configs(model_config)

    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        attention_config = model_layer.attention
        scale = vllm_layer.attention.inner_attention.scale
        if expected_scale is None:
            assert scale is None
        else:
            assert scale == pytest.approx(expected_scale)
            with torch.device("meta"):
                attention = attention_config.build()
            assert scale == attention.softmax_scale


@pytest.mark.parametrize(
    "build_config",
    [build_qwen3_config, build_deepseek_config, build_model_config, build_kimi_config],
    ids=["qwen3", "deepseek_v3", "qwen3_5", "kimi_k3"],
)
def test_vllm_replacement_preserves_full_attention_selection(build_config) -> None:
    model_config = build_config("debugmodel", parallelism=ParallelismConfig())
    full_attention_index = next(
        i
        for i, layer in enumerate(model_config.layers)
        if layer.attention is model_config.first_full_attention
    )

    vllm_config = _replace_vllm_layer_configs(model_config)

    assert (
        vllm_config.first_full_attention
        is vllm_config.layers[full_attention_index].attention
    )
    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        inner_attention = vllm_layer.attention.inner_attention
        assert isinstance(inner_attention, InnerAttention.Config)
        assert inner_attention._owner is not None
        assert issubclass(inner_attention._owner, InnerAttention)
        assert (
            inner_attention.is_full_attention
            == model_layer.attention.inner_attention.is_full_attention
        )
        assert inner_attention.build_attention_metadata(torch.arange(8)) is None


@pytest.mark.parametrize("sliding_window_size", [None, 128])
def test_vllm_attention_classifies_sliding_window(sliding_window_size) -> None:
    config = VLLMInnerAttention.Config(
        hidden_size=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        sliding_window_size=sliding_window_size,
    )
    assert config.is_full_attention is (sliding_window_size is None)


@pytest.mark.parametrize(
    "build_config", [build_model_config, build_kimi_config], ids=["gdn", "kda"]
)
def test_vllm_replacement_preserves_attention_sharding(build_config) -> None:
    """Pass the trainer attention's sharding config through unchanged."""
    model_config = build_config(
        "debugmodel",
        attn_backend="flex",
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2, enable_sequence_parallel=True
        ),
    )
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


class _Wrapper:
    """Stand-in `self` for VLLMModelWrapper's real model state dict load methods (no vLLM)."""

    load_state_dict = VLLMModelWrapper.load_state_dict

    def __init__(self, model: nn.Module):
        self.model = model


def test_load_state_dict_allocates_one_fsdp_group_at_a_time(
    tmp_path, monkeypatch
) -> None:
    """Load new weights with at most two FSDP groups' sharded buffers allocated at once.

    Guards that each group's sharded buffers are allocated before its copy, even when
    FSDP's own load pre-hook already resharded the group (tok_embeddings here).
    """
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        # The generator's layout: a tied [tok_embeddings, norm, lm_head] group, one per block.
        model = nn.Module()
        model.tok_embeddings = nn.Embedding(32, 16)
        model.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)) for _ in range(3)
        )
        model.norm = nn.LayerNorm(16)
        model.lm_head = nn.Linear(16, 32, bias=False)
        model.lm_head.weight = model.tok_embeddings.weight
        mesh = init_device_mesh("cpu", (1,))
        fully_shard([model.tok_embeddings, model.norm, model.lm_head], mesh=mesh)
        for block in model.layers:
            fully_shard(block, mesh=mesh)
        fully_shard(model, mesh=mesh)
        model.set_keep_unsharded_storage(True)
        new_state_dict = {k: v.detach() + 1.0 for k, v in model.state_dict().items()}
        groups = [
            group
            for module in (model.tok_embeddings, *model.layers)
            for group in module._get_fsdp_state()._fsdp_param_groups
        ]

        def is_allocated(group) -> bool:
            return all(
                p._sharded_param_data.untyped_storage().size() > 0
                for p in group.fsdp_params
            )

        wrapper = _Wrapper(model)
        max_allocated_groups = 0

        restore_sharded_params = model_wrapper._restore_sharded_params

        def checked_restore_sharded_params(module) -> None:
            nonlocal max_allocated_groups
            restore_sharded_params(module)
            # The load copies into these buffers next, so freed ones would be overrun.
            assert all(map(is_allocated, module._get_fsdp_state()._fsdp_param_groups))
            max_allocated_groups = max(
                max_allocated_groups, sum(map(is_allocated, groups))
            )

        monkeypatch.setattr(
            model_wrapper, "_restore_sharded_params", checked_restore_sharded_params
        )
        for module in model.modules():  # as after the initial load
            if isinstance(module, FSDPModule):
                model_wrapper._unshard_and_free_sharded_params(module)
        wrapper.load_state_dict(new_state_dict)

        assert max_allocated_groups <= 2
        assert all(
            p._sharded_param_data.untyped_storage().size() == 0
            for group in groups
            for p in group.fsdp_params
        )
        for name, param in model.named_parameters(remove_duplicate=False):
            assert param.equal(new_state_dict[name].to_local()), name
    finally:
        dist.destroy_process_group()


def test_unshard_hands_dist_moe_block_major_copy_of_loaded_weights(tmp_path) -> None:
    """Each unshard that gathers new expert weights rebuilds Dist-MoE's block-major copy in place.

    Covers the initial unshard and a weight sync through `load_state_dict` that
    replaces every weight with new random values, so a stale W13 or W2 fails.
    The copy keeps its address, which CUDA graphs read.
    """
    dist_moe = pytest.importorskip("dist_moe")

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Module()])
        model.layers[0].routed_experts = DistMoeRoutedExperts.Config(
            w13=GroupedLinear.Config(
                group_size=2, in_features=128, out_features=128, num_linears=2
            ),
            w2=GroupedLinear.Config(group_size=2, in_features=128, out_features=128),
            top_k=1,
            block_major_weights=True,
        ).build()
        initial = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                initial[name] = torch.randn(param.shape)
                param.copy_(initial[name])
        mesh = init_device_mesh("cpu", (1,))
        fully_shard(model.layers[0], mesh=mesh)
        fully_shard(model, mesh=mesh)
        model.set_keep_unsharded_storage(True)

        def expert_operands() -> dict[str, torch.Tensor]:
            return {
                name: param.operands.weight
                for name, param in model.named_parameters()
                if "routed_experts" in name
            }

        def assert_block_major(weights: dict[str, torch.Tensor]) -> None:
            for name, operand in expert_operands().items():
                expected = weights[name].flatten(1, -2).clone()
                dist_moe.relayout_to_block_major(expected)
                assert operand.equal(expected), name

        for module in model.modules():  # as after the initial load
            if isinstance(module, FSDPModule):
                model_wrapper._unshard_and_free_sharded_params(module)
        assert_block_major(initial)
        addresses = {name: t.data_ptr() for name, t in expert_operands().items()}

        new_state_dict = {
            k: torch.randn_like(v.detach()) for k, v in model.state_dict().items()
        }
        _Wrapper(model).load_state_dict(new_state_dict)
        assert_block_major({k: v.to_local() for k, v in new_state_dict.items()})
        assert {
            name: t.data_ptr() for name, t in expert_operands().items()
        } == addresses
    finally:
        dist.destroy_process_group()


def test_generator_gives_every_dist_moe_expert_one_layout() -> None:
    """The vLLM wrapper sets the resolved layout; LoRA on one layer keeps all row-major."""
    config = DistMoeTransform(inplace_wgrad_accum=False).transform(
        build_qwen3_config("debugmodel_moe", parallelism=ParallelismConfig())
    )
    first = next(fqn for fqn, *_ in config.traverse(DistMoeRoutedExperts.Config))
    lora_config = LoRATransform(
        rank=8, alpha=16.0, target_modules=[f"{first}.w2"]
    ).transform(copy.deepcopy(config))
    runtime_config = DistMoeRuntime.Config(inference=True, block_major_weights=True)
    for model_config, expected in ((config, True), (lora_config, False)):
        model_wrapper._set_dist_moe_weight_layout(model_config, runtime_config)
        experts = model_config.traverse(DistMoeRoutedExperts.Config)
        assert {cfg.block_major_weights for _, cfg, _, _ in experts} == {expected}
