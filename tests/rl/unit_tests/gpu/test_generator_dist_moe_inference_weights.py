# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-GPU vLLM generator test for Dist-MoE inference weights (MXFP8 Mega, block-major BF16)."""

import gc
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip(
    "dist_moe",
    reason="Dist-MoE integration tests require the optional dist_moe package",
)

from torchtitan.config import CommConfig
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.common.dist_moe import (
    _dist_moe as dist_moe,
    BlockMajorDistMoeRoutedExperts,
    DistMoeRuntime,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.models.qwen3 import build_model_config
from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.vllm.registry import register_to_vllm, VLLM_MODEL_NAME
from torchtitan.tools.utils import has_cuda_capability
from torchtitan_recipes.rl.model_configs import build_rl_model_configs
from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_moe_debug_varlen
from vllm import SamplingParams
from vllm.config import CUDAGraphMode
from vllm.model_executor.model_loader import utils as model_loader_utils
from vllm.sampling_params import RequestOutputKind

from tests.rl.unit_tests.gpu.test_bitwise_parity import (
    _run_engine,
    _sync_trainer_weights_to_vllm,
    build_inference_engine,
    make_prompt_tokens,
)


def _generator_config(expert_transform: DistMoeTransform) -> Controller.Config:
    """Build a one-GPU debug MoE generator whose experts use ``expert_transform``."""
    config = rl_grpo_qwen3_moe_debug_varlen()
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=1,
        expert_parallel_degree=1,
    )
    _, generator_model = build_rl_model_configs(
        build_model_config,
        "debugmodel_moe",
        seq_len=config.generator.model.max_context_length,
        attn_backend="varlen",
        trainer_parallelism=config.trainer.parallelism,
        generator_parallelism=generator_parallelism,
    )
    # Batch-invariant mode is process-wide and cannot be turned off, so an
    # earlier test in this process (e.g. Kimi K3 parity) may have enabled it.
    # Local compile regions reject that mode and this test does not need them.
    generator_model.local_compile.regions = []
    config.generator.model = generator_model
    config.generator.parallelism = generator_parallelism
    config.generator.gpu_memory_limit = 0.1
    config.generator.engine.cuda_graph.mode = "FULL"
    config.generator.dist_moe_runtime = DistMoeRuntime.Config(inference=True)
    config.generator = apply_transforms(config.generator, [expert_transform])
    return config


def _clear_cached_model_class() -> None:
    """Drop vLLM's cached model class, so the next engine builds the model
    registered next, not one an earlier test (e.g. Kimi K3) registered.
    """
    # TODO: Fix this in register_to_vllm (or vLLM) instead of the test.
    for key, (_, arch) in list(model_loader_utils._MODEL_ARCH_BY_HASH.items()):
        if arch == VLLM_MODEL_NAME:
            del model_loader_utils._MODEL_ARCH_BY_HASH[key]


def _state_dict_with_random_experts(wrapper, experts, *, seed: int) -> dict:
    """Build a full state dict to load into the vLLM model, with only the
    expert weights changed.

    - Expert W13/W2 weights: new random values from ``seed``
    - All other weights: copies of the model's current values, so loading
      them changes nothing (FSDP keeps their unsharded storage)
    """
    expert_names = {f"{name}.{p}.weight" for name in experts for p in ("w13", "w2")}
    live = {
        **dict(wrapper.model.named_parameters()),
        **dict(wrapper.model.named_buffers()),
    }
    rng = torch.Generator().manual_seed(seed)
    return {
        name: (
            0.02 * torch.randn(buffer.shape, generator=rng).to(buffer.dtype)
            if name in expert_names
            else live[name].detach().cpu()
        )
        for name, buffer in (
            wrapper.model_state_dict_loader.prefetched_model_state_dict.items()
        )
    }


def _trainer_model(wrapper, state_dict: dict[str, torch.Tensor]) -> SimpleNamespace:
    """Stand in for the trainer model: ``state_dict`` under the generator model's
    buffer and ``preserve_dtype_params`` names, which match the trainer's."""
    return SimpleNamespace(
        state_dict=lambda: state_dict,
        named_buffers=wrapper.model.named_buffers,
        named_preserve_dtype_params=wrapper.model.named_preserve_dtype_params,
    )


def _operands(
    experts, operand_fields: tuple[str, ...]
) -> list[tuple[torch.Tensor, ...]]:
    """Return each expert weight's FSDP-managed operand tensors, e.g. qdata and FPROP scale."""
    return [
        tuple(getattr(weight.operands, field) for field in operand_fields)
        for module in experts.values()
        for weight in (module.w13.weight, module.w2.weight)
    ]


def _generate(engine, prompt_ids, request_prefix: str) -> list[list[float]]:
    """Greedily generate two tokens per prompt and return their logprobs."""
    outputs = _run_engine(
        engine,
        request_prefix,
        prompt_ids,
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=2,
            ignore_eos=True,
            logprobs=1,
            output_kind=RequestOutputKind.FINAL_ONLY,
        ),
    )
    logprobs = [
        [list(step.values())[0].logprob for step in output.outputs[0].logprobs]
        for output in outputs
    ]
    assert all(len(sample) == 2 for sample in logprobs)
    assert all(math.isfinite(logprob) for sample in logprobs for logprob in sample)
    return logprobs


def _check_inference_operands(
    engine, experts_cls: type, operand_fields: tuple[str, ...]
) -> None:
    """Check the experts' inference operands across FULL-graph weight loads.

    - The engine replays FULL CUDA graphs rather than running eagerly
    - Every MXFP8 expert uses the Mega fused inference weights (no DGRAD
      scales) from the first unshard
    - Each weight load refills the same operand tensors at the same addresses
    - The operand contents and generated logprobs follow the loaded weights,
      and reloading the same weights reproduces the same logprobs
    """
    compilation_config = engine.vllm_config.compilation_config
    assert compilation_config.cudagraph_mode == CUDAGraphMode.FULL
    assert not engine.vllm_config.model_config.enforce_eager

    wrapper = engine.model_executor.driver_worker.get_model()
    experts = {
        name: module
        for name, module in wrapper.model.named_modules()
        if isinstance(module, experts_cls)
    }
    assert experts
    for module in experts.values():
        for weight in (module.w13.weight, module.w2.weight):
            # The wrapper already unsharded the weights, so their operands come
            # from the first preparation
            assert isinstance(weight, _UnshardedFSDPTensor)
            if experts_cls is MXFP8DistMoeRoutedExperts:
                assert module.mega_fused_inference
                assert weight.operands.dgrad_scale.numel() == 0

    weights_a = _state_dict_with_random_experts(wrapper, experts, seed=0)
    weights_b = _state_dict_with_random_experts(wrapper, experts, seed=1)
    prompt_ids = make_prompt_tokens(2, 32, engine.get_tokenizer())

    # The CUDA graphs captured at engine startup read these exact tensors and
    # addresses, so every load must refill them in place.
    operands = _operands(experts, operand_fields)
    data_ptrs = [tuple(t.data_ptr() for t in tensors) for tensors in operands]

    # Reuse the trainer weight sync, which loads through the generator's
    # weight-load path.
    _sync_trainer_weights_to_vllm(_trainer_model(wrapper, weights_a), engine)
    contents_a = [
        tuple(t.clone() for t in tensors)
        for tensors in _operands(experts, operand_fields)
    ]
    logprobs_a = _generate(engine, prompt_ids, "weights_a")

    _sync_trainer_weights_to_vllm(_trainer_model(wrapper, weights_b), engine)
    for tensors, tensors_a in zip(
        _operands(experts, operand_fields), contents_a, strict=True
    ):
        for tensor, tensor_a in zip(tensors, tensors_a, strict=True):
            assert not torch.equal(tensor, tensor_a)
    logprobs_b = _generate(engine, prompt_ids, "weights_b")

    _sync_trainer_weights_to_vllm(_trainer_model(wrapper, weights_a), engine)
    logprobs_a_again = _generate(engine, prompt_ids, "weights_a_again")

    for tensors, tensors_0, ptrs in zip(
        _operands(experts, operand_fields), operands, data_ptrs, strict=True
    ):
        assert all(t is t_0 for t, t_0 in zip(tensors, tensors_0, strict=True))
        assert tuple(t.data_ptr() for t in tensors) == ptrs
    assert logprobs_b != logprobs_a
    assert logprobs_a_again == logprobs_a


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_cuda_capability(10, 0),
    reason="Dist-MoE requires an SM100-or-newer GPU",
)
@pytest.mark.parametrize(
    "expert_transform, experts_cls, operand_fields, force_swiglu_interleave",
    [
        pytest.param(
            DistMoeTransform(
                expert_precision="mxfp8",
                block_scaled_config=dist_moe.BlockScaledConfig(pipeline="mega"),
                mxfp8_mega_fused_inference=True,
            ),
            MXFP8DistMoeRoutedExperts,
            ("qdata", "fprop_scale"),
            False,
            id="mxfp8_mega",
        ),
        pytest.param(
            DistMoeTransform(bf16_block_major_inference=True),
            BlockMajorDistMoeRoutedExperts,
            ("block_major_data",),
            False,
            id="bf16_block_major",
        ),
        pytest.param(
            DistMoeTransform(bf16_block_major_inference=True),
            BlockMajorDistMoeRoutedExperts,
            ("block_major_data",),
            True,
            id="bf16_block_major_interleaved",
        ),
    ],
)
def test_vllm_generator_refills_dist_moe_inference_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    expert_transform: DistMoeTransform,
    experts_cls: type,
    operand_fields: tuple[str, ...],
    force_swiglu_interleave: bool,
):
    """The generator serves Dist-MoE experts from inference weights that each load refills.

    MXFP8 Mega experts (``mxfp8_mega_fused_inference``) have no DGRAD scales from
    the first unshard. Under FULL CUDA graphs, weight loads refill the retained
    operands (MXFP8 qdata and scales, or the block-major BF16 copy) in place: their
    objects stay fixed while their contents and the generated logprobs follow the
    loaded weights. ``force_swiglu_interleave`` also interleaves W13's gate and up
    rows, which dist_moe's size rule skips for the debug experts.
    """
    world_size = (
        dist.get_world_size()
        if dist.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size != 1:
        pytest.skip(f"requires exactly 1 GPU, got {world_size}")

    if force_swiglu_interleave:
        monkeypatch.setattr("dist_moe.should_fuse_bf16_swiglu", lambda *args: True)
    config = _generator_config(expert_transform)
    engine = None
    try:
        if not dist.is_initialized():
            # Kept until the process exits: vLLM holds on to this group, and the next
            # engine in this process (e.g. the next parametrization) reuses it.
            dist_utils.init_distributed(
                CommConfig(), base_folder=str(tmp_path_factory.mktemp("rl_generator"))
            )
        _clear_cached_model_class()
        register_to_vllm(
            config.generator.model,
            parallelism=config.generator.parallelism,
            precision=config.precision,
            checkpointer_config=None,
            dist_moe_runtime=config.generator.dist_moe_runtime,
            model_state_dict_load_mode="cpu_prefetch",
        )
        engine = build_inference_engine(config)
        if force_swiglu_interleave:
            wrapper = engine.model_executor.driver_worker.get_model()
            w13_weights = [
                module._weight_operands()[0]
                for module in wrapper.model.modules()
                if isinstance(module, BlockMajorDistMoeRoutedExperts)
            ]
            assert w13_weights
            assert all(w13._interleaved_fc13 for w13 in w13_weights)
        _check_inference_operands(engine, experts_cls, operand_fields)
    finally:
        if engine is not None:
            renderer = getattr(engine, "renderer", None)
            if renderer is not None:
                renderer.shutdown()
            del engine
        _clear_cached_model_class()
        gc.collect()
        torch.cuda.empty_cache()
