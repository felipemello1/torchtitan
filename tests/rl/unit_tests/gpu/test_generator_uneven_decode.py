# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Four-GPU vLLM generator integration test."""

import gc
import math
import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist

from torchtitan.config import CommConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.qwen3_5 import build_model_config as build_qwen3_5_model_config
from torchtitan.rl.generator.vllm.registry import register_to_vllm
from torchtitan_recipes.rl.model_configs import build_rl_model_configs
from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_moe_debug_varlen
from vllm import SamplingParams
from vllm.config import CUDAGraphMode
from vllm.sampling_params import RequestOutputKind

from tests.rl.unit_tests.gpu.test_bitwise_parity import (
    _make_prompt_tokens,
    _run_engine,
    build_inference_engine,
)

# Runs under a four-GPU torchrun launch.
pytestmark = pytest.mark.multi_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "data_parallel_degree, gdn",
    [(1, False), (2, False), (2, True)],
    ids=["dp1_tp4", "dp2_tp2_one_idle_rank", "dp2_tp2_one_idle_rank_gdn"],
)
def test_vllm_uneven_decode_tp_padding(data_parallel_degree: int, gdn: bool):
    """Three decode tokens run through EP-internal TP sequence sharding.

    With DP2 only DP rank 0 gets prompts. DP rank 1 runs vLLM's dummy batches,
    whose MoE all-gather must match rank 0's DP-padded CUDA graph every step.
    With GDN layers, rank 0's prefill pads rank 1's dummy batch past max_num_seqs, so
    rank 1 replays the mixed graph instead of a decode graph.
    Run each case in its own torchrun launch, as CI does: vLLM keeps its process
    groups across engines.
    """
    world_size = (
        dist.get_world_size()
        if dist.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size != 4:
        pytest.skip(f"requires exactly 4 GPUs, got {world_size}")

    config = rl_grpo_qwen3_moe_debug_varlen()
    config.generator.parallelism.data_parallel_degree = data_parallel_degree
    config.generator.parallelism.tensor_parallel_degree = 4 // data_parallel_degree
    config.generator.gpu_memory_limit = 0.5
    if gdn:
        _, config.generator.model = build_rl_model_configs(
            build_qwen3_5_model_config,
            "debugmodel_moe",
            seq_len=2048,
            attn_backend="varlen",
            trainer_parallelism=config.trainer.parallelism,
            generator_parallelism=config.generator.parallelism,
        )

    temporary_dump_folder = None
    if not dist.is_initialized():
        temporary_dump_folder = tempfile.mkdtemp(prefix="rl_generator_moe_")
        dist_utils.init_distributed(
            CommConfig(),
            base_folder=temporary_dump_folder,
        )

    register_to_vllm(
        config.generator.model,
        parallelism=config.generator.parallelism,
        checkpointer_config=None,
    )

    engine = build_inference_engine(config)
    try:
        dp_rank = dist.get_rank() // config.generator.parallelism.tensor_parallel_degree
        prompt_ids = (
            _make_prompt_tokens(3, 100, engine.get_tokenizer()) if dp_rank == 0 else []
        )
        vllm_config = engine.vllm_config
        if gdn:
            # vLLM downgrades FULL when a backend can't graph mixed batches.
            assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        if gdn and prompt_ids:
            # Else rank 1's dummy batch would fit a decode graph and skip the fallback.
            assert sum(map(len, prompt_ids)) > vllm_config.scheduler_config.max_num_seqs
        outputs = _run_engine(
            engine,
            "uneven_decode",
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

        for output in outputs:
            sample = output.outputs[0]
            assert len(sample.token_ids) == 2
            assert len(sample.logprobs) == 2
            assert all(
                math.isfinite(list(logprobs.values())[0].logprob)
                for logprobs in sample.logprobs
            )
    finally:
        if dist.is_initialized():
            dist.barrier()
        renderer = getattr(engine, "renderer", None)
        if renderer is not None:
            renderer.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        if temporary_dump_folder is not None:
            shutil.rmtree(temporary_dump_folder, ignore_errors=True)
