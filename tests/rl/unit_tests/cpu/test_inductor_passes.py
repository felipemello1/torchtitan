# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch._inductor.utils import run_and_get_code

from torchtitan.rl.generator import VLLMCudaGraphConfig
from torchtitan.rl.model.inductor_passes import UnfuseResidualAddmmPass
from vllm.config.utils import Range


def _residual_block(x, weight, norm_weight):
    h = x + x @ weight.T
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * norm_weight


def _compiled_code(post_pass):
    torch._dynamo.reset()
    x, weight, norm_weight = torch.randn(4, 64), torch.randn(64, 64), torch.randn(64)
    with torch._inductor.config.patch(
        post_grad_custom_post_pass=post_pass, fx_graph_cache=False
    ):
        out, (code,) = run_and_get_code(
            torch.compile(_residual_block), x, weight, norm_weight
        )
    torch.testing.assert_close(out, _residual_block(x, weight, norm_weight))
    return code


def test_pass_turns_residual_addmm_back_into_mm_plus_add():
    assert "addmm" in _compiled_code(None)
    code = _compiled_code(UnfuseResidualAddmmPass())
    assert "addmm" not in code and "extern_kernels.mm(" in code


def test_pass_targets_only_the_single_token_graph():
    unfuse = UnfuseResidualAddmmPass()
    assert unfuse.is_applicable_for_range(Range(1, 1))
    assert not unfuse.is_applicable_for_range(Range(1, 8192))
    assert not unfuse.is_applicable_for_range(Range(16, 16))


def test_vllm_compile_config_compiles_a_single_token_graph():
    config = VLLMCudaGraphConfig(
        mode="FULL", capture_sizes=[1, 16], vllm_compile=True
    ).get_vllm_compilation_config(
        max_num_seqs=16, expert_sequence_parallel_size=1, enable_sequence_parallel=False
    )
    assert config.compile_sizes == [1]
    assert isinstance(
        config.inductor_compile_config["post_grad_custom_post_pass"],
        UnfuseResidualAddmmPass,
    )
