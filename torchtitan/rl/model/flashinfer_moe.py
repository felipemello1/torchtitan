# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlashInfer's TRT-LLM bf16 fused MoE for the generator's routed experts (inference only).

The router stays in TorchTitan; FlashInfer takes its top-k ids and scores and runs dispatch, both
expert GEMMs, SwiGLU and the weighted combine in its fused kernels. The kernels read the expert
weights in FlashInfer's shuffled block layout, so the generator rewrites ``w13`` and ``w2`` in place
after every weight load:

    load_state_dict (TorchTitan layout) -> to_flashinfer_layout_ (in place) -> forward (FlashInfer)

Only for single-rank expert groups (EP=1, TP=1): every expert is local.
"""

import functools

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import RoutedExperts


def _local(param: torch.Tensor) -> torch.Tensor:
    """The local storage of ``param``, detached so in-place layout rewrites skip autograd."""
    return (param.to_local() if isinstance(param, DTensor) else param).detach()


def _block_major_k(weight: torch.Tensor) -> torch.Tensor:
    """View ``(E, rows, cols)`` bf16 storage as FlashInfer's ``(E, cols // 64, rows, 64)`` BlockMajorK."""
    num_experts, rows, cols = weight.shape
    return weight.view(num_experts, cols // 64, rows, 64)


def _expert_weights(module: RoutedExperts) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``w13`` as ``(E, 2F, D)`` (gate rows, then up rows) and ``w2`` as ``(E, D, F)``."""
    w13 = _local(module.w13.weight)  # (E, 2, F, D)
    num_experts, _, hidden, dim = w13.shape
    return w13.view(num_experts, 2 * hidden, dim), _local(module.w2.weight)


def _forward(
    module: RoutedExperts,
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
) -> torch.Tensor:
    import flashinfer
    from flashinfer.fused_moe import WeightLayout

    w13, w2 = _expert_weights(module)
    out = flashinfer.fused_moe.trtllm_bf16_routed_moe(
        topk_ids=(topk_expert_ids_TK.to(torch.int32), topk_scores_TK.bfloat16()),
        hidden_states=x_TD.bfloat16(),
        gemm1_weights=_block_major_k(w13),
        gemm2_weights=_block_major_k(w2),
        num_experts=w13.shape[0],
        top_k=topk_expert_ids_TK.shape[1],
        n_group=None,
        topk_group=None,
        intermediate_size=w2.shape[2],
        local_expert_offset=0,
        local_num_experts=w13.shape[0],
        routed_scaling_factor=None,
        routing_method_type=1,  # unused: the routing comes in as ids and scores
        use_shuffled_weight=True,
        weight_layout=WeightLayout.BlockMajorK,
        do_finalize=True,
    )
    out = out[0] if isinstance(out, list) else out
    return out.type_as(x_TD)


def enable_flashinfer_routed_experts(model: torch.nn.Module) -> list[RoutedExperts]:
    """Route every stock ``RoutedExperts`` of ``model`` through FlashInfer when grad is off.

    Example:
        experts = enable_flashinfer_routed_experts(model)
        to_flashinfer_layout_(experts)  # after every weight load
    """
    experts = [m for m in model.modules() if type(m) is RoutedExperts]
    for module in experts:
        if module.output_postprocess is not None:
            raise ValueError("FlashInfer routed experts do not support output_postprocess")
        module.inference_forward = functools.partial(_forward, module)
    return experts


def _shuffle_to_block_major_k_(weight: torch.Tensor, rows: torch.Tensor) -> None:
    """Gather ``weight``'s rows by ``rows`` and lay each expert out in 128-byte K blocks, in place.

    ``rows`` depends only on the per-expert shape, so one gather covers every expert.
    """
    num_experts, m, k = weight.shape
    # Weights created under vLLM's inference mode can only be written in inference mode.
    with torch.inference_mode(weight.is_inference()):
        blocks = weight.view(torch.uint8).view(num_experts, m, 2 * k // 128, 128)
        shuffled = blocks.permute(0, 2, 1, 3)[:, :, rows, :].contiguous()
        weight.view(-1).copy_(shuffled.view(torch.bfloat16).view(-1))


@torch.no_grad()
def to_flashinfer_layout_(experts: list[RoutedExperts]) -> None:
    """Rewrite each module's ``w13`` and ``w2`` in place from TorchTitan's layout to FlashInfer's.

    Run it after every load of TorchTitan-layout weights; running it twice scrambles the weights.

    Example:
        model.load_state_dict(state_dict)
        to_flashinfer_layout_(experts)
    """
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m = 128
    cache: dict[torch.Size, torch.Tensor] = {}
    for module in experts:
        w13, w2 = _expert_weights(module)
        if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
            raise ValueError(f"FlashInfer's bf16 MoE needs bf16 expert weights, got {w13.dtype}")
        # FlashInfer's gated GEMM wants [up; gate]; shifting the w3/w1 indices by half reads
        # TorchTitan's [gate; up] rows in that order.
        w13_rows = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13[0].view(torch.uint8), epilogue_tile_m, is_gated_act_gemm=True
        )
        w13_rows = (w13_rows + w13.shape[1] // 2) % w13.shape[1]
        w2_rows = get_w2_permute_indices_with_cache(
            cache, w2[0].view(torch.uint8), epilogue_tile_m
        )
        _shuffle_to_block_major_k_(w13, w13_rows.to(w13.device))
        _shuffle_to_block_major_k_(w2, w2_rows.to(w2.device))
