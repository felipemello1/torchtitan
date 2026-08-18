# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch.distributed.tensor import DTensor

from torchtitan.distributed import ParallelDims
from torchtitan.models.common.moe import MoE

from .runtime import log_stats


@torch.no_grad()
def log_router_statistics(
    moe_layers: Sequence[MoE],
    *,
    local_tokens_per_expert_by_layer: Sequence[torch.Tensor],
    ep_group_tokens_per_expert_by_layer: torch.Tensor | None,
    global_tokens_per_expert_by_layer: torch.Tensor,
    entropy_logits_sum_LE: torch.Tensor,
    entropy_token_count_L1: torch.Tensor,
    sequence_expert_counts_LSE: torch.Tensor,
    recorded_sequence_count_L1: torch.Tensor,
    parallel_dims: ParallelDims,
) -> None:
    """Reconstruct topology-aware router metrics for every MoE layer.

    Args:
        moe_layers: Ordered MoE modules for `L` local layers.
        local_tokens_per_expert_by_layer: `L` local count tensors, each logically
            `[E]`; these retain local-shard load for diagnostics.
        ep_group_tokens_per_expert_by_layer: `[L, E]` counts reduced within each
            EP group, or `None` when EP is disabled.
        global_tokens_per_expert_by_layer: Counts with shape `[L, E]` already reconstructed over every token-sharding axis.
        entropy_logits_sum_LE: Preallocated `[L, E]` whole-step numerators.
        entropy_token_count_L1: Preallocated `[L, 1]` whole-step denominators.
        sequence_expert_counts_LSE: Preallocated `[L, S, E]` per-sequence counts.
        recorded_sequence_count_L1: Preallocated `[L, 1]` valid sequence counts.
        parallel_dims: Meshes used to reconstruct router logits and per-sequence counts.

    Example:

        # Two layers and four global experts.
        global_counts_LE = torch.tensor([[8, 8, 4, 12], [7, 9, 8, 8]])
        log_router_statistics(
            moe_layers,
            local_tokens_per_expert_by_layer=local_counts_by_layer,
            ep_group_tokens_per_expert_by_layer=ep_group_counts_LE,
            global_tokens_per_expert_by_layer=global_counts_LE,
            entropy_logits_sum_LE=entropy_sum_LE,
            entropy_token_count_L1=entropy_count_L1,
            sequence_expert_counts_LSE=sequence_counts_LSE,
            recorded_sequence_count_L1=recorded_sequence_count_L1,
            parallel_dims=parallel_dims,
        )
        # Layer 0 records expert_load=[1.0, 1.0, 0.5, 1.5].
    """

    # Reconstruct numerator and denominator over every partial token population.
    # These buffers are reduced in place, published, and cleared by the optimizer
    # hook before the next forward writes into the same storage.
    tp_mesh = parallel_dims.get_optional_mesh("tp")
    if tp_mesh is not None:
        torch.distributed.all_reduce(
            entropy_logits_sum_LE,
            group=tp_mesh.get_group(),
            op=torch.distributed.ReduceOp.SUM,
        )
        torch.distributed.all_reduce(
            entropy_token_count_L1,
            group=tp_mesh.get_group(),
            op=torch.distributed.ReduceOp.SUM,
        )
    loss_mesh = parallel_dims.get_optional_mesh("loss")
    if loss_mesh is not None:
        torch.distributed.all_reduce(
            entropy_logits_sum_LE,
            group=loss_mesh.get_group(),
            op=torch.distributed.ReduceOp.SUM,
        )
        torch.distributed.all_reduce(
            entropy_token_count_L1,
            group=loss_mesh.get_group(),
            op=torch.distributed.ReduceOp.SUM,
        )
    router_logits_mean_LE = entropy_logits_sum_LE / entropy_token_count_L1.clamp_min(1)

    # Derive entropy from reconstructed scores, but retain local expert imbalance.
    ep_mesh = parallel_dims.get_optional_mesh("ep")
    for layer_index, (moe, router_logits_mean_E, local_counts_E) in enumerate(
        zip(
            moe_layers,
            router_logits_mean_LE,
            local_tokens_per_expert_by_layer,
            strict=True,
        )
    ):
        router = moe.router
        if router.score_func == "sigmoid":
            router_scores_E = torch.sigmoid(router_logits_mean_E)
        elif router.score_func == "softmax":
            router_scores_E = torch.softmax(router_logits_mean_E, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {router.score_func}")
        if moe.expert_bias_E is not None:
            expert_bias_E = moe.expert_bias_E
            if isinstance(expert_bias_E, DTensor):
                expert_bias_E = expert_bias_E.to_local()
            router_scores_E = router_scores_E + (expert_bias_E - expert_bias_E.min())
        normalized_scores_E = torch.nn.functional.normalize(
            router_scores_E,
            dim=-1,
            p=1,
        )
        entropy = -(
            normalized_scores_E * normalized_scores_E.clamp_min(1e-12).log()
        ).sum()
        log_stats(moe.router, entropy=entropy.view(1))

        if isinstance(local_counts_E, DTensor):
            local_counts_E = local_counts_E.to_local()
        if ep_mesh is not None:
            num_local_experts = local_counts_E.numel() // ep_mesh.size()
            local_expert_start = ep_mesh.get_local_rank() * num_local_experts
            local_counts_E = local_counts_E[
                local_expert_start : local_expert_start + num_local_experts
            ]
        local_average = local_counts_E.float().mean().clamp_min(1)
        log_stats(
            moe.router,
            local_expert_imbalance=(local_counts_E.float() / local_average)
            .max()
            .view(1),
        )
        if ep_mesh is not None:
            assert ep_group_tokens_per_expert_by_layer is not None
            ep_group_counts_E = ep_group_tokens_per_expert_by_layer[layer_index]
            if isinstance(ep_group_counts_E, DTensor):
                ep_group_counts_E = ep_group_counts_E.to_local()
            ep_group_counts_E = ep_group_counts_E.float()
            expert_shard_counts = ep_group_counts_E.reshape(ep_mesh.size(), -1).sum(
                dim=-1
            )
            log_stats(
                moe.router,
                ep_shard_imbalance=(
                    expert_shard_counts.max()
                    * ep_mesh.size()
                    / ep_group_counts_E.sum().clamp_min(1)
                ).view(1),
            )

    # Reconstruct each sequence across CP (and TP when TP and EP share work).
    cp_mesh = parallel_dims.get_optional_mesh("cp")
    if cp_mesh is not None:
        torch.distributed.all_reduce(
            sequence_expert_counts_LSE,
            group=cp_mesh.get_group(),
            op=torch.distributed.ReduceOp.SUM,
        )
    if parallel_dims.ep_enabled and parallel_dims.tp_enabled:
        torch.distributed.all_reduce(
            sequence_expert_counts_LSE,
            group=parallel_dims.get_mesh("tp").get_group(),
            op=torch.distributed.ReduceOp.SUM,
        )

    if ep_mesh is None:
        local_expert_slice = slice(None)
    else:
        num_experts = sequence_expert_counts_LSE.shape[-1]
        num_local_experts = num_experts // ep_mesh.size()
        local_expert_start = ep_mesh.get_local_rank() * num_local_experts
        local_expert_slice = slice(
            local_expert_start,
            local_expert_start + num_local_experts,
        )
    for moe, sequence_counts_SE, sequence_position_1 in zip(
        moe_layers,
        sequence_expert_counts_LSE,
        recorded_sequence_count_L1,
        strict=True,
    ):
        local_counts_SE = sequence_counts_SE[:, local_expert_slice].float()
        average_counts_S1 = local_counts_SE.mean(dim=1, keepdim=True)
        sequence_imbalance_S = (local_counts_SE / average_counts_S1.clamp_min(1)).amax(
            dim=1
        )
        valid_sequences_S = (
            torch.arange(
                sequence_counts_SE.shape[0],
                device=sequence_counts_SE.device,
            )
            < sequence_position_1
        )
        selected_sequence_imbalance_S = torch.where(
            valid_sequences_S,
            sequence_imbalance_S,
            torch.zeros_like(sequence_imbalance_S),
        )
        log_stats(
            moe.router,
            seq_expert_imbalance_max=selected_sequence_imbalance_S.max().view(1),
            seq_expert_imbalance_mean=(
                selected_sequence_imbalance_S.sum()
                / sequence_position_1.clamp_min(1).float()
            ).view(1),
        )

    # Derive global expert load and expose the bias that produced the routing.
    for moe, tokens_per_expert_E in zip(
        moe_layers,
        global_tokens_per_expert_by_layer,
        strict=True,
    ):
        if isinstance(tokens_per_expert_E, DTensor):
            tokens_per_expert_E = tokens_per_expert_E.to_local()
        tokens_per_expert_E = tokens_per_expert_E.float()
        average_tokens = tokens_per_expert_E.mean().clamp_min(1)
        expert_load_E = tokens_per_expert_E / average_tokens
        log_stats(
            moe.router,
            expert_load=expert_load_E,
            experts_max_violation=(expert_load_E.max() - 1).view(1),
        )
        if moe.expert_bias_E is not None:
            log_stats(moe.router, expert_bias=moe.expert_bias_E)
