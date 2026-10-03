# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cascade decode attention for requests that share a cached prompt.

In RL rollouts, the samples of one prompt point their block tables at the same cached
pages. A plain decode call reads those pages once per request, and each FA4 decode tile
holds only ``num_heads // num_kv_heads`` real query rows. The cascade instead attends each
group's shared prefix once, with the group's decode tokens as the query tokens of one
sequence, attends each request's own suffix separately, and merges the two by LSE.

Every step is a fixed-shape GPU op, so the whole decode can be captured in a FULL CUDA graph.

Shape suffixes: B = requests (one decode token each), P = block-table pages,
H = query heads, D = head dim.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch.nn.attention.varlen import AuxRequest, varlen_attn_out
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states


@dataclass(frozen=True, slots=True)
class CascadePlan:
    """Per-step metadata for :func:`cascade_decode`, with requests sorted by group.

    The prefix call runs one sequence per request slot: slot ``g`` holds the queries of
    the group whose first request is ``g`` (empty for other slots) over the first
    ``prefix_seqused_k[g]`` tokens of row ``g`` of the block table.
    """

    order: torch.Tensor
    """``[B]`` request indices sorted by group, so each group's queries are contiguous."""
    prefix_cu_seqlens_q: torch.Tensor
    """``[B + 1]`` query offsets of the prefix call's request slots."""
    prefix_seqused_k: torch.Tensor
    """``[B]`` shared prefix tokens per slot (0: no group)."""
    suffix_cu_seqlens_q: torch.Tensor
    """``[B + 1]`` query offsets of the suffix call: one token per sorted request."""
    suffix_seqused_k: torch.Tensor
    """``[B]`` tokens after the shared prefix, per sorted request."""
    suffix_block_table: torch.Tensor
    """``[B, P]`` block table rows of the sorted requests, shifted past their prefix."""


@triton.jit
def _deepest_shared_page_kernel(
    block_table_ptr,
    table_stride,
    seq_lens_ptr,
    shared_pages_ptr,
    group_key_ptr,
    num_reqs,
    page_size,
    BLOCK_B: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Per request: how many leading pages it shares with some other request."""
    req = tl.program_id(0)
    reqs = tl.arange(0, BLOCK_B)
    others = (reqs < num_reqs) & (reqs != req)
    # Pages before the decode token; later entries are the token's page, stale or padding.
    cached = tl.load(seq_lens_ptr + reqs, mask=reqs < num_reqs, other=1)
    cached = tl.maximum(cached - 1, 0) // page_size
    cached_req = tl.maximum(tl.load(seq_lens_ptr + req) - 1, 0) // page_size
    shared = cached_req * 0
    sharing = cached_req > 0
    while sharing:
        pages = shared + tl.arange(0, BLOCK_P)
        own = tl.load(
            block_table_ptr + req * table_stride + pages, mask=pages < cached_req
        )
        table = tl.load(
            block_table_ptr + reqs[:, None] * table_stride + pages[None, :],
            mask=others[:, None] & (pages[None, :] < cached[:, None]),
            other=-1,
        )
        held = tl.max(
            ((table == own[None, :]) & (pages < cached_req)[None, :]).to(tl.int32), 0
        )
        run = tl.sum(tl.cumprod(held, 0), 0)
        shared += run
        sharing = (run == BLOCK_P) & (shared < cached_req)
    deepest = tl.load(block_table_ptr + req * table_stride + tl.maximum(shared - 1, 0))
    tl.store(shared_pages_ptr + req, shared)
    tl.store(
        group_key_ptr + req,
        tl.where(shared > 0, deepest.to(tl.int64), (-1 - req).to(tl.int64)),
    )


@triton.jit
def _group_kernel(
    group_key_ptr,
    shared_pages_ptr,
    leader_ptr,
    prefix_pages_ptr,
    num_reqs,
    page_size,
    min_prefix_tokens,
    BLOCK_B: tl.constexpr,
):
    """Requests with the same deepest shared page form a group, led by the first one."""
    req = tl.program_id(0)
    reqs = tl.arange(0, BLOCK_B)
    same = (reqs < num_reqs) & (
        tl.load(group_key_ptr + reqs, mask=reqs < num_reqs, other=0)
        == tl.load(group_key_ptr + req)
    )
    shared = tl.load(shared_pages_ptr + req)
    use_prefix = (tl.sum(same.to(tl.int32), 0) > 1) & (
        shared * page_size >= min_prefix_tokens
    )
    tl.store(leader_ptr + req, tl.min(tl.where(same, reqs, BLOCK_B), 0))
    tl.store(prefix_pages_ptr + req, tl.where(use_prefix, shared, 0))


@triton.jit
def _plan_kernel(
    block_table_ptr,
    table_stride,
    seq_lens_ptr,
    leader_ptr,
    prefix_pages_ptr,
    order_ptr,
    prefix_cu_seqlens_q_ptr,
    prefix_seqused_k_ptr,
    suffix_cu_seqlens_q_ptr,
    suffix_seqused_k_ptr,
    suffix_table_ptr,
    num_reqs,
    num_pages,
    page_size,
    BLOCK_B: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Sort requests by group (stable) and fill the prefix and suffix call metadata."""
    req = tl.program_id(0)
    reqs = tl.arange(0, BLOCK_B)
    leaders = tl.load(leader_ptr + reqs, mask=reqs < num_reqs, other=BLOCK_B)
    leader = tl.load(leader_ptr + req)
    prefix_pages = tl.load(prefix_pages_ptr + req)
    position = tl.sum(
        ((leaders < leader) | ((leaders == leader) & (reqs < req))).to(tl.int32), 0
    )
    tl.store(order_ptr + position, req.to(tl.int64))
    tl.store(
        suffix_seqused_k_ptr + position,
        tl.load(seq_lens_ptr + req) - prefix_pages * page_size,
    )
    # Slot `req` holds the queries of the group `req` leads.
    tl.store(
        prefix_seqused_k_ptr + req,
        tl.where(leader == req, prefix_pages * page_size, 0),
    )
    tl.store(
        prefix_cu_seqlens_q_ptr + req + 1, tl.sum((leaders <= req).to(tl.int32), 0)
    )
    tl.store(suffix_cu_seqlens_q_ptr + req + 1, req + 1)
    if req == 0:
        tl.store(prefix_cu_seqlens_q_ptr, 0)
        tl.store(suffix_cu_seqlens_q_ptr, 0)
    for start in range(0, num_pages, BLOCK_P):
        pages = start + tl.arange(0, BLOCK_P)
        source = tl.minimum(pages + prefix_pages, num_pages - 1)
        row = tl.load(
            block_table_ptr + req * table_stride + source, mask=pages < num_pages
        )
        tl.store(
            suffix_table_ptr + position * num_pages + pages, row, mask=pages < num_pages
        )


def plan_cascade(
    block_table_BP: torch.Tensor,
    seq_lens_B: torch.Tensor,
    *,
    page_size: int,
    min_prefix_tokens: int,
) -> CascadePlan:
    """Group decode requests by their deepest shared prefix.

    Prefix caching reuses a physical page only under the same prefix, so two requests
    holding the same page at position ``p`` share every page before it too. A request's
    group is the requests that share its deepest shared page; groups of one, or with a
    prefix under ``min_prefix_tokens``, get an empty prefix and decode as usual. Three
    small kernels, so the plan costs a few us per step.

    Example::

        block table rows (pages)   seq_len   group prefix
        [7, 8, 9, 20]              500       3 pages
        [7, 8, 9, 21]              510       3 pages
        [4, 5, 22, 0]              300       0 (no other request shares page 5)
    """
    num_reqs, num_pages = block_table_BP.shape
    device = block_table_BP.device
    int32 = dict(dtype=torch.int32, device=device)
    block_b = triton.next_power_of_2(num_reqs)
    shared_pages_B = torch.empty(num_reqs, **int32)
    group_key_B = torch.empty(num_reqs, dtype=torch.int64, device=device)
    leader_B = torch.empty(num_reqs, **int32)
    prefix_pages_B = torch.empty(num_reqs, **int32)
    plan = CascadePlan(
        order=torch.empty(num_reqs, dtype=torch.int64, device=device),
        prefix_cu_seqlens_q=torch.empty(num_reqs + 1, **int32),
        prefix_seqused_k=torch.empty(num_reqs, **int32),
        suffix_cu_seqlens_q=torch.empty(num_reqs + 1, **int32),
        suffix_seqused_k=torch.empty(num_reqs, **int32),
        suffix_block_table=torch.empty(num_reqs, num_pages, **int32),
    )
    grid = (num_reqs,)
    _deepest_shared_page_kernel[grid](
        block_table_BP,
        block_table_BP.stride(0),
        seq_lens_B,
        shared_pages_B,
        group_key_B,
        num_reqs,
        page_size,
        BLOCK_B=block_b,
        BLOCK_P=32,
    )
    _group_kernel[grid](
        group_key_B,
        shared_pages_B,
        leader_B,
        prefix_pages_B,
        num_reqs,
        page_size,
        min_prefix_tokens,
        BLOCK_B=block_b,
    )
    _plan_kernel[grid](
        block_table_BP,
        block_table_BP.stride(0),
        seq_lens_B,
        leader_B,
        prefix_pages_B,
        plan.order,
        plan.prefix_cu_seqlens_q,
        plan.prefix_seqused_k,
        plan.suffix_cu_seqlens_q,
        plan.suffix_seqused_k,
        plan.suffix_block_table,
        num_reqs,
        num_pages,
        page_size,
        BLOCK_B=block_b,
        BLOCK_P=256,
    )
    return plan


def cascade_decode(
    out_BHD: torch.Tensor,
    query_BHD: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_BP: torch.Tensor,
    plan: CascadePlan,
    *,
    max_seqlen_k: int,
    scale: float,
    enable_gqa: bool,
    num_splits: int,
    prefix_num_splits: int,
) -> None:
    """One decode token per request: shared-prefix attention + suffix attention, merged.

    Requests with an empty prefix get exactly the suffix output, which equals the plain
    decode output. ``prefix_num_splits`` is explicit because FA4's split heuristic counts
    every request slot of the prefix call, though only group leaders' slots hold queries.
    """
    num_reqs = query_BHD.shape[0]
    sorted_query_BHD = query_BHD.index_select(0, plan.order)
    kwargs = dict(return_aux=AuxRequest(lse=True), scale=scale, enable_gqa=enable_gqa)
    prefix_out_BHD, prefix_lse_HB = varlen_attn_out(
        torch.empty_like(sorted_query_BHD),
        sorted_query_BHD,
        key_cache,
        value_cache,
        plan.prefix_cu_seqlens_q,
        None,
        num_reqs,
        max_seqlen_k,
        window_size=(-1, -1),
        seqused_k=plan.prefix_seqused_k,
        block_table=block_table_BP,
        num_splits=prefix_num_splits,
        **kwargs,
    )
    suffix_out_BHD, suffix_lse_HB = varlen_attn_out(
        torch.empty_like(sorted_query_BHD),
        sorted_query_BHD,
        key_cache,
        value_cache,
        plan.suffix_cu_seqlens_q,
        None,
        1,
        max_seqlen_k,
        window_size=(-1, 0),
        seqused_k=plan.suffix_seqused_k,
        block_table=plan.suffix_block_table,
        num_splits=num_splits,
        **kwargs,
    )
    merged_BHD = torch.empty_like(sorted_query_BHD)
    merge_attn_states(
        merged_BHD, prefix_out_BHD, prefix_lse_HB, suffix_out_BHD, suffix_lse_HB
    )
    out_BHD.index_copy_(0, plan.order, merged_BHD)
