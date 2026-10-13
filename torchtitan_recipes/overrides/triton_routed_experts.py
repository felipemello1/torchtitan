# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Triton routed experts for inference at EP=1.

``triton_routed_experts`` replaces ``RoutedExperts`` with a subclass whose no-grad
forward runs dispatch, the w13 GEMM with SwiGLU, the w2 GEMM and the combine in four
Triton kernels instead of the stock dispatch, two grouped GEMMs, the activation and the
combine. Forwards with grad, batch-invariant mode, expert-parallel dispatchers and
unsupported shapes take the stock path, so training is unchanged.

::

    slot of route r = t * K + k:  its expert's padded M-block range (scan or scatter)
    w13 GEMM per (M-block, I-tile), SwiGLU on the fp32 accumulators -> h [slot, I]
    w2 GEMM per (M-block, D-tile)                                    -> y [route, D] bf16
    combine: out[t] = sum_k score[t, k] * y[t * K + k] in fp32       -> out [T, D] bf16

From ``_GROUPED_MIN_TOKENS`` tokens (prefill) the two GEMMs are ``torch._grouped_mm`` on
unpadded expert-sorted rows, which beats these Triton GEMMs there; routing, SwiGLU and the
combine stay in Triton.

Each GEMM program finds its M-block's expert from the router's per-expert counts, so no
block-to-expert buffer is written. At decode (at most ``_SCAN_MAX_ROUTES`` routes) each
program also finds its rows by scanning the route ids, so there is no scatter kernel.
Weight tiles load through TMA descriptors, and the w2 and combine kernels use
programmatic dependent launch to overlap the previous kernel's tail.

Numerics: SwiGLU runs on the fp32 accumulators (stock rounds gate and up to bf16 first), each
route's w2 output is rounded to bf16 as on the stock path, and the K routes are summed in fp32
with one rounding. The padded per-expert M-blocks and the fused SiLU-and-mul follow the design
of vLLM's ``fused_moe`` Triton kernel (Apache-2.0); the kernels here are written from scratch.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override
from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)

__all__ = ["TritonRoutedExperts", "triton_routed_experts", "triton_routed_experts_op"]

# At most this many routes (T * K), each GEMM program scans every route id for its rows.
_SCAN_MAX_ROUTES = 1024
# From this many tokens, the GEMMs run as torch._grouped_mm (prefill).
_GROUPED_MIN_TOKENS = 4096


@dataclass(frozen=True, slots=True)
class _KernelConfig:
    block_m: int
    scan: bool
    gate_up_block_n: int
    gate_up_block_k: int
    gate_up_warps: int
    gate_up_stages: int
    down_block_n: int
    down_block_k: int
    down_warps: int
    down_stages: int
    scatter_block: int = 32
    combine_block: int = 1024
    grouped: bool = False


def _kernel_config(num_tokens: int, num_routes: int) -> _KernelConfig:
    """Return the kernel config for a step of ``num_tokens`` tokens (tuned on GB300).

    Up to ~512 tokens each touched expert gets one 16-row block (weight-bandwidth bound);
    above that, larger blocks feed the tensor cores, and from ``_GROUPED_MIN_TOKENS`` the
    GEMMs are grouped GEMMs (the block fields are unused there).

    Example:
        >>> _kernel_config(34, 272).block_m
        16
    """
    if num_tokens <= 512:
        return _KernelConfig(
            block_m=16,
            scan=num_routes <= _SCAN_MAX_ROUTES,
            gate_up_block_n=64,
            gate_up_block_k=128,
            gate_up_warps=4,
            gate_up_stages=3,
            down_block_n=128,
            down_block_k=64,
            down_warps=4,
            down_stages=3,
        )
    if num_tokens <= 2048:
        return _KernelConfig(
            block_m=64,
            scan=False,
            gate_up_block_n=64,
            gate_up_block_k=64,
            gate_up_warps=4,
            gate_up_stages=4,
            down_block_n=128,
            down_block_k=64,
            down_warps=4,
            down_stages=3,
        )
    return _KernelConfig(
        block_m=128,
        scan=False,
        grouped=num_tokens >= _GROUPED_MIN_TOKENS,
        gate_up_block_n=64,
        gate_up_block_k=64,
        gate_up_warps=8,
        gate_up_stages=3,
        down_block_n=128,
        down_block_k=64,
        down_warps=8,
        down_stages=3,
    )


@triton.jit
def _expert_of_block(counts_ptr, block, E: tl.constexpr, BLOCK_M: tl.constexpr):
    """Return ``(expert, padded start, count)`` of M-block ``block``; ``expert == E`` past the end."""
    expert_ids = tl.arange(0, E)
    counts = tl.load(counts_ptr + expert_ids).to(tl.int32)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    padded_end = tl.cumsum(padded, axis=0)
    expert = tl.sum((padded_end <= block * BLOCK_M).to(tl.int32), axis=0)
    is_expert = expert_ids == expert
    start = tl.sum(tl.where(is_expert, padded_end - padded, 0), axis=0)
    count = tl.sum(tl.where(is_expert, counts, 0), axis=0)
    return expert, start, count


@triton.jit
def _routes_by_scan(
    ids_ptr,
    num_routes,
    expert,
    first_row,
    BLOCK_M: tl.constexpr,
    ROUTES_P2: tl.constexpr,
):
    """Return the routes of ``expert``'s rows ``first_row + [0, BLOCK_M)``, in route order."""
    route_ids = tl.arange(0, ROUTES_P2)
    mine = tl.load(ids_ptr + route_ids, mask=route_ids < num_routes, other=-1) == expert
    mine = mine.to(tl.int32)
    rank = tl.cumsum(mine, axis=0) - mine
    rows = first_row + tl.arange(0, BLOCK_M)
    hit = (mine[None, :] == 1) & (rank[None, :] == rows[:, None])
    return tl.sum(tl.where(hit, route_ids[None, :], 0), axis=1)


@triton.jit
def _block_rows(
    block,
    counts_ptr,
    slot_route_ptr,
    ids_ptr,
    num_routes,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SCAN: tl.constexpr,
    ROUTES_P2: tl.constexpr,
):
    """Return ``(expert, slots, valid, routes)`` of M-block ``block``."""
    expert, start, count = _expert_of_block(counts_ptr, block, E, BLOCK_M)
    slots = block * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = slots - start < count
    if SCAN:
        routes = _routes_by_scan(
            ids_ptr, num_routes, expert, block * BLOCK_M - start, BLOCK_M, ROUTES_P2
        )
    else:
        routes = tl.load(slot_route_ptr + slots, mask=valid, other=0)
    return expert, slots, valid, routes


@triton.jit
def _scatter_routes_kernel(
    ids_ptr,
    counts_ptr,
    fill_ptr,
    slot_route_ptr,
    num_routes,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Give each route a slot in its expert's padded block range.

    The slot order inside an expert follows the atomics, which doesn't change any row's result.
    """
    routes = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    in_range = routes < num_routes
    experts = tl.load(ids_ptr + routes, mask=in_range, other=0).to(tl.int32)
    expert_ids = tl.arange(0, E)
    counts = tl.load(counts_ptr + expert_ids).to(tl.int32)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    padded_start = tl.cumsum(padded, axis=0) - padded
    first_slot = tl.sum(
        tl.where(experts[:, None] == expert_ids[None, :], padded_start[None, :], 0),
        axis=1,
    )
    rank = tl.atomic_add(fill_ptr + experts, 1, mask=in_range, sem="relaxed")
    tl.store(slot_route_ptr + first_slot + rank, routes, mask=in_range)


@triton.jit
def _gate_up_swiglu_kernel(
    x_ptr,
    w13_desc,
    h_ptr,
    slot_route_ptr,
    counts_ptr,
    ids_ptr,
    num_routes,
    stride_x,
    stride_h,
    D: tl.constexpr,
    I: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCAN: tl.constexpr,
    ROUTES_P2: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """``h[slot] = silu(x @ w13[e, 0].T) * (x @ w13[e, 1].T)`` for one (M-block, I-tile)."""
    block = tl.program_id(0)
    tile_n = tl.program_id(1)
    expert, slots, valid, routes = _block_rows(
        block,
        counts_ptr,
        slot_route_ptr,
        ids_ptr,
        num_routes,
        E,
        BLOCK_M,
        SCAN,
        ROUTES_P2,
    )
    if expert >= E:
        return
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + (routes // K)[:, None].to(tl.int64) * stride_x + offs_k[None, :]
    gate_row = expert * 2 * I + tile_n * BLOCK_N
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, D // BLOCK_K, num_stages=NUM_STAGES):
        x = tl.load(x_ptrs, mask=valid[:, None], other=0.0)
        acc_gate = tl.dot(x, w13_desc.load([gate_row, k * BLOCK_K]).T, acc_gate)
        acc_up = tl.dot(x, w13_desc.load([gate_row + I, k * BLOCK_K]).T, acc_up)
        x_ptrs += BLOCK_K
    tl.extra.cuda.gdc_launch_dependents()
    h = acc_gate * tl.sigmoid(acc_gate) * acc_up
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    h_ptrs = h_ptr + slots[:, None].to(tl.int64) * stride_h + offs_n[None, :]
    tl.store(h_ptrs, h.to(h_ptr.dtype.element_ty), mask=valid[:, None])


@triton.jit
def _down_kernel(
    h_ptr,
    w2_desc,
    y_ptr,
    slot_route_ptr,
    counts_ptr,
    ids_ptr,
    num_routes,
    stride_h,
    stride_y,
    D: tl.constexpr,
    I: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCAN: tl.constexpr,
    ROUTES_P2: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """``y[route] = h[slot] @ w2[e].T``, rounded to bf16, for one (M-block, D-tile)."""
    block = tl.program_id(0)
    tile_n = tl.program_id(1)
    expert, slots, valid, routes = _block_rows(
        block,
        counts_ptr,
        slot_route_ptr,
        ids_ptr,
        num_routes,
        E,
        BLOCK_M,
        SCAN,
        ROUTES_P2,
    )
    if expert >= E:
        return
    offs_k = tl.arange(0, BLOCK_K)
    h_ptrs = h_ptr + slots[:, None].to(tl.int64) * stride_h + offs_k[None, :]
    w_row = expert * D + tile_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Programmatic dependent launch: everything above overlaps the w13 kernel's tail.
    tl.extra.cuda.gdc_wait()
    for k in tl.range(0, I // BLOCK_K, num_stages=NUM_STAGES):
        h = tl.load(h_ptrs, mask=valid[:, None], other=0.0)
        acc = tl.dot(h, w2_desc.load([w_row, k * BLOCK_K]).T, acc)
        h_ptrs += BLOCK_K
    tl.extra.cuda.gdc_launch_dependents()
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + routes[:, None].to(tl.int64) * stride_y + offs_n[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=valid[:, None])


@triton.jit
def _combine_kernel(
    y_ptr, scores_ptr, out_ptr, D: tl.constexpr, K: tl.constexpr, BLOCK_D: tl.constexpr
):
    """``out[t] = sum_k score[t, k] * y[t * K + k]`` in fp32, one rounding."""
    tl.extra.cuda.gdc_wait()
    token = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in tl.static_range(K):
        score = tl.load(scores_ptr + token * K + k).to(tl.float32)
        y = tl.load(y_ptr + (token * K + k).to(tl.int64) * D + offs).to(tl.float32)
        acc += score * y
    tl.store(out_ptr + token.to(tl.int64) * D + offs, acc.to(out_ptr.dtype.element_ty))


@triton.jit
def _sort_and_gather_kernel(
    ids_ptr,
    counts_ptr,
    fill_ptr,
    route_row_ptr,
    x_ptr,
    x_sorted_ptr,
    num_routes,
    stride_x,
    stride_x_sorted,
    D: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy each route's token row to its row in the unpadded expert-sorted order."""
    routes = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    in_range = routes < num_routes
    experts = tl.load(ids_ptr + routes, mask=in_range, other=0).to(tl.int32)
    expert_ids = tl.arange(0, E)
    counts = tl.load(counts_ptr + expert_ids).to(tl.int32)
    start = tl.cumsum(counts, axis=0) - counts
    first_row = tl.sum(
        tl.where(experts[:, None] == expert_ids[None, :], start[None, :], 0), axis=1
    )
    rows = first_row + tl.atomic_add(
        fill_ptr + experts, 1, mask=in_range, sem="relaxed"
    )
    tl.store(route_row_ptr + routes, rows, mask=in_range)
    tokens = routes // K
    for d0 in tl.static_range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        row = tl.load(
            x_ptr + tokens[:, None].to(tl.int64) * stride_x + cols[None, :],
            mask=in_range[:, None],
            other=0.0,
        )
        tl.store(
            x_sorted_ptr + rows[:, None].to(tl.int64) * stride_x_sorted + cols[None, :],
            row,
            mask=in_range[:, None],
        )


@triton.jit
def _swiglu_rows_kernel(
    gate_up_ptr,
    h_ptr,
    num_rows,
    I: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """``h = silu(gate) * up`` for ``gate_up`` rows laid out ``[gate (I) | up (I)]``."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    in_range = (rows[:, None] < num_rows) & (cols[None, :] < I)
    gate_ptrs = gate_up_ptr + rows[:, None].to(tl.int64) * (2 * I) + cols[None, :]
    gate = tl.load(gate_ptrs, mask=in_range, other=0.0).to(tl.float32)
    up = tl.load(gate_ptrs + I, mask=in_range, other=0.0).to(tl.float32)
    h = gate * tl.sigmoid(gate) * up
    tl.store(
        h_ptr + rows[:, None].to(tl.int64) * I + cols[None, :],
        h.to(h_ptr.dtype.element_ty),
        mask=in_range,
    )


@triton.jit
def _combine_rows_kernel(
    y_ptr,
    route_row_ptr,
    scores_ptr,
    out_ptr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """``out[t] = sum_k score[t, k] * y[row of route t * K + k]`` in fp32, one rounding."""
    token = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in tl.static_range(K):
        score = tl.load(scores_ptr + token * K + k).to(tl.float32)
        row = tl.load(route_row_ptr + token * K + k).to(tl.int64)
        acc += score * tl.load(y_ptr + row * D + offs).to(tl.float32)
    tl.store(out_ptr + token.to(tl.int64) * D + offs, acc.to(out_ptr.dtype.element_ty))


def _grouped_routed_experts(
    x_TD, topk_scores_TK, ids_N, num_tokens_per_expert_E, w13_E2FD, w2_EDF, cfg
) -> torch.Tensor:
    """Prefill: Triton routing, SwiGLU and combine around two ``torch._grouped_mm``."""
    num_tokens, dim = x_TD.shape
    top_k = topk_scores_TK.shape[1]
    num_experts, _, hidden, _ = w13_E2FD.shape
    num_routes = num_tokens * top_k
    device = x_TD.device
    fill_E = torch.zeros(num_experts, dtype=torch.int32, device=device)
    route_row = torch.empty(num_routes, dtype=torch.int32, device=device)
    x_sorted = torch.empty(num_routes, dim, dtype=x_TD.dtype, device=device)
    _sort_and_gather_kernel[(triton.cdiv(num_routes, 64),)](
        ids_N,
        num_tokens_per_expert_E,
        fill_E,
        route_row,
        x_TD,
        x_sorted,
        num_routes,
        x_TD.stride(0),
        x_sorted.stride(0),
        D=dim,
        K=top_k,
        E=num_experts,
        BLOCK_R=64,
        BLOCK_D=256,
    )
    offsets_E = torch.cumsum(num_tokens_per_expert_E, 0, dtype=torch.int32)
    gate_up = torch._grouped_mm(
        x_sorted,
        w13_E2FD.view(num_experts, 2 * hidden, dim).transpose(1, 2),
        offs=offsets_E,
    )
    h = torch.empty(num_routes, hidden, dtype=x_TD.dtype, device=device)
    _swiglu_rows_kernel[(triton.cdiv(num_routes, 32), triton.cdiv(hidden, 256))](
        gate_up, h, num_routes, I=hidden, BLOCK_R=32, BLOCK_I=256
    )
    y = torch._grouped_mm(h, w2_EDF.transpose(1, 2), offs=offsets_E)
    out_TD = torch.empty(num_tokens, dim, dtype=x_TD.dtype, device=device)
    _combine_rows_kernel[(num_tokens, dim // cfg.combine_block)](
        y, route_row, topk_scores_TK, out_TD, D=dim, K=top_k, BLOCK_D=cfg.combine_block
    )
    return out_TD


@torch.library.custom_op("torchtitan::triton_routed_experts", mutates_args=())
def triton_routed_experts_op(
    x_TD: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    topk_expert_ids_TK: torch.Tensor,
    num_tokens_per_expert_E: torch.Tensor,
    w13_E2FD: torch.Tensor,
    w2_EDF: torch.Tensor,
) -> torch.Tensor:
    """Routed-experts output ``(T, D)``: each token's ``K`` routes, weighted by their scores.

    Example:
        >>> out_TD = triton_routed_experts_op(x_TD, scores_TK, ids_TK, counts_E, w13, w2)
    """
    from triton.tools.tensor_descriptor import TensorDescriptor

    num_tokens, dim = x_TD.shape
    top_k = topk_expert_ids_TK.shape[1]
    num_experts, _, hidden, _ = w13_E2FD.shape
    num_routes = num_tokens * top_k
    cfg = _kernel_config(num_tokens, num_routes)
    block_m = cfg.block_m
    # Upper bound on M-blocks: every route's block plus one partial block per touched expert.
    max_blocks = num_routes // block_m + min(num_routes, num_experts)
    device = x_TD.device
    ids_N = topk_expert_ids_TK.reshape(-1)
    if cfg.grouped:
        return _grouped_routed_experts(
            x_TD, topk_scores_TK, ids_N, num_tokens_per_expert_E, w13_E2FD, w2_EDF, cfg
        )
    slot_route = torch.empty(
        1 if cfg.scan else max_blocks * block_m, dtype=torch.int32, device=device
    )
    if not cfg.scan:
        fill_E = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _scatter_routes_kernel[(triton.cdiv(num_routes, cfg.scatter_block),)](
            ids_N,
            num_tokens_per_expert_E,
            fill_E,
            slot_route,
            num_routes,
            E=num_experts,
            BLOCK_M=block_m,
            BLOCK_R=cfg.scatter_block,
        )
    h = torch.empty(max_blocks * block_m, hidden, dtype=x_TD.dtype, device=device)
    y = torch.empty(num_routes, dim, dtype=x_TD.dtype, device=device)
    out_TD = torch.empty(num_tokens, dim, dtype=x_TD.dtype, device=device)
    routes_p2 = triton.next_power_of_2(num_routes)
    w13_desc = TensorDescriptor.from_tensor(
        w13_E2FD.view(num_experts * 2 * hidden, dim),
        [cfg.gate_up_block_n, cfg.gate_up_block_k],
    )
    w2_desc = TensorDescriptor.from_tensor(
        w2_EDF.view(num_experts * dim, hidden), [cfg.down_block_n, cfg.down_block_k]
    )
    _gate_up_swiglu_kernel[(max_blocks, hidden // cfg.gate_up_block_n)](
        x_TD,
        w13_desc,
        h,
        slot_route,
        num_tokens_per_expert_E,
        ids_N,
        num_routes,
        x_TD.stride(0),
        h.stride(0),
        D=dim,
        I=hidden,
        K=top_k,
        E=num_experts,
        BLOCK_M=block_m,
        BLOCK_N=cfg.gate_up_block_n,
        BLOCK_K=cfg.gate_up_block_k,
        SCAN=cfg.scan,
        ROUTES_P2=routes_p2,
        NUM_STAGES=cfg.gate_up_stages,
        num_warps=cfg.gate_up_warps,
    )
    _down_kernel[(max_blocks, dim // cfg.down_block_n)](
        h,
        w2_desc,
        y,
        slot_route,
        num_tokens_per_expert_E,
        ids_N,
        num_routes,
        h.stride(0),
        y.stride(0),
        D=dim,
        I=hidden,
        E=num_experts,
        BLOCK_M=block_m,
        BLOCK_N=cfg.down_block_n,
        BLOCK_K=cfg.down_block_k,
        SCAN=cfg.scan,
        ROUTES_P2=routes_p2,
        NUM_STAGES=cfg.down_stages,
        num_warps=cfg.down_warps,
        launch_pdl=True,
    )
    _combine_kernel[(num_tokens, dim // cfg.combine_block)](
        y,
        topk_scores_TK,
        out_TD,
        D=dim,
        K=top_k,
        BLOCK_D=cfg.combine_block,
        launch_pdl=True,
    )
    return out_TD


@triton_routed_experts_op.register_fake
def _triton_routed_experts_fake(
    x_TD, topk_scores_TK, topk_expert_ids_TK, num_tokens_per_expert_E, w13_E2FD, w2_EDF
):
    return torch.empty_like(x_TD)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


class TritonRoutedExperts(RoutedExperts):
    """``RoutedExperts`` whose no-grad forward at EP=1 runs the Triton routed-experts op."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        pass

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w13_E2FD, w2_EDF = _local(self.w13.weight), _local(self.w2.weight)
        if not self._can_run_triton(x_TD, w13_E2FD, w2_EDF):
            return super().forward(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
                padding_mask_T=padding_mask_T,
            )
        # Padding rows are computed like real tokens, as on the stock path.
        return triton_routed_experts_op(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
            w13_E2FD,
            w2_EDF,
        )

    def _all_experts_local(self) -> bool:
        dispatcher = self.token_dispatcher
        if type(dispatcher) is LocalTokenDispatcher:
            return True
        if type(dispatcher) is not AllToAllTokenDispatcher:
            return False
        return dispatcher.ep_mesh is None or dispatcher.ep_mesh.size() == 1

    def _can_run_triton(
        self, x_TD: torch.Tensor, w13_E2FD: torch.Tensor, w2_EDF: torch.Tensor
    ) -> bool:
        if torch.is_grad_enabled() or is_in_batch_invariant_mode():
            return False
        if not self._all_experts_local():
            return False
        if (
            self.output_postprocess is not None
            or type(self.activation_fn) is not SwiGLU
        ):
            return False
        tensors = (x_TD, w13_E2FD, w2_EDF)
        if any(t.dtype != torch.bfloat16 or not t.is_cuda for t in tensors):
            return False
        if not (w13_E2FD.is_contiguous() and w2_EDF.is_contiguous()):
            return False
        dim, hidden = w2_EDF.shape[1], w2_EDF.shape[2]
        # The largest tiles of any config must divide the dimensions.
        return dim % 1024 == 0 and hidden % 128 == 0 and x_TD.shape[0] > 0


@override(
    target=RoutedExperts.Config,
    exact=True,
    description="Run no-grad routed experts at EP=1 as four Triton kernels.",
)
def triton_routed_experts(cfg: RoutedExperts.Config) -> TritonRoutedExperts.Config:
    return derive(cfg, TritonRoutedExperts.Config)
