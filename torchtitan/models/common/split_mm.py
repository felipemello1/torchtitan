# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""``split_mm(a, b)``: ``a @ b`` for an fp32 ``a`` and a bf16 ``b`` on bf16 tensor cores, H100 only.

``FP32OutputLinear``'s backward multiplies the fp32 grad_output by bf16 operands. The torch path
splits grad_output into stacked bf16 pieces in memory and runs cuBLAS GEMMs over them, each summing
up to ``linear._MAX_K_PER_GEMM`` (8192) of K inside the tensor core, which loses precision as it
goes (an LM head's grad_input sums 152k).
This kernel instead, per 128-row output tile:

    TMA-loads an fp32 tile of ``a`` and a bf16 tile of ``b`` into shared memory
    splits the fp32 tile into bf16 pieces in registers (rule of ``linear._split_into_bf16_pieces``)
        and, for a transposed ``a`` (grad_weight), stores them back to shared memory
    runs one async wgmma per piece, smallest first, into a partial sum that restarts every
        ``_PROMOTE_EVERY`` K tiles (256 products), and adds each partial into an fp32 accumulator
        with ordinary fp32 adds ("promoted accumulation", as in DeepSeek-V3's FP8 GEMMs); 2 pieces
        over a short K skip it, see ``_MAX_UNPROMOTED_K``

LM-head grad_input (Qwen3-8B, 2048 tokens, 2 pieces): 2.4e-6 relative error vs fp64, vs 1.3e-5 for
the torch path's split-K (2.9e-4 for one GEMM over the whole K) and 1.2e-4 for an IEEE fp32 matmul.
No pieces are written to global memory. ``linear._split_mm_supported`` says which operands this runs.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

# Shape suffix legend: M, N, K = matmul dims (a is [M, K], b is [K, N]); S = split-K partials.

# BLOCK_M: two consumer warp groups of wgmma's 64 rows.
_BLOCK_M, _BLOCK_K = 128, 64
_PROMOTE_EVERY = 4
# Below this K, a 2-piece sum's error is set by the 8 bits the split drops, not by the tensor
# core's accumulation: LM-head grad_weight (K = 2048 tokens) 6.8e-6 unpromoted (128 x 256 tiles,
# 8.3 ms) vs 2.3e-6 promoted (128 x 128 tiles, 11.2 ms); the torch path gets 7.6e-6.
_MAX_UNPROMOTED_K = 4096


@torch.library.custom_op("torchtitan::split_mm", mutates_args=())
def split_mm(
    a_MK: torch.Tensor, b_KN: torch.Tensor, num_pieces: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """``a_MK @ b_KN`` (fp32 x bf16), accumulated in fp32, returned in ``out_dtype``.

    Args:
        a_MK: fp32, row-major or a transposed view of a row-major tensor; see
            ``linear._split_mm_supported``.
        b_KN: bf16, row-major.
        num_pieces: bf16 pieces per element of ``a_MK``: 2 (16 of fp32's 24 bits) or 3 (exact).
        out_dtype: the result is rounded once from the fp32 accumulator.

    Example:
        grad_input_TD = split_mm(grad_output_TO, weight_OD, 2, torch.bfloat16)
        grad_weight_OD = split_mm(grad_output_TO.T, input_TD, 2, torch.float32)
    """
    (m, k), n = a_MK.shape, b_KN.shape[1]
    if m == 0 or n == 0 or k == 0:
        return torch.zeros(m, n, device=a_MK.device, dtype=out_dtype)
    a_is_transposed = a_MK.stride(1) != 1
    num_sms = torch.cuda.get_device_properties(a_MK.device).multi_processor_count
    num_splits, k_tiles_per_split, promote, block_n, num_buffers = _plan(
        m, n, k, num_pieces, a_is_transposed, num_sms
    )
    out_SMN = torch.empty(
        num_splits,
        m,
        n,
        device=a_MK.device,
        dtype=out_dtype if num_splits == 1 else torch.float32,
    )

    a_rows = a_MK.T if a_is_transposed else a_MK
    # A view can start off a 16-byte boundary (fresh allocations start on 512 bytes).
    a_rows = a_rows if a_rows.data_ptr() % 16 == 0 else _aligned_copy(a_rows)
    b_KN = b_KN if b_KN.data_ptr() % 16 == 0 else _aligned_copy(b_KN)
    a_block = [_BLOCK_K, _BLOCK_M] if a_is_transposed else [_BLOCK_M, _BLOCK_K]
    a_desc = TensorDescriptor.from_tensor(
        a_rows, a_block, gl.NVMMASharedLayout.get_default_for(a_block, gl.float32)
    )
    b_block = [_BLOCK_K, block_n]
    b_desc = TensorDescriptor.from_tensor(
        b_KN, b_block, gl.NVMMASharedLayout.get_default_for(b_block, gl.bfloat16)
    )
    num_tiles = triton.cdiv(m, _BLOCK_M) * triton.cdiv(n, block_n)
    _split_mm_kernel[(num_tiles, num_splits)](
        a_desc,
        b_desc,
        out_SMN,
        m,
        n,
        k,
        k_tiles_per_split,
        out_SMN.stride(0),
        out_SMN.stride(1),
        out_SMN.stride(2),
        NUM_PIECES=num_pieces,
        PROMOTE_EVERY=_PROMOTE_EVERY if promote else 0,
        NUM_BUFFERS=num_buffers,
        A_IS_TRANSPOSED=a_is_transposed,
        BLOCK_M=_BLOCK_M,
        GROUP_M=8,
        num_warps=4,
    )
    return out_SMN[0] if num_splits == 1 else out_SMN.sum(dim=0).to(out_dtype)


@split_mm.register_fake
def _(a_MK, b_KN, num_pieces, out_dtype):
    return a_MK.new_empty(a_MK.shape[0], b_KN.shape[1], dtype=out_dtype)


def _aligned_copy(tensor: torch.Tensor) -> torch.Tensor:
    """A copy at a fresh (aligned) address with the same strides: ``clone()`` would drop a padded
    row's 16-byte stride, which TMA needs."""
    return torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
    ).copy_(tensor)


def _plan(
    m: int, n: int, k: int, num_pieces: int, a_is_transposed: bool, num_sms: int
) -> tuple[int, int, bool, int, int]:
    """``(num_splits, k_tiles_per_split, promote, block_n, num_buffers)`` for an M x N x K product.

    Example (H100, 132 SMs):
        _plan(2048, 4096, 151936, 2, False, 132) -> (1, 2374, True, 128, 4)  # LM-head grad_input
        _plan(151936, 4096, 2048, 2, True, 132) -> (1, 32, False, 256, 3)  # LM-head grad_weight
        _plan(128, 2048, 16384, 3, True, 132) -> (16, 16, True, 128, 3)  # router grad_weight
    """
    k_tiles = triton.cdiv(k, _BLOCK_K)
    # Split K when the tiles can't fill two waves of SMs (a router's grad_weight: 16 tiles,
    # K = tokens), keeping at least 16 K tiles per split. Tiles are counted 128 wide: block_n
    # is picked below, from the K per split.
    num_tiles_128 = triton.cdiv(m, _BLOCK_M) * triton.cdiv(n, 128)
    num_splits = max(1, min(triton.cdiv(2 * num_sms, num_tiles_128), k_tiles // 16))
    k_tiles_per_split = triton.cdiv(k_tiles, num_splits)
    num_splits = triton.cdiv(k_tiles, k_tiles_per_split)
    promote = num_pieces == 3 or k_tiles_per_split * _BLOCK_K > _MAX_UNPROMOTED_K
    # A promoted 128 x 256 tile would need 256 accumulator registers per thread.
    block_n, num_buffers = (128, 4) if promote else (256, 3)
    if a_is_transposed and num_pieces == 3:
        # 3 stages of 48 KB and 48 KB of pieces (see `_consumer`) fill the 227 KB of shared memory.
        num_buffers = 3
    return num_splits, k_tiles_per_split, promote, block_n, num_buffers


@gluon.aggregate
class _KernelArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    piece_bufs: gl.shared_memory_descriptor
    ready: gl.shared_memory_descriptor
    empty: gl.shared_memory_descriptor
    out_ptr: gl.tensor
    M: gl.tensor
    N: gl.tensor
    off_m: gl.tensor
    off_n: gl.tensor
    k_tile_begin: gl.tensor
    num_k_tiles: gl.tensor
    stride_os: gl.tensor
    stride_om: gl.tensor
    stride_on: gl.tensor
    NUM_PIECES: gl.constexpr
    PROMOTE_EVERY: gl.constexpr
    A_IS_TRANSPOSED: gl.constexpr


@gluon.jit
def _split_mm_kernel(
    a_desc,
    b_desc,
    out_ptr,
    M,
    N,
    K,
    k_tiles_per_split,
    stride_os,
    stride_om,
    stride_on,
    NUM_PIECES: gl.constexpr,
    PROMOTE_EVERY: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    A_IS_TRANSPOSED: gl.constexpr,
    BLOCK_M: gl.constexpr,
    GROUP_M: gl.constexpr,
):
    """One 128 x BLOCK_N output tile: two consumer warp groups, 64 rows each; see ``_consumer``.

    There is no separate TMA warp: a 3rd warp group makes Triton cap every warp at 168 registers
    (``.maxnreg``), and ptxas then serializes every wgmma (C7512, "insufficient register resources").
    """
    gl.static_assert(BLOCK_M == 128, "two consumer warp groups of wgmma's 64 rows")
    BLOCK_K: gl.constexpr = b_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_type.shape[1]
    # Grouped tile order for L2 reuse; program_id(1) is the split-K partial.
    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    first_pid_m = pid // (GROUP_M * num_pid_n) * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % (GROUP_M * num_pid_n)) % group_size_m
    pid_n = (pid % (GROUP_M * num_pid_n)) // group_size_m
    k_tile_begin = gl.program_id(1) * k_tiles_per_split
    num_k_tiles = min(k_tiles_per_split, gl.cdiv(K, BLOCK_K) - k_tile_begin)

    a_bufs = gl.allocate_shared_memory(
        gl.float32, [NUM_BUFFERS] + a_desc.block_type.shape, a_desc.layout
    )
    b_bufs = gl.allocate_shared_memory(
        gl.bfloat16, [NUM_BUFFERS] + b_desc.block_type.shape, b_desc.layout
    )
    if A_IS_TRANSPOSED:
        # Each warp group's pieces, [BLOCK_K, 64] each, M contiguous; see `_consumer`.
        piece_bufs = gl.allocate_shared_memory(
            gl.bfloat16,
            [2 * NUM_PIECES, BLOCK_K, 64],
            gl.NVMMASharedLayout.get_default_for([BLOCK_K, 64], gl.bfloat16),
        )
    else:
        # Unused: a row-major `a`'s pieces stay in registers.
        piece_bufs = b_bufs
    ready = gl.allocate_shared_memory(
        gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout()
    )
    empty = gl.allocate_shared_memory(
        gl.int64, [NUM_BUFFERS, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(NUM_BUFFERS):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=2)  # one arrive per consumer warp group
    # Make the initialized barriers visible to the TMA (async proxy), as CUDA's TMA examples do.
    fence_async_shared()

    # gl.to_tensor: Triton turns integer arguments equal to 1 into constexprs.
    args = _KernelArgs(
        a_desc,
        b_desc,
        a_bufs,
        b_bufs,
        piece_bufs,
        ready,
        empty,
        out_ptr,
        gl.to_tensor(M),
        gl.to_tensor(N),
        gl.to_tensor(pid_m * BLOCK_M),
        gl.to_tensor(pid_n * BLOCK_N),
        gl.to_tensor(k_tile_begin),
        gl.to_tensor(num_k_tiles),
        gl.to_tensor(stride_os),
        gl.to_tensor(stride_om),
        gl.to_tensor(stride_on),
        NUM_PIECES,
        PROMOTE_EVERY,
        A_IS_TRANSPOSED,
    )
    gl.warp_specialize([(_consumer, (args, 0)), (_consumer, (args, 1))], [4])


@gluon.jit
def _consumer(args: _KernelArgs, WARP_GROUP: gl.constexpr):
    """Rows [64 WARP_GROUP, 64 WARP_GROUP + 64) of the tile: split, MMAs, promotion, store.

    Per K tile i:
        wait for this warp group's MMAs of tile i-1, release tile i-1's stage, promote
        wait for tile i's TMA, load and split the fp32 rows, issue one async wgmma per piece
        warp group 0 only: TMA-load tile i + NUM_BUFFERS - 1 once both released tile i-1

    The wait comes BEFORE the split: an in-flight wgmma reads its A registers asynchronously,
    ptxas reuses registers it thinks are dead, and writing the next pieces there returned NaN.
    The two warp groups never synchronize otherwise, so one splits while the other's MMAs run.

    A transposed ``a`` (grad_weight) is M-contiguous in shared memory. Loading it in the wgmma's
    register layout takes 4-byte loads, and ptxas then serializes the wgmmas (C7515). Instead this
    loads 16 bytes per thread, splits, stores the pieces to this warp group's shared buffers (free
    after the same wait), and the wgmmas read them there. Qwen3-8B LM-head grad_weight, 2 pieces:
    8.2 ms, vs 8.8 ms with the pieces in registers.
    """
    BLOCK_K: gl.constexpr = args.b_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = args.b_desc.block_type.shape[1]
    NUM_BUFFERS: gl.constexpr = args.a_bufs.type.shape[0]
    ROWS: gl.constexpr = 64  # wgmma's M: this warp group's rows of the tile
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BLOCK_N, 16]
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=2
    )
    # 4 consecutive M (16 bytes) per thread, along a transposed tile's contiguous dim.
    transposed_a_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [2, 16], [4, 1], [1, 0]
    )

    if WARP_GROUP == 0:
        for j in gl.static_range(NUM_BUFFERS - 1):
            if j < args.num_k_tiles:
                _load_k_tile(args, j)
    acc = gl.zeros((ROWS, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    part = warpgroup_mma_init(
        gl.zeros((ROWS, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    )
    for i in range(args.num_k_tiles):
        stage = i % NUM_BUFFERS
        part = warpgroup_mma_wait(0, deps=(part,))
        if i > 0:
            # Order this warp group's shared loads of tile i-1 before the TMA that refills it.
            fence_async_shared()
            mbarrier.arrive(args.empty.index((i - 1) % NUM_BUFFERS), count=1)
        if args.PROMOTE_EVERY > 0:
            if i % args.PROMOTE_EVERY == 0:
                acc += part
        mbarrier.wait(args.ready.index(stage), (i // NUM_BUFFERS) & 1)
        if args.A_IS_TRANSPOSED:
            # [BLOCK_K, 64]: this warp group's M columns.
            a_tile = args.a_bufs.index(stage).slice(WARP_GROUP * ROWS, ROWS, dim=1)
            a = a_tile.load(transposed_a_layout)
        else:
            # Logical row 8h + 4b + l reads row 8h + 2l + b (the epilogue stores it there), so each
            # half-warp's 8-byte loads hit 4 rows of different swizzle phase: no bank conflicts,
            # LM-head grad_input 9.02 vs 9.25 ms.
            a_tile = (
                args.a_bufs.index(stage)
                .reshape((2 * ROWS // 8, 4, 2, BLOCK_K))
                .permute((0, 2, 1, 3))
                .reshape((2 * ROWS, BLOCK_K))
                .slice(WARP_GROUP * ROWS, ROWS, dim=0)
            )
            a = a_tile.load(a_layout)
        # Same pieces as linear._split_into_bf16_pieces: hi rounds to nearest (ties away) with
        # integer ops, later pieces are what is left.
        hi = ((a.to(gl.int32, bitcast=True) + 0x8000) & -65536).to(
            gl.float32, bitcast=True
        )
        rest = a - hi
        b = args.b_bufs.index(stage)
        # use_acc=False starts a new partial sum (promoted) or the only sum (PROMOTE_EVERY = 0).
        if args.PROMOTE_EVERY > 0:
            use_acc = i % args.PROMOTE_EVERY != 0
        else:
            use_acc = i > 0
        if args.A_IS_TRANSPOSED:
            # This warp group's pieces: piece_bufs[WARP_GROUP * NUM_PIECES + q], smallest first.
            first = WARP_GROUP * args.NUM_PIECES
            if args.NUM_PIECES == 2:
                args.piece_bufs.index(first).store(rest.to(gl.bfloat16))
            else:
                mid = ((rest.to(gl.int32, bitcast=True) + 0x8000) & -65536).to(
                    gl.float32, bitcast=True
                )
                args.piece_bufs.index(first).store((rest - mid).to(gl.bfloat16))
                args.piece_bufs.index(first + 1).store(mid.to(gl.bfloat16))
            args.piece_bufs.index(first + args.NUM_PIECES - 1).store(hi.to(gl.bfloat16))
            # All 4 warps' stores, made visible to the wgmma (async proxy).
            fence_async_shared()
            gl.barrier()
            part = warpgroup_mma(
                args.piece_bufs.index(first).permute((1, 0)),
                b,
                part,
                use_acc=use_acc,
                is_async=True,
            )
            for q in gl.static_range(1, args.NUM_PIECES):
                part = warpgroup_mma(
                    args.piece_bufs.index(first + q).permute((1, 0)),
                    b,
                    part,
                    is_async=True,
                )
        else:
            if args.NUM_PIECES == 2:
                part = warpgroup_mma(
                    rest.to(gl.bfloat16), b, part, use_acc=use_acc, is_async=True
                )
            else:
                mid = ((rest.to(gl.int32, bitcast=True) + 0x8000) & -65536).to(
                    gl.float32, bitcast=True
                )
                part = warpgroup_mma(
                    (rest - mid).to(gl.bfloat16),
                    b,
                    part,
                    use_acc=use_acc,
                    is_async=True,
                )
                part = warpgroup_mma(mid.to(gl.bfloat16), b, part, is_async=True)
            part = warpgroup_mma(hi.to(gl.bfloat16), b, part, is_async=True)
        if WARP_GROUP == 0:
            if i + NUM_BUFFERS - 1 < args.num_k_tiles:
                if i > 0:
                    mbarrier.wait(
                        args.empty.index((i - 1) % NUM_BUFFERS),
                        ((i - 1) // NUM_BUFFERS) & 1,
                    )
                _load_k_tile(args, i + NUM_BUFFERS - 1)
    part = warpgroup_mma_wait(0, deps=(part,))
    if args.PROMOTE_EVERY > 0:
        acc += part
    else:
        acc = part

    rows = gl.arange(0, ROWS, layout=gl.SliceLayout(1, mma_layout))
    if not args.A_IS_TRANSPOSED:
        # The A load's row permutation: logical 8h + 4b + l is row 8h + 2l + b.
        rows = (rows & -8) | ((rows & 3) << 1) | ((rows >> 2) & 1)
    offs_m = args.off_m + WARP_GROUP * ROWS + rows
    offs_n = args.off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mma_layout))
    # int64: an output can have more than 2^31 elements (a 256k vocab x 8k dim grad_weight).
    out_ptrs = (
        args.out_ptr
        + gl.program_id(1).to(gl.int64) * args.stride_os
        + offs_m.to(gl.int64)[:, None] * args.stride_om
        + offs_n[None, :] * args.stride_on
    )
    mask = (offs_m[:, None] < args.M) & (offs_n[None, :] < args.N)
    gl.store(out_ptrs, acc.to(args.out_ptr.dtype.element_ty), mask=mask)


@gluon.jit
def _load_k_tile(args: _KernelArgs, i):
    """TMA-load this program's i-th K tile of ``a`` and ``b`` into stage i % NUM_BUFFERS."""
    BLOCK_K: gl.constexpr = args.b_desc.block_type.shape[0]
    NUM_BUFFERS: gl.constexpr = args.a_bufs.type.shape[0]
    stage = i % NUM_BUFFERS
    bar = args.ready.index(stage)
    mbarrier.expect(bar, args.a_desc.block_type.nbytes + args.b_desc.block_type.nbytes)
    k = (args.k_tile_begin + i) * BLOCK_K
    if args.A_IS_TRANSPOSED:
        tma.async_load(args.a_desc, [k, args.off_m], bar, args.a_bufs.index(stage))
    else:
        tma.async_load(args.a_desc, [args.off_m, k], bar, args.a_bufs.index(stage))
    tma.async_load(args.b_desc, [k, args.off_n], bar, args.b_bufs.index(stage))
