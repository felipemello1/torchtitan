# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn.attention import activate_flash_attention_impl
from torch.nn.attention.varlen import varlen_attn_out

from torchtitan.rl.model.cascade_attention import cascade_decode, plan_cascade
from torchtitan.tools.utils import get_cuda_flash_attention_impl

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE = 16, 4, 256, 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_plan_cascade_groups_requests_by_deepest_shared_prefix():
    # 4-token pages; page 1 is a system prompt shared by every request, page 0 pads.
    block_table = torch.tensor(
        [
            [1, 2, 3, 10, 0],  # prompt A (pages 2, 3), sample 1
            [1, 2, 3, 11, 0],  # prompt A, sample 2
            [1, 4, 5, 12, 0],  # prompt B (pages 4, 5), sample 1
            [1, 4, 5, 13, 0],  # prompt B, sample 2
            [1, 6, 14, 0, 0],  # shares only the system page: no group
            [0, 0, 0, 0, 0],  # CUDA graph padding row
        ],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([14, 15, 13, 16, 10, 0], dtype=torch.int32, device="cuda")

    plan = plan_cascade(block_table, seq_lens, page_size=4, min_prefix_tokens=8)

    prefix_tokens = seq_lens[plan.order] - plan.suffix_seqused_k
    assert prefix_tokens.tolist() == [12, 12, 12, 12, 0, 0]
    # Slot g attends the prefix of the group led by request g.
    assert plan.prefix_cu_seqlens_q.tolist() == [0, 2, 2, 4, 4, 5, 6]
    assert plan.prefix_seqused_k.tolist() == [12, 0, 12, 0, 0, 0]
    assert plan.suffix_block_table[:, 0].tolist() == [10, 11, 12, 13, 1, 0]

    plan = plan_cascade(block_table, seq_lens, page_size=4, min_prefix_tokens=16)
    assert plan.prefix_seqused_k.tolist() == [0] * 6
    assert plan.suffix_seqused_k.tolist() == seq_lens[plan.order].tolist()


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_cuda_flash_attention_impl() != "FA4",
    reason="requires FA4 (SM100)",
)
def test_cascade_decode_matches_plain_decode():
    """Two prompt groups, two unshared requests and a padding row, as in an RL decode step."""
    activate_flash_attention_impl("FA4")
    torch.manual_seed(0)
    prefix_pages, own_pages = 8, 3
    rows, seq_lens, next_page = [], [], 1  # page 0 is vLLM's null block
    for group_size in (4, 4, 1, 1):
        prefix = list(range(next_page, next_page + prefix_pages))
        next_page += prefix_pages
        for _ in range(group_size):
            rows.append(prefix + list(range(next_page, next_page + own_pages)))
            next_page += own_pages
            seq_lens.append((prefix_pages + own_pages - 1) * PAGE + 5)
    rows.append([0] * (prefix_pages + own_pages))
    seq_lens.append(0)
    block_table = torch.tensor(rows, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    num_reqs = block_table.shape[0]
    shape = (next_page, PAGE, NUM_KV_HEADS, HEAD_DIM)
    key_cache = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value_cache = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    query = torch.randn(
        num_reqs, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    max_seqlen_k = int(seq_lens.max())

    plain = torch.empty_like(query)
    varlen_attn_out(
        plain,
        query,
        key_cache,
        value_cache,
        torch.arange(num_reqs + 1, dtype=torch.int32, device="cuda"),
        None,
        1,
        max_seqlen_k,
        window_size=(-1, 0),
        enable_gqa=True,
        seqused_k=seq_lens,
        block_table=block_table,
        num_splits=-1,
    )
    plan = plan_cascade(block_table, seq_lens, page_size=PAGE, min_prefix_tokens=PAGE)
    cascade = torch.empty_like(query)
    cascade_decode(
        cascade,
        query,
        key_cache,
        value_cache,
        block_table,
        plan,
        max_seqlen_k=max_seqlen_k,
        scale=HEAD_DIM**-0.5,
        enable_gqa=True,
        num_splits=-1,
        prefix_num_splits=16,
    )

    grouped = slice(0, 8)
    torch.testing.assert_close(cascade[grouped], plain[grouped], atol=2e-2, rtol=2e-2)
    # Unshared requests take the suffix call alone, which is the plain decode.
    assert torch.equal(cascade[8:10], plain[8:10])
