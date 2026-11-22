# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan.rl.generator.vllm.attention.attention import (
    TorchTitanVarlenInnerAttentionImpl,
)
from torchtitan.tools.utils import (
    activate_cuda_flash_attention_impl,
    has_cuda_capability,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_cuda_capability(10, 0),
    reason="FA4 decode kernel requires SM100",
)


def test_verify_rows_as_decode_match_causal_attention():
    """Each verify token attends causally to its row's keys; a padded row (seq_len 0) is ignored."""
    activate_cuda_flash_attention_impl()
    torch.manual_seed(0)
    device = torch.device("cuda")
    query_len, num_heads, num_kv_heads, head_dim, page_size = 4, 24, 4, 256, 128
    # Shortest row: 4 keys, one shared by all tokens. Others cross page boundaries.
    seq_lens = [4, 130, 1000, 257, 0]
    num_pages = [-(-length // page_size) for length in seq_lens]
    key_cache = torch.randn(
        sum(num_pages) + 1,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    page_order = torch.randperm(key_cache.shape[0])
    block_table = torch.zeros(len(seq_lens), max(num_pages), dtype=torch.int32)
    start = 0
    for row, count in enumerate(num_pages):
        block_table[row, :count] = page_order[start : start + count]
        start += count
    query = torch.randn(
        len(seq_lens) * query_len,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    metadata = SimpleNamespace(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        max_seq_len=query_len,
        block_table=block_table.to(device),
    )
    impl = TorchTitanVarlenInnerAttentionImpl.__new__(
        TorchTitanVarlenInnerAttentionImpl
    )
    impl.scale = head_dim**-0.5
    output = torch.zeros_like(query)

    impl._verify_rows_as_decode(output, query, key_cache, value_cache, metadata)

    groups = num_heads // num_kv_heads
    for row, num_keys in enumerate(seq_lens[:-1]):
        pages = block_table[row, : num_pages[row]].long()
        keys = key_cache[pages].flatten(0, 1)[:num_keys].float()
        values = value_cache[pages].flatten(0, 1)[:num_keys].float()
        row_query = query[row * query_len : (row + 1) * query_len].float()
        # [heads, tokens, keys]; token j sees keys 0 .. num_keys - query_len + j.
        scores = impl.scale * torch.einsum(
            "thd,khd->htk", row_query, keys.repeat_interleave(groups, dim=1)
        )
        is_visible = torch.ones(
            query_len, num_keys, dtype=torch.bool, device=device
        ).tril(num_keys - query_len)
        scores = scores.masked_fill(~is_visible, float("-inf"))
        expected = torch.einsum(
            "htk,khd->thd",
            scores.softmax(dim=-1),
            values.repeat_interleave(groups, dim=1),
        )
        actual = output[row * query_len : (row + 1) * query_len].float()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
