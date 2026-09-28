# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Varlen attention over a shared prompt matches attention over duplicated prompts."""

# Shape suffix legend:
#   T = packed tokens, H = attention heads, K = query/key head dimension,
#   V = value head dimension

import unittest

import torch

from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    VarlenInnerAttention,
)


@unittest.skipUnless(torch.cuda.is_available(), "varlen attention needs CUDA")
class TestVarlenSharedPrefix(unittest.TestCase):
    def test_shared_prompt_matches_duplicated_prompt(self):
        # (-1, 0) is causal; (16, 0) is a sliding window shorter than the prompt.
        for window_size in ((-1, 0), (16, 0)):
            with self.subTest(window_size=window_size):
                self._check_shared_prompt(window_size)

    def _check_shared_prompt(self, window_size: tuple[int, int]) -> None:
        torch.manual_seed(0)
        prompt_len, completion_lens = 37, [21, 64, 5]
        num_heads, num_kv_heads, head_dim = 8, 2, 64

        # Shared: [prompt, completion_0, completion_1, ...]; each completion's
        # positions start after the prompt. Duplicated: [prompt, completion_i]
        # per sample, gathered from the shared rows.
        shared_positions = list(range(prompt_len))
        duplicated_rows, duplicated_positions = [], []
        next_row = prompt_len
        for completion_len in completion_lens:
            completion_rows = list(range(next_row, next_row + completion_len))
            shared_positions += range(prompt_len, prompt_len + completion_len)
            duplicated_rows += list(range(prompt_len)) + completion_rows
            duplicated_positions += range(prompt_len + completion_len)
            next_row += completion_len
        num_tokens = next_row
        duplicated_rows_T = torch.tensor(duplicated_rows, device="cuda")
        completion_mask_T = torch.arange(num_tokens, device="cuda") >= prompt_len

        inner_attention = VarlenInnerAttention.Config(window_size=window_size).build()
        q_THK, k_THK, v_THV = (
            torch.randn(
                num_tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16
            )
            for heads in (num_heads, num_kv_heads, num_kv_heads)
        )
        grad_out_THV = torch.randn(
            num_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        grad_out_THV[~completion_mask_T] = 0

        def run(duplicate: bool):
            q, k, v = (t.clone().requires_grad_() for t in (q_THK, k_THK, v_THV))
            positions = duplicated_positions if duplicate else shared_positions
            metadata = create_varlen_metadata_for_document(
                torch.tensor(positions, device="cuda"), allow_shared_prefixes=True
            )
            if duplicate:
                out = inner_attention(
                    q[duplicated_rows_T],
                    k[duplicated_rows_T],
                    v[duplicated_rows_T],
                    attention_masks=metadata,
                    enable_gqa=True,
                )
                out = torch.zeros_like(q).index_add(0, duplicated_rows_T, out)
            else:
                out = inner_attention(
                    q, k, v, attention_masks=metadata, enable_gqa=True
                )
            out.backward(grad_out_THV)
            return out[completion_mask_T], q.grad, k.grad, v.grad

        # Completion outputs, and gradients summed over every copy of the prompt.
        for shared, duplicated in zip(
            run(duplicate=False), run(duplicate=True), strict=True
        ):
            torch.testing.assert_close(shared, duplicated, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
