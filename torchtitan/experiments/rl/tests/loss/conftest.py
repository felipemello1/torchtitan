# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def assert_close(actual, expected, atol=1e-4, rtol=1e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def policy_logprobs_from(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Per-token logprobs via negative cross-entropy, matching the loss's
    `compute_logprobs` (temperature 1). Differentiable in `logits`."""
    B, S, V = logits.shape
    return -F.cross_entropy(
        logits.float().reshape(B * S, V),
        target_ids.reshape(B * S).long(),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).reshape(B, S)


def num_valid_tokens(loss_mask: torch.Tensor) -> int:
    """Global response-token count; in the single-batch test setup the global
    count equals the local valid-token count."""
    return int(loss_mask.sum().item())


@pytest.fixture
def inputs():
    """Fixed loss fixture; RNG draw order (logits, target_ids, ref, advantages)
    is load-bearing -- it produces the golden loss/grad values the per-loss
    tests assert."""
    torch.manual_seed(42)
    B, S, V = 2, 4, 10

    logits = torch.randn(B, S, V)
    target_ids = torch.randint(0, V, (B, S))

    # Seq 0: mild divergence, Seq 1: high divergence (triggers clipping)
    generator_logprobs = torch.tensor(
        [
            [-2.0, -2.1, -1.9, -2.0],
            [-6.0, -1.0, -5.0, -0.5],
        ]
    )
    ref_logprobs = torch.randn(B, S) * 0.5 - 2.0
    advantages = torch.randn(B, S)

    # Interleaved mask (multi-turn pattern)
    loss_mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.float)

    return {
        "B": B,
        "S": S,
        "V": V,
        "logits": logits,
        "target_ids": target_ids,
        "generator_logprobs": generator_logprobs,
        "ref_logprobs": ref_logprobs,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "policy_logprobs": policy_logprobs_from(logits, target_ids),
        "num_global_valid_tokens": num_valid_tokens(loss_mask),
    }
