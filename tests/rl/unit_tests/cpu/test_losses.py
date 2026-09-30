# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.rl.losses import DAPOLoss, GRPOLoss
from torchtitan.rl.losses.dapo import _normalize


def test_loss_normalization_uses_mutable_tensor_denominator() -> None:
    value = torch.tensor(1.2345679, dtype=torch.float32)
    global_valid_tokens = torch.tensor(7, dtype=torch.int64)

    normalized = _normalize(value, global_valid_tokens)

    assert torch.equal(
        normalized,
        value * global_valid_tokens.clamp_min(1).reciprocal(),
    )


@pytest.mark.parametrize(
    "loss_config", [DAPOLoss.Config(ratio_clip_high=0.28), GRPOLoss.Config()]
)
def test_chunked_loss_skips_tokens_outside_loss_mask(loss_config) -> None:
    # Same loss, metrics and gradients as running the lm_head on every token.
    torch.manual_seed(42)
    num_tokens, dim, vocab = 1024, 16, 64
    hidden = torch.randn(num_tokens, dim)
    labels = torch.randint(0, vocab, (num_tokens,))
    loss_mask = torch.rand(num_tokens) < 0.2
    generator_logprobs = -5 * torch.rand(num_tokens)
    generator_logprobs[:8] = float("-inf")
    loss_inputs = {
        "generator_logprobs": generator_logprobs,
        "advantages": torch.randn(num_tokens) * loss_mask,
        "loss_mask": loss_mask,
    }
    global_valid_tokens = loss_mask.sum().float()
    lm_head = nn.Linear(dim, vocab, bias=False)
    lm_head_rows = []
    lm_head.register_forward_hook(
        lambda module, args, output: lm_head_rows.append(args[0].shape[0])
    )

    results = []
    for skip in (False, True):
        chunked_loss = ChunkedLossWrapper.Config(
            num_chunks=4, loss_fn=loss_config
        ).build()
        chunked_loss.set_lm_head(lm_head)
        token_indices = chunked_loss._loss_token_indices(labels, loss_inputs)
        if not skip:
            chunked_loss._loss_token_indices = lambda *args: None
        lm_head.weight.grad = None
        lm_head_rows.clear()
        hidden_input = hidden.clone().requires_grad_()
        loss, metrics = chunked_loss(
            hidden_input, labels, global_valid_tokens, **loss_inputs
        )
        loss.backward()
        assert sum(lm_head_rows) == (token_indices.numel() if skip else num_tokens)
        results.append((loss, metrics, hidden_input.grad, lm_head.weight.grad))

    (loss, metrics, grad_hidden, grad_weight), skipped = results
    torch.testing.assert_close(skipped[0], loss)
    torch.testing.assert_close(skipped[1], metrics)
    torch.testing.assert_close(skipped[2], grad_hidden)
    torch.testing.assert_close(skipped[3], grad_weight)
