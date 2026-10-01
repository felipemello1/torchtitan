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
from torchtitan.rl.types import TrainingMicrobatch


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
    # Same loss, metrics and gradients as the unchunked loss, which also gets the RL loss kwargs.
    torch.manual_seed(42)
    num_tokens, dim, vocab = 1024, 16, 64
    hidden = torch.randn(num_tokens, dim)
    loss_mask = torch.rand(num_tokens) < 0.2
    generator_logprobs = -5 * torch.rand(num_tokens)
    generator_logprobs[:8] = float("-inf")
    microbatch = TrainingMicrobatch(
        input=torch.zeros(num_tokens, dtype=torch.long),
        labels=torch.randint(0, vocab, (num_tokens,)),
        positions=torch.arange(num_tokens),
        padding_mask=torch.zeros(num_tokens, dtype=torch.bool),
        num_valid_tokens=int(loss_mask.sum()),
        generator_logprobs=generator_logprobs,
        loss_mask=loss_mask,
        advantages=torch.randn(num_tokens) * loss_mask,
    )
    labels, loss_kwargs = microbatch.labels, microbatch.loss_kwargs()
    global_valid_tokens = loss_mask.sum().float()
    lm_head = nn.Linear(dim, vocab, bias=False)
    chunked_loss = ChunkedLossWrapper.Config(num_chunks=4, loss_fn=loss_config).build()
    chunked_loss.set_lm_head(lm_head)

    hidden_ref = hidden.clone().requires_grad_()
    loss_ref, metrics_ref = chunked_loss.loss_fn(
        lm_head(hidden_ref), labels, global_valid_tokens, **loss_kwargs
    )
    loss_ref.backward()
    grad_weight_ref, lm_head.weight.grad = lm_head.weight.grad, None

    lm_head_rows = []
    lm_head.register_forward_hook(
        lambda module, args, output: lm_head_rows.append(args[0].shape[0])
    )
    hidden_input = hidden.clone().requires_grad_()
    loss, metrics = chunked_loss(
        hidden_input, labels, global_valid_tokens, **loss_kwargs
    )
    loss.backward()

    assert sum(lm_head_rows) == int(loss_mask.sum())
    torch.testing.assert_close(loss, loss_ref)
    torch.testing.assert_close(metrics, metrics_ref)
    torch.testing.assert_close(hidden_input.grad, hidden_ref.grad)
    torch.testing.assert_close(lm_head.weight.grad, grad_weight_ref)
