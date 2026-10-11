# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.rl.losses import GRPOLoss
from torchtitan.rl.losses.dapo import _normalize, DAPOLoss


def test_loss_normalization_uses_mutable_tensor_denominator() -> None:
    value = torch.tensor(1.2345679, dtype=torch.float32)
    global_loss_token_counts = torch.tensor([7, 3], dtype=torch.int64)

    normalized = _normalize(value, global_loss_token_counts)

    assert torch.equal(
        normalized,
        value * global_loss_token_counts[0].clamp_min(1).reciprocal(),
    )


def test_dapo_loss_matches_vllm_tempered_logprobs_bitwise() -> None:
    torch.manual_seed(0)
    logits = (torch.randn(64, 128) * 3).to(torch.bfloat16)
    labels = torch.randint(0, 128, (64,))
    temperature = torch.full((64,), 0.7)
    # vLLM's sampler: logits.to(float32).div_(temperature), then log_softmax(dtype=float32).
    generator_logprobs = (
        logits.to(torch.float32)
        .div_(temperature[:, None])
        .log_softmax(dim=-1, dtype=torch.float32)
        .gather(-1, labels[:, None])
        .squeeze(-1)
    )

    _, metrics = DAPOLoss.Config().build()(
        logits,
        labels,
        torch.tensor(64),
        generator_logprobs=generator_logprobs,
        temperature=temperature,
        advantages=torch.ones(64),
        loss_mask=torch.ones(64, dtype=torch.bool),
    )

    assert metrics["bit_wise/logprob_diff/max"] == 0


def _inputs_with_ratios(logits: torch.Tensor, ratio: torch.Tensor) -> dict:
    """DAPOLoss inputs whose generator logprobs give each token the requested ratio."""
    num_tokens, vocab_size = logits.shape
    labels = torch.randint(0, vocab_size, (num_tokens,))
    trainer_logprobs = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    return dict(
        labels=labels,
        global_loss_token_counts=torch.tensor(num_tokens),
        generator_logprobs=trainer_logprobs - ratio.log(),
        temperature=torch.ones(num_tokens),
        advantages=torch.where(torch.arange(num_tokens) % 2 == 0, 1.0, -1.0),
        loss_mask=torch.ones(num_tokens, dtype=torch.bool),
    )


def _loss_and_grad(loss_fn, logits: torch.Tensor, inputs: dict):
    logits = logits.detach().clone().requires_grad_(True)
    loss, metrics = loss_fn(logits, **inputs)
    loss.backward()
    return loss.detach(), logits.grad, metrics


@pytest.mark.parametrize("config_cls", [DAPOLoss.Config, GRPOLoss.Config])
def test_ratio_mask_drops_tokens_outside_band(config_cls) -> None:
    torch.manual_seed(0)
    logits = torch.randn(7, 32)
    # Advantages alternate +1, -1. Tokens 0-3 are outside the band, 4-5 inside.
    ratio = torch.tensor([0.1, 100.0, 6.0, 0.4, 0.55, 4.5, 100.0])
    inputs = _inputs_with_ratios(logits, ratio)
    # Token 6 is a prompt token, so it is never counted as masked.
    inputs["loss_mask"][6] = False
    inputs["global_loss_token_counts"] = torch.tensor(6)

    _, grad_off, metrics_off = _loss_and_grad(config_cls().build(), logits, inputs)
    _, grad_on, metrics_on = _loss_and_grad(
        config_cls(ratio_mask=(0.5, 5.0)).build(), logits, inputs
    )

    # The clip trains tokens 0 and 1 (the advantage pushes the ratio back toward 1) and
    # zeroes 2 and 3 (pushed further from 1); the mask zeroes all four.
    assert torch.all(grad_off[:2].norm(dim=-1) > 0)
    assert torch.all(grad_off[2:4] == 0)
    assert torch.all(grad_on[:4] == 0)
    # Masked tokens stay in the denominator, so kept tokens' gradients are unchanged.
    assert torch.equal(grad_on[4:], grad_off[4:])
    assert metrics_on.pop("loss/ratio_masked_frac").item() == pytest.approx(4 / 6)
    # The other metrics still see every loss token.
    assert metrics_on.keys() == metrics_off.keys()
    for key in metrics_off.keys() - {"loss/mean"}:
        assert torch.equal(metrics_on[key], metrics_off[key]), key
