# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

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


def _prime_ipo_loss_sum(
    trainer_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    eps: float,
    max_importance_ratio: float,
) -> torch.Tensor:
    """Prime-RL's IPOLoss.loss (trainer/rl/loss.py) on already-masked tokens."""
    log_importance_ratio = trainer_logprobs - inference_logprobs
    larger = torch.maximum(trainer_logprobs, inference_logprobs)
    smaller = torch.minimum(trainer_logprobs, inference_logprobs)
    abs_probs_diff = torch.exp(larger) * -torch.expm1(smaller - larger)
    keep = ~(abs_probs_diff > eps)
    kept = log_importance_ratio[keep]
    capped = kept.detach().clamp(max=kept.new_tensor(max_importance_ratio).log())
    importance_ratio = torch.exp(capped + (kept - kept.detach()))
    return (-advantages[keep] * importance_ratio).sum()


def test_ipo_loss_matches_prime_rl_loss_and_gradient() -> None:
    from torchtitan.rl.losses import IPOLoss

    torch.manual_seed(0)
    num_tokens, vocab = 256, 64
    logits = (torch.randn(num_tokens, vocab) * 4).requires_grad_(True)
    labels = torch.randint(0, vocab, (num_tokens,))
    # The first 64 labels are each row's likeliest token, so moving q_y there crosses eps.
    labels[:64] = logits[:64].argmax(-1)
    with torch.no_grad():
        exact = torch.log_softmax(logits, -1).gather(-1, labels[:, None])[:, 0]
    generator_logprobs = exact + torch.randn(num_tokens) * 0.5
    # Rare under the generator: dropped where p_y is large, a capped huge ratio elsewhere.
    generator_logprobs[:8] = -30.0
    generator_logprobs[64:72] = -30.0
    advantages = torch.randn(num_tokens)
    loss_mask = torch.rand(num_tokens) > 0.2
    num_loss_tokens = int(loss_mask.sum())

    loss, metrics = IPOLoss.Config(eps=0.3, max_importance_ratio=1e4).build()(
        logits,
        labels,
        torch.tensor(num_loss_tokens),
        generator_logprobs=generator_logprobs,
        temperature=torch.ones(num_tokens),
        advantages=advantages,
        loss_mask=loss_mask,
    )
    (grad,) = torch.autograd.grad(loss, logits)

    reference_logits = logits.detach().clone().requires_grad_(True)
    trainer_logprobs = torch.log_softmax(reference_logits, -1).gather(
        -1, labels[:, None]
    )[:, 0]
    reference_loss = (
        _prime_ipo_loss_sum(
            trainer_logprobs[loss_mask],
            generator_logprobs[loss_mask],
            advantages[loss_mask],
            eps=0.3,
            max_importance_ratio=1e4,
        )
        / num_loss_tokens
    )
    (reference_grad,) = torch.autograd.grad(reference_loss, reference_logits)

    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(grad, reference_grad)
    assert 0 < metrics["loss/ipo_dropped_frac"] < 1


def test_ipo_loss_drops_tokens_outside_the_probability_trust_region() -> None:
    from torchtitan.rl.losses import IPOLoss

    # p_y = [0.5, 0.6, 0.2] under the trainer; q_y = [0.9, 0.5, 0.01] under the generator.
    trainer_probs = torch.tensor([0.5, 0.6, 0.2])
    logits = torch.stack(
        [torch.stack([p.log(), (1 - p).log()]) for p in trainer_probs]
    ).requires_grad_(True)
    labels = torch.zeros(3, dtype=torch.long)

    loss, metrics = IPOLoss.Config(eps=0.3).build()(
        logits,
        labels,
        torch.tensor(3),
        generator_logprobs=torch.tensor([0.9, 0.5, 0.01]).log(),
        temperature=torch.ones(3),
        advantages=torch.ones(3),
        loss_mask=torch.ones(3, dtype=torch.bool),
    )

    # Token 0 (|0.5 - 0.9| = 0.4 > 0.3) is dropped but stays in the denominator.
    torch.testing.assert_close(loss, -(0.6 / 0.5 + 0.2 / 0.01) / 3 * torch.ones(()))
    torch.testing.assert_close(metrics["loss/ipo_dropped_frac"], torch.tensor(1 / 3))
