# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.experiments.rl.losses`: DAPO, GRPO (Dr. GRPO), DPPO and
the primitives in `ops.py`. All inputs are flattened ``[T]`` / ``[T, V]`` like the
trainer's."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from torchtitan.components.loss import ChunkedLossWrapper, compute_logprobs
from torchtitan.experiments.rl.losses import DAPOLoss, DPPOLoss, GRPOLoss
from torchtitan.experiments.rl.losses.ops import (
    compute_ratio,
    pg_ppo_clip,
    pg_truncated_reinforce,
    trust_region_mask,
    valid_response_mask,
)

_NUM_TOKENS, _VOCAB = 12, 16
_MAX_RESPONSE_TOKENS = 5


@pytest.fixture
def batch() -> dict[str, torch.Tensor | int]:
    """One flattened microbatch: 4 prompt tokens, 8 response tokens (one with a NaN
    generator logprob), mixed-sign advantages, a generator that disagrees with the
    trainer by up to 1.5 nats so both clip bounds trigger."""
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(_NUM_TOKENS, _VOCAB, generator=generator)
    labels = torch.randint(0, _VOCAB, (_NUM_TOKENS,), generator=generator)
    loss_mask = torch.tensor([False] * 4 + [True] * 8)
    with torch.no_grad():
        trainer_logprobs = compute_logprobs(logits, labels)
    offsets = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 1.5, -1.5, 0.3, -0.3, 0.0, 1.0, -1.0, 0.1]
    )
    generator_logprobs = trainer_logprobs - offsets
    generator_logprobs[8] = float("nan")
    advantages = torch.tensor([0.0] * 4 + [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 0.0])
    return {
        "logits": logits,
        "labels": labels,
        "loss_mask": loss_mask,
        "generator_logprobs": generator_logprobs,
        "advantages": advantages,
        "global_valid_tokens": 7,  # 8 response tokens minus the NaN one
        "num_global_training_samples": 2,
    }


def _call(loss_fn, batch, logits=None):
    return loss_fn(
        batch["logits"] if logits is None else logits,
        batch["labels"],
        batch["global_valid_tokens"],
        generator_logprobs=batch["generator_logprobs"],
        advantages=batch["advantages"],
        loss_mask=batch["loss_mask"],
        num_global_training_samples=batch["num_global_training_samples"],
    )


def _reference_clipped_surrogate(batch, clip_low, clip_high, denominator):
    """The pre-refactor DAPO/GRPO math, op for op, so the refactor must be bit-exact."""
    trainer_logprobs = compute_logprobs(batch["logits"], batch["labels"])
    generator_logprobs, advantages = batch["generator_logprobs"], batch["advantages"]
    effective_loss_mask = batch["loss_mask"] & torch.isfinite(generator_logprobs)
    raw_log_ratio = trainer_logprobs - generator_logprobs
    masked_log_ratio = torch.where(
        effective_loss_mask, raw_log_ratio, torch.zeros_like(raw_log_ratio)
    )
    ratio = torch.exp(torch.clamp(masked_log_ratio, -10.0, 10.0))
    clipped_ratio = torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    token_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    return (token_loss * effective_loss_mask).sum() / max(denominator, 1)


def _two_way_logits(target_probabilities: list[float]) -> torch.Tensor:
    """``[T, 2]`` logits whose label-0 softmax equals each requested probability."""
    probabilities = torch.tensor(target_probabilities)
    return torch.stack((probabilities.log(), (1 - probabilities).log()), dim=-1)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class TestOps:
    def test_valid_response_mask_drops_nan_generator_logprobs(self, batch):
        mask = valid_response_mask(batch["loss_mask"], batch["generator_logprobs"])
        assert mask.tolist() == [False] * 4 + [True] * 4 + [False] + [True] * 3

    def test_compute_ratio_masked_tokens_are_one(self):
        trainer = torch.tensor([-1.0, -1.0, -3.0])
        generator = torch.tensor([-1.0, -2.0, -1.0])
        mask = torch.tensor([True, True, False])
        ratio, raw_log_ratio = compute_ratio(trainer, generator, mask)
        torch.testing.assert_close(ratio, torch.tensor([1.0, math.e, 1.0]))
        torch.testing.assert_close(raw_log_ratio, torch.tensor([0.0, 1.0, -2.0]))

    def test_compute_ratio_clamps_log_ratio(self):
        trainer, generator = torch.tensor([0.0]), torch.tensor([-50.0])
        ratio, _ = compute_ratio(trainer, generator, torch.tensor([True]))
        assert ratio.item() == pytest.approx(math.exp(10.0))

    def test_pg_ppo_clip_example(self):
        ratio = torch.tensor([1.5, 0.5, 1.0])
        advantages = torch.tensor([1.0, -1.0, 1.0])
        token_loss, metrics = pg_ppo_clip(
            ratio,
            advantages,
            torch.ones(3, dtype=torch.bool),
            clip_low=0.2,
            clip_high=0.2,
        )
        torch.testing.assert_close(token_loss, torch.tensor([-1.2, 0.8, -1.0]))
        assert metrics["loss/ratio_clipped_frac"].item() == 2

    def test_pg_ppo_clip_higher_only_widens_positive_side(self):
        ratio = torch.tensor([1.25, 0.75])
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones(2, dtype=torch.bool)
        symmetric, _ = pg_ppo_clip(ratio, advantages, mask, clip_low=0.2, clip_high=0.2)
        higher, _ = pg_ppo_clip(ratio, advantages, mask, clip_low=0.2, clip_high=0.28)
        torch.testing.assert_close(symmetric, torch.tensor([-1.2, 0.8]))
        torch.testing.assert_close(higher, torch.tensor([-1.25, 0.8]))

    def test_trust_region_tv_example(self):
        trainer = torch.log(torch.tensor([0.50, 0.05, 0.10, 0.30]))
        generator = torch.log(torch.tensor([0.20, 0.01, 0.40, 0.01]))
        advantages = torch.tensor([1.0, 1.0, -1.0, 0.0])
        keep = trust_region_mask(
            trainer,
            generator,
            advantages,
            divergence_type="binary_tv",
            divergence_threshold=0.2,
        )
        # +0.30 dropped; 5x ratio but only +0.04 kept; -0.30 dropped; A == 0 always kept.
        assert keep.tolist() == [False, True, False, True]

    def test_trust_region_ignores_moves_against_the_advantage(self):
        trainer = torch.log(torch.tensor([0.10, 0.60]))
        generator = torch.log(torch.tensor([0.60, 0.10]))
        advantages = torch.tensor([1.0, -1.0])
        for divergence_type, threshold in (("binary_tv", 0.2), ("binary_kl", 0.05)):
            keep = trust_region_mask(
                trainer,
                generator,
                advantages,
                divergence_type=divergence_type,
                divergence_threshold=threshold,
            )
            assert keep.tolist() == [True, True], divergence_type

    def test_trust_region_keeps_move_at_exact_threshold(self):
        """The comparison is strict, so a move equal to delta stays trainable."""
        trainer = torch.log(torch.tensor([0.7]))
        generator = torch.log(torch.tensor([0.5]))
        threshold = (trainer.exp() - generator.exp()).item()
        keep = trust_region_mask(
            trainer,
            generator,
            torch.tensor([1.0]),
            divergence_type="binary_tv",
            divergence_threshold=threshold,
        )
        assert keep.item()

    def test_trust_region_kl_matches_closed_form_and_is_finite_at_extremes(self):
        # Bernoulli KL(q || p) for q = 0.5, p = 0.9:  0.5 ln(0.5/0.9) + 0.5 ln(0.5/0.1) = 0.5108
        trainer = torch.log(torch.tensor([0.9, 0.51, 1.0, 1e-30]))
        generator = torch.log(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        advantages = torch.tensor([1.0, 1.0, 1.0, -1.0])
        keep = trust_region_mask(
            trainer,
            generator,
            advantages,
            divergence_type="binary_kl",
            divergence_threshold=0.05,
        )
        # 0.5108 > 0.05 dropped; KL(0.5||0.51) = 2e-4 kept; p = 1.0 exactly stays finite
        # (clamped to 1 - eps) and is dropped; p -> 0 with A < 0 is a huge KL, dropped.
        assert keep.tolist() == [False, True, False, False]

    def test_pg_truncated_reinforce_gradient_matches_ratio_times_advantage(self):
        """d/dθ[-A * sg(r) * log p] == d/dθ[-A * r] when r <= max_ratio."""
        trainer = torch.tensor([-1.0, -0.5, -2.0], requires_grad=True)
        generator = torch.tensor([-1.2, -0.4, -1.5])
        advantages = torch.tensor([1.0, -1.0, 0.5])
        mask = torch.ones(3, dtype=torch.bool)
        ratio, _ = compute_ratio(trainer, generator, mask)
        token_loss, _ = pg_truncated_reinforce(
            ratio, trainer, advantages, mask, mask, max_ratio=100.0
        )
        (reinforce_grad,) = torch.autograd.grad(token_loss.sum(), trainer)
        ratio, _ = compute_ratio(trainer, generator, mask)
        (ppo_grad,) = torch.autograd.grad((-advantages * ratio).sum(), trainer)
        torch.testing.assert_close(reinforce_grad, ppo_grad)

    def test_pg_truncated_reinforce_truncates_and_masks(self):
        trainer = torch.tensor([0.0, -1.0, -1.0], requires_grad=True)
        generator = torch.tensor([-3.0, -1.0, -1.0])  # ratio e^3 ~ 20 on token 0
        advantages = torch.tensor([1.0, 1.0, 1.0])
        keep_mask = torch.tensor([True, True, False])
        mask = torch.ones(3, dtype=torch.bool)
        ratio, _ = compute_ratio(trainer, generator, mask)
        token_loss, metrics = pg_truncated_reinforce(
            ratio, trainer, advantages, keep_mask, mask, max_ratio=5.0
        )
        (grad,) = torch.autograd.grad(token_loss.sum(), trainer)
        # Truncated weight 5 (not 20), unit weight on token 1, no gradient on the dropped token.
        torch.testing.assert_close(grad, torch.tensor([-5.0, -1.0, 0.0]))
        assert metrics["loss/ratio_truncated_frac"].item() == 1
        assert metrics["loss/trust_region_dropped_frac"].item() == 1


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class TestGRPOLoss:
    def test_fixed_normalizer_denominator(self, batch):
        loss_fn = GRPOLoss.Config(max_response_tokens=_MAX_RESPONSE_TOKENS).build()
        loss, _ = _call(loss_fn, batch)
        expected = _reference_clipped_surrogate(
            batch, 0.2, 0.2, denominator=2 * _MAX_RESPONSE_TOKENS
        )
        assert torch.equal(loss, expected)

    def test_denominator_ignores_valid_token_count(self, batch):
        """Dr. GRPO: dropping a response token changes the sum, never the denominator."""
        loss_fn = GRPOLoss.Config(max_response_tokens=_MAX_RESPONSE_TOKENS).build()
        full_loss, _ = _call(loss_fn, batch)
        batch["loss_mask"][11] = False  # an A == 0 token: no loss contribution
        batch["global_valid_tokens"] -= 1
        shorter_loss, _ = _call(loss_fn, batch)
        assert torch.equal(full_loss, shorter_loss)

    def test_matches_old_grpo_when_normalizer_equals_valid_tokens(self, batch):
        """The only change from the previous GRPOLoss is the denominator."""
        batch["num_global_training_samples"] = 1
        loss_fn = GRPOLoss.Config(
            max_response_tokens=batch["global_valid_tokens"]
        ).build()
        loss, _ = _call(loss_fn, batch)
        expected = _reference_clipped_surrogate(
            batch, 0.2, 0.2, denominator=batch["global_valid_tokens"]
        )
        assert torch.equal(loss, expected)

    def test_metrics_are_normalized_by_global_valid_tokens(self, batch):
        loss_fn = GRPOLoss.Config(max_response_tokens=_MAX_RESPONSE_TOKENS).build()
        loss, metrics = _call(loss_fn, batch)
        assert metrics["loss/mean"] == loss.detach()
        assert 0.0 < metrics["loss/ratio_clipped_frac"].item() <= 1.0
        assert metrics["bit_wise/logprob_diff/max"].item() == pytest.approx(1.5)
        assert metrics["trainer/entropy/mean"].item() > 0.0
        assert set(metrics) == {
            "loss/mean",
            "loss/ratio_mean",
            "loss/ratio_clipped_frac",
            "bit_wise/logprob_diff/mean",
            "bit_wise/ratio_tokens_different/mean",
            "bit_wise/logprob_diff/max",
            "trainer/entropy/mean",
        }

    def test_nan_generator_logprob_token_gets_no_gradient(self, batch):
        logits = batch["logits"].clone().requires_grad_(True)
        loss_fn = GRPOLoss.Config(max_response_tokens=_MAX_RESPONSE_TOKENS).build()
        loss, _ = _call(loss_fn, batch, logits=logits)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.all(logits.grad[8] == 0)
        assert torch.all(logits.grad[:4] == 0)

    def test_all_masked_batch_is_finite_zero(self, batch):
        batch["loss_mask"][:] = False
        loss_fn = GRPOLoss.Config(max_response_tokens=_MAX_RESPONSE_TOKENS).build()
        loss, metrics = _call(loss_fn, batch)
        assert loss.item() == 0.0
        assert all(torch.isfinite(value) for value in metrics.values())


class TestDAPOLoss:
    def test_bit_exact_with_previous_dapo_math(self, batch):
        loss_fn = DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28).build()
        loss, _ = _call(loss_fn, batch)
        expected = _reference_clipped_surrogate(
            batch, 0.2, 0.28, denominator=batch["global_valid_tokens"]
        )
        assert torch.equal(loss, expected)

    def test_directional_clipping(self):
        """A > 0 clips at 1 + high (0.65/0.5 = 1.3 > 1.28); A < 0 at 1 - low (0.7 < 0.8)."""
        loss_fn = DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28).build()
        loss, metrics = loss_fn(
            _two_way_logits([0.65, 0.35]),
            torch.zeros(2, dtype=torch.long),
            2,
            generator_logprobs=torch.full((2,), math.log(0.5)),
            advantages=torch.tensor([1.0, -1.0]),
            loss_mask=torch.ones(2, dtype=torch.bool),
            num_global_training_samples=1,
        )
        # (-1.28 * 1 + 0.8 * 1) / 2
        torch.testing.assert_close(loss, torch.tensor(-0.24))
        assert metrics["loss/ratio_clipped_frac"].item() == pytest.approx(1.0)


class TestDPPOLoss:
    def test_loss_is_finite_and_masked_tokens_get_no_gradient(self, batch):
        logits = batch["logits"].clone().requires_grad_(True)
        loss, metrics = _call(DPPOLoss.Config().build(), batch, logits=logits)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.all(logits.grad[:4] == 0)
        assert torch.all(logits.grad[8] == 0)
        assert {"loss/trust_region_dropped_frac", "loss/ratio_truncated_frac"} <= set(
            metrics
        )

    def test_tv_keeps_rare_token_jump_and_drops_confident_move(self):
        """The paper's motivating case: a rare token whose probability went 5x (0.01 ->
        0.05) is kept, a confident token that moved +0.3 (0.2 -> 0.5) is dropped."""
        loss_fn = DPPOLoss.Config(divergence_type="binary_tv").build()
        _, metrics = loss_fn(
            _two_way_logits([0.05, 0.5]),
            torch.zeros(2, dtype=torch.long),
            2,
            generator_logprobs=torch.log(torch.tensor([0.01, 0.2])),
            advantages=torch.tensor([1.0, 1.0]),
            loss_mask=torch.ones(2, dtype=torch.bool),
            num_global_training_samples=1,
        )
        assert metrics["loss/trust_region_dropped_frac"].item() == pytest.approx(0.5)

    def test_kl_default_threshold_drops_large_binary_kl(self):
        """KL(0.5 || 0.9) = 0.51 > 0.05 dropped; KL(0.5 || 0.51) = 2e-4 kept."""
        loss_fn = DPPOLoss.Config(divergence_type="binary_kl").build()
        assert loss_fn.divergence_threshold == 0.05
        _, metrics = loss_fn(
            _two_way_logits([0.9, 0.51]),
            torch.zeros(2, dtype=torch.long),
            2,
            generator_logprobs=torch.full((2,), math.log(0.5)),
            advantages=torch.tensor([1.0, 1.0]),
            loss_mask=torch.ones(2, dtype=torch.bool),
            num_global_training_samples=1,
        )
        assert metrics["loss/trust_region_dropped_frac"].item() == pytest.approx(0.5)

    def test_gradient_equals_ppo_unclipped_when_inside_trust_region(self, batch):
        """With a wide trust region and no truncation, DPPO's gradient equals the
        unclipped ratio * A policy gradient (Eq. 23 with M = 1, C = inf)."""
        logits = batch["logits"].clone().requires_grad_(True)
        loss_fn = DPPOLoss.Config(divergence_threshold=1.0, max_ratio=1e6).build()
        loss, _ = _call(loss_fn, batch, logits=logits)
        (dppo_grad,) = torch.autograd.grad(loss, logits)

        logits = batch["logits"].clone().requires_grad_(True)
        trainer_logprobs = compute_logprobs(logits, batch["labels"])
        mask = valid_response_mask(batch["loss_mask"], batch["generator_logprobs"])
        ratio, _ = compute_ratio(trainer_logprobs, batch["generator_logprobs"], mask)
        ppo_loss = (-batch["advantages"] * ratio * mask).sum() / batch[
            "global_valid_tokens"
        ]
        (ppo_grad,) = torch.autograd.grad(ppo_loss, logits)
        torch.testing.assert_close(dppo_grad, ppo_grad)


# ---------------------------------------------------------------------------
# ChunkedLossWrapper integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loss_config",
    [
        GRPOLoss.Config(max_response_tokens=_MAX_RESPONSE_TOKENS),
        DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
        DPPOLoss.Config(),
        DPPOLoss.Config(divergence_type="binary_kl"),
    ],
    ids=["grpo", "dapo", "dppo_tv", "dppo_kl"],
)
def test_chunked_wrapper_matches_unchunked(batch, loss_config):
    """Loss, decoder-hidden gradient, lm_head gradient and every metric agree between
    ``ChunkedLossWrapper(num_chunks=3)`` and one unchunked call."""
    hidden_dim = 8
    torch.manual_seed(0)
    lm_head = nn.Linear(hidden_dim, _VOCAB, bias=False)
    hidden = torch.randn(_NUM_TOKENS, hidden_dim)
    loss_inputs = {
        key: batch[key]
        for key in (
            "generator_logprobs",
            "advantages",
            "loss_mask",
            "num_global_training_samples",
        )
    }

    unchunked_hidden = hidden.clone().requires_grad_(True)
    unchunked_loss, unchunked_metrics = loss_config.build()(
        lm_head(unchunked_hidden),
        batch["labels"],
        batch["global_valid_tokens"],
        **loss_inputs,
    )
    unchunked_loss.backward()
    unchunked_head_grad = lm_head.weight.grad.clone()
    lm_head.weight.grad = None

    chunked = ChunkedLossWrapper.Config(num_chunks=3, loss_fn=loss_config).build()
    chunked.set_lm_head(lm_head)
    chunked_hidden = hidden.clone().requires_grad_(True)
    chunked_loss, chunked_metrics = chunked(
        chunked_hidden, batch["labels"], batch["global_valid_tokens"], **loss_inputs
    )
    chunked_loss.backward()

    torch.testing.assert_close(chunked_loss, unchunked_loss)
    torch.testing.assert_close(chunked_hidden.grad, unchunked_hidden.grad)
    torch.testing.assert_close(lm_head.weight.grad, unchunked_head_grad)
    assert set(chunked_metrics) == set(unchunked_metrics)
    for key, value in unchunked_metrics.items():
        torch.testing.assert_close(chunked_metrics[key], value, msg=key)
