# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Golden tests for the shared loss primitives, ported from forge's test_ops.py.

The denominators differ in form (forge's local `mask.sum()` / `loss_scale` vs the
titan `LossNormalization` constants) but the values match: the `inputs` fixture
sets num_global_valid_tokens == mask.sum() == 4 and num_global_fixed_horizon ==
B*S == 8, so every golden below is forge's exact constant. The sequence ratio is
the one deliberate divergence: it is sample-aware (per source episode, valid
positions only) instead of forge's whole-row fill.
"""

import pytest
import torch

from torchtitan.experiments.rl.loss.ops import (
    aggregate_loss,
    compute_entropy,
    compute_kl,
    compute_logprobs,
    compute_sequence_ratio,
    compute_token_ratio,
    masked_token_mean,
)
from torchtitan.experiments.rl.loss.types import LossNormalization

from .conftest import assert_close, make_normalization, make_sample_ids


class TestMaskedTokenMean:
    def test_basic(self, inputs):
        d = inputs
        norm = make_normalization(d["loss_mask"], d["B"], d["S"])  # global tokens == 4
        result = masked_token_mean(d["advantages"], d["loss_mask"], norm)
        assert_close(result, torch.tensor(-0.348463))

    def test_zero_mask(self, inputs):
        d = inputs
        norm = make_normalization(d["loss_mask"], d["B"], d["S"])
        result = masked_token_mean(
            d["advantages"], torch.zeros_like(d["loss_mask"]), norm
        )
        assert_close(result, torch.tensor(0.0))

    def test_larger_global_denominator(self, inputs):
        """Global token count of 8 (vs the local 4) halves the per-rank share,
        matching forge's loss_scale=8 case."""
        d = inputs
        norm = LossNormalization(
            num_global_valid_tokens=8,
            num_global_sequences=d["B"],
            num_global_fixed_horizon_tokens=d["B"] * d["S"],
        )
        result = masked_token_mean(d["advantages"], d["loss_mask"], norm)
        assert_close(result, torch.tensor(-0.174231))


class TestComputeLogprobs:
    def test_forward(self, inputs):
        d = inputs
        logprobs = compute_logprobs(d["logits"], d["target_ids"])
        expected = torch.tensor(
            [
                [-2.455715, -3.950112, -2.637205, -3.512223],
                [-3.542688, -2.388949, -3.638923, -4.686581],
            ]
        )
        assert_close(logprobs, expected)

    def test_backward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)
        logprobs = compute_logprobs(logits, d["target_ids"])
        loss = (logprobs * d["loss_mask"]).sum()
        loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(2.077044))


class TestComputeEntropy:
    def test_forward(self, inputs):
        d = inputs
        norm = make_normalization(d["loss_mask"], d["B"], d["S"])
        entropy, metrics = compute_entropy(d["logits"], d["loss_mask"], norm)
        expected = torch.tensor(
            [
                [1.801453, 1.862737, 2.120112, 1.875997],
                [1.429505, 2.056069, 1.953664, 1.997996],
            ]
        )
        assert_close(entropy, expected)
        assert (entropy >= 0).all()
        assert_close(metrics["loss/entropy/mean"].value, torch.tensor(1.851785))

    def test_backward(self, inputs):
        d = inputs
        norm = make_normalization(d["loss_mask"], d["B"], d["S"])
        logits = d["logits"].clone().requires_grad_(True)
        entropy, _ = compute_entropy(logits, d["loss_mask"], norm)
        loss = masked_token_mean(entropy, d["loss_mask"], norm)
        loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.164508))


class TestComputeTokenRatio:
    def test_forward(self, inputs):
        d = inputs
        norm = make_normalization(d["loss_mask"], d["B"], d["S"])
        ratio, log_ratio, _ = compute_token_ratio(
            d["policy_logprobs"], d["generator_logprobs"], d["loss_mask"], norm
        )
        expected_ratio = torch.tensor(
            [
                [0.633994, 0.157220, 0.478449, 0.220419],
                [11.673395, 0.249337, 3.900393, 0.015198],
            ]
        )
        expected_log_ratio = torch.tensor(
            [
                [-0.455715, -1.850112, -0.737205, -1.512223],
                [2.457312, -1.388949, 1.361077, -4.186581],
            ]
        )
        assert_close(ratio, expected_ratio)
        assert_close(log_ratio, expected_log_ratio)

    def test_backward(self, inputs):
        d = inputs
        norm = make_normalization(d["loss_mask"], d["B"], d["S"])
        logprobs = d["policy_logprobs"].clone().requires_grad_(True)
        ratio, _, _ = compute_token_ratio(
            logprobs, d["generator_logprobs"], d["loss_mask"], norm
        )
        loss = masked_token_mean(ratio, d["loss_mask"], norm)
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(2.925761))


class TestComputeSequenceRatio:
    """Titan's sequence ratio is sample-aware: at valid positions it equals
    forge's per-row sequence ratio (one episode per row here), but masked
    positions are filled with 1.0 instead of forge's whole-row broadcast."""

    def test_forward(self, inputs):
        d = inputs
        norm = make_normalization(d["loss_mask"], d["B"], d["S"])
        sample_ids = make_sample_ids(d["B"], d["S"])  # one episode per row
        ratio, _log_ratio, _ = compute_sequence_ratio(
            d["policy_logprobs"],
            d["generator_logprobs"],
            d["loss_mask"],
            sample_ids,
            norm,
        )
        # Valid positions match forge's per-row sequence ratio.
        assert_close(ratio[0, 0], torch.tensor(0.550758))
        assert_close(ratio[0, 2], torch.tensor(0.550758))
        assert_close(ratio[1, 0], torch.tensor(1.706051))
        assert_close(ratio[1, 1], torch.tensor(1.706051))
        # Masked positions are 1.0 (sample-aware fill, not forge's broadcast).
        assert_close(ratio[0, 1], torch.tensor(1.0))
        assert_close(ratio[0, 3], torch.tensor(1.0))
        assert_close(ratio[1, 2], torch.tensor(1.0))
        assert_close(ratio[1, 3], torch.tensor(1.0))

    def test_backward(self, inputs):
        d = inputs
        norm = make_normalization(d["loss_mask"], d["B"], d["S"])
        sample_ids = make_sample_ids(d["B"], d["S"])
        logprobs = d["policy_logprobs"].clone().requires_grad_(True)
        ratio, _, _ = compute_sequence_ratio(
            logprobs, d["generator_logprobs"], d["loss_mask"], sample_ids, norm
        )
        loss = masked_token_mean(ratio, d["loss_mask"], norm)
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(0.633832))


class TestComputeKl:
    @pytest.mark.parametrize(
        "kl_type,expected_kl,expected_mean,expected_grad_norm",
        [
            pytest.param(
                "k1",
                torch.tensor(
                    [
                        [-1.415665, -1.837418, -0.466356, -1.664230],
                        [-1.198181, 0.174410, -1.496045, -2.139825],
                    ]
                ),
                -0.726448,
                0.500000,
                id="k1",
            ),
            pytest.param(
                "k2",
                torch.tensor(
                    [
                        [1.002053, 1.688052, 0.108744, 1.384830],
                        [0.717819, 0.015209, 1.119076, 2.289426],
                    ]
                ),
                0.460956,
                0.480081,
                id="k2",
            ),
            pytest.param(
                "k3",
                torch.tensor(
                    [
                        [1.703559, 3.442883, 0.127818, 2.617373],
                        [1.115902, 0.014362, 1.967954, 5.358127],
                    ]
                ),
                0.740411,
                0.983082,
                id="k3",
            ),
        ],
    )
    def test_kl_types(
        self, inputs, kl_type, expected_kl, expected_mean, expected_grad_norm
    ):
        d = inputs
        norm = make_normalization(d["loss_mask"], d["B"], d["S"])
        logprobs = d["policy_logprobs"].clone().requires_grad_(True)
        kl, metrics = compute_kl(
            logprobs, d["ref_logprobs"], d["loss_mask"], norm, kl_type=kl_type
        )

        assert_close(kl, expected_kl)
        assert_close(metrics["loss/kl_ref/mean"].value, torch.tensor(expected_mean))

        loss = masked_token_mean(kl, d["loss_mask"], norm)
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(expected_grad_norm))


class TestAggregate:
    @pytest.mark.parametrize(
        "agg_type,expected_loss,expected_grad_norm",
        [
            pytest.param(
                "token_mean", torch.tensor(3.258794), 0.500000, id="token_mean"
            ),
            pytest.param(
                "fixed_horizon", torch.tensor(1.629397), 0.250000, id="fixed_horizon"
            ),
            pytest.param(
                "sequence_mean", torch.tensor(3.258794), 0.500000, id="sequence_mean"
            ),
        ],
    )
    def test_agg_types(self, inputs, agg_type, expected_loss, expected_grad_norm):
        d = inputs
        norm = make_normalization(d["loss_mask"], d["B"], d["S"])
        sample_ids = make_sample_ids(d["B"], d["S"])  # one episode per row
        ratio, _, _ = compute_token_ratio(
            d["policy_logprobs"], d["generator_logprobs"], d["loss_mask"], norm
        )
        per_token_loss = ratio.detach().clone().requires_grad_(True)
        loss = aggregate_loss(
            per_token_loss,
            d["loss_mask"],
            agg_type=agg_type,
            normalization=norm,
            sample_ids=sample_ids,
        )

        assert_close(loss, expected_loss)

        loss.backward()
        assert_close(per_token_loss.grad.norm(), torch.tensor(expected_grad_norm))
