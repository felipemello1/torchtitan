# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Golden-value tests for the shared loss primitives."""

import pytest
import torch

from torchtitan.experiments.rl.loss.ops import (
    compute_entropy,
    compute_kl,
    compute_logprobs,
    compute_token_ratio,
    masked_token_mean,
)

from .conftest import assert_close, num_valid_tokens


class TestMaskedTokenMean:
    def test_basic(self, inputs):
        d = inputs
        result = masked_token_mean(
            d["advantages"], d["loss_mask"], num_valid_tokens(d["loss_mask"])
        )
        assert_close(result, torch.tensor(-0.348463))

    def test_zero_mask(self, inputs):
        d = inputs
        result = masked_token_mean(
            d["advantages"],
            torch.zeros_like(d["loss_mask"]),
            num_valid_tokens(d["loss_mask"]),
        )
        assert_close(result, torch.tensor(0.0))

    def test_larger_global_denominator(self, inputs):
        """A global token count of 8 (vs the local 4) halves the per-rank share."""
        d = inputs
        result = masked_token_mean(d["advantages"], d["loss_mask"], 8)
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
        nvt = num_valid_tokens(d["loss_mask"])
        entropy, metrics = compute_entropy(d["logits"], d["loss_mask"], nvt)
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
        nvt = num_valid_tokens(d["loss_mask"])
        logits = d["logits"].clone().requires_grad_(True)
        entropy, _ = compute_entropy(logits, d["loss_mask"], nvt)
        loss = masked_token_mean(entropy, d["loss_mask"], nvt)
        loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.164508))


class TestComputeTokenRatio:
    def test_forward(self, inputs):
        d = inputs
        nvt = num_valid_tokens(d["loss_mask"])
        ratio, log_ratio, _ = compute_token_ratio(
            d["policy_logprobs"], d["generator_logprobs"], d["loss_mask"], nvt
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
        nvt = num_valid_tokens(d["loss_mask"])
        logprobs = d["policy_logprobs"].clone().requires_grad_(True)
        ratio, _, _ = compute_token_ratio(
            logprobs, d["generator_logprobs"], d["loss_mask"], nvt
        )
        loss = masked_token_mean(ratio, d["loss_mask"], nvt)
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(2.925761))


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
        nvt = num_valid_tokens(d["loss_mask"])
        logprobs = d["policy_logprobs"].clone().requires_grad_(True)
        kl, metrics = compute_kl(
            logprobs, d["ref_logprobs"], d["loss_mask"], nvt, kl_type=kl_type
        )

        assert_close(kl, expected_kl)
        assert_close(metrics["loss/kl_ref/mean"].value, torch.tensor(expected_mean))

        loss = masked_token_mean(kl, d["loss_mask"], nvt)
        loss.backward()
        assert_close(logprobs.grad.norm(), torch.tensor(expected_grad_norm))
