# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.experiments.rl.loss import CISPOLoss

from .conftest import assert_close, make_normalization, make_sample_ids


class TestCISPOLoss:
    def test_forward(self, inputs):
        d = inputs
        loss_fn = CISPOLoss.Config(clip_low=1.0, clip_high=4.0).build()
        output = loss_fn(
            logits=d["logits"],
            target_ids=d["target_ids"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
            advantages=d["advantages"],
            normalization=d["normalization"],
            sample_ids=d["sample_ids"],
        )

        assert_close(output.loss, torch.tensor(-0.083283))

    def test_backward(self, inputs):
        d = inputs
        logits = d["logits"].clone().requires_grad_(True)

        loss_fn = CISPOLoss.Config(clip_low=1.0, clip_high=4.0).build()
        output = loss_fn(
            logits=logits,
            target_ids=d["target_ids"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
            advantages=d["advantages"],
            normalization=d["normalization"],
            sample_ids=d["sample_ids"],
        )

        output.loss.backward()
        assert_close(logits.grad.norm(), torch.tensor(0.492673))

    def test_zero_advantages(self, inputs):
        d = inputs
        advantages = torch.zeros_like(d["advantages"])

        loss_fn = CISPOLoss.Config().build()
        output = loss_fn(
            logits=d["logits"],
            target_ids=d["target_ids"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=d["loss_mask"],
            advantages=advantages,
            normalization=d["normalization"],
            sample_ids=d["sample_ids"],
        )

        assert output.loss.isfinite()

    def test_empty_mask(self, inputs):
        """Loss should be finite (zero) when mask is all zeros (no trainable tokens)."""
        d = inputs
        empty_mask = torch.zeros_like(d["loss_mask"])

        loss_fn = CISPOLoss.Config().build()
        output = loss_fn(
            logits=d["logits"],
            target_ids=d["target_ids"],
            generator_logprobs=d["generator_logprobs"],
            loss_mask=empty_mask,
            advantages=d["advantages"],
            normalization=make_normalization(empty_mask, d["B"], d["S"]),
            sample_ids=d["sample_ids"],
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0

    def test_empty_sequence(self):
        """Loss should be zero when sequence length is 0."""
        B, V = 2, 10
        logits = torch.empty(B, 0, V)
        target_ids = torch.empty(B, 0, dtype=torch.long)
        advantages = torch.empty(B, 0)
        generator_logprobs = torch.empty(B, 0)
        loss_mask = torch.empty(B, 0)

        loss_fn = CISPOLoss.Config().build()
        output = loss_fn(
            logits=logits,
            target_ids=target_ids,
            generator_logprobs=generator_logprobs,
            loss_mask=loss_mask,
            advantages=advantages,
            normalization=make_normalization(loss_mask, B, 0),
            sample_ids=make_sample_ids(B, 0),
        )

        assert output.loss.isfinite()
        assert output.loss == 0.0
