# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.rl.losses.dapo import _normalize, DAPOLoss


def test_loss_normalization_uses_mutable_tensor_denominator() -> None:
    value = torch.tensor(1.2345679, dtype=torch.float32)
    global_valid_tokens = torch.tensor(7, dtype=torch.int64)

    normalized = _normalize(value, global_valid_tokens)

    assert torch.equal(
        normalized,
        value * global_valid_tokens.clamp_min(1).reciprocal(),
    )


def test_dapo_logs_logprob_gap_metrics() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 8)
    labels = torch.tensor([1, 2, 3, 4])
    trainer_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None])[
        :, 0
    ]
    # log p_trainer - log q_generator per token; the last token is not trained.
    logprob_diffs = torch.tensor([0.1, -0.2, 1.0, 5.0])
    loss_fn = DAPOLoss.Config().build()

    _, metrics = loss_fn(
        logits,
        labels,
        torch.tensor(3),
        generator_logprobs=trainer_logprobs - logprob_diffs,
        advantages=torch.zeros(4),
        loss_mask=torch.tensor([True, True, True, False]),
    )

    trained = logprob_diffs[:3]
    torch.testing.assert_close(
        metrics["bit_wise/logprob_diff_abs/mean"], trained.abs().mean()
    )
    torch.testing.assert_close(
        metrics["bit_wise/kl_k3/mean"], (trained.exp() - 1 - trained).mean()
    )
    # Only the 1.0 diff has p/q above 2.
    torch.testing.assert_close(
        metrics["bit_wise/ratio_beyond_2x/mean"], torch.tensor(1 / 3)
    )
