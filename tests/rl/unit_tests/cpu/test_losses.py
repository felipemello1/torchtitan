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
        is_policy_age_0=torch.ones(64, dtype=torch.bool),
    )

    assert metrics["bit_wise/logprob_diff/max"] == 0


def test_dapo_loss_sums_the_logprob_gap_over_policy_age_0_tokens() -> None:
    # Uniform logits over 4 tokens: every trainer logprob is log(1/4).
    trainer_logprob = torch.tensor(0.25).log()
    generator_logprobs = trainer_logprob - torch.tensor([0.1, 0.2, 0.4, 0.8])
    #                                        policy age: 0    0    1    0 (no loss)

    _, metrics = DAPOLoss.Config().build()(
        torch.zeros(4, 4),
        torch.zeros(4, dtype=torch.long),
        torch.tensor(3),
        generator_logprobs=generator_logprobs,
        temperature=torch.ones(4),
        advantages=torch.ones(4),
        loss_mask=torch.tensor([True, True, True, False]),
        is_policy_age_0=torch.tensor([True, True, False, True]),
    )

    # Shares of the 3 loss tokens; the trainer divides the first by the second.
    torch.testing.assert_close(
        metrics["bit_wise/logprob_diff_age_0/mean"], torch.tensor((0.1 + 0.2) / 3)
    )
    torch.testing.assert_close(
        metrics["bit_wise/age_0_tokens/frac"], torch.tensor(2 / 3)
    )
