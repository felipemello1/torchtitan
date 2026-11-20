# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.components.loss import loss_metric_reduce_op
from torchtitan.rl.trainer.losses.dapo import _normalize, DAPOLoss


def test_loss_normalization_uses_mutable_tensor_denominator() -> None:
    value = torch.tensor(1.2345679, dtype=torch.float32)
    global_loss_token_counts = torch.tensor([7, 3], dtype=torch.int64)

    normalized = _normalize(value, global_loss_token_counts)

    assert torch.equal(
        normalized,
        value * global_loss_token_counts[0].clamp_min(1).reciprocal(),
    )


def test_kl_k3_metric_where_signed_mean_cancels() -> None:
    # Log ratios +0.5 and -0.5: their signed mean is 0, their k3 KL is not.
    torch.manual_seed(0)
    logits = torch.randn(2, 32)
    labels = torch.tensor([3, 7])
    trainer_logprobs = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    log_ratio = torch.tensor([0.5, -0.5])

    _, metrics = DAPOLoss.Config().build()(
        logits=logits,
        labels=labels,
        global_loss_token_counts=torch.tensor(2),
        generator_logprobs=trainer_logprobs - log_ratio,
        temperature=torch.ones(2),
        advantages=torch.tensor([1.0, -1.0]),
        loss_mask=torch.ones(2, dtype=torch.bool),
    )

    torch.testing.assert_close(
        metrics["bit_wise/kl_k3/mean"],
        (torch.expm1(log_ratio) - log_ratio).mean(),
    )
    assert abs(metrics["bit_wise/logprob_diff/mean"].item()) < 1e-6
    # ChunkedLossWrapper and the trainer reduce each key by its suffix, and raise on any other.
    for key in metrics:
        loss_metric_reduce_op(key)


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
        logits=logits,
        labels=labels,
        global_loss_token_counts=torch.tensor(64),
        generator_logprobs=generator_logprobs,
        temperature=temperature,
        advantages=torch.ones(64),
        loss_mask=torch.ones(64, dtype=torch.bool),
    )

    assert metrics["bit_wise/logprob_diff/max"] == 0
