# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the RL loss package + trainer grad-accum path.

These pin three invariants that the GPU smoke can't catch cheaply:

1. `DAPOLoss` sanitizes non-finite `ref_logprobs` so the loss stays finite
   (and `loss/ratio/mean` SUM-reduces to ~1.0, not `num_microbatches * 1`).
2. `MetricAccumulator` combines SUM-reduced contributions across microsteps
   by addition and MAX-reduced ones by elementwise maximum.
3. `forward_backward` zeroes gradients once, then backprops all microsteps.
"""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from torchtitan.experiments.rl.actors.trainer import (
    compute_logprobs,
    MetricAccumulator,
    PartialLogprobDrift,
    PolicyTrainer,
)
from torchtitan.experiments.rl.loss import DAPOLoss, LossOutput
from torchtitan.experiments.rl.sampling import TrainingLogprobConfig
from torchtitan.experiments.rl.types import TrainingBatch


def _basic_inputs(extra_logprob: float = 0.0) -> dict[str, torch.Tensor]:
    """Single-row [1, 4] sample with prompt[0:2] masked off.

    Tokens 2 and 3 carry advantages of +0.5 and -0.2 respectively.
    `extra_logprob` is added to position 2's `ref_logprob` so callers can
    inject NaN / +inf / -inf there without touching the others.
    """
    return {
        "policy_logprobs": torch.tensor([[-1.0, -1.0, -0.30, -0.50]]),
        "ref_logprobs": torch.tensor([[0.0, 0.0, -0.40 + extra_logprob, -0.55]]),
        "loss_mask": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "advantages": torch.tensor([[0.0, 0.0, 0.5, -0.2]]),
        "num_global_valid_tokens": torch.tensor(2.0),
    }


def test_compute_logprobs_applies_sampling_temperature() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [0.5, 0.0, 1.0],
            ]
        ]
    )
    token_ids = torch.tensor([[0, 1, 2]])

    out = compute_logprobs(logits, token_ids, temperature=0.5)
    expected = torch.log_softmax(logits[:, :-1].float() / 0.5, dim=-1).gather(
        -1, token_ids[:, 1:].unsqueeze(-1)
    )

    torch.testing.assert_close(out, expected.squeeze(-1))
    with pytest.raises(ValueError, match="temperature must be positive"):
        compute_logprobs(logits, token_ids, temperature=0.0)


def test_training_logprob_config_rejects_nucleus_sampling() -> None:
    sampling = SimpleNamespace(temperature=0.8, top_p=1.0)
    assert TrainingLogprobConfig.from_sampling(sampling).temperature == 0.8

    with pytest.raises(ValueError, match="top_p=1.0"):
        TrainingLogprobConfig.from_sampling(
            SimpleNamespace(temperature=0.8, top_p=0.95)
        )


@pytest.mark.parametrize(
    "bad_ref",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "posinf", "neginf"],
)
def test_dapo_sanitizes_nonfinite_ref_logprobs(bad_ref: float) -> None:
    """A non-finite ref logprob at a trainable token must not poison the loss."""
    loss_fn = DAPOLoss(DAPOLoss.Config())
    inputs = _basic_inputs()
    inputs["ref_logprobs"][0, 2] = bad_ref

    out = loss_fn(**inputs)

    assert math.isfinite(out.loss.item()), f"loss is {out.loss.item()!r}"
    assert math.isfinite(out.sum_metrics["loss/ratio/mean"].item())
    # NaN maps to ratio=1; +/- inf are clamped well above/below 1.
    assert out.sum_metrics["health/loss/ref_logprob_nonfinite_frac"].item() > 0.0


def test_dapo_global_denominator_yields_unit_ratio_sum() -> None:
    """With `policy == ref`, `loss/ratio/mean` from each microbatch
    SUM-reduces to ~1.0, not `num_microbatches`.

    The denominator is `num_global_valid_tokens` (global), not the local
    mask sum, so summing per-microbatch contributions reproduces the
    global mean.
    """
    loss_fn = DAPOLoss(DAPOLoss.Config())
    # Two microbatches share the same global N=4 (2 valid tokens each).
    global_n = torch.tensor(4.0)

    summed = 0.0
    for _ in range(2):
        out = loss_fn(
            policy_logprobs=torch.tensor([[-1.0, -1.0, -0.30, -0.50]]),
            ref_logprobs=torch.tensor([[-1.0, -1.0, -0.30, -0.50]]),  # ratio == 1
            loss_mask=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            advantages=torch.tensor([[0.0, 0.0, 0.5, -0.2]]),
            num_global_valid_tokens=global_n,
        )
        summed += out.sum_metrics["loss/ratio/mean"].item()

    assert summed == pytest.approx(1.0, abs=1e-5), summed


def test_metric_accumulator_combines_sum_and_max() -> None:
    """SUM keys add; MAX keys take elementwise max; missing-key first add wins.

    These are the only two reductions the trainer uses across grad-accum
    microsteps; both must compose cleanly with running state.
    """
    acc = MetricAccumulator()
    acc.add_sum({"loss/mean": torch.tensor(0.3)})
    acc.add_sum({"loss/mean": torch.tensor(0.2), "loss/ratio/mean": torch.tensor(0.5)})
    acc.add_max({"loss/ratio/max_abs": torch.tensor(1.10)})
    acc.add_max({"loss/ratio/max_abs": torch.tensor(1.25)})

    assert acc.sum_reduced_metrics["loss/mean"].item() == pytest.approx(0.5)
    assert acc.sum_reduced_metrics["loss/ratio/mean"].item() == pytest.approx(0.5)
    assert acc.max_reduced_metrics["loss/ratio/max_abs"].item() == pytest.approx(1.25)


def _training_batch(sample_id: int) -> TrainingBatch:
    return TrainingBatch(
        token_ids=torch.tensor([[sample_id, sample_id + 1]]),
        positions=torch.tensor([[0, 1]]),
        ref_logprobs=torch.zeros(1, 2),
        loss_mask=torch.tensor([[0.0, 1.0]]),
        advantages=torch.ones(1, 2),
    )


def test_forward_backward_accumulates_microsteps_before_optim_step() -> None:
    """One optimizer step zeroes grads once and backprops every microstep.

    Example: four grad-accum microsteps become one `forward_backward` call,
    not four calls that each reset gradients.
    """
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    trainer.policy_version = 0
    trainer.model_parts = [object()]
    trainer.device = torch.device("cpu")
    trainer.dp_rank = 0
    trainer.optimizers = MagicMock()

    backward_calls: list[int] = []

    def fake_forward_one_microbatch(
        microbatch: TrainingBatch,
        *,
        num_global_valid_tokens: torch.Tensor,
        logprob_config: TrainingLogprobConfig,
    ) -> tuple[LossOutput, PartialLogprobDrift]:
        sample_id = int(microbatch.token_ids[0, 0].item())
        loss = torch.tensor(float(sample_id), requires_grad=True)

        def record_backward(_grad, *, sample_id=sample_id) -> None:
            backward_calls.append(sample_id)

        loss.register_hook(record_backward)
        value = torch.tensor(float(sample_id))
        return (
            LossOutput(
                loss=loss,
                sum_metrics={"loss/mean": value},
                max_metrics={"loss/ratio/max_abs": value},
            ),
            PartialLogprobDrift(
                logprob_diff_mean=torch.tensor(0.0),
                logprob_diff_max=value,
                ratio_tokens_different=torch.tensor(0.0),
            ),
        )

    reduced_inputs: dict[str, dict[str, torch.Tensor]] = {}

    def fake_reduce_forward_backward_metrics(
        *,
        sum_reduced_metrics: dict[str, torch.Tensor],
        max_reduced_metrics: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        reduced_inputs["sum"] = sum_reduced_metrics
        reduced_inputs["max"] = max_reduced_metrics
        return {"loss/mean": float(sum_reduced_metrics["loss/mean"].item())}

    trainer._forward_one_microbatch = fake_forward_one_microbatch
    trainer.reduce_forward_backward_metrics = fake_reduce_forward_backward_metrics

    forward_backward_impl = PolicyTrainer.forward_backward._method
    out = asyncio.run(
        forward_backward_impl(
            trainer,
            [[_training_batch(i)] for i in range(1, 5)],
            num_global_valid_tokens=4,
            logprob_config=TrainingLogprobConfig(temperature=1.0),
        )
    )

    trainer.optimizers.zero_grad.assert_called_once_with()
    assert backward_calls == [1, 2, 3, 4]
    assert out["loss/mean"] == pytest.approx(10.0)
    assert reduced_inputs["sum"]["loss/mean"].item() == pytest.approx(10.0)
    assert reduced_inputs["max"]["loss/ratio/max_abs"].item() == pytest.approx(4.0)
