# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Titan-specific normalization tests: global-denominator invariance under
gradient accumulation, sample-aware aggregation/ratio under packing, and the
batcher's LossNormalization + sample_ids."""

import math

import pytest
import torch

from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.loss import DAPOLoss
from torchtitan.experiments.rl.loss.ops import aggregate_loss, compute_sequence_ratio
from torchtitan.experiments.rl.loss.types import LossNormalization
from torchtitan.experiments.rl.types import Episode

from .conftest import assert_close, make_normalization


@pytest.mark.parametrize("agg_type", ["token_mean", "fixed_horizon", "sequence_mean"])
def test_grad_accum_invariance(agg_type):
    """Loss(full batch) == sum of Loss(microbatch) when every microbatch shares
    the same global LossNormalization. This is what makes grad accumulation
    equivalent to a single large-batch step."""
    torch.manual_seed(0)
    B, S, V = 4, 6, 10
    logits = torch.randn(B, S, V)
    target_ids = torch.randint(0, V, (B, S))
    gen = torch.randn(B, S)
    adv = torch.randn(B, S)
    loss_mask = torch.ones(B, S, dtype=torch.bool)
    sample_ids = torch.arange(B).unsqueeze(1).expand(B, S).contiguous()
    norm = LossNormalization(
        num_global_valid_tokens=int(loss_mask.sum()),
        num_global_sequences=B,
        num_global_fixed_horizon_tokens=B * S,
    )

    loss_fn = DAPOLoss.Config(agg_type=agg_type).build()

    def run(rows):
        return loss_fn(
            logits=logits[rows],
            target_ids=target_ids[rows],
            generator_logprobs=gen[rows],
            loss_mask=loss_mask[rows],
            advantages=adv[rows],
            normalization=norm,
            sample_ids=sample_ids[rows],
        ).loss

    full = run(slice(0, B))
    split = run(slice(0, 2)) + run(slice(2, 4))
    assert_close(split, full)


def test_sequence_mean_uses_samples_not_rows():
    """sequence_mean averages per source episode, not per packed row.

    Two episodes packed in one row: episode 0 tokens [2, 4] -> mean 3, episode 1
    token [10] -> mean 10, global sequence mean = (3 + 10) / 2 = 6.5.
    """
    per_token_loss = torch.tensor([[2.0, 4.0, 10.0, 0.0]])
    loss_mask = torch.tensor([[True, True, True, False]])
    sample_ids = torch.tensor([[0, 0, 1, -1]])
    norm = LossNormalization(
        num_global_valid_tokens=3,
        num_global_sequences=2,
        num_global_fixed_horizon_tokens=4,
    )
    loss = aggregate_loss(
        per_token_loss,
        loss_mask,
        agg_type="sequence_mean",
        normalization=norm,
        sample_ids=sample_ids,
    )
    assert_close(loss, torch.tensor(6.5))


def test_sequence_ratio_is_per_episode():
    """The sequence ratio is one value per episode (sample), not per row.

    One row, two episodes: episode 0 log-ratio 0 -> ratio 1; episode 1 log-ratio
    ln(2) -> ratio 2.
    """
    policy = torch.tensor([[0.0, 0.0, math.log(2.0), math.log(2.0)]])
    gen = torch.zeros(1, 4)
    loss_mask = torch.ones(1, 4, dtype=torch.bool)
    sample_ids = torch.tensor([[0, 0, 1, 1]])
    norm = make_normalization(loss_mask, 1, 4)
    ratio, _log_ratio, _metrics = compute_sequence_ratio(
        policy, gen, loss_mask, sample_ids, norm
    )
    assert_close(ratio[0, 0], torch.tensor(1.0))
    assert_close(ratio[0, 2], torch.tensor(2.0))


def test_batcher_normalization_and_sample_ids():
    """Batcher reports global denominators and globally-unique sample ids."""
    episodes = [
        Episode(
            policy_version=0,
            prompt_idx=i,
            prompt_token_ids=[1, 2],
            text="",
            token_ids=[3, 4, 5],
            token_logprobs=[0.0, 0.0, 0.0],
            reward=0.0,
            advantage=float(i),
        )
        for i in range(4)
    ]
    # seq_len=4 == per-episode length after the [:-1]/[1:] shift, so each episode
    # is its own packed row: 4 rows, local_batch_size=2 -> 2 microbatches.
    batcher = Batcher(
        Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=4, seq_len=4)
        ),
        pad_id=0,
    )
    microbatches, norm, _ = batcher.batch(episodes, dp_degree=1)

    # 3 response tokens per episode (5 raw -> 4 after shift, mask [F,T,T,T]).
    assert norm.num_global_valid_tokens == 12
    assert norm.num_global_sequences == 4
    assert norm.num_global_fixed_horizon_tokens == 4 * 4

    seen = set()
    for step in microbatches:
        for tb in step:
            ids = tb.sample_ids[tb.sample_ids >= 0].tolist()
            seen.update(ids)
    assert seen == {0, 1, 2, 3}


def test_batcher_packs_multiple_episodes_into_one_row():
    """Short episodes greedy-pack into a single row; sample_ids delineate each
    source episode within that row (not one episode per row)."""
    episodes = [
        Episode(
            policy_version=0,
            prompt_idx=i,
            prompt_token_ids=[1],
            text="",
            token_ids=[2, 3],
            token_logprobs=[0.0, 0.0],
            reward=0.0,
            advantage=float(i),
        )
        for i in range(3)
    ]
    # Each episode is 2 tokens after the [:-1]/[1:] shift (raw len 3 -> 2), so
    # all three pack into one [1, 6] row with no padding.
    batcher = Batcher(
        Batcher.Config(
            batch=BatchConfig(local_batch_size=1, global_batch_size=1, seq_len=6)
        ),
        pad_id=0,
    )
    microbatches, norm, _ = batcher.batch(episodes, dp_degree=1)

    # One grad-accum step, one rank, one packed row holding all three episodes.
    assert len(microbatches) == 1 and len(microbatches[0]) == 1
    tb = microbatches[0][0]
    assert tb.sample_ids.shape == (1, 6)
    # Episode boundaries are contiguous within the single row.
    assert tb.sample_ids.tolist() == [[0, 0, 1, 1, 2, 2]]
    assert norm.num_global_sequences == 3
    assert norm.num_global_valid_tokens == 6


def test_log_entropy_flag_controls_entropy_metric():
    """log_entropy gates loss/entropy/mean; entropy is logging-only, so the loss
    value is identical with the flag on or off."""
    torch.manual_seed(0)
    B, S, V = 2, 4, 10
    loss_mask = torch.ones(B, S, dtype=torch.bool)
    kwargs = dict(
        logits=torch.randn(B, S, V),
        target_ids=torch.randint(0, V, (B, S)),
        generator_logprobs=torch.randn(B, S),
        loss_mask=loss_mask,
        advantages=torch.randn(B, S),
        normalization=make_normalization(loss_mask, B, S),
        sample_ids=torch.arange(B).unsqueeze(1).expand(B, S).contiguous(),
    )
    with_entropy = DAPOLoss.Config(log_entropy=True).build()(**kwargs)
    without_entropy = DAPOLoss.Config(log_entropy=False).build()(**kwargs)

    assert "loss/entropy/mean" in with_entropy.metrics
    assert torch.isfinite(with_entropy.metrics["loss/entropy/mean"].value)
    assert "loss/entropy/mean" not in without_entropy.metrics
    assert_close(with_entropy.loss, without_entropy.loss)
