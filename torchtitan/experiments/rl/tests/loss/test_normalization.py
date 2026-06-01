# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for global-token-count invariance under gradient accumulation, the
batcher's num_global_valid_tokens, and the log_entropy flag."""

import torch

from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.loss import DAPOLoss
from torchtitan.experiments.rl.types import Episode

from .conftest import assert_close, num_valid_tokens


def test_grad_accum_invariance():
    """Loss(full batch) == sum of Loss(microbatch) when every microbatch shares
    the same global num_global_valid_tokens. This is what makes grad accumulation
    equivalent to a single large-batch step."""
    torch.manual_seed(0)
    B, S, V = 4, 6, 10
    logits = torch.randn(B, S, V)
    target_ids = torch.randint(0, V, (B, S))
    gen = torch.randn(B, S)
    adv = torch.randn(B, S)
    loss_mask = torch.ones(B, S, dtype=torch.bool)
    num_global_valid_tokens = int(loss_mask.sum())

    loss_fn = DAPOLoss.Config().build()

    def run(rows):
        return loss_fn(
            logits=logits[rows],
            target_ids=target_ids[rows],
            generator_logprobs=gen[rows],
            loss_mask=loss_mask[rows],
            advantages=adv[rows],
            num_global_valid_tokens=num_global_valid_tokens,
        ).loss

    full = run(slice(0, B))
    split = run(slice(0, 2)) + run(slice(2, 4))
    assert_close(split, full)


def test_batcher_reports_global_valid_tokens():
    """Batcher returns the global response-token count (the token-mean denominator)."""
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
    microbatches, num_global_valid_tokens, _ = batcher.batch(episodes, dp_degree=1)

    # 3 response tokens per episode (5 raw -> 4 after shift, mask [F,T,T,T]).
    assert num_global_valid_tokens == 12
    assert len(microbatches) == 2 and all(len(step) == 1 for step in microbatches)


def test_batcher_packs_multiple_episodes_into_one_row():
    """Short episodes greedy-pack into a single row; the global token count still
    reflects only response tokens."""
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
    # Each episode is 2 tokens after the [:-1]/[1:] shift (raw len 3 -> 2), so all
    # three pack into one [1, 6] row with no padding.
    batcher = Batcher(
        Batcher.Config(
            batch=BatchConfig(local_batch_size=1, global_batch_size=1, seq_len=6)
        ),
        pad_id=0,
    )
    microbatches, num_global_valid_tokens, _ = batcher.batch(episodes, dp_degree=1)

    assert len(microbatches) == 1 and len(microbatches[0]) == 1
    assert microbatches[0][0].token_ids.shape == (1, 6)
    assert num_global_valid_tokens == 6


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
        num_global_valid_tokens=num_valid_tokens(loss_mask),
    )
    with_entropy = DAPOLoss.Config(log_entropy=True).build()(**kwargs)
    without_entropy = DAPOLoss.Config(log_entropy=False).build()(**kwargs)

    assert "loss/entropy/mean" in with_entropy.metrics
    assert torch.isfinite(with_entropy.metrics["loss/entropy/mean"].value)
    assert "loss/entropy/mean" not in without_entropy.metrics
    assert_close(with_entropy.loss, without_entropy.loss)
