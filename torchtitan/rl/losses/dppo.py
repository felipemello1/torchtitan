# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DPPO (Divergence PPO) loss: policy gradient inside a trust region on each token's probability."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss, compute_logprobs
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.rl.losses.dapo import _MAX_LOG_RATIO, _normalize


class DPPOLoss(BaseLoss):
    """Policy-gradient loss that drops a token once its probability moved too far from
    the generator's (https://arxiv.org/abs/2602.04879).

    PPO bounds the ratio `p / p_gen` of the sampled token, which is tight for rare
    tokens and loose for common ones. Raising a token from 1e-5 to 1e-3 is a ratio of
    100 and gets clipped. Lowering one from 0.99 to 0.80 is a ratio of 0.81 and is not,
    though it moves ~190x more probability. DPPO bounds the probability change instead.

    For each response token with advantage `A`:

        dropped    = sign(A) * (p - p_gen) > divergence_threshold
        token_loss = -A * detach(min(p / p_gen, max_ratio)) * log p,  0 if dropped
        loss       = sum(token_loss) / global_loss_token_counts

    This is the paper's Eq. 23 gradient: on kept tokens below `max_ratio`, it equals
    PPO's unclipped `-A * (p / p_gen) * grad log p`. Like `DAPOLoss`, it skips tokens
    with a non-finite generator logprob.

    Example (threshold 0.2):

        p_gen      = [0.20, 0.01,  0.40,  0.01]
        p          = [0.50, 0.05,  0.10,  0.30]
        advantages = [ 1.0,  1.0,  -1.0,  -1.0]
        dropped    = [True, False, True, False]
        # Token 1 moved +0.04: kept, though PPO clips its ratio of 5.
        # Token 3 moved against its advantage: kept, so its gradient pulls p back.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        divergence_threshold: float = 0.2
        """Drop a token whose probability moved more than this from the generator's, in
        the direction its advantage pushes it (the paper's binary-TV delta)."""

        max_ratio: float = 5.0
        """Truncate the detached ratio `p / p_gen` at this value. Unlike a PPO clip, a
        truncated token keeps its gradient, scaled by `max_ratio`."""

        global_vocab_size: int | None = None
        """Full vocabulary size; required when TP shards the vocabulary."""

    def __init__(self, config: Config) -> None:
        self.divergence_threshold = config.divergence_threshold
        self.max_ratio = config.max_ratio
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        *,
        generator_logprobs: torch.Tensor,
        temperature: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the trust-region-masked policy-gradient loss.

        Args:
            logits: [T, V] current-policy output.
            labels: [T] pre-shifted target token ids.
            generator_logprobs: [T] logprobs from the sampling policy.
            temperature: [T] temperature each token was sampled at.
            loss_mask: [T] bool mask; True for response tokens.
            advantages: [T] per-token advantages (0.0 for prompt/padding).
            global_loss_token_counts: total response tokens with finite generator logprobs
                across all microbatches and DP ranks; the loss denominator.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a dict of
            scalar tensors pre-normalized for SUM reduction across DP ranks.
        """
        logits = logits / temperature.unsqueeze(-1)
        trainer_logprobs, token_entropy = compute_logprobs(
            logits,
            labels,
            vocab_parallel_group=spmd_mesh_group("tp"),
            return_entropy=True,
            global_vocab_size=self.global_vocab_size,
        )
        effective_loss_mask = loss_mask & torch.isfinite(generator_logprobs)

        # The ratio and the trust region are constants in the gradient.
        with torch.no_grad():
            masked_log_ratio = torch.where(
                effective_loss_mask, trainer_logprobs - generator_logprobs, 0.0
            )
            ratio = torch.exp(
                torch.clamp(masked_log_ratio, -_MAX_LOG_RATIO, _MAX_LOG_RATIO)
            )

            # TODO: add binary KL (paper Eq. 14) to reproduce verl/SkyRL `dppo_kl` runs.
            # Left out to keep one variant with one default; the paper finds them on par.
            # How far p moved in the direction A pushes it; negative if it moved back.
            prob_change = advantages.sign() * (
                torch.exp(trainer_logprobs) - torch.exp(generator_logprobs)
            )
            dropped = (prob_change > self.divergence_threshold) & effective_loss_mask
            keep_mask = effective_loss_mask & ~dropped

        token_loss = (
            -advantages * torch.clamp(ratio, max=self.max_ratio) * trainer_logprobs
        )
        loss = _normalize((token_loss * keep_mask).sum(), global_loss_token_counts)

        with torch.no_grad():
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": _normalize(
                    (ratio * effective_loss_mask).sum(), global_loss_token_counts
                ),
                "loss/trust_region_dropped_frac": _normalize(
                    dropped.float().sum(), global_loss_token_counts
                ),
                "loss/ratio_truncated_frac": _normalize(
                    ((ratio > self.max_ratio) & keep_mask).float().sum(),
                    global_loss_token_counts,
                ),
                # Mean of log p - log p_gen over loss tokens: the k1 estimate of
                # -KL(p_gen || p).
                "bit_wise/logprob_diff/mean": _normalize(
                    masked_log_ratio.float().sum(), global_loss_token_counts
                ),
                "bit_wise/ratio_tokens_different/mean": _normalize(
                    (masked_log_ratio.abs() > 1e-6).float().sum(),
                    global_loss_token_counts,
                ),
                "bit_wise/logprob_diff/max": masked_log_ratio.abs().max(),
                # Mean entropy of softmax(logits / temperature) over tokens used by the loss.
                "trainer/entropy/mean": _normalize(
                    (token_entropy * effective_loss_mask).sum(),
                    global_loss_token_counts,
                ),
            }

        return loss, metrics
