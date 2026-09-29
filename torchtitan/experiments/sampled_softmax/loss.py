# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sampled-softmax training loss (modded-nanogpt record #360, adapted to vocab-parallel TP).

Early in training, each rank computes the lm_head and CE only over a candidate set
``C`` of its local vocab rows: every target of the microbatch that falls in the local
shard, plus negatives, up to a fixed per-rank budget ``P``. Logits are
``h @ W[C].T`` and the lm_head gradient is dense. Validation always runs the parent
full-vocab path; once the schedule ends, training runs the full softmax through the
same fused kernel.

Corrections, i.e. how the loss accounts for the rows ``R`` outside ``C``:

* ``none``: record #360, the normalizer is ``Z_C`` and rows of ``R`` get no gradient.
* ``importance``: uniform negatives get ``log(num_nontargets / num_negatives)`` added
  to their logits, so their summed ``exp`` estimates the mass of every non-target.
* ``meanfield``: one extra logit per token, ``log|R| + h @ mean(W[R])``, stands for
  all of ``R``. Its gradient reaches every row of ``R`` as one shared rank-1 update.

Negatives: ``uniform`` walks a stride permutation of the shard (record #360);
``frequent`` draws without replacement by running target counts (Gumbel top-k).

Shape suffixes: T = tokens, D = model dim, V = local vocab rows, P = candidates.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.loss import ChunkedLossWrapper, IGNORE_INDEX
from torchtitan.distributed.spmd_types import current_spmd_mesh, spmd_mesh_size
from torchtitan.experiments.sampled_softmax import kernels


class SampledSoftmaxChunkedLoss(ChunkedLossWrapper):
    """``ChunkedLossWrapper`` whose training loss is a sampled softmax early in training."""

    @dataclass(kw_only=True, slots=True)
    class Config(ChunkedLossWrapper.Config):
        total_steps: int = 100
        """Training steps; the schedule is expressed as fractions of it."""

        schedule: list[tuple[float, int]] = field(
            default_factory=lambda: [(0.57, 8192), (0.81, 12288), (0.93, 24576)]
        )
        """``(end_fraction, budget)``: candidates per TP rank until ``end_fraction``
        of ``total_steps``. After the last entry training uses the full softmax.
        The default mirrors record #360's ramp and ends 7% of steps early."""

        correction: str = "none"
        """``none`` (record #360), ``importance``, or ``meanfield``; see the module docstring."""

        negatives: str = "uniform"
        """``uniform`` (record #360) or ``frequent``; see the module docstring."""

        negative_count_power: float = 0.75
        """``frequent`` negatives are drawn with probability ~ ``count ** power``."""

        sampled_num_chunks: int = 1
        """Sequence chunks while sampling. Sampled logits are small, so one chunk
        avoids repeated lm_head weight-gradient writes."""

        fused_full_softmax: bool = True
        """Full-softmax training steps use the fused kernel; False uses the eager parent CE."""

        probe_tokens: int = 4096
        """Tokens of each sampled microbatch whose full-vocab CE is measured (no grad),
        so the logged loss is comparable with the baseline. 0 disables."""

        probe_freq: int = 1
        """Probe every ``probe_freq`` steps."""

    def __init__(self, config: Config, *, compile_config=None):
        super().__init__(config, compile_config=compile_config)
        if config.correction not in ("none", "importance", "meanfield"):
            raise ValueError(f"Unknown sampled-softmax correction {config.correction}")
        if config.negatives not in ("uniform", "frequent"):
            raise ValueError(f"Unknown sampled-softmax negatives {config.negatives}")
        if config.correction == "importance" and config.negatives != "uniform":
            raise ValueError("importance weights assume uniform negatives")
        self.config = config
        self.step = 1
        """1-indexed training step, set by the trainer before each step."""
        self.step_metrics: dict[str, torch.Tensor | float] = {}
        self._active: tuple | None = None
        self._stride: int | None = None
        self._target_counts_V: torch.Tensor | None = None
        self._generator: torch.Generator | None = None

    def set_lm_head(self, lm_head: nn.Module) -> None:
        super().set_lm_head(lm_head)
        loss = self

        # Linear.forward calls self._linear(input, weight, bias) on the unsharded
        # (TP-local) weight inside FSDP's hooks. Routing the fused path through
        # it keeps FSDP unshard / reduce-scatter unchanged; the "output" is the
        # chunk's summed NLL.
        def _linear(input, weight, bias):
            if loss._active is None:
                return F.linear(input, weight, bias)
            assert bias is None
            return _CandidateHeadCrossEntropy.apply(input, weight, *loss._active)

        lm_head._linear = _linear  # pyrefly: ignore[missing-attribute]

    def budget_at(self, step: int) -> int:
        """Candidates per TP rank at 1-indexed ``step``; 0 means full softmax."""
        for end_fraction, budget in self.config.schedule:
            if step <= end_fraction * self.config.total_steps:
                return budget
        return 0

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not pred.requires_grad:
            return super().__call__(pred, labels, global_valid_tokens, **loss_inputs)
        assert isinstance(pred, torch.Tensor) and not loss_inputs, "single output only"
        budget = self.budget_at(self.step)
        if budget == 0 and not self.config.fused_full_softmax:
            self.step_metrics = {"sampled_softmax/budget": 0.0}
            return super().__call__(pred, labels, global_valid_tokens)
        return self._fused_call(pred, labels, global_valid_tokens, budget=budget)

    def _fused_call(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        *,
        budget: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from torch.distributed._composable.fsdp import FSDPModule

        lm_head = self.lm_head
        assert lm_head is not None
        num_chunks = self.config.sampled_num_chunks if budget > 0 else self.num_chunks
        fsdp_enabled = isinstance(lm_head, FSDPModule)
        probe_tokens = self.config.probe_tokens
        run_probe = (
            budget > 0 and probe_tokens > 0 and self.step % self.config.probe_freq == 0
        )
        with spmd.local():
            if fsdp_enabled:
                lm_head.set_reshard_after_forward(False)
                lm_head.set_reshard_after_backward(False)
                lm_head.set_requires_gradient_sync(False, recurse=False)
                with spmd.no_typecheck():
                    lm_head.unshard()
            # The unsharded weight is only reachable inside FSDP's hooks; its
            # row count is the TP-local vocab size.
            weight = lm_head.weight
            num_rows = weight.shape[0]
            vocab_start, tp_group = _vocab_shard(num_rows)
            cands = self._build_candidates(
                labels, budget=budget, num_rows=num_rows, vocab_start=vocab_start
            )
            lse_T = torch.empty(labels.shape, dtype=torch.float32, device=labels.device)

            # Per chunk: fused lm_head + CE forward (which also computes the logits
            # gradient), then backward into the chunk's hidden states and the weight.
            seq_len = pred.shape[0]
            chunk_len = seq_len // num_chunks
            assert chunk_len * num_chunks == seq_len
            if num_chunks > 1:
                grad_pred = torch.empty(
                    pred.shape, dtype=torch.float32, device=pred.device
                )
            total_loss = pred.new_zeros((), dtype=torch.float32)
            for i in range(num_chunks):
                if fsdp_enabled and i == num_chunks - 1:
                    lm_head.set_requires_gradient_sync(True, recurse=False)
                chunk = slice(i * chunk_len, (i + 1) * chunk_len)
                h_chunk = pred[chunk].detach().requires_grad_()
                self._active = (cands.chunk(chunk), tp_group, lse_T[chunk])
                nll = lm_head(h_chunk)
                self._active = None
                chunk_loss = nll / global_valid_tokens
                total_loss = total_loss + chunk_loss.detach()
                with spmd.no_typecheck():
                    chunk_loss.backward()
                if num_chunks > 1:
                    grad_pred[chunk] = h_chunk.grad
                else:
                    grad_pred = h_chunk.grad

            probe_metrics = {}
            if run_probe:
                with torch.no_grad():
                    probe_metrics = _probe_full_softmax(
                        h_TD=pred[:probe_tokens].detach(),
                        weight_VD=weight.detach(),
                        labels_T=labels[:probe_tokens],
                        cand_rows_P=cands.rows,
                        loss_lse_T=lse_T[:probe_tokens],
                        vocab_start=vocab_start,
                        tp_group=tp_group,
                    )
            if fsdp_enabled:
                lm_head.set_reshard_after_forward(True)
                lm_head.set_reshard_after_backward(True)
                lm_head.reshard()

        self.step_metrics = {
            "sampled_softmax/budget": float(budget),
            "sampled_softmax/num_targets": cands.num_targets.float(),
        }
        if budget > 0:
            self.step_metrics[
                "sampled_softmax/dropped_tokens"
            ] = cands.num_dropped.float()
        if probe_metrics:
            self.step_metrics.update(probe_metrics)
            self.step_metrics["sampled_softmax/train_ce"] = total_loss.detach() * (
                global_valid_tokens / (labels != IGNORE_INDEX).sum().clamp(min=1)
            )
        with spmd.no_typecheck():
            loss = self._gradient_backprop(
                (pred,), (grad_pred.to(pred.dtype),), total_loss
            )
        return loss, {}

    def _build_candidates(
        self, labels_T: torch.Tensor, *, budget: int, num_rows: int, vocab_start: int
    ) -> "CandidateSet":
        """Pick this rank's ``budget`` candidate rows without a host sync.

        Targets come first. If the microbatch has more distinct local targets than
        ``budget``, the highest target rows are dropped and their tokens leave the
        loss (``labels_T = -1``); ``dropped_tokens`` counts them.

        Example: 4 local rows, budget 3, targets {2}, uniform negatives visiting
        rows 3, 2, 0, 1 -> rows_P [0, 2, 3], a target of row 2 maps to column 1.
        """
        device = labels_T.device
        valid_T = labels_T != IGNORE_INDEX
        local_T = labels_T - vocab_start
        in_shard_T = valid_T & (local_T >= 0) & (local_T < num_rows)
        # Out-of-shard labels write the sentinel row num_rows, dropped below.
        safe_local_T = torch.where(in_shard_T, local_T, num_rows)
        is_target_V = torch.zeros(num_rows + 1, dtype=torch.bool, device=device)
        is_target_V[safe_local_T] = True
        is_target_V = is_target_V[:num_rows]
        num_targets = is_target_V.sum()
        if self.config.negatives == "frequent":
            if self._target_counts_V is None:
                self._target_counts_V = torch.zeros(num_rows + 1, device=device)
            self._target_counts_V.index_add_(
                0, safe_local_T, torch.ones_like(safe_local_T, dtype=torch.float32)
            )
        if budget <= 0 or budget >= num_rows:
            return CandidateSet(
                rows=None,
                labels=torch.where(in_shard_T, local_T, -1),
                valid=valid_T.float(),
                col_bias=None,
                num_targets=num_targets,
                num_dropped=torch.zeros_like(num_targets),
                num_rest=0,
            )

        num_negatives = (budget - num_targets).clamp(min=0)
        is_negative_V = self._pick_negatives(
            is_target_V, num_negatives=num_negatives, budget=budget
        )
        is_candidate_V = is_target_V | is_negative_V
        rows_P = torch.nonzero_static(is_candidate_V, size=budget).squeeze(1)
        column_V = torch.cumsum(is_candidate_V, 0) - 1
        column_T = column_V[safe_local_T.clamp(max=num_rows - 1)]
        is_kept_T = column_T < budget
        labels_local_T = torch.where(in_shard_T & is_kept_T, column_T, -1)

        col_bias_P = None
        if self.config.correction == "importance":
            # Negatives are a uniform sample of the non-target rows; weighting them by
            # num_nontargets / num_negatives makes sum(exp) estimate the missing mass.
            log_weight = torch.log(
                (num_rows - num_targets).float() / num_negatives.clamp(min=1).float()
            )
            col_bias_P = torch.where(is_target_V[rows_P], 0.0, log_weight).float()
        return CandidateSet(
            rows=rows_P,
            labels=labels_local_T,
            valid=valid_T.float(),
            col_bias=col_bias_P,
            num_targets=num_targets,
            num_dropped=(in_shard_T & ~is_kept_T).sum(),
            num_rest=num_rows - budget if self.config.correction == "meanfield" else 0,
        )

    def _pick_negatives(
        self, is_target_V: torch.Tensor, *, num_negatives: torch.Tensor, budget: int
    ) -> torch.Tensor:
        """Mark ``num_negatives`` non-target rows, different every step and rank."""
        num_rows = is_target_V.shape[0]
        device = is_target_V.device
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.config.negatives == "uniform":
            # The first num_negatives non-targets along a stride permutation of the
            # shard, starting at a per-step, per-rank offset.
            if self._stride is None:
                self._stride = _coprime_stride(num_rows)
            offset = (self.step * budget + rank * 7919 * budget) % num_rows
            order_V = (
                offset + torch.arange(num_rows, device=device) * self._stride
            ) % num_rows
            nontarget_V = ~is_target_V[order_V]
            take_V = nontarget_V & (torch.cumsum(nontarget_V, 0) <= num_negatives)
        else:
            # Gumbel top-k draws rows without replacement with probability ~ count ** power.
            if self._generator is None:
                self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.step * 1_000_003 + rank)
            uniform_V = torch.rand(num_rows, device=device, generator=self._generator)
            assert self._target_counts_V is not None
            keys_V = self.config.negative_count_power * torch.log1p(
                self._target_counts_V[:num_rows]
            ) - torch.log(-torch.log(uniform_V))
            keys_V = keys_V.masked_fill(is_target_V, float("-inf"))
            order_V = torch.topk(keys_V, budget).indices
            take_V = torch.arange(budget, device=device) < num_negatives
        is_negative_V = torch.zeros_like(is_target_V)
        is_negative_V[order_V] = take_V
        return is_negative_V


@dataclass
class CandidateSet:
    """One microbatch's candidate rows on this TP rank and its labels remapped into them."""

    rows: torch.Tensor | None
    """[P] local vocab rows, ascending; None means every row (full softmax)."""

    labels: torch.Tensor
    """[T] column of each token's target in the logits; -1 if it is not a local candidate."""

    valid: torch.Tensor
    """[T] 1.0 where the global label is not IGNORE_INDEX."""

    col_bias: torch.Tensor | None
    """[P] fp32 log importance weight added to each column's logit."""

    num_targets: torch.Tensor
    """Distinct local targets in the microbatch."""

    num_dropped: torch.Tensor
    """Tokens whose local target did not fit in the budget; they leave the loss."""

    num_rest: int
    """Rows outside the candidate set that the meanfield logit stands for; 0 disables it."""

    def chunk(self, tokens: slice) -> "CandidateSet":
        return CandidateSet(
            rows=self.rows,
            labels=self.labels[tokens],
            valid=self.valid[tokens],
            col_bias=self.col_bias,
            num_targets=self.num_targets,
            num_dropped=self.num_dropped,
            num_rest=self.num_rest,
        )


class _CandidateHeadCrossEntropy(torch.autograd.Function):
    """Summed NLL of ``h_TD @ weight_VD[rows_P].T`` (plus the meanfield logit) over valid tokens.

    The forward pass overwrites the bf16 logits with their gradient, so backward is
    two GEMMs and a dense row scatter. ``lse_out_T`` receives each token's normalizer.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        h_TD: torch.Tensor,
        weight_VD: torch.Tensor,
        cands: CandidateSet,
        tp_group: dist.ProcessGroup | None,
        lse_out_T: torch.Tensor,
    ) -> torch.Tensor:
        rows_P = cands.rows
        weight_PD = weight_VD if rows_P is None else weight_VD.index_select(0, rows_P)
        logits_TP = torch.mm(h_TD, weight_PD.t())
        row_max_T, row_sumexp_T, target_T = kernels.row_stats(
            logits_TP, cands.col_bias, cands.labels
        )
        rest_mean_D = rest_logit_T = None
        if cands.num_rest > 0:
            rest_sum_D = weight_VD.sum(0, dtype=torch.float32) - weight_PD.sum(
                0, dtype=torch.float32
            )
            rest_mean_D = (rest_sum_D / cands.num_rest).to(h_TD.dtype)
            rest_logit_T = torch.mv(h_TD, rest_mean_D).float() + math.log(
                cands.num_rest
            )
            new_max_T = torch.maximum(row_max_T, rest_logit_T)
            row_sumexp_T = row_sumexp_T * torch.exp(row_max_T - new_max_T) + torch.exp(
                rest_logit_T - new_max_T
            )
            row_max_T = new_max_T
        lse_T, target_T, has_target_T = _combine_vocab_parallel_stats(
            row_max_T=row_max_T,
            row_sumexp_T=row_sumexp_T,
            target_T=target_T,
            has_target_T=(cands.labels >= 0).float(),
            tp_group=tp_group,
        )
        # Tokens whose target overflowed the budget on its shard leave the loss on every shard.
        valid_T = cands.valid * has_target_T
        lse_out_T.copy_(lse_T)
        nll = ((lse_T - target_T) * valid_T).sum()
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_logits_TP = kernels.softmax_grad_(
                logits_TP, cands.col_bias, cands.labels, lse_T, valid_T
            )
            rest_prob_T = (
                None
                if rest_logit_T is None
                else (torch.exp(rest_logit_T - lse_T) * valid_T).to(h_TD.dtype)
            )
            ctx.save_for_backward(
                h_TD, weight_PD, grad_logits_TP, rows_P, rest_mean_D, rest_prob_T
            )
            ctx.num_weight_rows = weight_VD.shape[0]
            ctx.num_rest = cands.num_rest
        return nll

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore[bad-override]
        (
            h_TD,
            weight_PD,
            grad_logits_TP,
            rows_P,
            rest_mean_D,
            rest_prob_T,
        ) = ctx.saved_tensors
        grad_output = grad_output.to(h_TD.dtype)
        grad_h_TD = torch.mm(grad_logits_TP, weight_PD)
        grad_weight_PD = torch.mm(grad_logits_TP.t(), h_TD).mul_(grad_output)
        if rows_P is None:
            grad_weight_VD = grad_weight_PD
        elif rest_prob_T is None:
            grad_weight_VD = grad_weight_PD.new_zeros(
                ctx.num_weight_rows, h_TD.shape[1]
            )
            grad_weight_VD.index_copy_(0, rows_P, grad_weight_PD)
        else:
            # The meanfield logit is h @ mean(W[R]): every row of R gets the same gradient.
            grad_h_TD.addr_(rest_prob_T, rest_mean_D)
            grad_rest_row_D = torch.mv(h_TD.t(), rest_prob_T).mul_(
                grad_output / ctx.num_rest
            )
            grad_weight_VD = grad_rest_row_D.expand(
                ctx.num_weight_rows, -1
            ).contiguous()
            grad_weight_VD.index_copy_(0, rows_P, grad_weight_PD)
        return grad_h_TD.mul_(grad_output), grad_weight_VD, None, None, None


def _combine_vocab_parallel_stats(
    *,
    row_max_T: torch.Tensor,
    row_sumexp_T: torch.Tensor,
    target_T: torch.Tensor,
    has_target_T: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return global ``(lse_T, target_logit_T, has_target_T)`` from per-shard statistics.

    One all-gather of the stacked ``[4, T]`` statistics replaces the three
    all-reduces (max, sumexp, target) of ``_LossParallelCrossEntropy``.
    """
    if tp_group is None:
        return row_max_T + torch.log(row_sumexp_T), target_T, has_target_T
    stats_4T = torch.stack((row_max_T, row_sumexp_T, target_T, has_target_T))
    tp_size = dist.get_world_size(tp_group)
    gathered = funcol.all_gather_tensor(stats_4T, gather_dim=0, group=tp_group)
    shard_max, shard_sumexp, shard_target, shard_has_target = gathered.view(
        tp_size, 4, -1
    ).unbind(1)
    global_max_T = shard_max.amax(0)
    sumexp_T = (shard_sumexp * torch.exp(shard_max - global_max_T)).sum(0)
    return (
        global_max_T + torch.log(sumexp_T),
        shard_target.sum(0),
        shard_has_target.sum(0),
    )


def _probe_full_softmax(
    *,
    h_TD: torch.Tensor,
    weight_VD: torch.Tensor,
    labels_T: torch.Tensor,
    cand_rows_P: torch.Tensor | None,
    loss_lse_T: torch.Tensor,
    vocab_start: int,
    tp_group: dist.ProcessGroup | None,
) -> dict[str, torch.Tensor]:
    """Full-vocab CE, the mass outside the candidate set, and the loss normalizer's bias.

    ``missing_mass`` is ``1 - Z_C / Z``; the logit-gradient L1 error of record #360's
    loss is ``2 * missing_mass`` per token. ``lse_bias`` is the mean of
    ``loss_lse - lse``: 0 for an exact normalizer, ``log(Z_C / Z)`` for ``none``.
    """
    num_rows = weight_VD.shape[0]
    local_T = labels_T - vocab_start
    in_shard_T = (local_T >= 0) & (local_T < num_rows)
    valid_T = (labels_T != IGNORE_INDEX).float()
    num_valid = valid_T.sum().clamp(min=1)
    logits_TV = torch.mm(h_TD, weight_VD.t())
    row_max_T, row_sumexp_T, target_T = kernels.row_stats(
        logits_TV, None, torch.where(in_shard_T, local_T, -1)
    )
    lse_T, target_T, _ = _combine_vocab_parallel_stats(
        row_max_T=row_max_T,
        row_sumexp_T=row_sumexp_T,
        target_T=target_T,
        has_target_T=torch.zeros_like(row_max_T),
        tp_group=tp_group,
    )
    metrics = {
        "sampled_softmax/probe_full_ce": ((lse_T - target_T) * valid_T).sum()
        / num_valid,
        "sampled_softmax/probe_lse_bias": ((loss_lse_T - lse_T) * valid_T).sum()
        / num_valid,
    }
    if cand_rows_P is not None:
        cand_max_T, cand_sumexp_T, _ = kernels.row_stats(
            logits_TV.index_select(1, cand_rows_P), None, torch.full_like(labels_T, -1)
        )
        cand_lse_T, _, _ = _combine_vocab_parallel_stats(
            row_max_T=cand_max_T,
            row_sumexp_T=cand_sumexp_T,
            target_T=torch.zeros_like(cand_max_T),
            has_target_T=torch.zeros_like(cand_max_T),
            tp_group=tp_group,
        )
        missing_mass_T = 1 - torch.exp(cand_lse_T - lse_T)
        metrics["sampled_softmax/probe_missing_mass"] = (
            missing_mass_T * valid_T
        ).sum() / num_valid
    return metrics


def _vocab_shard(weight_rows: int) -> tuple[int, dist.ProcessGroup | None]:
    """Return this rank's first vocab id and the TP group (None without TP)."""
    if spmd_mesh_size("tp") > 1:
        tp_group = current_spmd_mesh().get_group("tp")  # pyrefly: ignore
        return dist.get_rank(tp_group) * weight_rows, tp_group
    return 0, None


def _coprime_stride(n: int) -> int:
    """A stride near n * 0.618 with gcd(stride, n) == 1, so k * stride mod n is a permutation."""
    stride = int(n * 0.6180339887) | 1
    while math.gcd(stride, n) != 1:
        stride += 2
    return stride
