# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sampled-softmax training loss (modded-nanogpt record #360, adapted to vocab-parallel TP).

Early in training, each rank computes the lm_head and CE only over a candidate
set ``C`` of vocabulary rows instead of the full vocabulary:

* every target in the rank's microbatch that falls in the local vocab shard, plus
* negatives from a stride permutation of the local shard until the per-rank
  budget ``P`` is reached. The window start moves every step and differs per
  rank, so every row is visited uniformly and no RNG is needed.

``C`` is shared by every token of the microbatch. Logits are ``h @ W[C].T``;
the lm_head gradient is dense with zeros outside ``C``. With TP the local shards
differ in how many targets they hold (Qwen's frequent tokens have low ids, so
shard 0 holds ~95% of them), so each TP rank gets its own budget. That keeps the
per-rank GEMMs equal; ``P`` only grows past the budget if a shard has more
distinct targets than ``P``.

``correction="importance"`` adds ``log(w)`` to negative logits with
``w = (V_local - num_targets) / num_negatives``: the negatives are a uniform
sample of the shard's non-target rows, so ``w * sum(exp(z_neg))`` estimates the
missing softmax mass. ``correction="none"`` is record #360's loss.

Validation always runs the parent ``ChunkedLossWrapper`` full-vocab path. Once
the schedule ends, training switches to the full softmax, optionally through the
same fused kernel (``fused_full_softmax``).

Shape suffixes: T = tokens, D = model dim, V = local vocab rows, P = candidates.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.components.loss import ChunkedLossWrapper, IGNORE_INDEX
from torchtitan.distributed.spmd_types import current_spmd_mesh, spmd_mesh_size
from torchtitan.experiments.sampled_softmax import kernels

logger = logging.getLogger(__name__)

import spmd_types as spmd


def _combine_vocab_parallel_stats(
    row_max_T: torch.Tensor,
    row_sumexp_T: torch.Tensor,
    target_T: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global ``(lse_T, target_logit_T)`` from per-shard statistics.

    One all-gather of the stacked ``[3, T]`` statistics replaces the three
    all-reduces (max, sumexp, target) of ``_LossParallelCrossEntropy``.
    """
    if tp_group is None:
        return row_max_T + torch.log(row_sumexp_T), target_T
    stats_3T = torch.stack((row_max_T, row_sumexp_T, target_T))
    tp_size = dist.get_world_size(tp_group)
    gathered = funcol.all_gather_tensor(stats_3T, gather_dim=0, group=tp_group)
    gathered = gathered.view(tp_size, 3, -1)
    shard_max, shard_sumexp, shard_target = gathered.unbind(1)
    global_max_T = shard_max.amax(0)
    sumexp_T = (shard_sumexp * torch.exp(shard_max - global_max_T)).sum(0)
    return global_max_T + torch.log(sumexp_T), shard_target.sum(0)


class _SampledHeadCrossEntropy(torch.autograd.Function):
    """Summed NLL of ``h_TD @ weight_VD[cand_P].T`` over the candidate columns.

    The gradient of the logits is computed in the forward pass and stored in
    the bf16 logits buffer, so backward is two GEMMs and a row scatter.
    ``cand_P=None`` means all local rows (fused full softmax).
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        h_TD: torch.Tensor,
        weight_VD: torch.Tensor,
        cand_P: torch.Tensor | None,
        labels_T: torch.Tensor,
        valid_T: torch.Tensor,
        col_bias_P: torch.Tensor | None,
        tp_group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        weight_PD = weight_VD if cand_P is None else weight_VD.index_select(0, cand_P)
        logits_TP = torch.mm(h_TD, weight_PD.t())
        row_max_T, row_sumexp_T, target_T = kernels.row_stats(
            logits_TP, col_bias_P, labels_T
        )
        lse_T, target_T = _combine_vocab_parallel_stats(
            row_max_T, row_sumexp_T, target_T, tp_group
        )
        nll = ((lse_T - target_T) * valid_T).sum()
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_logits_TP = kernels.softmax_grad_(
                logits_TP, col_bias_P, labels_T, lse_T, valid_T
            )
            ctx.save_for_backward(h_TD, weight_PD, grad_logits_TP, cand_P)
            ctx.weight_shape = weight_VD.shape
        return nll

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore[bad-override]
        h_TD, weight_PD, grad_logits_TP, cand_P = ctx.saved_tensors
        grad_output = grad_output.to(h_TD.dtype)
        grad_h_TD = torch.mm(grad_logits_TP, weight_PD).mul_(grad_output)
        grad_weight_PD = torch.mm(grad_logits_TP.t(), h_TD).mul_(grad_output)
        if cand_P is None:
            grad_weight_VD = grad_weight_PD
        else:
            grad_weight_VD = grad_weight_PD.new_zeros(ctx.weight_shape)
            grad_weight_VD.index_copy_(0, cand_P, grad_weight_PD)
        return grad_h_TD, grad_weight_VD, None, None, None, None, None


def _coprime_stride(n: int) -> int:
    """A stride near n * 0.618 with gcd(stride, n) == 1, so k * stride mod n is a permutation."""
    stride = int(n * 0.6180339887) | 1
    while math.gcd(stride, n) != 1:
        stride += 2
    return stride


@dataclass
class _Candidates:
    cand_P: torch.Tensor | None
    labels_T: torch.Tensor  # column index into the local logits, -1 if not local
    valid_T: torch.Tensor  # global label is not IGNORE_INDEX
    col_bias_P: torch.Tensor | None
    num_targets: torch.Tensor
    num_cols: int


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
        """``none`` (record #360) or ``importance`` (log-weight the negatives)."""

        sampled_num_chunks: int = 1
        """Sequence chunks while sampling. Sampled logits are small, so one chunk
        avoids repeated lm_head weight-gradient writes."""

        fused_full_softmax: bool = False
        """Use the fused kernel (not the eager parent CE) for full-softmax steps."""

        full_ce_probe_tokens: int = 4096
        """Tokens per microbatch whose full-vocab CE is measured (no grad) every
        sampled step, so the logged loss is comparable with the baseline. 0 disables."""

        num_microbatches_per_step: int = 1

    def __init__(self, config: Config, *, compile_config=None):
        super().__init__(config, compile_config=compile_config)
        if config.correction not in ("none", "importance"):
            raise ValueError(f"Unknown sampled-softmax correction {config.correction}")
        self.config = config
        self._num_train_microbatches = 0
        self._active: tuple[torch.Tensor | None, ...] | None = None
        self.step_metrics: dict[str, torch.Tensor | float] = {}
        self._stride: int | None = None

    def set_lm_head(self, lm_head: nn.Module) -> None:
        super().set_lm_head(lm_head)
        loss = self

        # Linear.forward calls self._linear(input, weight, bias) on the unsharded
        # (TP-local) weight inside FSDP's hooks. Routing the sampled path through
        # it keeps FSDP unshard / reduce-scatter unchanged; the "output" is the
        # chunk's summed NLL.
        def _linear(input, weight, bias):
            if loss._active is None:
                return F.linear(input, weight, bias)
            assert bias is None
            return _SampledHeadCrossEntropy.apply(input, weight, *loss._active)

        lm_head._linear = _linear  # pyrefly: ignore[missing-attribute]

    def budget_at(self, step: int) -> int:
        """Candidates per TP rank at 1-indexed ``step``; 0 means full softmax."""
        for end_fraction, budget in self.config.schedule:
            if step <= end_fraction * self.config.total_steps:
                return budget
        return 0

    def _vocab_shard(self, weight_rows: int) -> tuple[int, dist.ProcessGroup | None]:
        if spmd_mesh_size("tp") > 1:
            tp_group = current_spmd_mesh().get_group("tp")  # pyrefly: ignore
            return dist.get_rank(tp_group) * weight_rows, tp_group
        return 0, None

    def _build_candidates(
        self, labels_T: torch.Tensor, *, budget: int, step: int, num_rows: int, vocab_start: int
    ) -> _Candidates:
        """Pick this rank's candidate rows without a host sync.

        Every shape is fixed by ``budget``, so the step is CUDA-graph and compile
        friendly. The budget must cover the microbatch's distinct local targets;
        an async device assert fires otherwise.
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
        if budget <= 0 or budget >= num_rows:
            return _Candidates(
                cand_P=None,
                labels_T=torch.where(in_shard_T, local_T, -1),
                valid_T=valid_T.float(),
                col_bias_P=None,
                num_targets=num_targets,
                num_cols=num_rows,
            )
        torch._assert_async(
            num_targets <= budget,
            "sampled-softmax budget is smaller than the microbatch's distinct targets",
        )
        num_negatives = budget - num_targets

        # Negatives: the first num_negatives non-targets along a stride permutation
        # of the shard, starting at a per-step, per-rank offset.
        if self._stride is None:
            self._stride = _coprime_stride(num_rows)
        rank = dist.get_rank() if dist.is_initialized() else 0
        offset = (step * budget + rank * 7919 * budget) % num_rows
        perm_V = (offset + torch.arange(num_rows, device=device) * self._stride) % num_rows
        nontarget_V = ~is_target_V[perm_V]
        take_V = nontarget_V & (torch.cumsum(nontarget_V, 0) <= num_negatives)
        is_chosen_V = torch.empty_like(is_target_V)
        is_chosen_V.scatter_(0, perm_V, take_V | ~nontarget_V)
        cand_P = torch.nonzero_static(is_chosen_V, size=budget).squeeze(1)
        column_V = torch.cumsum(is_chosen_V, 0) - 1
        labels_local_T = torch.where(
            in_shard_T, column_V[safe_local_T.clamp(max=num_rows - 1)], -1
        )

        col_bias_P = None
        if self.config.correction == "importance":
            # Negatives are a uniform sample of the non-target rows; weighting them by
            # num_nontargets / num_negatives makes sum(exp) estimate the missing mass.
            log_weight = torch.log(
                (num_rows - num_targets).float() / num_negatives.clamp(min=1).float()
            )
            col_bias_P = torch.where(is_target_V[cand_P], 0.0, log_weight).float()
        return _Candidates(
            cand_P=cand_P,
            labels_T=labels_local_T,
            valid_T=valid_T.float(),
            col_bias_P=col_bias_P,
            num_targets=num_targets,
            num_cols=budget,
        )

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
        step = self._num_train_microbatches // self.config.num_microbatches_per_step + 1
        self._num_train_microbatches += 1
        budget = self.budget_at(step)
        if budget == 0 and not self.config.fused_full_softmax:
            self.step_metrics = {"sampled_softmax/budget": 0.0}
            return super().__call__(pred, labels, global_valid_tokens)
        return self._fused_call(pred, labels, global_valid_tokens, budget=budget, step=step)

    def _fused_call(self, pred, labels, global_valid_tokens, *, budget: int, step: int):
        from torch.distributed._composable.fsdp import FSDPModule

        lm_head = self.lm_head
        assert lm_head is not None
        num_chunks = self.config.sampled_num_chunks if budget > 0 else self.num_chunks
        fsdp_enabled = isinstance(lm_head, FSDPModule)
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
            vocab_start, tp_group = self._vocab_shard(num_rows)
            cands = self._build_candidates(
                labels, budget=budget, step=step, num_rows=num_rows, vocab_start=vocab_start
            )

            probe = self.config.full_ce_probe_tokens if budget > 0 else 0
            probe_nll = None
            if probe > 0:
                with torch.no_grad():
                    probe_nll = _SampledHeadCrossEntropy.apply(
                        pred[:probe].detach(), weight.detach(), None,
                        torch.where(
                            (labels[:probe] >= vocab_start) & (labels[:probe] < vocab_start + num_rows),
                            labels[:probe] - vocab_start, -1,
                        ),
                        (labels[:probe] != IGNORE_INDEX).float(), None, tp_group,
                    )

            seq_len = pred.shape[0]
            chunk_len = seq_len // num_chunks
            assert chunk_len * num_chunks == seq_len
            grad_acc = torch.empty(pred.shape, dtype=torch.float32, device=pred.device)
            total_loss = pred.new_zeros((), dtype=torch.float32)
            for i in range(num_chunks):
                if fsdp_enabled and i == num_chunks - 1:
                    lm_head.set_requires_gradient_sync(True, recurse=False)
                sl = slice(i * chunk_len, (i + 1) * chunk_len)
                h_chunk = pred[sl].detach().requires_grad_()
                self._active = (
                    cands.cand_P, cands.labels_T[sl], cands.valid_T[sl], cands.col_bias_P, tp_group
                )
                nll = lm_head(h_chunk)
                self._active = None
                chunk_loss = nll / global_valid_tokens
                total_loss = total_loss + chunk_loss.detach()
                with spmd.no_typecheck():
                    chunk_loss.backward()
                grad_acc[sl] = h_chunk.grad
            if fsdp_enabled:
                lm_head.set_reshard_after_forward(True)
                lm_head.set_reshard_after_backward(True)
                lm_head.reshard()

        self.step_metrics = {
            "sampled_softmax/budget": float(budget),
            "sampled_softmax/num_cols": float(cands.num_cols),
            "sampled_softmax/num_targets": cands.num_targets.float(),
        }
        if probe_nll is not None:
            num_probe = (labels[:probe] != IGNORE_INDEX).sum().clamp(min=1)
            self.step_metrics["sampled_softmax/probe_full_ce"] = probe_nll / num_probe
            self.step_metrics["sampled_softmax/train_ce"] = total_loss.detach() * (
                global_valid_tokens / (labels != IGNORE_INDEX).sum().clamp(min=1)
            )
        with spmd.no_typecheck():
            loss = self._gradient_backprop((pred,), (grad_acc.to(pred.dtype),), total_loss)
        return loss, {}
