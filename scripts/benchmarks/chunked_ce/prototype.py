# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone prototype for the chunked CE reducer design.

* ``reduce_selected_token_logprobs`` streams chunk-local selected logprobs into
  a scalar reducer. This is the GRPO-style path.
* ``compute_selected_token_logprobs`` returns a real ``[batch, seq]`` tensor for
  sequence-wise losses. This is the GSPO-style path.

Run ``python prototype.py``. The smoke is CPU-only and does not validate
DTensor, FSDP, TP, or loss-parallel behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


def selected_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Selected label logprobs via ``F.cross_entropy(reduction="none")``."""

    losses = F.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(labels)
    return -losses


def _chunk_slices(seq_len: int, num_chunks: int) -> list[slice]:
    """Split ``seq_len`` into at most ``num_chunks`` non-empty slices."""

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")

    chunks = min(seq_len, num_chunks)
    base = seq_len // chunks
    extra = seq_len % chunks
    slices: list[slice] = []
    start = 0
    for idx in range(chunks):
        length = base + (1 if idx < extra else 0)
        end = start + length
        slices.append(slice(start, end))
        start = end
    return slices


def _valid_mask(labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    return labels.ne(ignore_index)


def _clear_param_grads(params: list[nn.Parameter]) -> None:
    for param in params:
        param.grad = None


class GradAccumulator:
    """Accumulates per-chunk hidden-state gradients into ``[batch, seq, hidden]``.

    The integrated TorchTitan version additionally handles DTensor placement:
        if isinstance(chunk_grad, DTensor):
            chunk_grad = chunk_grad.redistribute(device_mesh, placements)
            chunk_grad = chunk_grad.to_local()
    """

    def __init__(self, target: torch.Tensor) -> None:
        self.buffer = torch.zeros_like(target)
        self._next_start = 0

    def add(self, chunk_grad: torch.Tensor) -> None:
        """Write the next chunk gradient at its cumulative sequence offset."""

        chunk_seq_len = chunk_grad.shape[1]
        start = self._next_start
        end = start + chunk_seq_len
        if end > self.buffer.shape[1]:
            raise ValueError("received more chunk gradient tokens than expected")
        self.buffer[:, start:end, :] = chunk_grad
        self._next_start = end

    def finish(self) -> torch.Tensor:
        if self._next_start != self.buffer.shape[1]:
            raise ValueError(
                f"missing gradients for {self.buffer.shape[1] - self._next_start} tokens"
            )
        return self.buffer


class _ChunkedLossWithParamGrads(torch.autograd.Function):
    """Connect inner chunk backprops to the caller's outer autograd graph.

    The reducer path calls ``chunk_loss.backward()`` inside the chunk loop. This
    bridge captures hidden and ``lm_head`` parameter grads, clears
    ``param.grad``, and returns those grads from its own backward so external
    scaling like ``(0.125 * loss).backward()`` is applied exactly once.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden_states: torch.Tensor,
        accumulated_hidden_grad: torch.Tensor,
        total_loss: torch.Tensor,
        hidden_requires_grad: bool,
        *params: nn.Parameter,
    ) -> torch.Tensor:
        ctx.hidden_requires_grad = hidden_requires_grad
        ctx.param_grads = tuple(
            None if param.grad is None else param.grad.detach().clone()
            for param in params
        )
        ctx.save_for_backward(accumulated_hidden_grad.detach())
        for param in params:
            param.grad = None
        return total_loss.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (accumulated_hidden_grad,) = ctx.saved_tensors
        hidden_grad = (
            accumulated_hidden_grad * grad_output if ctx.hidden_requires_grad else None
        )
        param_grads = tuple(
            None if grad is None else grad * grad_output for grad in ctx.param_grads
        )
        return hidden_grad, None, None, None, *param_grads


class _ChunkedSelectedTokenLogprobs(torch.autograd.Function):
    """Autograd tensor path for sequence-wise losses.

    Forward returns ``[batch, seq]`` selected logprobs. Backward replays
    ``lm_head`` one chunk at a time. Under FSDP this can trigger another
    ``lm_head`` unshard/all-gather, so token-local losses should prefer the
    reducer path.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        lm_head: nn.Module,
        num_chunks: int,
        ignore_index: int,
        *params: nn.Parameter,
    ) -> torch.Tensor:
        ctx.lm_head = lm_head
        ctx.num_chunks = num_chunks
        ctx.ignore_index = ignore_index
        ctx.save_for_backward(hidden_states.detach(), labels)

        logprob_chunks = []
        for chunk in _chunk_slices(hidden_states.shape[1], num_chunks):
            logits = lm_head(hidden_states[:, chunk, :].contiguous())
            logprob_chunks.append(
                selected_token_logprobs(
                    logits,
                    labels[:, chunk].contiguous(),
                    ignore_index=ignore_index,
                )
            )
        return torch.cat(logprob_chunks, dim=1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        hidden_states, labels = ctx.saved_tensors
        lm_head: nn.Module = ctx.lm_head
        params = list(lm_head.parameters())
        _clear_param_grads(params)

        grad_accumulator = GradAccumulator(hidden_states)
        with torch.enable_grad():
            for chunk in _chunk_slices(hidden_states.shape[1], ctx.num_chunks):
                h_chunk = (
                    hidden_states[:, chunk, :]
                    .contiguous()
                    .detach()
                    .requires_grad_(True)
                )
                label_chunk = labels[:, chunk].contiguous()
                logits = lm_head(h_chunk)
                logprobs = selected_token_logprobs(
                    logits,
                    label_chunk,
                    ignore_index=ctx.ignore_index,
                )
                torch.autograd.backward(
                    logprobs,
                    grad_tensors=grad_output[:, chunk].contiguous(),
                )
                grad_accumulator.add(h_chunk.grad.detach())

        param_grads = tuple(
            None if param.grad is None else param.grad.detach().clone()
            for param in params
        )
        _clear_param_grads(params)
        return grad_accumulator.finish(), None, None, None, None, *param_grads


class ChunkedCELoss:
    """Small standalone version of the proposed TorchTitan API.

    The integrated TorchTitan version uses ``ChunkedCELoss(config)`` plus
    ``set_lm_head(lm_head)`` because the trainer builds the loss before the
    model head is available. This prototype takes ``lm_head`` in ``__init__``
    to keep the smoke self-contained.
    """

    def __init__(self, lm_head: nn.Module, *, num_chunks: int = 8) -> None:
        self.lm_head = lm_head
        self.num_chunks = num_chunks

    def __call__(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        *,
        ignore_index: int = IGNORE_INDEX,
    ) -> torch.Tensor:
        """Scalar SFT CE path, implemented through the reducer primitive."""

        def sft_reducer(
            logprobs_chunk: torch.Tensor,
            labels_chunk: torch.Tensor,
            token_slice: slice,
        ) -> torch.Tensor:
            del token_slice
            valid = _valid_mask(labels_chunk, ignore_index)
            # Prototype uses sum to match dense_sft_loss below. Real Titan SFT
            # normalizes by global valid-token count.
            return -logprobs_chunk[valid].sum()

        return self.reduce_selected_token_logprobs(
            hidden_states,
            labels,
            sft_reducer,
            ignore_index=ignore_index,
        )

    def reduce_selected_token_logprobs(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        reducer: Callable[[torch.Tensor, torch.Tensor, slice], torch.Tensor],
        *,
        ignore_index: int = IGNORE_INDEX,
    ) -> torch.Tensor:
        """Stream chunk-local selected logprobs into a scalar reducer."""

        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must be [batch, seq, hidden], got {hidden_states.shape}"
            )
        if labels.shape != hidden_states.shape[:2]:
            raise ValueError(
                f"labels must be {hidden_states.shape[:2]}, got {labels.shape}"
            )

        params = list(self.lm_head.parameters())
        _clear_param_grads(params)
        hidden_requires_grad = hidden_states.requires_grad
        needs_backward = hidden_requires_grad or any(
            param.requires_grad for param in params
        )

        grad_accumulator = GradAccumulator(hidden_states)
        total_loss = hidden_states.new_zeros(())

        for chunk in _chunk_slices(hidden_states.shape[1], self.num_chunks):
            h_chunk = (
                hidden_states[:, chunk, :]
                .contiguous()
                .detach()
                .requires_grad_(hidden_requires_grad)
            )
            label_chunk = labels[:, chunk].contiguous()
            logits = self.lm_head(h_chunk)
            logprobs = selected_token_logprobs(
                logits,
                label_chunk,
                ignore_index=ignore_index,
            )
            chunk_loss = reducer(logprobs, label_chunk, chunk)
            total_loss = total_loss + chunk_loss.detach()
            if needs_backward:
                chunk_loss.backward()
                if hidden_requires_grad:
                    grad_accumulator.add(h_chunk.grad.detach())

        accumulated_hidden_grad = (
            grad_accumulator.finish()
            if hidden_requires_grad
            else torch.zeros_like(hidden_states)
        )
        return _ChunkedLossWithParamGrads.apply(
            hidden_states,
            accumulated_hidden_grad,
            total_loss,
            hidden_requires_grad,
            *params,
        )

    def compute_selected_token_logprobs(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        *,
        ignore_index: int = IGNORE_INDEX,
    ) -> torch.Tensor:
        """Return ``[batch, seq]`` selected logprobs for sequence-wise losses."""

        params = tuple(self.lm_head.parameters())
        return _ChunkedSelectedTokenLogprobs.apply(
            hidden_states,
            labels,
            self.lm_head,
            self.num_chunks,
            ignore_index,
            *params,
        )


@dataclass(frozen=True)
class PackedPolicyLossInputs:
    """Side tensors aligned to ``labels`` for token-local RL reducers."""

    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor


class GRPOTokenReducer:
    """Token-local PPO/GRPO clipping reducer."""

    def __init__(self, inputs: PackedPolicyLossInputs, *, clip_eps: float) -> None:
        self.inputs = inputs
        self.clip_eps = clip_eps

    def __call__(
        self,
        policy_logprobs: torch.Tensor,
        labels_chunk: torch.Tensor,
        token_slice: slice,
    ) -> torch.Tensor:
        del labels_chunk
        old = self.inputs.old_logprobs[:, token_slice]
        adv = self.inputs.advantages[:, token_slice]
        weights = self.inputs.weights[:, token_slice]
        ratio = torch.exp(policy_logprobs - old)
        clipped_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
        token_loss = -torch.minimum(ratio * adv, clipped_ratio * adv)
        return (token_loss * weights).sum()


def clone_linear(module: nn.Linear) -> nn.Linear:
    clone = nn.Linear(
        module.in_features, module.out_features, bias=module.bias is not None
    )
    clone.load_state_dict(module.state_dict())
    return clone


def assert_allclose(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Tight fp32 tolerances for this CPU smoke; bf16 distributed tests need looser bars.
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6, msg=name)


def dense_sft_loss(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logprobs = selected_token_logprobs(lm_head(hidden_states), labels)
    return -logprobs[_valid_mask(labels)].sum()


def dense_grpo_loss(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    inputs: PackedPolicyLossInputs,
    *,
    clip_eps: float,
) -> torch.Tensor:
    policy_logprobs = selected_token_logprobs(lm_head(hidden_states), labels)
    return GRPOTokenReducer(inputs, clip_eps=clip_eps)(
        policy_logprobs,
        labels,
        slice(0, labels.shape[1]),
    )


def sequence_loss_from_selected_logprobs(
    selected_logprobs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Simple sequence-wise loss that needs all tokens before reducing."""

    valid = _valid_mask(labels)
    lengths = valid.sum(dim=1).clamp(min=1)
    per_sequence_mean = (selected_logprobs * valid).sum(dim=1) / lengths
    return -per_sequence_mean.sum()


def compare_grads(
    prefix: str,
    dense_hidden: torch.Tensor,
    chunked_hidden: torch.Tensor,
    dense_head: nn.Linear,
    chunked_head: nn.Linear,
) -> None:
    assert_allclose(f"{prefix}: hidden grad", chunked_hidden.grad, dense_hidden.grad)
    assert_allclose(
        f"{prefix}: weight grad",
        chunked_head.weight.grad,
        dense_head.weight.grad,
    )
    if dense_head.bias is not None:
        assert_allclose(
            f"{prefix}: bias grad", chunked_head.bias.grad, dense_head.bias.grad
        )


def make_inputs() -> tuple[nn.Linear, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    batch, seq_len, hidden, vocab = 2, 17, 8, 32
    lm_head = nn.Linear(hidden, vocab)
    hidden_states = torch.randn(batch, seq_len, hidden)
    labels = torch.randint(0, vocab, (batch, seq_len))
    labels[0, 1] = IGNORE_INDEX
    labels[1, 7] = IGNORE_INDEX
    return lm_head, hidden_states, labels


def run_sft_smoke() -> None:
    lm_head, hidden_base, labels = make_inputs()
    dense_head = clone_linear(lm_head)
    chunked_head = clone_linear(lm_head)
    dense_hidden = hidden_base.detach().clone().requires_grad_(True)
    chunked_hidden = hidden_base.detach().clone().requires_grad_(True)

    dense_loss = dense_sft_loss(dense_head, dense_hidden, labels)
    chunked_loss = ChunkedCELoss(chunked_head, num_chunks=5)(chunked_hidden, labels)
    assert_allclose("sft loss", chunked_loss, dense_loss)

    scale = torch.tensor(0.125)
    (dense_loss * scale).backward()
    (chunked_loss * scale).backward()
    compare_grads("sft", dense_hidden, chunked_hidden, dense_head, chunked_head)


def run_grpo_smoke() -> None:
    lm_head, hidden_base, labels = make_inputs()
    dense_head = clone_linear(lm_head)
    chunked_head = clone_linear(lm_head)
    dense_hidden = hidden_base.detach().clone().requires_grad_(True)
    chunked_hidden = hidden_base.detach().clone().requires_grad_(True)

    torch.manual_seed(13)
    valid = _valid_mask(labels).float()
    inputs = PackedPolicyLossInputs(
        old_logprobs=torch.randn_like(labels.float()) * 0.1,
        advantages=torch.randn_like(labels.float()),
        weights=valid / valid.sum().clamp(min=1.0),
    )

    dense_loss = dense_grpo_loss(
        dense_head,
        dense_hidden,
        labels,
        inputs,
        clip_eps=0.2,
    )
    reducer = GRPOTokenReducer(inputs, clip_eps=0.2)
    chunked_loss = ChunkedCELoss(
        chunked_head, num_chunks=5
    ).reduce_selected_token_logprobs(
        chunked_hidden,
        labels,
        reducer,
    )
    assert_allclose("grpo loss", chunked_loss, dense_loss)

    scale = torch.tensor(0.25)
    (dense_loss * scale).backward()
    (chunked_loss * scale).backward()
    compare_grads("grpo", dense_hidden, chunked_hidden, dense_head, chunked_head)


def run_selected_logprob_smoke() -> None:
    lm_head, hidden_base, labels = make_inputs()
    dense_head = clone_linear(lm_head)
    chunked_head = clone_linear(lm_head)
    dense_hidden = hidden_base.detach().clone().requires_grad_(True)
    chunked_hidden = hidden_base.detach().clone().requires_grad_(True)

    dense_logprobs = selected_token_logprobs(dense_head(dense_hidden), labels)
    chunked_logprobs = ChunkedCELoss(
        chunked_head,
        num_chunks=5,
    ).compute_selected_token_logprobs(chunked_hidden, labels)
    assert_allclose("selected logprobs", chunked_logprobs, dense_logprobs)

    dense_loss = sequence_loss_from_selected_logprobs(dense_logprobs, labels)
    chunked_loss = sequence_loss_from_selected_logprobs(chunked_logprobs, labels)
    assert_allclose("sequence loss", chunked_loss, dense_loss)
    scale = torch.tensor(0.5)
    (dense_loss * scale).backward()
    (chunked_loss * scale).backward()
    compare_grads(
        "selected path", dense_hidden, chunked_hidden, dense_head, chunked_head
    )


def run_grad_accumulator_smoke() -> None:
    target = torch.zeros(1, 10, 1)
    accumulator = GradAccumulator(target)
    for idx, chunk in enumerate(_chunk_slices(seq_len=10, num_chunks=8)):
        grad = torch.full((1, chunk.stop - chunk.start, 1), float(idx + 1))
        accumulator.add(grad)
    actual = accumulator.finish().flatten()
    expected = torch.tensor([1, 1, 2, 2, 3, 4, 5, 6, 7, 8], dtype=actual.dtype)
    assert_allclose("uneven GradAccumulator offsets", actual, expected)


def main() -> None:
    run_sft_smoke()
    run_grpo_smoke()
    run_selected_logprob_smoke()
    run_grad_accumulator_smoke()
    print("OK: chunked CE reducer prototype smoke passed")


if __name__ == "__main__":
    main()
