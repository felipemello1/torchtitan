# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.experimental import local_map
from torchtitan.config import CompileConfig, Configurable
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def _prepare_labels_for_dtensor_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.distributed.tensor import DTensor, Replicate

    if not isinstance(logits, DTensor):
        return logits, labels

    if all(isinstance(placement, Replicate) for placement in logits.placements):
        return logits.to_local(), labels

    if not isinstance(labels, DTensor):
        labels = DTensor.from_local(
            labels,
            device_mesh=logits.device_mesh,
            placements=tuple(Replicate() for _ in logits.placements),
        )
    return logits, labels


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with sum reduction for token-based normalization."""
    if isinstance(pred, DTensor) and isinstance(labels, DTensor):
        return _cross_entropy_via_local_map(pred, labels)

    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def _cross_entropy_via_local_map(pred: DTensor, labels: DTensor) -> torch.Tensor:
    mesh = pred.device_mesh
    # Labels don't have a vocab dim.
    expected_labels_placements = tuple(
        Replicate() if isinstance(p, Shard) and p.dim == 2 else p
        for p in pred.placements
    )
    if labels.placements != expected_labels_placements:
        raise ValueError(
            f"cross_entropy_loss: expected labels placements {expected_labels_placements}, "
            f"got {labels.placements}"
        )

    # After local flatten(0, 1), tensor dims are [batch*seq, vocab].
    # Per-axis placement:
    #   Shard on batch/seq -> Shard(0) (valid because reduction is sum)
    #   Shard on vocab -> Shard(1)
    def _flatten_placement(p):
        if isinstance(p, Shard):
            return Shard(0 if p.dim == 0 else p.dim - 1)
        return p

    vocab_sharded = any(isinstance(p, Shard) and p.dim == 2 for p in pred.placements)

    # Per-axis output placement for sum reduction:
    #   Shard on non-vocab-dim -> Partial
    #   Shard on vocab-dim -> Replicate
    out_placements = [
        Partial() if isinstance(p, Shard) and p.dim != 2 else Replicate()
        for p in pred.placements
    ]

    @local_map(
        out_placements=out_placements,
        in_placements=(pred.placements, labels.placements),
        in_grad_placements=(pred.placements, labels.placements),
        device_mesh=mesh,
    )
    def _local_cross_entropy(
        pred_local: torch.Tensor, labels_local: torch.Tensor
    ) -> torch.Tensor:
        flat_pred = pred_local.flatten(0, 1).float()
        flat_labels = labels_local.flatten(0, 1)
        if not vocab_sharded:
            return torch.nn.functional.cross_entropy(
                flat_pred,
                flat_labels,
                reduction="sum",
                ignore_index=IGNORE_INDEX,
            )

        # vocab_sharded == True => loss parallel case
        # TODO: rewrite the entire loss parallel using megatron style.
        flat_pred_placements = tuple(_flatten_placement(p) for p in pred.placements)
        flat_labels_placements = tuple(_flatten_placement(p) for p in labels.placements)
        pred_dtensor = DTensor.from_local(
            flat_pred, mesh, flat_pred_placements, run_check=False
        )
        labels_dtensor = DTensor.from_local(
            flat_labels, mesh, flat_labels_placements, run_check=False
        )
        loss_dtensor = torch.nn.functional.cross_entropy(
            pred_dtensor,
            labels_dtensor,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        assert isinstance(loss_dtensor, DTensor)
        return loss_dtensor.to_local()

    return _local_cross_entropy(pred, labels)


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MSE loss with sum reduction for Transformer models training."""
    return torch.nn.functional.mse_loss(
        pred.float(), labels.float().detach(), reduction="sum"
    )


def selected_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Return log probability for each selected label.

    This is the `reduction="none"` equivalent of cross entropy. Ignored labels
    return zero, so downstream masks can be applied without a separate gather
    special case.
    """
    from torch.distributed.tensor import DTensor

    logits, labels = _prepare_labels_for_dtensor_logits(logits, labels)

    losses = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(labels)
    logprobs = -losses
    if isinstance(logprobs, DTensor):
        logprobs = logprobs.to_local()
    return logprobs


def _chunk_slices(seq_len: int, num_chunks: int) -> list[slice]:
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    if seq_len < 1:
        raise ValueError("Cannot chunk an empty sequence")

    quotient, remainder = divmod(seq_len, num_chunks)
    start = 0
    slices = []
    for idx in range(num_chunks):
        chunk_len = quotient + int(idx < remainder)
        if chunk_len == 0:
            continue
        end = start + chunk_len
        slices.append(slice(start, end))
        start = end
    return slices


def _replicate_tp_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor, Replicate

    if not isinstance(hidden_states, DTensor):
        return hidden_states

    mesh = hidden_states.device_mesh
    if mesh.mesh_dim_names is None or "tp" not in mesh.mesh_dim_names:
        return hidden_states

    tp_axis = mesh.mesh_dim_names.index("tp")
    placements = list(hidden_states.placements)
    if isinstance(placements[tp_axis], Replicate):
        return hidden_states

    placements[tp_axis] = Replicate()
    return hidden_states.redistribute(mesh, tuple(placements))


class _FSDPChunkedLossContext:
    """Temporarily keep an FSDP lm_head unsharded across sequence chunks."""

    def __init__(self, lm_head: nn.Module, *, manage_gradient_sync: bool):
        self.lm_head = lm_head
        self.manage_gradient_sync = manage_gradient_sync

    def __enter__(self):
        self.lm_head.set_reshard_after_forward(False)
        self.lm_head.set_reshard_after_backward(False)
        if self.manage_gradient_sync:
            self.lm_head.set_requires_gradient_sync(False, recurse=False)
        return self

    def enable_gradient_sync_for_last_chunk(self) -> None:
        if self.manage_gradient_sync:
            self.lm_head.set_requires_gradient_sync(True, recurse=False)

    def __exit__(self, *args) -> None:
        self.lm_head.set_reshard_after_forward(True)
        self.lm_head.set_reshard_after_backward(True)
        if self.manage_gradient_sync:
            self.lm_head.set_requires_gradient_sync(True, recurse=False)
        self.lm_head.reshard()


def _maybe_fsdp_chunked_loss_context(
    lm_head: nn.Module,
    fsdp_enabled: bool,
    *,
    manage_gradient_sync: bool,
):
    if not fsdp_enabled:
        return nullcontext(None)
    return _FSDPChunkedLossContext(
        lm_head,
        manage_gradient_sync=manage_gradient_sync,
    )


def _chunked_selected_token_logprobs(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    *,
    num_chunks: int = 8,
    ignore_index: int = IGNORE_INDEX,
    selected_logprobs_fn: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute selected-token logprobs without full-sequence logits.

    The returned tensor has shape ``[batch, seq_len]`` and behaves like
    ``selected_token_logprobs(lm_head(hidden_states), labels)``. Backward
    replays ``lm_head`` one sequence chunk at a time.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape [batch, seq, hidden], got "
            f"{tuple(hidden_states.shape)}"
        )
    if labels.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"labels must have shape {tuple(hidden_states.shape[:2])}, got "
            f"{tuple(labels.shape)}"
        )

    hidden_states = _replicate_tp_hidden_states(hidden_states)
    selected_logprobs_fn = selected_logprobs_fn or selected_token_logprobs
    return _ChunkedSelectedTokenLogprobs.apply(
        hidden_states,
        labels,
        lm_head,
        num_chunks,
        ignore_index,
        selected_logprobs_fn,
        *lm_head.parameters(),
    )


def _chunked_token_loss(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    *,
    reducer: Callable[[torch.Tensor, torch.Tensor, slice], torch.Tensor],
    num_chunks: int = 8,
    ignore_index: int = IGNORE_INDEX,
    selected_logprobs_fn: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run a scalar token loss over selected-logprob chunks.

    ``reducer(logprobs, labels, chunk)`` receives one contiguous sequence
    chunk and must return a scalar. The helper immediately backprops each chunk
    through ``lm_head`` and bridges the accumulated hidden-state gradient back
    to the decoder, matching ``ChunkedCELoss``'s memory pattern.
    """
    from torch.distributed._composable.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape [batch, seq, hidden], got "
            f"{tuple(hidden_states.shape)}"
        )
    if labels.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"labels must have shape {tuple(hidden_states.shape[:2])}, got "
            f"{tuple(labels.shape)}"
        )

    hidden_states = _replicate_tp_hidden_states(hidden_states)
    selected_logprobs_fn = selected_logprobs_fn or selected_token_logprobs
    requires_grad = hidden_states.requires_grad
    h_detached = hidden_states.detach().requires_grad_(requires_grad)
    chunks = _chunk_slices(h_detached.shape[1], num_chunks)

    local_hidden_states = (
        hidden_states.to_local()
        if isinstance(hidden_states, DTensor)
        else hidden_states
    )
    grad_accumulator = (
        GradAccumulator(h_detached, num_chunks=len(chunks), dtype=torch.float32)
        if requires_grad
        else None
    )
    total_loss = local_hidden_states.new_zeros((), dtype=torch.float32)

    fsdp_enabled = isinstance(lm_head, FSDPModule)
    last_idx = len(chunks) - 1
    grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
    with _maybe_fsdp_chunked_loss_context(
        lm_head,
        fsdp_enabled,
        manage_gradient_sync=requires_grad,
    ) as fsdp_context, grad_context:
        for idx, chunk in enumerate(chunks):
            if fsdp_context is not None and requires_grad and idx == last_idx:
                fsdp_context.enable_gradient_sync_for_last_chunk()

            h_chunk = (
                h_detached[:, chunk, :]
                .contiguous()
                .detach()
                .requires_grad_(requires_grad)
            )
            label_chunk = labels[:, chunk].contiguous()
            logits = lm_head(h_chunk)
            # TODO(chunked-rl): If RL needs entropy metrics, compute them here
            # under no_grad from the chunk logits instead of carrying entropy
            # through PackedPolicyLossInputs.
            logprobs = selected_logprobs_fn(
                logits,
                label_chunk,
                ignore_index=ignore_index,
            )
            chunk_loss = reducer(logprobs, label_chunk, chunk)
            if chunk_loss.ndim != 0:
                raise ValueError(
                    "reducer must return a scalar tensor, got "
                    f"shape {tuple(chunk_loss.shape)}"
                )
            total_loss = total_loss + chunk_loss.detach()

            if requires_grad:
                chunk_loss.backward()
                assert h_chunk.grad is not None
                assert grad_accumulator is not None
                grad_accumulator.add(h_chunk.grad)
                h_chunk.grad = None

    if not requires_grad:
        return total_loss

    assert grad_accumulator is not None
    return _ChunkedLossWithParamGrads.apply(
        hidden_states,
        grad_accumulator.result().to(hidden_states.dtype),
        total_loss,
        lm_head,
        fsdp_enabled,
        *lm_head.parameters(),
    )


class _ChunkedSelectedTokenLogprobs(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        lm_head: nn.Module,
        num_chunks: int,
        ignore_index: int,
        selected_logprobs_fn: Callable[..., torch.Tensor],
        *lm_params: torch.Tensor,
    ) -> torch.Tensor:
        from torch.distributed._composable.fsdp import FSDPModule

        ctx.save_for_backward(hidden_states, labels)
        ctx.lm_head = lm_head
        ctx.num_chunks = num_chunks
        ctx.ignore_index = ignore_index
        ctx.selected_logprobs_fn = selected_logprobs_fn
        ctx.lm_params = lm_params

        fsdp_enabled = isinstance(lm_head, FSDPModule)
        with _maybe_fsdp_chunked_loss_context(
            lm_head,
            fsdp_enabled,
            manage_gradient_sync=False,
        ):
            logprob_chunks = []
            for chunk in _chunk_slices(hidden_states.shape[1], num_chunks):
                h_chunk = hidden_states[:, chunk, :].contiguous()
                label_chunk = labels[:, chunk].contiguous()
                logits = lm_head(h_chunk)
                logprob_chunks.append(
                    selected_logprobs_fn(
                        logits,
                        label_chunk,
                        ignore_index=ignore_index,
                    )
                )

        return torch.cat(logprob_chunks, dim=1)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        from torch.distributed._composable.fsdp import FSDPModule

        hidden_states, labels = ctx.saved_tensors
        lm_head = ctx.lm_head
        num_chunks = ctx.num_chunks
        ignore_index = ctx.ignore_index
        selected_logprobs_fn = ctx.selected_logprobs_fn
        lm_params = ctx.lm_params
        chunks = _chunk_slices(hidden_states.shape[1], num_chunks)

        fsdp_enabled = isinstance(lm_head, FSDPModule)
        grad_accumulator = GradAccumulator(
            hidden_states,
            num_chunks=len(chunks),
            dtype=torch.float32,
        )

        last_idx = len(chunks) - 1
        with _maybe_fsdp_chunked_loss_context(
            lm_head,
            fsdp_enabled,
            manage_gradient_sync=True,
        ) as fsdp_context, torch.enable_grad():
            for idx, chunk in enumerate(chunks):
                if fsdp_context is not None and idx == last_idx:
                    fsdp_context.enable_gradient_sync_for_last_chunk()

                h_chunk = (
                    hidden_states[:, chunk, :]
                    .contiguous()
                    .detach()
                    .requires_grad_(True)
                )
                label_chunk = labels[:, chunk].contiguous()
                logits = lm_head(h_chunk)
                logprobs = selected_logprobs_fn(
                    logits,
                    label_chunk,
                    ignore_index=ignore_index,
                )
                torch.autograd.backward(
                    logprobs,
                    grad_tensors=grad_output[:, chunk].contiguous(),
                )
                assert h_chunk.grad is not None
                grad_accumulator.add(h_chunk.grad)
                h_chunk.grad = None

        lm_param_grads = _capture_lm_head_param_grads(lm_head, fsdp_enabled, lm_params)
        if fsdp_enabled:
            torch.autograd.Variable._execution_engine.queue_callback(
                lambda: _restore_lm_head_gradient_sync(lm_head)
            )
        return (
            grad_accumulator.result().to(hidden_states.dtype),
            None,
            None,
            None,
            None,
            None,
            *lm_param_grads,
        )


class BaseLoss(ABC, Configurable):
    """Abstract base class for all loss functions.

    Provides compile support and a unified ``__call__`` signature:
    ``(pred, labels, global_valid_tokens) -> scaled_loss``.
    Subclasses must implement ``__init__`` and set ``self.fn``.
    """

    fn: LossFunction

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        ...

    def _maybe_compile(self, compile_config: CompileConfig | None) -> None:
        if (
            compile_config is not None
            and compile_config.enable
            and "loss" in compile_config.components
        ):
            logger.info("Compiling the loss function with torch.compile")
            self.fn = torch.compile(self.fn, backend=compile_config.backend)

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.fn(pred, labels)
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        return loss


class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss with sum reduction for token-based normalization."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)


class MSELoss(BaseLoss):
    """MSE loss with sum reduction for Transformer models training (e.g. Flux)."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.fn: LossFunction = mse_loss
        self._maybe_compile(compile_config)


class GradAccumulator:
    """Accumulates chunk gradients into a pre-allocated buffer.

    Instead of collecting chunk gradients in a list and concatenating at the end,
    this uses a pre-allocated buffer with in-place copies for better memory efficiency.

    Args:
        reference: Reference tensor to derive shape, device, and DTensor metadata.
            If a DTensor, result() returns a DTensor with matching placements.
        num_chunks: Number of chunks that will be added.
        seq_dim: The sequence dimension along which chunks are accumulated.
        dtype: Dtype for the buffer.

    Usage:
        accumulator = GradAccumulator(hidden_states, num_chunks=4, dtype=torch.float32)
        for chunk_grad in chunk_grads:
            accumulator.add(chunk_grad)
        full_grad = accumulator.result()
    """

    def __init__(
        self,
        reference: torch.Tensor,
        *,
        num_chunks: int,
        seq_dim: int = 1,
        dtype: torch.dtype,
    ):
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor, Placement

        self.num_chunks = num_chunks
        self.seq_dim = seq_dim
        self._next_idx = 0
        self._next_start = 0
        self._device_mesh: DeviceMesh | None = None
        self._placements: tuple[Placement, ...] | None = None

        # Track DTensor metadata for transparent wrap-back in result()
        if isinstance(reference, DTensor):
            self._device_mesh = reference.device_mesh
            self._placements = reference.placements
            local = reference.to_local()
        else:
            local = reference

        self._buffer = torch.zeros_like(local, dtype=dtype)

    def add(self, chunk_grad: torch.Tensor) -> None:
        """Add the next chunk gradient sequentially.

        Chunks must be added in order (0, 1, 2, ..., num_chunks - 1).
        """
        from torch.distributed.tensor import DTensor

        if self._next_idx >= self.num_chunks:
            raise ValueError(f"Already added {self.num_chunks} chunks, cannot add more")

        # Extract local tensor if DTensor
        if isinstance(chunk_grad, DTensor):
            if (
                self._device_mesh is not None
                and self._placements is not None
                and chunk_grad.placements != self._placements
            ):
                chunk_grad = chunk_grad.redistribute(
                    self._device_mesh,
                    self._placements,
                )
            chunk_grad = chunk_grad.to_local()

        if chunk_grad.dtype != self._buffer.dtype:
            chunk_grad = chunk_grad.to(self._buffer.dtype)

        chunk_seq_len = chunk_grad.shape[self.seq_dim]
        start = self._next_start
        end = start + chunk_seq_len

        slices = [slice(None)] * self._buffer.ndim
        slices[self.seq_dim] = slice(start, end)
        self._buffer[tuple(slices)] = chunk_grad

        self._next_start = end
        self._next_idx += 1

    def result(self) -> torch.Tensor:
        """Return the accumulated gradient tensor, wrapped as DTensor if needed."""
        from torch.distributed.tensor import DTensor

        if self._device_mesh is not None:
            return DTensor.from_local(
                self._buffer,
                device_mesh=self._device_mesh,
                placements=self._placements,
            )
        return self._buffer


class ChunkedCELoss(BaseLoss):
    """Chunked cross-entropy loss that splits the sequence dimension to reduce peak memory.

    Instead of materializing the full [B, L, V] logits tensor at once, this splits
    the hidden states into N chunks along the sequence dimension and computes
    lm_head + cross_entropy_loss on each chunk sequentially. This reduces peak memory
    from O(B*L*V) to O(B*L/N*V).

    The flow:
    1. Model forward with _skip_lm_head=True to get hidden states [B, L, D]
    2. Detach hidden states at the boundary
    3. Split detached hidden states into N chunks along seq dim
    4. Disable FSDP reshard on lm_head to keep weight unsharded across chunks
    5. For each chunk: lm_head(chunk) -> ce_loss -> backward()
    6. Assemble chunk gradients into a full gradient [B, L, D] via GradAccumulator
    7. Backward through the decoder via hidden_states.backward(accumulated_grad)

    FSDP2 composability:
        The lm_head's FSDP reshard-after-forward and reshard-after-backward are
        temporarily disabled during the chunked loop so that the weight stays
        unsharded across all chunks (avoiding repeated all-gathers). Reduce-scatter
        fires per-chunk, and FSDP2 accumulates the sharded gradients correctly.

    TP / SP composability:
        Hidden states are redistributed to ``Replicate()`` on the TP mesh
        before chunking, so each chunk enters the lm_head as ``Replicate()``
        input regardless of whether SP is enabled. With SP, this is an
        all-gather from ``Shard(1)``; without SP, it's a no-op.

        When loss parallel is applied, each TP rank
        computes partial CE on its ``V/tp`` slice, with an internal
        all-reduce for the correct log-sum-exp.

    CP: Further chunks the local sequence dimension. Works out of the box.

    Compile: ce_loss can be compiled independently; lm_head is not compiled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        # TODO(chunked-rl): Consider a max_tokens_per_chunk knob after the
        # reducer API lands. Keeping a single chunk-count knob makes this
        # review match current TorchTitan behavior.
        num_chunks: int = 8
        """Number of chunks to split the sequence into."""

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)
        self._selected_logprobs_fn: Callable[
            ..., torch.Tensor
        ] = selected_token_logprobs
        if (
            compile_config is not None
            and compile_config.enable
            and "loss" in compile_config.components
        ):
            self._selected_logprobs_fn = torch.compile(
                selected_token_logprobs,
                backend=compile_config.backend,
            )
        self.num_chunks = config.num_chunks
        self.lm_head: nn.Module | None = None

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Set the lm_head module. Must be called before the first __call__."""
        self.lm_head = lm_head

    def _require_lm_head(self) -> nn.Module:
        if self.lm_head is None:
            raise ValueError("Set lm_head before calling ChunkedCELoss")
        return self.lm_head

    def selected_token_logprobs(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        *,
        ignore_index: int = IGNORE_INDEX,
    ) -> torch.Tensor:
        """Return selected-token logprobs with the same chunking policy as CE."""
        return _chunked_selected_token_logprobs(
            pred,
            self._require_lm_head(),
            labels,
            num_chunks=self.num_chunks,
            ignore_index=ignore_index,
            selected_logprobs_fn=self._selected_logprobs_fn,
        )

    def reduce_selected_token_logprobs(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        reducer: Callable[[torch.Tensor, torch.Tensor, slice], torch.Tensor],
        *,
        ignore_index: int = IGNORE_INDEX,
    ) -> torch.Tensor:
        """Run a scalar reducer over selected-token logprobs one chunk at a time."""
        return _chunked_token_loss(
            pred,
            self._require_lm_head(),
            labels,
            reducer=reducer,
            num_chunks=self.num_chunks,
            ignore_index=ignore_index,
            selected_logprobs_fn=self._selected_logprobs_fn,
        )

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute chunked cross-entropy loss.

        ``pred`` should be hidden states from model forward with
        ``_skip_lm_head=True``.

        When ``pred`` does not require grad (e.g. validation), runs chunked
        forward only — no per-chunk backward or gradient accumulation.

        Returns a differentiable loss. When ``.backward()`` is called on it
        (either by the trainer or the PP schedule), it triggers backward
        through the decoder via a custom autograd Function.
        """
        from torch.distributed._composable.fsdp import FSDPModule

        hidden_states = pred
        num_chunks = self.num_chunks
        lm_head = self._require_lm_head()
        fsdp_enabled = isinstance(lm_head, FSDPModule)

        # If SP is enabled, hidden states are Shard(1) on the TP mesh dim.
        # Redistribute only the TP dim to Replicate before chunking so that
        # the lm_head receives Replicate input on TP.
        if isinstance(hidden_states, DTensor):
            mesh = hidden_states.device_mesh
            if mesh.mesh_dim_names is not None and "tp" in mesh.mesh_dim_names:
                tp_dim = mesh.mesh_dim_names.index("tp")
                placements = list(hidden_states.placements)
                if not isinstance(placements[tp_dim], Replicate):
                    placements[tp_dim] = Replicate()
                    hidden_states = hidden_states.redistribute(mesh, tuple(placements))

        # Check if it's training model or validation mode
        requires_grad = hidden_states.requires_grad

        # Chunking always operates on the *local* view: when ``t`` is a
        # Shard(1) DTensor, chunking the global view would distribute whole
        # chunks across ranks (e.g. size=2, num_chunks=8: chunks 0-3 on
        # rank 0, 4-7 on rank 1), leaving half the per-chunk DTensors with
        # local seq=0 and breaking GradAccumulator's slice writes.
        # ``local_map`` runs the chunking body on plain tensors; under the
        # non-DTensor (eager) path we call ``_chunk_local`` directly.
        # ``.contiguous()`` breaks shared storage from ``torch.chunk``.
        def _chunk_local(t):
            return tuple(c.contiguous() for c in torch.chunk(t, num_chunks, dim=1))

        def _chunk(t):
            if not isinstance(t, DTensor):
                return _chunk_local(t)
            p = t.placements
            wrapped = local_map(
                _chunk_local,
                out_placements=(p,) * num_chunks,
                in_placements=(p,),
                device_mesh=t.device_mesh,
            )
            return wrapped(t)

        # ``detach`` + ``requires_grad_`` makes each chunk a leaf so it
        # accumulates ``.grad`` for ``GradAccumulator``.
        h_chunks = [
            c.detach().requires_grad_(requires_grad) for c in _chunk(hidden_states)
        ]
        label_chunks = list(_chunk(labels))

        grad_accumulator = GradAccumulator(
            hidden_states,
            num_chunks=num_chunks,
            dtype=torch.float32,
        )

        total_loss = hidden_states.new_zeros((), dtype=torch.float32)

        # Disable FSDP reshard on lm_head to keep weight unsharded across
        # all chunks, avoiding repeated all-gathers. Coalesce per-chunk
        # grad sync into a single reduce-scatter at the last chunk by
        # disabling gradient sync for chunks 0..N-2.
        if fsdp_enabled:
            lm_head.set_reshard_after_forward(False)
            lm_head.set_reshard_after_backward(False)
            lm_head.set_requires_gradient_sync(False, recurse=False)

        last_idx = len(h_chunks) - 1
        for i, (h_chunk, label_chunk) in enumerate(zip(h_chunks, label_chunks)):
            if fsdp_enabled and i == last_idx:
                lm_head.set_requires_gradient_sync(  # pyrefly: ignore[not-callable]
                    True, recurse=False
                )

            logits = lm_head(h_chunk)

            chunk_loss = self.fn(logits, label_chunk)
            if global_valid_tokens is not None:
                chunk_loss = chunk_loss / global_valid_tokens
            total_loss = total_loss + chunk_loss.detach()

            if requires_grad:
                chunk_loss.backward()
                assert h_chunk.grad is not None
                grad_accumulator.add(h_chunk.grad)
                h_chunk.grad = None

        if fsdp_enabled:
            lm_head.set_reshard_after_forward(True)
            lm_head.set_reshard_after_backward(True)
            lm_head.set_requires_gradient_sync(True, recurse=False)
            lm_head.reshard()
        if not requires_grad:
            return total_loss

        accumulated_grad = grad_accumulator.result().to(hidden_states.dtype)

        return self._gradient_backprop(
            hidden_states, accumulated_grad, total_loss, lm_head, fsdp_enabled
        )

    @staticmethod
    def _gradient_backprop(
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        total_loss: torch.Tensor,
        lm_head: nn.Module,
        fsdp_enabled: bool,
    ) -> torch.Tensor:
        """Return a differentiable loss via _DecoderOutputGradientBackProp.
        When ``.backward()`` is called (by the trainer or PP schedule),
        autograd calls ``_DecoderOutputGradientBackProp.backward`` which
        returns ``accumulated_grad`` as the gradient for ``hidden_states``,
        propagating through the decoder. Subclasses override to swap in a
        different autograd Function.
        """
        # TODO(chunked-rl): Move the existing SFT path to
        # _ChunkedLossWithParamGrads after separate SFT parity tests. The new
        # reducer paths already use that bridge so external loss scaling is
        # preserved there.
        return _DecoderOutputGradientBackProp.apply(
            hidden_states, accumulated_grad, total_loss
        )


def _grad_output_scale_tensor(
    grad_output: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor, Replicate

    if isinstance(grad_output, DTensor):
        grad_output = grad_output.redistribute(
            grad_output.device_mesh,
            tuple(Replicate() for _ in grad_output.placements),
        ).to_local()
    if grad_output.numel() != 1:
        raise RuntimeError(
            "Chunked loss bridge only supports scalar grad_output, got "
            f"shape {tuple(grad_output.shape)}"
        )
    if isinstance(grad, DTensor):
        local_grad = grad.to_local()
        local_scale = grad_output.to(
            device=local_grad.device,
            dtype=local_grad.dtype,
        )
        return DTensor.from_local(
            local_scale,
            device_mesh=grad.device_mesh,
            placements=tuple(Replicate() for _ in grad.placements),
        )
    return grad_output.to(device=grad.device, dtype=grad.dtype)


def _scale_grad_by_output(
    grad: torch.Tensor | None,
    grad_output: torch.Tensor,
) -> torch.Tensor | None:
    if grad is None:
        return None
    return grad * _grad_output_scale_tensor(grad_output, grad)


def _capture_lm_head_param_grads(
    lm_head: nn.Module,
    fsdp_enabled: bool,
    lm_params: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor | None, ...]:
    """Move chunk-loop parameter grads from side effects into autograd."""
    param_grads: list[torch.Tensor | None] = []
    for param in lm_params:
        if param.grad is None:
            param_grads.append(None)
        else:
            param_grads.append(param.grad.detach())
            param.grad = None

    if fsdp_enabled:
        # The returned grads are already in the placement produced by the
        # chunk-loop's final FSDP reduction. Keep FSDP from reducing them again
        # when AccumulateGrad receives this Function's outputs.
        lm_head.set_requires_gradient_sync(False, recurse=False)

    return tuple(param_grads)


def _restore_lm_head_gradient_sync(lm_head: nn.Module) -> None:
    lm_head.set_requires_gradient_sync(True, recurse=False)


class _ChunkedLossWithParamGrads(torch.autograd.Function):
    """Bridge chunked hidden-state and lm_head parameter grads into autograd."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        accumulated_h_grad: torch.Tensor,
        total_loss: torch.Tensor,
        lm_head: nn.Module,
        fsdp_enabled: bool,
        *lm_params: torch.Tensor,
    ) -> torch.Tensor:
        param_grads = _capture_lm_head_param_grads(
            lm_head,
            fsdp_enabled,
            lm_params,
        )
        ctx.param_grad_is_none = tuple(grad is None for grad in param_grads)
        ctx.save_for_backward(
            accumulated_h_grad,
            *(grad for grad in param_grads if grad is not None),
        )
        ctx.lm_head = lm_head
        ctx.fsdp_enabled = fsdp_enabled
        return total_loss.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pyrefly: ignore[bad-override]
        saved = ctx.saved_tensors
        accumulated_h_grad = saved[0]
        saved_param_grads = iter(saved[1:])
        param_grads = tuple(
            None if is_none else next(saved_param_grads)
            for is_none in ctx.param_grad_is_none
        )
        if ctx.fsdp_enabled:
            torch.autograd.Variable._execution_engine.queue_callback(
                lambda: _restore_lm_head_gradient_sync(ctx.lm_head)
            )
        return (
            _scale_grad_by_output(accumulated_h_grad, grad_output),
            None,
            None,
            None,
            None,
            *(
                _scale_grad_by_output(param_grad, grad_output)
                for param_grad in param_grads
            ),
        )


class _DecoderOutputGradientBackProp(torch.autograd.Function):
    """Bridges chunked lm_head backward with decoder backward via autograd.

    Forward takes hidden_states (connected to decoder graph), the accumulated
    gradient from chunked lm_head backward, and the loss value. Returns a
    detached loss with this Function as its grad_fn.

    Backward returns accumulated_grad as the gradient for hidden_states.
    Autograd then propagates this through the decoder layers automatically —
    no explicit hidden_states.backward() needed.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        accumulated_grad: torch.Tensor,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(accumulated_grad)
        return loss.detach()

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (accumulated_grad,) = ctx.saved_tensors
        # Return accumulated_grad as the gradient for hidden_states.
        # Autograd then propagates this through hidden_states' existing
        # decoder graph — equivalent to hidden_states.backward(accumulated_grad)
        # but expressed as a return value so autograd handles the traversal
        # in a single pass (no "backward through graph twice" error).
        # Note: this is not safe if downstream accidentally runs tensor ops after
        # the loss returns, which would produce a non-trivial grad_output that we need
        # to properly handle. The complicated part is that grad_output might not be
        # on the same device mesh as accumlated_grad.
        return accumulated_grad, None, None
