# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer-state CPU offload."""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

__all__ = ["OptimizerStateOffloadConfig", "OptimizerStateOffloader"]

_MOMENT_KEYS = ("exp_avg", "exp_avg_sq")


@dataclass(kw_only=True, slots=True)
class OptimizerStateOffloadConfig:
    """Configure streamed optimizer-state CPU offload.

    Args:
        chunk_size_mb: Target MiB of moments moved per H2D/D2H chunk. Two chunks
            are resident on the GPU during ``step()``.

    Example:

        config = OptimizerStateOffloadConfig(chunk_size_mb=256)
    """

    chunk_size_mb: int = 256

    def __post_init__(self) -> None:
        if self.chunk_size_mb <= 0:
            raise ValueError("chunk_size_mb must be positive")


class OptimizerStateOffloader(Optimizer):
    """Run Adam/AdamW while keeping its moment tensors in CPU memory.

    Adam keeps a first and second moment for every weight. In fp32 these add 8
    bytes per weight, or about 112 GB for a 14B model before distributed
    sharding. Forward and backward do not use these moments; only
    ``optimizer.step()`` does.

    Definitions:
        Pinned memory: CPU RAM that CUDA can transfer asynchronously.
        Chunk: One or more whole local parameter shards grouped by moment size.
            A chunk can contain parameters from several layers or only part of
            one layer, but a single parameter shard is never split.
        Slot: Temporary GPU memory for one chunk's first and second moments.
        NUMA locality: Placing pinned RAM near the GPU's CPU socket so transfers
            do not cross CPU sockets.

    The offloader allocates contiguous pinned CPU buffers for the moments and
    records each parameter's slice. Before updating a chunk, it places
    parameter-shaped views of the GPU slot in ``optimizer.state`` so Adam sees
    the moments that belong to each weight. During ``step()``, it alternates
    chunks between two GPU slots::

                       warm-up | iteration 0       | iteration 1       | iteration 2
        GPU compute:           | update 0 in A     | update 1 in B     | update 2 in A
        slot A copy:   load 0  |                   | store 0 -> load 2 |
        slot B copy:           | load 1            |                   | store 1 -> load 3

    While one slot is updated, the other returns its previous chunk to the CPU
    and then loads its next chunk. Separate CUDA streams overlap these copies
    with the update; events prevent a slot from being reused too early.

    CPU offload does not fundamentally require a fused optimizer. This wrapper
    was designed and tested around one fused Adam/AdamW update per chunk.
    Supporting the for-loop or ``foreach`` paths is possible, but their
    per-parameter launches or additional temporary tensors need separate
    correctness, memory, and performance validation.

    Args:
        optimizer: A `torch.optim.Adam` or `torch.optim.AdamW` built with `fused=True`.
        chunk_size_mb: Target bytes of moments per chunk. A single parameter larger than this
            forms its own chunk.
        state_dtype: dtype of the moments (`torch.bfloat16` under `fused_opt_states_bf16`).

    Note:
        The wrapper shares ``param_groups`` and ``state`` with the inner optimizer. Each
        chunk temporarily binds GPU staging views into that shared state before invoking
        the unchanged fused optimizer step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        chunk_size_mb: int,
        state_dtype: torch.dtype,
    ) -> None:
        # Reject modes whose correctness or memory behavior this wrapper does not support.
        if chunk_size_mb <= 0:
            raise ValueError("chunk_size_mb must be positive")
        if any(state for state in optimizer.state.values()):
            raise ValueError(
                "OptimizerStateOffloader must wrap a fresh optimizer before its first step"
            )
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            if not group.get("fused"):
                raise ValueError(
                    "optimizer_state_offload requires fused Adam/AdamW "
                    "(optimizer.implementation='fused' or 'fused_opt_states_bf16')"
                )
            if group.get("amsgrad"):
                raise ValueError("optimizer_state_offload does not support amsgrad")
            for param in group["params"]:
                if param.device.type != "cuda":
                    raise ValueError(
                        "optimizer_state_offload requires GPU-resident parameters; it cannot be "
                        "combined with training.enable_cpu_offload"
                    )
                if not _local(param).is_contiguous():
                    raise ValueError(
                        "optimizer_state_offload requires contiguous parameters"
                    )

        # Register the wrapped optimizer's own group dicts, then share its state dict, so that
        # narrowing a group or binding a state tensor is visible to both objects.
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

        # Partition parameters, allocate their canonical CPU state, then create the
        # independent streams that move chunks in each direction.
        params = [param for group in self.param_groups for param in group["params"]]
        _apply_numa_binding(_local(params[0]).device)
        self._chunks = _pack_params_by_state_bytes(
            params, chunk_size_bytes=chunk_size_mb << 20, state_dtype=state_dtype
        )
        self._allocate_pinned_state(params, state_dtype)
        self._h2d_stream = torch.cuda.Stream()
        self._d2h_stream = torch.cuda.Stream()

        pinned_gb = sum(_local(param).numel() for param in params)
        pinned_gb *= len(_MOMENT_KEYS) * state_dtype.itemsize / 1e9
        logger.info(
            f"Optimizer-state offload: {len(params)} params in {len(self._chunks)} chunks "
            f"({chunk_size_mb} MiB target), {pinned_gb:.2f} GB pinned host memory per rank"
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        """Run one optimizer step while streaming moment chunks through the GPU.

        Args:
            closure: Unsupported; it must be ``None``.
        """
        assert closure is None, "OptimizerStateOffloader does not support closures"
        compute_stream = torch.cuda.current_stream()

        # Select only parameters Adam will update. Complete chunks use one contiguous
        # slab copy; chunks with missing gradients copy the remaining parameter slices.
        all_params_have_grad = all(
            param.grad is not None for chunk in self._chunks for param in chunk
        )
        chunks = (
            self._chunks
            if all_params_have_grad
            else [
                active_chunk
                for chunk in self._chunks
                if (
                    active_chunk := [param for param in chunk if param.grad is not None]
                )
            ]
        )
        if not chunks:
            return

        # Two staging slots, allocated on the compute stream so the caching allocator can hand
        # the memory back to forward/backward after the step. Freed only after the final D2H
        # synchronize below, so no cross-stream lifetime hazard remains.
        max_numel = max(_chunk_numel(chunk) for chunk in chunks)
        state_dtype = self.state[chunks[0][0]]["exp_avg"].dtype
        slots = [
            {
                key: torch.empty(max_numel, dtype=state_dtype, device="cuda")
                for key in _MOMENT_KEYS
            }
            for _ in range(2)
        ]
        slot_free = [self._d2h_stream.record_event(), self._d2h_stream.record_event()]
        self._h2d_stream.wait_stream(compute_stream)

        def h2d(chunk_index: int) -> torch.cuda.Event:
            slot = slots[chunk_index % 2]
            with torch.cuda.stream(self._h2d_stream):
                self._h2d_stream.wait_event(slot_free[chunk_index % 2])
                chunk = chunks[chunk_index]
                chunk_numel = _chunk_numel(chunk)
                if all_params_have_grad:
                    slab_offset = self._slab_offsets[chunk[0]]
                    for key in _MOMENT_KEYS:
                        slot[key][:chunk_numel].copy_(
                            self._slabs[key][slab_offset : slab_offset + chunk_numel],
                            non_blocking=True,
                        )
                else:
                    offset = 0
                    for param in chunk:
                        numel = _local(param).numel()
                        for key in _MOMENT_KEYS:
                            slot[key][offset : offset + numel].copy_(
                                _local(self.state[param][key]).view(-1),
                                non_blocking=True,
                            )
                        offset += numel
                return self._h2d_stream.record_event()

        # Prime slot A, then alternate slots so the next H2D and previous D2H overlap
        # the current chunk's optimizer kernels.
        saved_group_params = [list(group["params"]) for group in self.param_groups]
        h2d_done = h2d(0)
        try:
            for chunk_index, chunk in enumerate(chunks):
                slot = slots[chunk_index % 2]
                compute_stream.wait_event(h2d_done)
                if chunk_index + 1 < len(chunks):
                    h2d_done = h2d(chunk_index + 1)

                # Bind GPU staging views as the state, step only this chunk with the unchanged
                # optimizer, then copy the updated moments back and rebind the pinned CPU state.
                cpu_state = {
                    param: {key: self.state[param][key] for key in _MOMENT_KEYS}
                    for param in chunk
                }
                offset = 0
                for param in chunk:
                    numel = _local(param).numel()
                    for key in _MOMENT_KEYS:
                        view = slot[key][offset : offset + numel].view(
                            _local(param).shape
                        )
                        self.state[param][key] = _state_like(param, view)
                    offset += numel
                chunk_set = set(map(id, chunk))
                for group, params in zip(self.param_groups, saved_group_params):
                    group["params"] = [p for p in params if id(p) in chunk_set]
                self.optimizer.step()

                self._d2h_stream.wait_stream(compute_stream)
                with torch.cuda.stream(self._d2h_stream):
                    if all_params_have_grad:
                        slab_offset = self._slab_offsets[chunk[0]]
                        chunk_numel = _chunk_numel(chunk)
                        for key in _MOMENT_KEYS:
                            self._slabs[key][
                                slab_offset : slab_offset + chunk_numel
                            ].copy_(slot[key][:chunk_numel], non_blocking=True)
                    for param in chunk:
                        if not all_params_have_grad:
                            for key in _MOMENT_KEYS:
                                _local(cpu_state[param][key]).copy_(
                                    _local(self.state[param][key]), non_blocking=True
                                )
                        for key in _MOMENT_KEYS:
                            self.state[param][key] = cpu_state[param][key]
                    slot_free[chunk_index % 2] = self._d2h_stream.record_event()
        finally:
            # The inner optimizer must expose its complete parameter groups outside step().
            for group, params in zip(self.param_groups, saved_group_params):
                group["params"] = params
        # CPU state is canonical again; checkpoint and state_dict readers need no extra fence.
        self._d2h_stream.synchronize()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Copy a loaded state dict into the existing pinned state, in place.

        Args:
            state_dict: Integer-keyed optimizer state produced by
                ``_unflatten_optim_state_dict`` in parameter order.

        Note:
            The base optimizer implementation would move state tensors to the parameter
            device. This override preserves the pinned slabs and copies values into them.
        """
        params = [param for group in self.param_groups for param in group["params"]]

        # Copy checkpoint values into the existing slab views instead of replacing
        # them with independently allocated tensors.
        for param_index, loaded in state_dict["state"].items():
            state = self.state[params[param_index]]
            for key, value in loaded.items():
                _local(state[key]).copy_(_local(value))
        for group, loaded_group in zip(self.param_groups, state_dict["param_groups"]):
            for key, value in loaded_group.items():
                if key not in ("params", "param_names"):
                    group[key] = value

    def _allocate_pinned_state(
        self, params: list[torch.Tensor], state_dtype: torch.dtype
    ) -> None:
        """One pinned slab per moment, each parameter's moment a view into it."""
        # Allocate the only CPU storage owned by the offloader.
        total_numel = sum(_local(param).numel() for param in params)
        self._slabs = {
            key: torch.zeros(total_numel, dtype=state_dtype, pin_memory=True)
            for key in _MOMENT_KEYS
        }
        self._slab_offsets: dict[torch.Tensor, int] = {}

        # Recreate optimizer.state with parameter-shaped views into those slabs.
        offset = 0
        for param in params:
            local = _local(param)
            state = self.state[param]
            self._slab_offsets[param] = offset
            state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
            for key in _MOMENT_KEYS:
                view = self._slabs[key][offset : offset + local.numel()].view(
                    local.shape
                )
                state[key] = _state_like(param, view)
            offset += local.numel()


def _pack_params_by_state_bytes(
    params: list[torch.Tensor], *, chunk_size_bytes: int, state_dtype: torch.dtype
) -> list[list[torch.Tensor]]:
    """Greedily pack params (in optimizer order) so each chunk's moments stay under the target.

    Whole parameters are atomic: one larger than the target forms its own chunk.

    Example:

        # target 256 MiB, fp32 moments (8 B per element); params of 20M, 20M and 40M elements
        # p0 = 160 MB, p0 + p1 = 320 MB > 256 MiB  -> chunk [p0]
        # p1 = 160 MB, p1 + p2 = 480 MB > 256 MiB  -> chunk [p1]
        # p2 = 320 MB > 256 MiB on its own          -> chunk [p2], with a one-time warning
    """
    bytes_per_element = len(_MOMENT_KEYS) * state_dtype.itemsize
    chunks: list[list[torch.Tensor]] = []
    current: list[torch.Tensor] = []
    current_bytes = 0
    oversized: list[torch.Tensor] = []

    # Preserve optimizer order while filling each chunk up to the byte target.
    for param in params:
        param_bytes = _local(param).numel() * bytes_per_element
        if param_bytes > chunk_size_bytes:
            oversized.append(param)
        if current and current_bytes + param_bytes > chunk_size_bytes:
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(param)
        current_bytes += param_bytes
    if current:
        chunks.append(current)

    # Oversized parameters remain valid one-parameter chunks, but determine the
    # minimum possible size of a GPU staging slot.
    if oversized:
        largest = max(oversized, key=lambda param: _local(param).numel())
        logger.warning(
            f"Optimizer-state offload: {len(oversized)} parameters' moments exceed chunk_size_mb "
            f"({chunk_size_bytes >> 20} MiB); each forms its own chunk. The largest, shape "
            f"{tuple(largest.shape)}, stages {_local(largest).numel() * bytes_per_element >> 20} MiB "
            "and sets the GPU staging peak"
        )
    return chunks


def _chunk_numel(chunk: list[torch.Tensor]) -> int:
    return sum(_local(param).numel() for param in chunk)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _state_like(param: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Give `local` (CPU or GPU) the parameter's mesh and placements so DCP saves it sharded.

    Args:
        param: Parameter whose DTensor metadata should be preserved.
        local: Local optimizer-state tensor to wrap.

    Returns:
        A DTensor with ``param``'s mesh and placements, or ``local`` for a plain tensor.

    Example:

        state = _state_like(param, local_state)

    Note:
        ``DTensor.from_local`` would move CPU state onto the CUDA mesh. The direct
        constructor keeps it on CPU and records the moment dtype in its metadata.
    """
    if not isinstance(param, DTensor):
        return local
    spec = DTensorSpec(
        mesh=param.device_mesh,
        placements=param.placements,
        tensor_meta=TensorMeta(
            shape=param.shape, stride=param.stride(), dtype=local.dtype
        ),
    )
    return DTensor(local, spec, requires_grad=False)


def _apply_numa_binding(device: torch.device) -> None:
    """Best-effort bind CPU threads before allocating pinned optimizer state."""
    try:
        from torch.numa.binding import (
            _bind_all_threads_in_current_process_to_logical_cpus,
            _get_numa_node_index_for_device_index,
            _node_get_logical_cpus_to_bind_to,
        )

        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        numa_node = _get_numa_node_index_for_device_index(device_index=device_index)
        cpus = _node_get_logical_cpus_to_bind_to(device_index=device_index)
        _bind_all_threads_in_current_process_to_logical_cpus(logical_cpu_indices=cpus)
    except (
        ImportError,
        AttributeError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        logger.warning(f"NUMA binding skipped for {device}: {exc}")
        return
    logger.info(
        f"NUMA binding: GPU {device_index} -> node {numa_node}, "
        f"{len(cpus)} logical CPUs"
    )
