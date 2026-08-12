# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import re
import weakref
from collections.abc import Iterable, Iterator, Sequence
from typing import cast, TypeAlias

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from .statistics import accumulate_tensor_statistics, StatisticBuffers


Owner: TypeAlias = nn.Module | nn.Parameter
ReducedBuffers: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

_registered_keys: weakref.WeakKeyDictionary[
    Owner, list[str]
] = weakref.WeakKeyDictionary()
_active_runtime: TensorLoggingRuntime | None = None
_enabled = False
_suppression_depth = 0


def _physical_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return the rank-local tensor represented by an observation."""

    return value.to_local() if isinstance(value, DTensor) else value


def register(owner: Owner, keys: Sequence[str]) -> None:
    """Declare the fixed tensor-statistic names that `owner` may emit.

    Args:
        owner: Module or parameter that owns the stable metric names.
        keys: Names that the owner may pass to `log_stats`.

    Example:

        register(router, ["entropy", "expert_load"])
    """

    _registered_keys.setdefault(owner, []).extend(keys)


def register_fwd_bwd(owner: nn.Module, keys: Sequence[str]) -> None:
    """Register paired `.x` and `.dx` statistics for each tensor name.

    Args:
        owner: Module that owns the observed tensors.
        keys: Base names passed to `log_fwd_bwd_stats`.

    Example:

        register_fwd_bwd(attention, ["xq", "head_out"])
    """

    register(owner, [f"{key}.x" for key in keys])
    register(owner, [f"{key}.dx" for key in keys])


@contextlib.contextmanager
def set_enabled(value: bool) -> Iterator[None]:
    """Set recording state for one scope and restore the prior state on exit.

    Args:
        value: Whether recording calls mutate their fixed buffers.

    Example:

        with set_enabled(step % tensor_logging_freq == 0):
            loss = train_step(batch)
    """

    global _enabled
    previous = _enabled
    _enabled = value
    runtime = _active_runtime
    if runtime is not None:
        runtime.buffers.enabled.fill_(value)
    try:
        yield
    finally:
        _enabled = previous
        if runtime is not None:
            runtime.buffers.enabled.fill_(previous)


@contextlib.contextmanager
def disable() -> Iterator[None]:
    """Suppress mutations while preserving the activation-checkpoint graph."""

    global _suppression_depth
    _suppression_depth += 1
    try:
        yield
    finally:
        _suppression_depth -= 1


def is_enabled() -> bool:
    """Return whether the current trainer scope records tensor statistics."""

    return _enabled and _suppression_depth == 0


def _infer_device(roots: Sequence[nn.Module]) -> torch.device:
    for root in roots:
        tensor = next(root.parameters(), None)
        if tensor is None:
            tensor = next(root.buffers(), None)
        if tensor is not None:
            return tensor.device
    return torch.device("cpu")


def _owner_names(roots: Sequence[nn.Module]) -> dict[Owner, str]:
    names: dict[Owner, str] = {}
    root_prefixes = (
        [""] if len(roots) == 1 else [f"model_parts.{i}" for i in range(len(roots))]
    )
    for root, root_prefix in zip(roots, root_prefixes, strict=True):
        for module_name, module in root.named_modules():
            module_name = ".".join(
                part
                for part in module_name.split(".")
                if part != "_checkpoint_wrapped_module"
            )
            name = ".".join(part for part in (root_prefix, module_name) if part)
            names[module] = name
            for parameter_name, parameter in module.named_parameters(recurse=False):
                names[parameter] = ".".join(
                    part for part in (name, parameter_name) if part
                )
    return names


def _gather_global_keys(local_keys: set[str]) -> list[str]:
    if not dist.is_initialized():
        return sorted(local_keys)
    keys_by_rank: list[set[str]] = [set() for _ in range(dist.get_world_size())]
    dist.all_gather_object(keys_by_rank, local_keys)
    return sorted(set().union(*keys_by_rank))


def _add_statistic_metrics(
    metrics: dict[str, int | float],
    key: str,
    counts: torch.Tensor,
    sums: torch.Tensor,
    maximum: torch.Tensor,
) -> None:
    numel, nonfinite_count, zero_count, observation_count = (
        int(value) for value in counts
    )
    if observation_count == 0:
        return
    finite_count = numel - nonfinite_count
    prefix = f"{key}."
    metrics[prefix + "numel"] = numel
    metrics[prefix + "nonfinite_count"] = nonfinite_count
    metrics[prefix + "observation_count"] = observation_count
    if finite_count == 0:
        return

    absolute_sum, square_sum, fourth_moment_sum = (float(value) for value in sums)
    absolute_mean = absolute_sum / finite_count
    square_mean = square_sum / finite_count
    metrics[prefix + "zero_count"] = zero_count
    metrics[prefix + "zero_frac"] = zero_count / finite_count
    metrics[prefix + "abs_sum"] = absolute_sum
    metrics[prefix + "abs_mean"] = absolute_mean
    metrics[prefix + "square_mean"] = square_mean
    metrics[prefix + "rms"] = square_mean**0.5
    metrics[prefix + "abs_max"] = float(maximum)
    if square_mean > 0:
        metrics[prefix + "kurtosis"] = (
            fourth_moment_sum / finite_count / square_mean**2 - 3
        )


class TensorLoggingRuntime:
    """Setup-static slots and lifecycle for the active model parts."""

    def __init__(
        self,
        roots: Sequence[nn.Module],
        *,
        device: torch.device | None = None,
        metrics_filter_regex: str = "",
    ) -> None:
        self._owners: list[Owner] = []
        self._owner_key_to_slot: dict[tuple[int, str], int] = {}
        self._metrics_filter = (
            re.compile(metrics_filter_regex) if metrics_filter_regex else None
        )

        owner_names = _owner_names(roots)
        local_full_keys: set[str] = set()
        local_bindings: list[tuple[Owner, str, str]] = []
        for owner, keys in _registered_keys.items():
            if owner not in owner_names:
                continue
            owner_name = owner_names[owner]
            for key in keys:
                full_key = ".".join(part for part in (owner_name, key) if part)
                if full_key in local_full_keys:
                    raise ValueError(f"tensor logging key registered twice: {full_key}")
                local_full_keys.add(full_key)
                local_bindings.append((owner, key, full_key))

        self.keys = _gather_global_keys(local_full_keys)
        self._gradient_indices = {
            "gradients.all": [
                index for index, key in enumerate(self.keys) if key.endswith(".dw")
            ],
            "gradients.moe": [
                index
                for index, key in enumerate(self.keys)
                if key.endswith(".dw") and ".moe." in f".{key}."
            ],
        }
        full_key_to_slot = {key: slot for slot, key in enumerate(self.keys)}
        for owner, key, full_key in local_bindings:
            self._owners.append(owner)
            self._owner_key_to_slot[(id(owner), key)] = full_key_to_slot[full_key]

        self.buffers = StatisticBuffers(
            len(self.keys),
            device=device or _infer_device(roots),
        )
        self._state_owner = roots[0]
        self._state_owner.add_module("_tensor_logging_state", self.buffers)
        self._closed = False

    def _accumulate(self, owner: Owner, key: str, value: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("tensor logging runtime is closed")
        try:
            slot = self._owner_key_to_slot[(id(owner), key)]
        except KeyError as error:
            raise KeyError(f"unregistered tensor logging key: {key}") from error
        accumulate_tensor_statistics(
            _physical_tensor(value),
            self.buffers.counts[slot],
            self.buffers.sums[slot],
            self.buffers.maxima[slot],
            self.buffers.enabled,
        )

    def _statistic_buffers(
        self,
        owner: Owner,
        key: str,
        *,
        cotangent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._closed:
            raise RuntimeError("tensor logging runtime is closed")
        try:
            slot = self._owner_key_to_slot[(id(owner), key)]
        except KeyError as error:
            raise KeyError(f"unregistered tensor logging key: {key}") from error
        if cotangent:
            cotangent_buffers = self.buffers.cotangents[slot]
            return cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                (
                    cotangent_buffers.counts,
                    cotangent_buffers.sums,
                    cotangent_buffers.maximum,
                ),
            )
        return (
            self.buffers.counts[slot],
            self.buffers.sums[slot],
            self.buffers.maxima[slot],
        )

    def clear(self) -> None:
        self.buffers.clear()

    def raw_snapshot(self) -> dict[str, dict[str, torch.Tensor]]:
        """Clone unreduced slots for focused correctness tests."""

        (
            cotangent_counts,
            cotangent_sums,
            cotangent_maxima,
        ) = self.buffers.stack_cotangents()
        counts = self.buffers.counts + cotangent_counts
        sums = self.buffers.sums + cotangent_sums
        maxima = torch.maximum(
            self.buffers.maxima,
            cotangent_maxima,
        )
        return {
            key: {
                "counts": counts[index].detach().cpu().clone(),
                "sums": sums[index].detach().cpu().clone(),
                "maximum": maxima[index].detach().cpu().clone(),
            }
            for index, key in enumerate(self.keys)
        }

    def reduce_buffers(self) -> ReducedBuffers:
        """Clone and reduce every registered key in three packed WORLD slabs."""

        (
            cotangent_counts,
            cotangent_sums,
            cotangent_maxima,
        ) = self.buffers.stack_cotangents()
        counts = self.buffers.counts + cotangent_counts
        sums = self.buffers.sums + cotangent_sums
        maxima = torch.maximum(
            self.buffers.maxima,
            cotangent_maxima,
        )
        if dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        return counts, sums, maxima

    def buffers_to_metrics(
        self,
        reduced_buffers: ReducedBuffers,
    ) -> dict[str, int | float]:
        """Derive scalar metric leaves from reduced sufficient statistics."""

        counts, sums, maxima = (buffer.detach().cpu() for buffer in reduced_buffers)
        metrics: dict[str, int | float] = {}
        for index, key in enumerate(self.keys):
            _add_statistic_metrics(
                metrics,
                key,
                counts[index],
                sums[index],
                maxima[index],
            )
        for key, indices in self._gradient_indices.items():
            if not indices:
                continue
            _add_statistic_metrics(
                metrics,
                key,
                counts[indices].sum(dim=0),
                sums[indices].sum(dim=0),
                maxima[indices].amax(),
            )
        if self._metrics_filter is not None:
            metrics = {
                key: value
                for key, value in metrics.items()
                if self._metrics_filter.search(":".join(key.rsplit(".", maxsplit=1)))
            }
        return metrics

    def close(self) -> None:
        global _active_runtime
        if self._closed:
            return
        self._closed = True
        self._owners.clear()
        self._owner_key_to_slot.clear()
        self._state_owner._modules.pop("_tensor_logging_state")
        if _active_runtime is self:
            _active_runtime = None


def init(
    roots: nn.Module | Iterable[nn.Module],
    *,
    device: torch.device | None = None,
    metrics_filter_regex: str = "",
) -> TensorLoggingRuntime:
    """Freeze registrations and install one active runtime for the model roots.

    Args:
        roots: Model or model parts whose registered owners should be active.
        device: Device for fixed statistic buffers; inferred when omitted.
        metrics_filter_regex: Publication allowlist over `<name>:<statistic>`.

    Example:

        register_fwd_bwd(model.layers[0], ["residual"])
        runtime = init(model, device=torch.device("cuda"))
    """

    global _active_runtime
    if _active_runtime is not None:
        raise RuntimeError("tensor logging already has an active runtime")
    root_list = [roots] if isinstance(roots, nn.Module) else list(roots)
    runtime = TensorLoggingRuntime(
        root_list,
        device=device,
        metrics_filter_regex=metrics_filter_regex,
    )
    _active_runtime = runtime
    return runtime


def _runtime() -> TensorLoggingRuntime:
    if _active_runtime is None:
        raise RuntimeError("tensor logging is enabled before init()")
    return _active_runtime


def log_stats(owner: Owner, **named_tensors: torch.Tensor) -> None:
    """Accumulate current-pass statistics for registered named tensors.

    Args:
        owner: Module or parameter used during registration.
        **named_tensors: Registered names mapped to their current tensors.

    Example:

        log_stats(router, entropy=entropy, expert_load=expert_load)
    """

    if not is_enabled():
        return
    runtime = _runtime()
    for key, value in named_tensors.items():
        runtime._accumulate(owner, key, value)


class _RecordForwardAndBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        value: torch.Tensor,
        forward_counts: torch.Tensor,
        forward_sums: torch.Tensor,
        forward_maximum: torch.Tensor,
        backward_counts: torch.Tensor,
        backward_sums: torch.Tensor,
        backward_maximum: torch.Tensor,
        enabled: torch.Tensor,
    ) -> torch.Tensor:
        accumulate_tensor_statistics(
            _physical_tensor(value),
            forward_counts,
            forward_sums,
            forward_maximum,
            enabled,
        )
        ctx.backward_counts = backward_counts
        ctx.backward_sums = backward_sums
        ctx.backward_maximum = backward_maximum
        ctx.enabled = enabled
        return value

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, gradient: torch.Tensor):
        accumulate_tensor_statistics(
            _physical_tensor(gradient),
            ctx.backward_counts,
            ctx.backward_sums,
            ctx.backward_maximum,
            ctx.enabled,
        )
        return gradient, None, None, None, None, None, None, None


def log_fwd_bwd_stats(
    owner: nn.Module,
    **named_tensors: torch.Tensor,
) -> torch.Tensor:
    """Record one tensor now and its incoming cotangent during backward.

    Args:
        owner: Module used during `register_fwd_bwd`.
        **named_tensors: Exactly one registered base name and tensor.

    Returns:
        The input tensor with the backward recorder attached.

    Example:

        xq = log_fwd_bwd_stats(attention, xq=xq)
    """

    if len(named_tensors) != 1:
        raise ValueError("log_fwd_bwd_stats() records exactly one named tensor")
    key, value = next(iter(named_tensors.items()))

    if not _enabled:
        return value
    if not value.requires_grad:
        raise ValueError(
            f"log_fwd_bwd_stats({key}=...) requires a differentiable tensor"
        )

    runtime = _runtime()
    forward_counts, forward_sums, forward_maximum = runtime._statistic_buffers(
        owner,
        f"{key}.x",
    )
    backward_counts, backward_sums, backward_maximum = runtime._statistic_buffers(
        owner,
        f"{key}.dx",
        cotangent=True,
    )
    return _RecordForwardAndBackward.apply(
        value,
        forward_counts,
        forward_sums,
        forward_maximum,
        backward_counts,
        backward_sums,
        backward_maximum,
        (
            runtime.buffers.suppressed
            if _suppression_depth > 0
            else runtime.buffers.enabled
        ),
    )
