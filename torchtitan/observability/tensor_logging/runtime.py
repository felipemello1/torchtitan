# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import cast, TypeAlias

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from .statistics import accumulate_tensor_statistics, StatisticBuffers


Owner: TypeAlias = nn.Module | nn.Parameter
ReducedBuffers: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
StatisticBinding: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

_REGISTERED_KEYS_ATTR = "_tensor_logging_registered_keys"
_STATISTIC_BINDINGS_ATTR = "_tensor_logging_statistic_bindings"
_active_runtime: TensorLoggingRuntime | None = None
_enabled = False
_suppression_depth = 0


def _registered_keys(owner: Owner) -> list[str] | None:
    """Return keys stored directly on an owner, without module proxy lookup."""

    return cast(
        list[str] | None,
        owner.__dict__.get(_REGISTERED_KEYS_ATTR),
    )


def _statistic_bindings(owner: Owner) -> dict[str, StatisticBinding] | None:
    return cast(
        dict[str, StatisticBinding] | None,
        owner.__dict__.get(_STATISTIC_BINDINGS_ATTR),
    )


def _statistic_binding(owner: Owner, key: str) -> StatisticBinding:
    try:
        return owner.__dict__[_STATISTIC_BINDINGS_ATTR][key]
    except KeyError as error:
        raise KeyError(f"unregistered tensor logging key: {key}") from error


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

    registered_keys = _registered_keys(owner)
    if registered_keys is None:
        registered_keys = []
        setattr(owner, _REGISTERED_KEYS_ATTR, registered_keys)
    registered_keys.extend(keys)


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
        with spmd.no_typecheck():
            runtime.buffers.enabled.fill_(value)
    try:
        yield
    finally:
        _enabled = previous
        if runtime is not None:
            with spmd.no_typecheck():
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


def _owner_names(
    roots: Sequence[nn.Module],
    root_name_overrides: Mapping[nn.Module, str],
) -> dict[Owner, str]:
    names: dict[Owner, str] = {}
    root_prefixes = (
        [""] if len(roots) == 1 else [f"model_parts.{i}" for i in range(len(roots))]
    )
    for root, root_prefix in zip(roots, root_prefixes, strict=True):
        root_prefix = root_name_overrides.get(root, root_prefix)
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
    counts: Sequence[int],
    sums: Sequence[float],
    maximum: float,
) -> None:
    numel, nonfinite_count, zero_count, observation_count = counts
    if observation_count == 0:
        return
    finite_count = numel - nonfinite_count
    prefix = f"{key}."
    metrics[prefix + "numel"] = numel
    metrics[prefix + "nonfinite_count"] = nonfinite_count
    metrics[prefix + "observation_count"] = observation_count
    if finite_count == 0:
        return

    absolute_sum, square_sum, fourth_moment_sum = sums
    metrics[prefix + "zero_count"] = zero_count
    metrics[prefix + "zero_frac"] = zero_count / finite_count
    if math.isfinite(absolute_sum):
        metrics[prefix + "abs_sum"] = absolute_sum
        metrics[prefix + "abs_mean"] = absolute_sum / finite_count
    if math.isfinite(square_sum):
        square_mean = square_sum / finite_count
        metrics[prefix + "square_mean"] = square_mean
        metrics[prefix + "rms"] = square_mean**0.5
        if square_mean > 0 and math.isfinite(fourth_moment_sum):
            kurtosis = fourth_moment_sum / finite_count / square_mean**2 - 3
            if math.isfinite(kurtosis):
                metrics[prefix + "kurtosis"] = kurtosis
    if math.isfinite(maximum):
        metrics[prefix + "abs_max"] = maximum


class TensorLoggingRuntime:
    """Setup-static slots and lifecycle for the active model parts."""

    def __init__(
        self,
        roots: Sequence[nn.Module],
        *,
        device: torch.device | None = None,
        metrics_filter_regex: str = "",
        root_name_overrides: Mapping[nn.Module, str] | None = None,
    ) -> None:
        self._owners: list[Owner] = []
        self._metrics_filter = (
            re.compile(metrics_filter_regex) if metrics_filter_regex else None
        )

        root_name_overrides = root_name_overrides or {}
        owner_names = _owner_names(roots, root_name_overrides)
        local_owner_by_full_key: dict[str, Owner] = {}
        local_bindings: list[tuple[Owner, str, str]] = []
        for owner, owner_name in owner_names.items():
            keys = _registered_keys(owner)
            if keys is None:
                continue
            for key in keys:
                full_key = ".".join(part for part in (owner_name, key) if part)
                previous_owner = local_owner_by_full_key.get(full_key)
                if previous_owner is not None:
                    explicitly_shared_root = (
                        previous_owner is not owner
                        and previous_owner in root_name_overrides
                        and owner in root_name_overrides
                        and root_name_overrides[previous_owner]
                        == root_name_overrides[owner]
                    )
                    if not explicitly_shared_root:
                        raise ValueError(
                            f"tensor logging key registered twice: {full_key}"
                        )
                else:
                    local_owner_by_full_key[full_key] = owner
                local_bindings.append((owner, key, full_key))

        self.keys = _gather_global_keys(set(local_owner_by_full_key))
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

        self.buffers = StatisticBuffers(
            len(self.keys),
            device=device or _infer_device(roots),
        )
        self._slot_indices = torch.arange(len(self.keys), dtype=torch.int64)
        for owner, key, full_key in local_bindings:
            bindings = _statistic_bindings(owner)
            if bindings is None:
                bindings = {}
                setattr(owner, _STATISTIC_BINDINGS_ATTR, bindings)
                self._owners.append(owner)
            slot = full_key_to_slot[full_key]
            bindings[key] = (
                self.buffers.counts[slot],
                self.buffers.sums[slot],
                self.buffers.maxima[slot],
                self._slot_indices[slot],
            )
        self._state_owner = roots[0]
        self._state_owner.add_module("_tensor_logging_state", self.buffers)
        self._closed = False

    def _accumulate(self, owner: Owner, key: str, value: torch.Tensor) -> None:
        binding = _statistic_binding(owner, key)
        with spmd.no_typecheck():
            accumulate_tensor_statistics(
                _physical_tensor(value),
                *binding[:3],
                self.buffers.enabled,
            )

    def _statistic_slot(self, owner: Owner, key: str) -> torch.Tensor:
        return _statistic_binding(owner, key)[3]

    def clear(self) -> None:
        self.buffers.clear()

    def raw_snapshot(self) -> dict[str, dict[str, torch.Tensor]]:
        """Clone unreduced slots for focused correctness tests."""

        return {
            key: {
                "counts": self.buffers.counts[index].detach().cpu().clone(),
                "sums": self.buffers.sums[index].detach().cpu().clone(),
                "maximum": self.buffers.maxima[index].detach().cpu().clone(),
            }
            for index, key in enumerate(self.keys)
        }

    def reduce_buffers(self) -> ReducedBuffers:
        """Clone and reduce every registered key in three packed WORLD slabs."""

        counts = self.buffers.counts.clone()
        sums = self.buffers.sums.clone()
        maxima = self.buffers.maxima.clone()
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
        count_rows = cast(list[list[int]], counts.tolist())
        sum_rows = cast(list[list[float]], sums.tolist())
        maximum_rows = cast(list[float], maxima.tolist())
        metrics: dict[str, int | float] = {}
        for index, key in enumerate(self.keys):
            _add_statistic_metrics(
                metrics,
                key,
                count_rows[index],
                sum_rows[index],
                maximum_rows[index],
            )
        for key, indices in self._gradient_indices.items():
            if not indices:
                continue
            _add_statistic_metrics(
                metrics,
                key,
                cast(list[int], counts[indices].sum(dim=0).tolist()),
                cast(list[float], sums[indices].sum(dim=0).tolist()),
                float(maxima[indices].amax()),
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
        for owner in self._owners:
            owner.__dict__.pop(_STATISTIC_BINDINGS_ATTR, None)
        self._owners.clear()
        self._state_owner._modules.pop("_tensor_logging_state")
        if _active_runtime is self:
            _active_runtime = None


def init(
    roots: nn.Module | Iterable[nn.Module],
    *,
    device: torch.device | None = None,
    metrics_filter_regex: str = "",
    root_name_overrides: Mapping[nn.Module, str] | None = None,
) -> TensorLoggingRuntime:
    """Freeze registrations and install one active runtime for the model roots.

    Args:
        roots: Model or model parts whose registered owners should be active.
        device: Device for fixed statistic buffers; inferred when omitted.
        metrics_filter_regex: Publication allowlist over `<name>:<statistic>`.
        root_name_overrides: Logical names for explicitly split model roots.

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
        root_name_overrides=root_name_overrides,
    )
    _active_runtime = runtime
    return runtime


def _runtime() -> TensorLoggingRuntime:
    if _active_runtime is None:
        raise RuntimeError("tensor logging is enabled before init()")
    return _active_runtime


@torch.library.custom_op(
    "torchtitan::record_tensor_statistics_cotangent",
    mutates_args=(),
)
def _record_tensor_statistics_cotangent(
    value: torch.Tensor,
    slot: torch.Tensor,
) -> None:
    """Record one cotangent without exposing mutable buffers to autograd."""

    runtime = _runtime()
    slot_index = int(slot.item())
    buffers = (
        runtime.buffers.counts[slot_index],
        runtime.buffers.sums[slot_index],
        runtime.buffers.maxima[slot_index],
        runtime.buffers.enabled,
    )
    accumulate_tensor_statistics(value, *buffers)


@_record_tensor_statistics_cotangent.register_fake
def _(
    value: torch.Tensor,
    slot: torch.Tensor,
) -> None:
    return None


_record_tensor_statistics_cotangent.register_effect(torch.library.EffectType.ORDERED)


def log_stats(owner: Owner, **named_tensors: torch.Tensor) -> None:
    """Accumulate current-pass statistics for registered named tensors.

    Args:
        owner: Module or parameter used during registration.
        **named_tensors: Registered names mapped to their current tensors.

    Example:

        log_stats(router, entropy=entropy, expert_load=expert_load)
    """

    if torch.compiler.is_compiling():
        if _active_runtime is None:
            return
    elif not is_enabled():
        return
    runtime = _runtime()
    for key, value in named_tensors.items():
        runtime._accumulate(owner, key, value)


def log_fwd_bwd_stats(
    owner: nn.Module,
    **named_tensors: torch.Tensor,
) -> None:
    """Record one tensor now and its incoming cotangent during backward.

    Args:
        owner: Module used during `register_fwd_bwd`.
        **named_tensors: Registered base names mapped to differentiable tensors.

    Example:

        log_fwd_bwd_stats(attention, xq=xq)
    """

    if not torch.is_grad_enabled():
        return
    if torch.compiler.is_compiling():
        if _active_runtime is None:
            return
    elif not is_enabled():
        return

    runtime = _runtime()
    with spmd.no_typecheck():
        for key, value in named_tensors.items():
            if not value.requires_grad:
                raise ValueError(
                    f"log_fwd_bwd_stats({key}=...) requires a differentiable tensor"
                )
            backward_slot = runtime._statistic_slot(owner, f"{key}.dx")
            runtime._accumulate(owner, f"{key}.x", value)

            def record_cotangent(
                cotangent: torch.Tensor,
                slot=backward_slot,
            ) -> torch.Tensor:
                with spmd.no_typecheck():
                    _record_tensor_statistics_cotangent(
                        _physical_tensor(cotangent),
                        slot,
                    )
                    return cotangent

            value.register_hook(record_cotangent)
