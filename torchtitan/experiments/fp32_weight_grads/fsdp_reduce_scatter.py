# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP2 reduce-scatter without the copy-in when gradients already have the reduce dtype.

FSDP2's ``foreach_reduce`` copies every gradient of an FSDP unit into one flat reduce-dtype buffer
(``chunk_cat``) and reduce-scatters that buffer. The copy converts dtype (bf16 -> fp32) and packs
the gradients. With fp32 weight gradients (``training.mixed_precision_grad=float32``) there is
nothing to convert, so this version reduce-scatters each gradient straight into its slice of the
output (one coalesced collective) and skips the copy.

``install()`` swaps in a copy of PyTorch's ``foreach_reduce`` with that change. The copy is the
proposed upstream PyTorch change (``pytorch_patches/fsdp_reduce_scatter_in_place.patch``), so it
is tied to one PyTorch version: ``install()`` refuses to run unless the installed
``foreach_reduce`` is exactly the one it was copied from.
"""

import hashlib
import inspect
import logging
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group
from torch.distributed.fsdp._fully_shard._fsdp_api import _ReduceOp, ReduceScatter
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _default_reduce_scatter_input_fn,
    _div_if_needed,
    _get_device_handle,
    _get_gradient_divide_factors,
    _raise_assert_with_print,
    _to_dtype_if_needed,
    DefaultReduceScatter,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)

# sha256 of inspect.getsource(foreach_reduce) in the PyTorch this copy comes from
# (nightly 2.15.0.dev20260926, identical in PyTorch main e0fe13c).
_ORIGINAL_FOREACH_REDUCE_SHA256 = (
    "598b5da568fc4f455a3d9f70ce2438e53861df708eb609b0f498a77b21f9a6fc"
)


def install() -> None:
    """Make FSDP2 reduce-scatter reduce-dtype gradients in place (see module docstring)."""
    original = _fsdp_collectives.foreach_reduce
    if original is foreach_reduce:
        return
    source_sha256 = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if source_sha256 != _ORIGINAL_FOREACH_REDUCE_SHA256:
        raise RuntimeError(
            "fsdp_reduce_scatter.install() only supports the PyTorch whose FSDP2 foreach_reduce "
            f"it was copied from (sha256 {_ORIGINAL_FOREACH_REDUCE_SHA256}); this PyTorch has "
            f"{source_sha256}. Use torch==2.15.0.dev20260926."
        )
    _fsdp_collectives.foreach_reduce = foreach_reduce
    _fsdp_param_group.foreach_reduce = foreach_reduce
    logger.info(
        "FSDP2 foreach_reduce: reduce-scatter in place for reduce-dtype gradients"
    )


# Below: PyTorch's foreach_reduce with the in-place change, and its helper, verbatim from the
# upstream patch. Only the `reduce_scatter_in_place` branches differ from PyTorch.


@torch.no_grad()
def foreach_reduce(
    fsdp_params: list[FSDPParam],
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    reduce_scatter_comm: ReduceScatter,
    orig_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    device: torch.device,
    gradient_divide_factor: float | None,
    all_reduce_group: dist.ProcessGroup | None,  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: torch.Tensor | None,  # only used for HSDP
    all_reduce_hook: Callable[[torch.Tensor], None] | None,
    force_sum_reduction_for_comms: bool = False,
    *,
    prepare_reduce_scatter_inputs: Callable = _default_reduce_scatter_input_fn,
) -> tuple[
    torch.Tensor | list[torch.Tensor],
    torch.Event,
    torch.Stream,
    torch.Event,
    torch.Tensor | None,
    torch.Event | None,
    torch.Tensor | None,
]:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """

    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) != 1:
        # Check this at runtime since it could be a real runtime error if e.g.
        # fp8 weights do not produce the correct higher precision gradients
        _raise_assert_with_print(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    (
        predivide_factor,
        postdivide_factor,
        reduce_scatter_op,
        all_reduce_op,
    ) = _get_gradient_divide_factors(
        reduce_scatter_group,
        all_reduce_group,
        reduce_dtype,
        device.type,
        gradient_divide_factor,
        force_sum_reduction_for_comms,
    )

    if reduce_scatter_group is None:
        world_size = 1
    else:
        world_size = reduce_scatter_group.size()
    device_handle = _get_device_handle(device.type)
    current_stream = device_handle.current_stream()

    padded_unsharded_sizes = prepare_reduce_scatter_inputs(
        fsdp_params, unsharded_grads, world_size
    )
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    # Gradients that already have the reduce dtype need no copy-in: each one is
    # reduce-scattered straight into its slice of the output. Needs a coalesced
    # reduce-scatter (NCCL, gloo).
    reduce_scatter_in_place = (
        type(reduce_scatter_comm) is DefaultReduceScatter
        and predivide_factor is None
        and device.type in ("cuda", "cpu")
        and all(grad.dtype == reduce_dtype for grad in unsharded_grads)
    )
    reduce_scatter_input: torch.Tensor | list[torch.Tensor]
    if reduce_scatter_in_place:
        # Returned in place of the copy-in buffer to keep the gradients alive
        # until the reduce-scatter finishes
        reduce_scatter_input = []
        for grad, padded_size in zip(unsharded_grads, padded_unsharded_sizes):
            if grad.size() != padded_size or not grad.is_contiguous():
                # Only uneven dim-0 (e.g. small norm weights) or non-contiguous
                # gradients are copied, padded like the copy-in buffer
                padded_grad = grad.new_zeros(padded_size)
                padded_grad[: grad.size(0)].copy_(grad)
                grad = padded_grad
            reduce_scatter_input.append(grad)
    else:
        reduce_scatter_input = reduce_scatter_comm.allocate(
            (reduce_scatter_input_numel,),
            dtype=reduce_dtype,
            device=device,
        )
        foreach_reduce_scatter_copy_in(
            unsharded_grads, reduce_scatter_input, world_size
        )

    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None

    with device_handle.stream(reduce_scatter_stream):
        reduce_output = reduce_scatter_comm.allocate(
            (reduce_scatter_output_numel,),
            dtype=reduce_dtype,
            device=device,
        )
        if isinstance(reduce_scatter_input, list):
            foreach_reduce_scatter_in_place(
                reduce_scatter_input,
                reduce_output,
                reduce_scatter_comm,
                reduce_scatter_group,
                reduce_scatter_op,
                gradient_divide_factor,
                world_size,
            )
        else:
            _div_if_needed(reduce_scatter_input, predivide_factor)
            if world_size > 1:
                reduce_scatter_comm(
                    output_tensor=reduce_output,
                    input_tensor=reduce_scatter_input,
                    group=reduce_scatter_group,
                    op=reduce_scatter_op,
                )
            else:
                # For single GPU, just copy the input to output (no actual reduce-scatter needed), and
                # account for a possible gradient_divide_factor.
                if gradient_divide_factor is not None and gradient_divide_factor != 1.0:
                    reduce_output.copy_(reduce_scatter_input / gradient_divide_factor)
                else:
                    reduce_output.copy_(reduce_scatter_input)
        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        if all_reduce_group is not None:  # HSDP or DDP/replicate
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream,
                    post_reduce_stream.record_event(),
                    all_reduce_input,
                    all_reduce_event,
                    partial_reduce_output,
                )
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            if world_size >= 1:
                all_reduce_stream.wait_stream(reduce_scatter_stream)
            else:
                all_reduce_stream.wait_stream(current_stream)
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=all_reduce_op,
                )
                # Keep refs to the reduce-dtype AR buffer + completion
                # event so FSDPParamGroup._all_reduce_state can hold them
                # across layers. This keeps the buffer off the caching
                # allocator's free list; otherwise the next layer's
                # reduce-scatter can reuse the same physical block while
                # this layer's AR is still in flight, causing cross-layer
                # gradient aliasing under slow AR. See PR #140044,
                # regression test PR #180900.
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    # -- END: ops in reduce_scatter stream

    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        # Rebinds to a new orig_dtype tensor when reduce_dtype !=
        # orig_dtype. Do NOT rely on this stream-scoped rebind to manage
        # the old reduce-dtype buffer's lifetime: the rebind orders the
        # cast before the free-event on AR stream, but the freed block
        # lands on the caching allocator's free list and the next layer's
        # RS on RS stream can reuse it without waiting for this layer's
        # AR to finish. The reduce-dtype buffer is held across layers by
        # FSDPParamGroup._all_reduce_state (captured above) to prevent
        # this. See PR #140044, regression test PR #180900.
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            if fsdp_param.offload_to_cpu:
                # Only overlap the D2H copy (copying to pinned memory) when no
                # in-backward CPU consumer of the grad exists. Two such
                # consumers suppress the overlap:
                #   - Accumulating grads: the CPU add kernel depends on the
                #     copy result and we cannot run the add as a callback.
                #   - Post-accumulate-grad hooks: user code (e.g.
                #     optimizer-in-backward) reads ``param.grad`` on CPU
                #     synchronously. With ``non_blocking=True`` the hook would
                #     observe in-flight pinned memory — silently wrong
                #     optimizer updates.
                has_post_acc_grad_hook = bool(
                    getattr(
                        fsdp_param.sharded_param,
                        "_post_accumulate_grad_hooks",
                        None,
                    )
                )
                non_blocking = (
                    fsdp_param.pin_memory
                    and not to_accumulate_grad
                    and not has_post_acc_grad_hook
                )
                # Since the GPU sharded gradient is allocated in the RS stream,
                # we can free it here by not keeping a ref without waiting for
                # the D2H copy since future RS-stream ops run after the copy
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                if non_blocking:
                    # Record an event on which to block the CPU thread to
                    # ensure that the D2H copy finishes before the optimizer
                    fsdp_param.grad_offload_event = post_reduce_stream.record_event()
            if to_accumulate_grad:
                if not isinstance(fsdp_param.sharded_param.grad, DTensor):
                    raise AssertionError(
                        f"Expected fsdp_param.sharded_param.grad to be DTensor, got {type(fsdp_param.sharded_param.grad)}"
                    )
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(
                    new_sharded_grad
                )
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            for hook in (
                getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {})
                or {}
            ).values():
                hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
    # The RS output is allocated in the RS stream and used in the default
    # stream (for optimizer). To ensure its memory is not reused for later
    # RSs, we do not need extra synchronization since the sharded parameters
    # hold refs through the end of backward.
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_stream,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )


def foreach_reduce_scatter_in_place(
    unsharded_grads: list[torch.Tensor],
    reduce_output: torch.Tensor,
    reduce_scatter_comm: ReduceScatter,
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_op: _ReduceOp,
    gradient_divide_factor: float | None,
    world_size: int,
) -> None:
    """
    Same result as ``foreach_reduce_scatter_copy_in`` followed by one
    reduce-scatter, without the copy-in: reduce-scatters each gradient into
    its slice of ``reduce_output``. Each gradient must already have the reduce
    dtype, be contiguous, and have its dim-0 padded size.
    """
    if world_size == 1:
        torch.cat([grad.view(-1) for grad in unsharded_grads], out=reduce_output)
        _div_if_needed(reduce_output, gradient_divide_factor)
        return
    output_slices = reduce_output.split(
        [grad.numel() // world_size for grad in unsharded_grads]
    )
    with dist._coalescing_manager(group=reduce_scatter_group):
        for grad, output_slice in zip(unsharded_grads, output_slices):
            reduce_scatter_comm(
                output_tensor=output_slice,
                input_tensor=grad.view(-1),
                group=reduce_scatter_group,
                op=reduce_scatter_op,
            )
