# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 linear training with FSDP-managed 32x32 weight caches.

Tensor shape suffixes:
    M: flattened token rows
    N: output features
    K: input features
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from torchao.prototype.mx_formats.kernels import (
    mxfp8_quantize_cuda,
    triton_mx_block_rearrange,
)

from torchtitan.models.common.linear import Linear

from .._fsdp_tensor import _UnshardedFSDPTensor
from .tensor import (
    _LinearShardedTensorWithMXFP8Compute,
    _MXFP8_BLOCK_SIZE,
    _MXFP8LinearOperands,
    _quantize_mxfp8_weight,
)


__all__ = [
    "InputActivationFormatForBackward",
    "MXFP8Linear",
    "mxfp8_linears_shared_input",
]

# Activation and gradient quantization takes a scaling mode; the 32x32 weight
# cast hardcodes RCEIL. Pin the two to match, so both operands of a GEMM round
# their E8M0 scales the same way. This is TorchAO's current default too, but
# relying on that would let a default change silently desync them.
_MXFP8_SCALING_MODE = "rceil"

InputActivationFormatForBackward = Literal["bf16", "mxfp8"]
_INPUT_ACTIVATION_FORMATS_FOR_BACKWARD = ("bf16", "mxfp8")


def _pad_rows(x_MK: torch.Tensor) -> tuple[torch.Tensor, int]:
    num_rows = x_MK.shape[0]
    num_padded_rows = (
        (num_rows + _MXFP8_BLOCK_SIZE - 1) // _MXFP8_BLOCK_SIZE
    ) * _MXFP8_BLOCK_SIZE
    if num_padded_rows == num_rows:
        return x_MK, num_rows
    return F.pad(x_MK, (0, 0, 0, num_padded_rows - num_rows)), num_rows


def _unpad_output(
    output_MN: torch.Tensor, num_rows: int, input_shape: torch.Size
) -> torch.Tensor:
    """Drop the padded rows and restore the input's leading dimensions.

    A 2D unpadded output is returned as the GEMM result itself rather than a
    view of it: autograd forbids in-place updates of a view created inside a
    custom Function, and callers such as fused RoPE rotate the output in place.
    """
    if len(input_shape) == 2 and num_rows == output_MN.shape[0]:
        return output_MN
    return output_MN[:num_rows].reshape(*input_shape[:-1], output_MN.shape[-1])


def _quantize_wgrad_input(
    x_hp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a saved BF16 input columnwise for WGRAD."""
    x_MK, _ = _pad_rows(x_hp.reshape(-1, x_hp.shape[-1]).contiguous())
    _, x_qdata_col_MK, _, x_scale_col = mxfp8_quantize_cuda(
        x_MK,
        rowwise=False,
        colwise=True,
        scaling_mode=_MXFP8_SCALING_MODE,
    )
    return x_qdata_col_MK, triton_mx_block_rearrange(x_scale_col)


def _weight_gradient(
    grad_output_col_MN: torch.Tensor,
    grad_output_col_scales: torch.Tensor,
    x_qdata_col_MK: torch.Tensor,
    x_scale_col: torch.Tensor,
    *,
    weight_param: torch.Tensor | None,
    wgrad_dtype: torch.dtype,
    weight_shape: torch.Size,
) -> torch.Tensor:
    """Return WGRAD, folding it into ``weight_param.grad`` in place if it has one.

    Args:
        grad_output_col_MN: Columnwise MXFP8 output gradient.
        grad_output_col_scales: Its unswizzled scales.
        x_qdata_col_MK: Columnwise MXFP8 input.
        x_scale_col: Its swizzled scales.
        weight_param: The leaf parameter whose running ``.grad`` may absorb this
            contribution, or None for an ordinary WGRAD.
        wgrad_dtype: Dtype of a freshly computed gradient.
        weight_shape: Parameter shape the gradient is returned in.
    """
    wgrad_scale_kwargs = dict(
        scale_a=triton_mx_block_rearrange(grad_output_col_scales),
        scale_recipe_a=F.ScalingType.BlockWise1x32,
        scale_b=x_scale_col,
        scale_recipe_b=F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
    )
    running_grad = None if weight_param is None else weight_param.grad
    if weight_param is None or running_grad is None:
        # First contribution since the gradient was last consumed, or a traced
        # execution. Nothing to accumulate into.
        grad_weight_NK = F.scaled_mm(
            grad_output_col_MN.t(),
            x_qdata_col_MK,
            output_dtype=wgrad_dtype,
            **wgrad_scale_kwargs,
        )
        return grad_weight_NK.view(weight_shape)
    # A later microbatch, e.g. under PP with gradient sync disabled. Fold this
    # contribution into the running gradient, in its grad_dtype, in the GEMM
    # epilogue instead of a separate AccumulateGrad add. Then hand the same
    # buffer back and clear the parameter, so AccumulateGrad reattaches it
    # instead of adding it to itself. While grad_dtype differs from FSDP's
    # reduce dtype, FSDP moves the gradient into its own accumulator after
    # every microbatch, so running_grad stays None here.
    F.scaled_addmm_(
        running_grad.view(-1, running_grad.shape[-1]),
        grad_output_col_MN.t(),
        x_qdata_col_MK,
        **wgrad_scale_kwargs,
    )
    weight_param.grad = None
    return running_grad


# Adapted from torchao.prototype.moe_training.mxfp8_linear.mx_mm. This variant
# lives in TorchTitan so its autograd state and weight cache can integrate with
# FSDP and other parallelisms.
@torch._dynamo.allow_in_graph
class _MXFP8LinearFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_qdata_fprop_KN: torch.Tensor,
        weight_scale_fprop_swizzled: torch.Tensor,
        weight_qdata_dgrad_NK: torch.Tensor,
        weight_scale_dgrad_swizzled: torch.Tensor,
        bias_N: torch.Tensor | None,
        input_activation_format_for_backward: InputActivationFormatForBackward,
        accumulate_into_weight_grad: bool,
    ) -> torch.Tensor:
        # ``weight`` is the module parameter itself, ``[N, K]`` or stacked
        # ``[num_linears, N, K]``, rather than a flattened view of it. WGRAD is
        # returned in its shape so the gradient reaches AccumulateGrad directly:
        # a view's backward would cast it to the view's dtype, ignoring the
        # parameter's grad_dtype. The quantized operands are already flattened.
        # FPROP always consumes rowwise MXFP8. WGRAD can either retain the
        # original BF16 input and quantize it columnwise in backward, or retain
        # a columnwise MXFP8 operands produced in forward. The former is
        # memory-safe when another operation already keeps BF16 x alive; the
        # latter reduces storage for a single-consumer input at the cost of an
        # extra cached operands when BF16 x is retained elsewhere. Under
        # full activation checkpointing, the selected state is created by
        # recompute.
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError(
                "MXFP8Linear requires BF16 activations and weights; "
                f"got activation dtype {x.dtype} and weight dtype {weight.dtype}."
            )
        if bias_N is not None and bias_N.dtype != torch.bfloat16:
            raise ValueError(
                f"MXFP8Linear requires a BF16 bias; got bias dtype {bias_N.dtype}."
            )
        local_in_features = weight.shape[-1]
        local_out_features = math.prod(weight.shape[:-1])
        if x.shape[-1] != local_in_features:
            raise ValueError(
                "MXFP8Linear activation and weight contraction dimensions must "
                f"match; got {x.shape[-1]} and {local_in_features}."
            )
        for name, value in (
            ("local in_features", local_in_features),
            ("local out_features", local_out_features),
        ):
            if value % _MXFP8_BLOCK_SIZE:
                raise ValueError(
                    f"MXFP8Linear requires {name} divisible by "
                    f"{_MXFP8_BLOCK_SIZE}; got {value}."
                )

        input_shape = x.shape
        x_MK, num_rows = _pad_rows(x.reshape(-1, input_shape[-1]).contiguous())
        requires_wgrad = ctx.needs_input_grad[1]
        quantize_wgrad_input_in_forward = (
            requires_wgrad and input_activation_format_for_backward == "mxfp8"
        )

        # The save format controls both computation and saved state. BF16 mode
        # requests only the rowwise FPROP operand here; backward produces the
        # columnwise WGRAD operand from the saved BF16 input.
        # TODO(anijain2305): torchao's mxfp8_quantize_2d_{1x32,32x1}_cutedsl
        # fuse the cast and the scale swizzle into one kernel, replacing this
        # call plus the triton_mx_block_rearrange below. Measured 2.4-3.2x
        # faster than the pair on a GB200, bitwise identical on both outputs.
        # Three things to settle before switching:
        #   - They require the token count to be a multiple of 128, where this
        #     path needs only 32. The 32 is the MX scaling granularity, and the
        #     128 is the tcgen05 scale-tile height that scaled_mm wants either
        #     way -- splitting the two kernels is what lets
        #     triton_mx_block_rearrange pad the *scales* up to 128 rows and
        #     leave the token count alone. Fusing pushes that padding onto the
        #     activations, so a 64-token microbatch would run the quantizer and
        #     the GEMM over 128 rows. The speedup above was measured on shapes
        #     that already divide 128 and should not be assumed to hold once
        #     small token counts pay for the extra rows.
        #   - They need nvidia-cutlass-dsl and apache-tvm-ffi, which torchao
        #     does not depend on: MXFP8 dense linears work without them today,
        #     and switching would make them mandatory for every MXFP8 user.
        #   - The usable cutlass-dsl range is narrow. torchao's README asks for
        #     4.5.2; 4.6.0 changed the nvvm.cvt_packfloat* builders and breaks
        #     torchao's CuTeDSL kernels outright.
        # Their availability check also raises from inside the kernel, so a
        # missing package would surface on the first forward. Gate it in
        # MXFP8LinearConverter.__init__ instead, beside the torchao check.
        x_qdata_row_MK, x_qdata_col_MK, x_scale_row, x_scale_col = mxfp8_quantize_cuda(
            x_MK,
            rowwise=True,
            colwise=quantize_wgrad_input_in_forward,
            scaling_mode=_MXFP8_SCALING_MODE,
        )
        x_scale_row = triton_mx_block_rearrange(x_scale_row)
        if quantize_wgrad_input_in_forward:
            x_scale_col = triton_mx_block_rearrange(x_scale_col)

        # The 32x32 weight quantizer returns both qdata/scale pairs ready for
        # this exact BlockWise1x32 and SWIZZLE_32_4_4 B-operand contract.
        output_MN = F.scaled_mm(
            x_qdata_row_MK,
            weight_qdata_fprop_KN,
            scale_a=x_scale_row,
            scale_recipe_a=F.ScalingType.BlockWise1x32,
            scale_b=weight_scale_fprop_swizzled,
            scale_recipe_b=F.ScalingType.BlockWise1x32,
            swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            bias=bias_N,
            output_dtype=torch.bfloat16,
        )

        # Save exactly one input-activation operands for WGRAD. BF16 mode
        # keeps the original tensor and builds the columnwise operand in
        # backward. MXFP8 mode keeps the columnwise qdata and scales produced
        # above. FPROP and DGRAD share the same weight qdata allocation.
        # An unsharded tensor's storage is FSDP's to free at reshard and refill
        # before backward, so save the wrapper and read the operands off it
        # then. Anything else carries no operands to refill, so save them.
        has_unsharded_tensor = isinstance(weight, _UnshardedFSDPTensor)
        saved_weight_tensors = (
            (weight,)
            if has_unsharded_tensor
            else (weight_qdata_dgrad_NK, weight_scale_dgrad_swizzled)
        )
        if requires_wgrad and input_activation_format_for_backward == "bf16":
            ctx.save_for_backward(x, *saved_weight_tensors)
        else:
            ctx.save_for_backward(x_qdata_col_MK, x_scale_col, *saved_weight_tensors)
        ctx.has_unsharded_tensor = has_unsharded_tensor
        ctx.input_shape = input_shape
        ctx.num_rows = num_rows
        ctx.requires_dgrad = ctx.needs_input_grad[0]
        ctx.requires_wgrad = requires_wgrad
        ctx.input_activation_format_for_backward = input_activation_format_for_backward
        ctx.has_bias = bias_N is not None
        # The WGRAD GEMM yields a 2D [num_linears * N, K], but the returned
        # gradient must match the parameter, which is [num_linears, N, K] for a
        # stacked weight. Backward may hold only the flattened quantized
        # operands, so record the shape.
        ctx.weight_shape = weight.shape
        # Produce WGRAD directly in the parameter's gradient dtype so
        # AccumulateGrad needs no cast. FSDP will set the unsharded parameter's
        # grad_dtype to the reduce dtype.
        # TODO(anijain2305): drop the dtype fallback once FSDP always sets
        # grad_dtype on the unsharded parameter
        # (https://github.com/pytorch/pytorch/pull/194434).
        ctx.wgrad_dtype = weight.grad_dtype or weight.dtype
        # Kept on ctx rather than saved: backward needs this exact parameter
        # object, and saved-tensor hooks may unpack a different one. A leaf
        # parameter does not reference its graph, so this forms no cycle.
        ctx.weight_param = weight if accumulate_into_weight_grad else None

        return _unpad_output(output_MN, num_rows, input_shape)

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        # WGRAD consumes either the saved columnwise activation pair or a pair
        # rebuilt from the saved BF16 input. DGRAD consumes the weight pair.
        x_hp = None
        x_qdata_col_MK = None
        x_scale_col = None
        saved_tensors = ctx.saved_tensors
        if ctx.requires_wgrad and ctx.input_activation_format_for_backward == "bf16":
            x_hp = saved_tensors[0]
            saved_weight_tensors = saved_tensors[1:]
        else:
            x_qdata_col_MK, x_scale_col = saved_tensors[:2]
            saved_weight_tensors = saved_tensors[2:]

        if ctx.has_unsharded_tensor:
            (weight,) = saved_weight_tensors
            if not isinstance(weight, _UnshardedFSDPTensor):
                raise RuntimeError("FSDP restored an incompatible MXFP8 weight")
            operands = weight.operands
            weight_qdata_dgrad_NK = operands.weight_qdata_dgrad_NK
            weight_scale_dgrad_swizzled = operands.weight_scale_dgrad_swizzled
        else:
            weight_qdata_dgrad_NK, weight_scale_dgrad_swizzled = saved_weight_tensors

        grad_output_MN = grad_output.contiguous().reshape(-1, grad_output.shape[-1])
        grad_bias_N = grad_output_MN.sum(dim=0) if ctx.has_bias else None

        grad_input = None
        grad_weight = None
        if ctx.requires_dgrad or ctx.requires_wgrad:
            padded_grad_output_MN, _ = _pad_rows(grad_output_MN)
            (
                grad_output_row_MN,
                grad_output_col_MN,
                grad_output_row_scales,
                grad_output_col_scales,
            ) = mxfp8_quantize_cuda(
                padded_grad_output_MN,
                rowwise=ctx.requires_dgrad,
                colwise=ctx.requires_wgrad,
                scaling_mode=_MXFP8_SCALING_MODE,
            )

            if ctx.requires_dgrad:
                grad_output_row_scales = triton_mx_block_rearrange(
                    grad_output_row_scales
                )
                grad_input_MK = F.scaled_mm(
                    grad_output_row_MN,
                    weight_qdata_dgrad_NK,
                    scale_a=grad_output_row_scales,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=weight_scale_dgrad_swizzled,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                )
                grad_input = grad_input_MK[: ctx.num_rows].reshape(ctx.input_shape)

            if ctx.requires_wgrad:
                if ctx.input_activation_format_for_backward == "bf16":
                    assert x_hp is not None
                    x_qdata_col_MK, x_scale_col = _quantize_wgrad_input(x_hp)
                assert x_qdata_col_MK is not None
                assert x_scale_col is not None
                grad_weight = _weight_gradient(
                    grad_output_col_MN,
                    grad_output_col_scales,
                    x_qdata_col_MK,
                    x_scale_col,
                    weight_param=ctx.weight_param,
                    wgrad_dtype=ctx.wgrad_dtype,
                    weight_shape=ctx.weight_shape,
                )

        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            None,
            grad_bias_N,
            None,
            None,
        )


# Marks the function local-only so SPMD type checking can propagate through
# an autograd function it cannot see into.
# TODO(anijain2305, pianpwk): drop this once register_local_autograd_function
# is removed tree-wide. nvfp4 and qwen3_5's gdn still rely on the same
# registration, so it has to go everywhere at once.
spmd.register_local_autograd_function(_MXFP8LinearFunction)


# Each linear contributes four inputs: its parameter, DGRAD qdata, FPROP scales,
# and DGRAD scales. The FPROP qdata is the transposed DGRAD qdata.
_INPUTS_PER_SHARED_INPUT_LINEAR = 4


@torch._dynamo.allow_in_graph
class _MXFP8SharedInputLinearsFunction(torch.autograd.Function):
    """Bias-free MXFP8 linears that read one input, quantized once.

    Separate ``_MXFP8LinearFunction`` calls quantize the shared input once per
    linear in forward and again per linear for WGRAD, then sum the DGRADs with a
    BF16 add. Here the input is quantized once, and each DGRAD after the first
    accumulates into the previous one in the GEMM epilogue. FPROP and WGRAD are
    unchanged per linear.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x: torch.Tensor,
        input_activation_format_for_backward: InputActivationFormatForBackward,
        accumulate_into_weight_grad: tuple[bool, ...],
        *linear_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        weights = linear_inputs[0::_INPUTS_PER_SHARED_INPUT_LINEAR]
        weight_qdata_dgrad_NK = linear_inputs[1::_INPUTS_PER_SHARED_INPUT_LINEAR]
        weight_scale_fprop_swizzled = linear_inputs[2::_INPUTS_PER_SHARED_INPUT_LINEAR]
        weight_scale_dgrad_swizzled = linear_inputs[3::_INPUTS_PER_SHARED_INPUT_LINEAR]
        if x.dtype != torch.bfloat16 or any(
            weight.dtype != torch.bfloat16 for weight in weights
        ):
            raise ValueError("MXFP8 linears require BF16 activations and weights.")
        input_shape = x.shape
        x_MK, num_rows = _pad_rows(x.reshape(-1, input_shape[-1]).contiguous())
        requires_wgrad = tuple(
            ctx.needs_input_grad[3 + _INPUTS_PER_SHARED_INPUT_LINEAR * index]
            for index in range(len(weights))
        )
        save_bf16_input = (
            any(requires_wgrad) and input_activation_format_for_backward == "bf16"
        )
        quantize_wgrad_input_in_forward = (
            any(requires_wgrad) and input_activation_format_for_backward == "mxfp8"
        )
        x_qdata_row_MK, x_qdata_col_MK, x_scale_row, x_scale_col = mxfp8_quantize_cuda(
            x_MK,
            rowwise=True,
            colwise=quantize_wgrad_input_in_forward,
            scaling_mode=_MXFP8_SCALING_MODE,
        )
        x_scale_row = triton_mx_block_rearrange(x_scale_row)
        if quantize_wgrad_input_in_forward:
            x_scale_col = triton_mx_block_rearrange(x_scale_col)

        outputs = tuple(
            _unpad_output(
                F.scaled_mm(
                    x_qdata_row_MK,
                    qdata_dgrad_NK.t(),
                    scale_a=x_scale_row,
                    scale_recipe_a=F.ScalingType.BlockWise1x32,
                    scale_b=scale_fprop_swizzled,
                    scale_recipe_b=F.ScalingType.BlockWise1x32,
                    swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                    swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    output_dtype=torch.bfloat16,
                ),
                num_rows,
                input_shape,
            )
            for qdata_dgrad_NK, scale_fprop_swizzled in zip(
                weight_qdata_dgrad_NK, weight_scale_fprop_swizzled, strict=True
            )
        )

        # As in _MXFP8LinearFunction: save an unsharded weight's wrapper so FSDP
        # can refill its operands before backward, else the DGRAD operands.
        has_unsharded_tensor = tuple(
            isinstance(weight, _UnshardedFSDPTensor) for weight in weights
        )
        saved_weight_tensors = []
        for weight, has_unsharded, qdata_dgrad_NK, scale_dgrad_swizzled in zip(
            weights,
            has_unsharded_tensor,
            weight_qdata_dgrad_NK,
            weight_scale_dgrad_swizzled,
            strict=True,
        ):
            saved_weight_tensors += (
                [weight] if has_unsharded else [qdata_dgrad_NK, scale_dgrad_swizzled]
            )
        saved_input_tensors = (x,) if save_bf16_input else (x_qdata_col_MK, x_scale_col)
        ctx.save_for_backward(*saved_input_tensors, *saved_weight_tensors)
        ctx.save_bf16_input = save_bf16_input
        ctx.has_unsharded_tensor = has_unsharded_tensor
        ctx.input_shape = input_shape
        ctx.num_rows = num_rows
        ctx.requires_dgrad = ctx.needs_input_grad[0]
        ctx.requires_wgrad = requires_wgrad
        ctx.weight_shapes = tuple(weight.shape for weight in weights)
        ctx.wgrad_dtypes = tuple(
            weight.grad_dtype or weight.dtype for weight in weights
        )
        ctx.weight_params = tuple(
            weight if accumulate else None
            for weight, accumulate in zip(
                weights, accumulate_into_weight_grad, strict=True
            )
        )
        # An unused output's gradient stays None instead of becoming zeros that
        # backward would quantize and multiply.
        ctx.set_materialize_grads(False)
        return outputs

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, *grad_outputs: torch.Tensor | None):
        saved_tensors = ctx.saved_tensors
        if ctx.save_bf16_input:
            x_qdata_col_MK, x_scale_col = _quantize_wgrad_input(saved_tensors[0])
            saved_weight_tensors = list(saved_tensors[1:])
        else:
            x_qdata_col_MK, x_scale_col = saved_tensors[:2]
            saved_weight_tensors = list(saved_tensors[2:])

        grad_input_MK = None
        grad_linear_inputs = []
        for index, grad_output in enumerate(grad_outputs):
            if ctx.has_unsharded_tensor[index]:
                weight = saved_weight_tensors.pop(0)
                if not isinstance(weight, _UnshardedFSDPTensor):
                    raise RuntimeError("FSDP restored an incompatible MXFP8 weight")
                weight_qdata_dgrad_NK = weight.operands.weight_qdata_dgrad_NK
                weight_scale_dgrad_swizzled = (
                    weight.operands.weight_scale_dgrad_swizzled
                )
            else:
                weight_qdata_dgrad_NK = saved_weight_tensors.pop(0)
                weight_scale_dgrad_swizzled = saved_weight_tensors.pop(0)

            requires_wgrad = ctx.requires_wgrad[index] and grad_output is not None
            grad_weight = None
            if grad_output is not None and (ctx.requires_dgrad or requires_wgrad):
                padded_grad_output_MN, _ = _pad_rows(
                    grad_output.contiguous().reshape(-1, grad_output.shape[-1])
                )
                (
                    grad_output_row_MN,
                    grad_output_col_MN,
                    grad_output_row_scales,
                    grad_output_col_scales,
                ) = mxfp8_quantize_cuda(
                    padded_grad_output_MN,
                    rowwise=ctx.requires_dgrad,
                    colwise=requires_wgrad,
                    scaling_mode=_MXFP8_SCALING_MODE,
                )
                if ctx.requires_dgrad:
                    dgrad_kwargs = dict(
                        scale_a=triton_mx_block_rearrange(grad_output_row_scales),
                        scale_recipe_a=F.ScalingType.BlockWise1x32,
                        scale_b=weight_scale_dgrad_swizzled,
                        scale_recipe_b=F.ScalingType.BlockWise1x32,
                        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                    )
                    if grad_input_MK is None:
                        grad_input_MK = F.scaled_mm(
                            grad_output_row_MN,
                            weight_qdata_dgrad_NK,
                            output_dtype=torch.bfloat16,
                            **dgrad_kwargs,
                        )
                    else:
                        F.scaled_addmm_(
                            grad_input_MK,
                            grad_output_row_MN,
                            weight_qdata_dgrad_NK,
                            **dgrad_kwargs,
                        )
                if requires_wgrad:
                    grad_weight = _weight_gradient(
                        grad_output_col_MN,
                        grad_output_col_scales,
                        x_qdata_col_MK,
                        x_scale_col,
                        weight_param=ctx.weight_params[index],
                        wgrad_dtype=ctx.wgrad_dtypes[index],
                        weight_shape=ctx.weight_shapes[index],
                    )
            grad_linear_inputs += [grad_weight, None, None, None]

        grad_input = (
            None
            if grad_input_MK is None
            else grad_input_MK[: ctx.num_rows].reshape(ctx.input_shape)
        )
        return (grad_input, None, None, *grad_linear_inputs)


spmd.register_local_autograd_function(_MXFP8SharedInputLinearsFunction)


class MXFP8Linear(Linear):
    """Linear using 1D activations and cached 32x32 weight quantization."""

    WEIGHT_BLOCK_SIZE = _MXFP8_BLOCK_SIZE

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for ``Linear.Config``."""

        input_activation_format_for_backward: InputActivationFormatForBackward = "bf16"
        """Format used to save the input activation needed by WGRAD.

        ``"bf16"`` saves the original input and quantizes it columnwise during
        backward. ``"mxfp8"`` produces the columnwise operands during
        forward and saves its qdata and scales for backward.
        """

        def __post_init__(self) -> None:
            if (
                self.input_activation_format_for_backward
                not in _INPUT_ACTIVATION_FORMATS_FOR_BACKWARD
            ):
                raise ValueError(
                    "MXFP8 input_activation_format_for_backward must be one of "
                    f"{_INPUT_ACTIVATION_FORMATS_FOR_BACKWARD}; got "
                    f"{self.input_activation_format_for_backward!r}."
                )
            for name in ("in_features", "out_features"):
                value = getattr(self, name)
                if value % _MXFP8_BLOCK_SIZE:
                    raise ValueError(
                        f"MXFP8 requires {name} divisible by {_MXFP8_BLOCK_SIZE}; "
                        f"got {name}={value}."
                    )

    def __init__(self, config: Config):
        super().__init__(config)
        self.input_activation_format_for_backward = (
            config.input_activation_format_for_backward
        )
        # Install the unsharded-tensor wrapper up front so no caller has to
        # remember to do it. The wrapper is inert until a data parallel
        # implementation drives its unshard lifecycle: until then it just holds
        # the BF16 weight, and forward rejects it.
        self.weight = nn.Parameter(
            _LinearShardedTensorWithMXFP8Compute(self.weight.data),
            requires_grad=self.weight.requires_grad,
        )

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        physical_weight, operands, accumulate_into_weight_grad = self._weight_operands()
        output = _MXFP8LinearFunction.apply(
            input,
            physical_weight,
            operands.weight_qdata_fprop_KN,
            operands.weight_scale_fprop_swizzled,
            operands.weight_qdata_dgrad_NK,
            operands.weight_scale_dgrad_swizzled,
            bias,
            self.input_activation_format_for_backward,
            accumulate_into_weight_grad,
        )
        return output

    def _weight_operands(
        self,
    ) -> tuple[torch.Tensor, _MXFP8LinearOperands, bool]:
        """Return the parameter, its MXFP8 operands, and whether WGRAD may
        accumulate into the parameter's ``.grad`` in place."""
        # The autograd function takes the parameter itself rather than
        # ``weight``, its flattened view, so a stacked parameter's gradient
        # reaches AccumulateGrad without the view's backward casting it. Always
        # a plain tensor: spmd_types carries TP and EP as annotations instead
        # of wrapping the weight as a model-parallel DTensor.
        physical_weight = self.weight
        local_out_features = physical_weight.shape[-2]
        if local_out_features % _MXFP8_BLOCK_SIZE:
            raise ValueError(
                "MXFP8 requires local out_features divisible by "
                f"{_MXFP8_BLOCK_SIZE}; got {local_out_features}. Adjust the "
                "Linear out_features or TP degree so quantization blocks do "
                "not span projection boundaries."
            )
        # __init__ installs a _LinearShardedTensorWithMXFP8Compute, but that is
        # not what forward usually sees. Under FSDP the post-all-gather hook has
        # already replaced it for this unshard lifetime with the storage-free
        # _UnshardedFSDPTensor holding the quantized operands, so the weight
        # arrives here already quantized and the type identifies which state we
        # are in.
        if isinstance(physical_weight, _UnshardedFSDPTensor):
            # Read operands from the physical wrapper. Dynamo can source the
            # module parameter, but not a temporary tensor-subclass view of it.
            operands = physical_weight.operands
        else:
            # No data parallel implementation owns this weight's lifecycle, so
            # it still holds high-precision storage and the operands are built
            # per invocation. Eager FSDP2 always installs an unsharded tensor, but
            # GraphTrainer under the spmd_types backend does not: its runtime
            # hands forward a plain annotated local tensor, so the wrapper
            # SimpleFSDP's parametrization built never reaches here. Quantize
            # the storage rather than the wrapper, which the kernels cannot
            # consume; the parameter itself stays wrapped so autograd returns
            # the gradient to it.
            with torch.no_grad():
                high_precision_weight = (
                    physical_weight._tensor
                    if isinstance(physical_weight, _LinearShardedTensorWithMXFP8Compute)
                    else physical_weight
                )
                operands = _quantize_mxfp8_weight(high_precision_weight.flatten(0, -2))
            # Nothing caches this across calls, so a frozen weight is
            # requantized on every forward. Training pays that anyway, since
            # the weight changes each optimizer step; inference does not.
            # TODO(anijain2305): key the operands on the parameter's
            # version counter so a frozen weight is quantized once.
        # Backward folds later WGRADs into the leaf parameter's running .grad.
        # Dynamo sets is_compiling; GraphTrainer's make_fx tracer does not, so
        # ask the proxy mode as well. A traced backward cannot represent this
        # read-and-clear of parameter.grad, so it uses an ordinary WGRAD.
        # TODO(graph_trainer): add a GraphTrainer graph pass that rewrites the
        # WGRAD scaled_mm plus gradient accumulation into scaled_addmm_.
        # SimpleFSDP hands forward a parametrization output rather than the
        # leaf, whose .grad autograd never populates.
        is_tracing = torch.compiler.is_compiling() or get_proxy_mode() is not None
        accumulate_into_weight_grad = not is_tracing and physical_weight.is_leaf
        return physical_weight, operands, accumulate_into_weight_grad


def mxfp8_linears_shared_input(
    input: torch.Tensor, linears: Sequence[MXFP8Linear]
) -> tuple[torch.Tensor, ...]:
    """Apply MXFP8 linears that read the same input, quantizing the input once.

    Matches ``tuple(linear(input) for linear in linears)`` bitwise: the same
    parameters, outputs, and gradients. Each DGRAD after the first is added in
    the GEMM epilogue, which rounds like the separate BF16 add it replaces.

    The input is saved for backward once: in MXFP8 if every linear's
    ``input_activation_format_for_backward`` is ``"mxfp8"``, else in BF16.

    Args:
        input: Activation shaped ``(..., in_features)``.
        linears: Bias-free, unstacked MXFP8 linears with equal ``in_features``.

    Returns:
        One output per linear, in order.

    Example:
        q_a, kv_down = mxfp8_linears_shared_input(x, (attn.wq_a, attn.wkv_a))
    """
    if any(linear.bias is not None or linear.num_linears != 1 for linear in linears):
        raise ValueError(
            "mxfp8_linears_shared_input supports only bias-free, unstacked linears."
        )
    linear_inputs = []
    accumulate_into_weight_grad = []
    for linear in linears:
        physical_weight, operands, accumulate = linear._weight_operands()
        linear_inputs += [
            physical_weight,
            operands.weight_qdata_dgrad_NK,
            operands.weight_scale_fprop_swizzled,
            operands.weight_scale_dgrad_swizzled,
        ]
        accumulate_into_weight_grad.append(accumulate)
    input_activation_format_for_backward = (
        "mxfp8"
        if all(
            linear.input_activation_format_for_backward == "mxfp8" for linear in linears
        )
        else "bf16"
    )
    return _MXFP8SharedInputLinearsFunction.apply(
        input,
        input_activation_format_for_backward,
        tuple(accumulate_into_weight_grad),
        *linear_inputs,
    )
