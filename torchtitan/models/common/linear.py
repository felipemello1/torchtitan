# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurable linear modules.

``Linear`` uses diamond inheritance (``nn.Linear`` + ``Module``) so that:
- The module hierarchy stays flat (no extra wrapper layer).
- Standard ``nn.Linear`` parameter and state-dict behavior is retained.
- The ``Module`` protocol is satisfied and ``build()`` is inherited
  from ``Configurable.Config``.
"""

import functools
import logging
import math
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.protocols.module import Module
from torchtitan.tools.utils import device_type

logger = logging.getLogger(__name__)

# Shape suffix legend:
#   X = leading input dims, T = num tokens, D = input features (model dimension for the router
#   gate), O = output features, E = num experts, R = routed rows grouped by expert,
#   I = grouped linear input features


class Linear(nn.Linear, Module):
    """Configurable linear that can store multiple stacked projections.

    A single projection keeps the standard ``[out_features, in_features]``
    parameter shape. Multiple projections use
    ``[num_linears, out_features, in_features]``, keeping each projection
    contiguous for blockwise weight quantization, and return
    ``[..., num_linears, out_features]``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        num_linears: int = 1
        bias: bool = False

    fp32_weight_grad: bool = False
    """Write the weight gradient in fp32 instead of bf16 (see ``enable_fp32_weight_grads``)."""

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.num_linears * config.out_features,
            bias=config.bias,
        )
        self.out_features = config.out_features
        self.num_linears = config.num_linears
        if config.num_linears > 1:
            self.weight = nn.Parameter(
                self.weight.detach().unflatten(
                    0, (config.num_linears, config.out_features)
                ),
                requires_grad=self.weight.requires_grad,
            )
            if self.bias is not None:
                self.bias = nn.Parameter(
                    self.bias.detach().unflatten(
                        0, (config.num_linears, config.out_features)
                    ),
                    requires_grad=self.bias.requires_grad,
                )

    def reset_parameters(self) -> None:
        # Flattening handles both ordinary and stacked projections while
        # keeping fan-in equal to in_features.
        nn.init.kaiming_uniform_(self.weight.flatten(0, -2), a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _flatten_weight_and_bias(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flatten stacked parameters for one linear operation."""
        weight = self.weight.flatten(0, -2)
        bias = None if self.bias is None else self.bias.flatten()
        return weight, bias

    def _unflatten_output(self, output: torch.Tensor) -> torch.Tensor:
        """Restore the logical stacked output dimensions after a linear operation."""
        if self.num_linears == 1:
            return output
        return output.unflatten(-1, self.weight.shape[:-1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self._flatten_weight_and_bias()
        output = self._linear(input, weight, bias)
        return self._unflatten_output(output)

    def extra_repr(self) -> str:
        result = nn.Linear.extra_repr(self)
        if self.num_linears > 1:
            result += f", num_linears={self.num_linears}"
        return result

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply local projection compute without outer communication.

        LoRA and quantized subclasses override this method so column- and
        row-parallel ``forward`` methods continue to own their collectives.
        Explicit operands let those boundaries adjust an operand's SPMD type
        before invoking the selected local compute implementation.
        """
        if self.fp32_weight_grad:
            return _Fp32WeightGradLinearFunction.apply(input, weight, bias, self.weight)
        return F.linear(input, weight, bias)


class CastLinear(Linear):
    """``Linear`` whose forward matmul runs in ``compute_dtype``.

    Inputs, weight, and bias are cast to ``compute_dtype`` before
    ``F.linear`` and the output is returned in that dtype. The stored
    parameters retain their original dtype, including under weight tying.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        compute_dtype: str = "float32"
        """Dtype for the forward matmul (key into ``TORCH_DTYPE_MAP``)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.compute_dtype = TORCH_DTYPE_MAP[config.compute_dtype]

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # The optimizer updates the weight each step, so training cannot cache
        # the upcast copy. Inference may be able to cache it between syncs.
        bias = None if bias is None else bias.to(self.compute_dtype)
        return F.linear(
            input.to(self.compute_dtype), weight.to(self.compute_dtype), bias
        )


@spmd.register_local_autograd_function
class _Fp32WeightGradLinearFunction(torch.autograd.Function):
    """``F.linear`` whose weight and bias gradients come out in fp32.

    With bf16 input and weight, every product in ``grad_output.T @ input`` is exact in fp32 and
    the GEMM accumulates in fp32, but ``F.linear``'s backward then writes the weight gradient in
    bf16. Writing it in fp32 keeps those bits for a parameter that accumulates fp32 gradients
    (``Tensor.grad_dtype``). The forward and the input gradient are the same as ``F.linear``'s.

    ``weight_OD`` may be a flattened view of the stacked ``weight_param``. The weight gradient is
    returned for ``weight_param`` itself: through the view, autograd would round it back to the
    view's bf16 dtype.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        input_XD: torch.Tensor,
        weight_OD: torch.Tensor,
        bias_O: torch.Tensor | None,
        weight_param: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(input_XD, weight_OD)
        ctx.weight_param_shape = weight_param.shape
        return F.linear(input_XD, weight_OD, bias_O)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_XO: torch.Tensor):  # pyrefly: ignore[bad-override]
        input_XD, weight_OD = ctx.saved_tensors
        grad_input_XD = grad_weight_OD = grad_bias_O = None
        grad_output_TO = grad_output_XO.reshape(-1, grad_output_XO.shape[-1])

        if ctx.needs_input_grad[0]:
            grad_input_XD = grad_output_XO.matmul(weight_OD)
        if ctx.needs_input_grad[3]:
            input_TD = input_XD.reshape(-1, input_XD.shape[-1])
            if grad_output_TO.is_cuda:
                grad_weight_OD = torch.mm(
                    grad_output_TO.T, input_TD, out_dtype=torch.float32
                )
            else:
                # aten::mm.dtype (bf16 inputs, fp32 output) is only implemented for CUDA/ROCm.
                grad_weight_OD = torch.mm(grad_output_TO.T.float(), input_TD.float())
        if ctx.needs_input_grad[2]:
            grad_bias_O = grad_output_TO.sum(0, dtype=torch.float32)

        grad_weight_param = (
            None
            if grad_weight_OD is None
            else grad_weight_OD.view(ctx.weight_param_shape)
        )
        return grad_input_XD, None, grad_bias_O, grad_weight_param


class ColumnParallelLinear(Linear):
    """Prepare an input for a column-parallel Linear.

    This is a ``Linear`` rather than a wrapper around one, so its parameter
    FQNs remain unchanged. The same module handles both tensor-parallel modes.
    With sequence parallelism, ``Shard(0) -> Replicate`` is an input all-gather.
    Without sequence parallelism, ``Invariant -> Replicate`` is a forward no-op
    whose backward performs the required all-reduce.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            input = spmd.redistribute(
                input,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
                backward_options={"op_dtype": input.dtype},
            )
        return super().forward(input)


class RowParallelLinear(Linear):
    """Reduce the partial output of an independently configured Linear.

    This is a ``Linear`` rather than a wrapper around one, so its parameter
    FQNs remain unchanged. ``Partial -> Shard(0)`` is a reduce-scatter, while
    ``Partial -> Invariant`` is an all-reduce without it. Dense SP state selects
    between the two. An invariant bias is converted to a partial contribution
    before local compute so the reduction adds it exactly once.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        weight, bias = self._flatten_weight_and_bias()
        if bias is not None and tp_group is not None:
            bias = spmd.convert(
                bias,
                tp_group,
                src=spmd.I,
                dst=spmd.P,
                expert_mode=True,
            )
            # The selected local compute may be native, LoRA, or quantized.
            # Its row-sharded operands and bias jointly produce a partial output.
            # TODO: Remove this suppression once spmd_types recognizes the
            # rowwise F.linear type combination [V, V, P] -> P.
            with spmd.no_typecheck():
                output = self._unflatten_output(self._linear(input, weight, bias))
            if spmd.is_type_checking():
                spmd.assert_local_type_like(
                    output,
                    input,
                    {tp_group: spmd.P},  # pyrefly: ignore [bad-argument-type]
                )
        else:
            output = self._unflatten_output(self._linear(input, weight, bias))
        if tp_group is None:
            return output

        return spmd.redistribute(
            output,
            tp_group,
            src=spmd.P,
            dst=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
            backward_options={"op_dtype": output.dtype},
        )


class GroupedLinear(Module):
    """A collection of linears selected by cumulative group offsets.

    Like :class:`Linear`, ``num_linears`` retains a projection axis in parameter
    storage. For example, a fused gate/up projection stores ``[E, 2, F, D]``
    and returns ``[R, 2, F]`` while grouped GEMM consumes its zero-copy
    ``[E, 2F, D]`` view.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Configure a grouped linear.

        Attributes:
            group_size: Number of independently selected linear weights.
            in_features: Input features for each linear.
            out_features: Output features for each linear.
            num_linears: Number of stacked linears per element in the group.
                Values greater than one retain a projection axis before
                ``out_features``.
        """

        group_size: int
        in_features: int
        out_features: int
        num_linears: int = 1

    fp32_weight_grad: bool = False
    """Write the weight gradient in fp32 instead of bf16 (see ``enable_fp32_weight_grads``)."""

    def __init__(self, config: Config):
        super().__init__()
        self.group_size = config.group_size
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.num_linears = config.num_linears
        output_shape = (
            (config.out_features,)
            if config.num_linears == 1
            else (config.num_linears, config.out_features)
        )
        self.weight = nn.Parameter(
            torch.empty(config.group_size, *output_shape, config.in_features)
        )

    def forward(self, input_RI: torch.Tensor, offsets_E: torch.Tensor) -> torch.Tensor:
        """Apply each grouped linear to rows selected by cumulative offsets.

        Args:
            input_RI: Input rows grouped by the selected linear.
            offsets_E: Exclusive cumulative row end for each group.

        Returns:
            Output rows with an optional ``num_linears`` axis before the output
            feature axis.
        """
        output_shape = self.weight.shape[1:-1]
        weight_EOI = self.weight.flatten(1, -2)
        output_RO = self._grouped_mm(
            input_RI=input_RI,
            weight_EOI=weight_EOI,
            offsets_E=offsets_E,
        )
        return output_RO.reshape(*output_RO.shape[:-1], *output_shape)

    def _grouped_mm(
        self,
        *,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        """Execute ``input_RI @ weight_EOI.transpose(-2, -1)`` by expert."""
        if self.fp32_weight_grad:
            return _Fp32WeightGradGroupedMMFunction.apply(
                input_RI, weight_EOI.bfloat16(), offsets_E, self.weight
            )
        return torch._grouped_mm(
            input_RI,
            weight_EOI.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )


@spmd.register_local_autograd_function
class _Fp32WeightGradGroupedMMFunction(torch.autograd.Function):
    """``torch._grouped_mm`` whose weight gradient comes out in fp32.

    The grouped counterpart of ``_Fp32WeightGradLinearFunction``: the forward and the input
    gradient are the grouped GEMMs autograd runs for ``torch._grouped_mm``, and the weight
    gradient is written in fp32 and returned for ``weight_param`` (``weight_EOI`` may be its
    flattened view).
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
        weight_param: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(input_RI, weight_EOI, offsets_E)
        ctx.weight_param_shape = weight_param.shape
        return torch._grouped_mm(input_RI, weight_EOI.transpose(-2, -1), offs=offsets_E)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_RO: torch.Tensor):  # pyrefly: ignore[bad-override]
        input_RI, weight_EOI, offsets_E = ctx.saved_tensors
        grad_input_RI = grad_weight_param = None
        if ctx.needs_input_grad[0]:
            grad_input_RI = torch._grouped_mm(
                grad_output_RO, weight_EOI, offs=offsets_E
            )
        if ctx.needs_input_grad[3]:
            grad_weight_EOI = torch._grouped_mm(
                grad_output_RO.transpose(-2, -1),
                input_RI,
                offs=offsets_E,
                out_dtype=torch.float32,
            )
            grad_weight_param = grad_weight_EOI.view(ctx.weight_param_shape)
        return grad_input_RI, None, None, grad_weight_param


@functools.cache
def _grouped_mm_writes_fp32() -> bool:
    """Whether ``torch._grouped_mm`` writes an fp32 output from bf16 inputs on this device."""
    # TODO: drop once torch._grouped_mm supports out_dtype=float32 for bf16 inputs upstream.
    try:
        input_RI = torch.zeros(16, 16, dtype=torch.bfloat16, device=device_type)
        offsets_E = torch.full((1,), 16, dtype=torch.int32, device=device_type)
        torch._grouped_mm(input_RI.T, input_RI, offs=offsets_E, out_dtype=torch.float32)
    except RuntimeError as error:
        logger.warning(
            "GroupedLinear weight gradients stay bf16: torch._grouped_mm cannot write fp32 "
            f"from bf16 inputs here ({error})"
        )
        return False
    return True


@spmd.register_local_autograd_function
class _RouterGateLinearFunction(torch.autograd.Function):
    """Router projection with FP32 output and backward GEMMs."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, input_TD: torch.Tensor, weight_ED: torch.Tensor, fp32_weight_grad: bool
    ) -> torch.Tensor:
        use_cuda_bf16_forward = (
            input_TD.device.type == "cuda"
            and input_TD.dtype is torch.bfloat16
            and weight_ED.dtype is torch.bfloat16
        )
        if use_cuda_bf16_forward:
            input_forward_TD = input_TD
            weight_forward_ED = weight_ED
            # CUDA supports BF16 matmul with FP32 accumulation and output via
            # out_dtype. The portable path below promotes the operands because
            # this mixed input/output dtype is not supported by all devices.
            output_TE = torch.mm(
                input_forward_TD, weight_forward_ED.T, out_dtype=torch.float32
            )
        else:
            input_forward_TD = input_TD.float()
            weight_forward_ED = weight_ED.float()
            output_TE = torch.mm(input_forward_TD, weight_forward_ED.T)

        ctx.save_for_backward(input_forward_TD, weight_forward_ED)
        ctx.input_dtype = input_TD.dtype
        ctx.weight_grad_dtype = torch.float32 if fp32_weight_grad else weight_ED.dtype
        return output_TE

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TE: torch.Tensor):  # pyrefly: ignore[bad-override]
        input_forward_TD, weight_forward_ED = ctx.saved_tensors
        grad_output_fp32_TE = grad_output_TE.float()

        grad_input_TD = None
        if ctx.needs_input_grad[0]:
            grad_input_TD = torch.mm(grad_output_fp32_TE, weight_forward_ED.float()).to(
                ctx.input_dtype
            )

        grad_weight_ED = None
        if ctx.needs_input_grad[1]:
            grad_weight_ED = torch.mm(
                grad_output_fp32_TE.T, input_forward_TD.float()
            ).to(ctx.weight_grad_dtype)

        return grad_input_TD, grad_weight_ED, None


class RouterGateLinear(Linear):
    """Router projection with FP32 output and backward compute.

    CUDA uses BF16 forward compute when both operands are BF16. All other
    forward paths use FP32 compute.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        output_TE = _RouterGateLinearFunction.apply(
            input, weight, self.fp32_weight_grad
        )
        if bias is not None:
            output_TE = output_TE + bias.float()
        return output_TE


__all__ = [
    "CastLinear",
    "ColumnParallelLinear",
    "GroupedLinear",
    "Linear",
    "RowParallelLinear",
    "RouterGateLinear",
]
