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

import math
from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.protocols.module import Module

# Shape suffix legend:
#   T = num tokens, D = input features, O = output features, E = num experts (router gate)


@spmd.register_local_autograd_function
class _Fp32OutputLinearFunction(torch.autograd.Function):
    """``input @ weight.T`` on bf16 operands, returning fp32, with a backward close to fp32's.

    Why a custom autograd Function: ``torch.mm(..., out_dtype=torch.float32)`` has no autograd
    formula (calling it on tensors that require grad raises "derivative for aten::mm is not
    implemented"), so both passes are written here. For ``output = input @ weight.T``:

        grad_input = grad_output @ weight        grad_weight = grad_output.T @ input

    Forward: tensor cores multiply bf16 numbers exactly and add the products in fp32, so
    ``out_dtype=torch.float32`` only skips the final rounding of the result to bf16. No fp32
    copy of the weight is made.

    Backward: ``grad_output`` is fp32 (the output was fp32), but tensor cores only take bf16,
    which keeps 8 significant bits of fp32's 24. Two ways to handle that:
    - Round ``grad_output`` to bf16: one bf16 GEMM per gradient, but those bits are lost.
    - An fp32 matmul (on Blackwell, TorchTitan runs these as BF16x9: split both operands into
      3 bf16 pieces each and run 9 bf16 GEMMs): keeps the bits, ~9x the work.

    Here ``input`` and ``weight`` are already exactly bf16; only ``grad_output`` has extra bits.
    So splitting ``grad_output`` into 2 bf16 pieces is enough, and each gradient is the sum of
    2 bf16 GEMMs:

        grad_output = hi + lo     hi: its top 16 bits (exactly a bf16), lo: bf16(grad_output - hi)
        e.g. 0.1 = 0.099609375 + 0.000391006   (off by 4e-7; bf16(0.1) = 0.1000977 is off by 1e-4)

        grad_input  = hi @ weight + lo @ weight
        grad_weight = hi.T @ input + lo.T @ input

    Each gradient stays one GEMM call: ``[hi; lo]`` is stacked either along the dimension the
    GEMM sums over (the GEMM adds the two terms in its fp32 accumulator), or along an output
    dimension whose two halves are added afterwards. The layout duplicates the smaller tensor:
    an LM head has out_features (vocab) >> tokens, a router has out_features (experts) << tokens.

    Qwen3.5-27B LM head (248320 x 5120 weight), 2048 real tokens, fwd + cross-entropy + bwd:

        backward                            time    grad_input error vs fp64
        round grad_output to bf16           12 ms   2.29e-3
        hi + lo (this)                      19 ms   1.70e-3
        fp32 matmul via BF16x9              54 ms   1.66e-3
        exact gradients rounded to bf16             1.66e-3   <- floor: gradients return in bf16

    The split itself loses almost nothing: summing the same hi + lo with an fp32 GEMM gives
    1.66e-3. The extra 2% comes from the bf16 GEMM adding up 248320 products per output in its
    fp32 accumulator; grad_weight (2048 products per output) matches fp32 exactly (1.64e-3).

    Qwen3.5-35B-A3B routers (2048 -> 256 experts, 40 layers), 64k tokens, fwd + bwd GPU time:
    44.7 ms with BF16x9, 14.4 ms with this, same gradient error (1.66e-3).
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, input_TD: torch.Tensor, weight_OD: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(input_TD, weight_OD)
        return torch.mm(input_TD, weight_OD.T, out_dtype=torch.float32)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TO: torch.Tensor):  # pyrefly: ignore[bad-override]
        input_TD, weight_OD = ctx.saved_tensors
        hi_TO, lo_TO = _split_into_bf16_hi_lo(grad_output_TO)
        num_tokens, out_features = hi_TO.shape
        grad_input_TD = grad_weight_OD = None

        if out_features > num_tokens:
            # LM head: stack [hi; lo] along T, so only T-sized tensors get duplicated.
            stacked_2TO = torch.cat([hi_TO, lo_TO])
            if ctx.needs_input_grad[0]:
                # [hi; lo] @ W = [hi @ W; lo @ W]: add the two halves.
                halves_2TD = torch.mm(stacked_2TO, weight_OD, out_dtype=torch.float32)
                grad_input_TD = halves_2TD[:num_tokens] + halves_2TD[num_tokens:]
                grad_input_TD = grad_input_TD.to(input_TD.dtype)
            if ctx.needs_input_grad[1]:
                # [hi; lo].T @ [x; x] = hi.T @ x + lo.T @ x, summed in the GEMM.
                grad_weight_OD = torch.mm(
                    stacked_2TO.T, torch.cat([input_TD, input_TD])
                )
        else:
            # Router: stack [hi | lo] along O, so only the small weight gets duplicated.
            if ctx.needs_input_grad[0]:
                # [hi | lo] @ [W; W] = hi @ W + lo @ W, summed in the GEMM.
                grad_input_TD = torch.mm(
                    torch.cat([hi_TO, lo_TO], dim=1), torch.cat([weight_OD, weight_OD])
                )
            if ctx.needs_input_grad[1]:
                # Two GEMMs with small [O, D] fp32 outputs.
                grad_weight_OD = torch.mm(hi_TO.T, input_TD, out_dtype=torch.float32)
                grad_weight_OD += torch.mm(lo_TO.T, input_TD, out_dtype=torch.float32)
                grad_weight_OD = grad_weight_OD.to(weight_OD.dtype)

        return grad_input_TD, grad_weight_OD


# The fp32 bits that bf16 keeps: sign, exponent and the top 7 mantissa bits (0xFFFF0000).
_BF16_BITS_OF_FP32 = -65536


def _split_into_bf16_hi_lo(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an fp32 tensor into bf16 ``hi`` and ``lo`` with ``hi + lo`` within 2^-15 of it."""
    # Clearing the low bits makes hi exactly a bf16 value and tensor - hi exact in fp32. A bit
    # mask rather than .to(bf16): torch.compile folds a bf16 round trip away, making lo zero.
    hi = (tensor.view(torch.int32) & _BF16_BITS_OF_FP32).view(torch.float32)
    return hi.to(torch.bfloat16), (tensor - hi).to(torch.bfloat16)


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
        matmul_mode: Literal[
            "default", "bf16_matmul_fp32_out", "upcast_fp32_matmul"
        ] = "default"
        """How the matmul treats its operands.
        "default": F.linear on the operands as given; output in their dtype.
        "bf16_matmul_fp32_out": bf16 operands as given, fp32 accumulation and output;
        backward within ~2% of an fp32 backward's error (see _Fp32OutputLinearFunction).
        "upcast_fp32_matmul": copies both operands to fp32 every call, then an fp32 matmul.

        Qwen3.5-27B LM head, 2048 real tokens (2048x5120 input, 248320x5120 weight), bf16
        operands, fwd + cross-entropy + bwd; errors vs fp64:

            mode                   output   time    mean logprob error   grad_input error
            default                bf16     12 ms   1.2e-2               1.1e-2
            bf16_matmul_fp32_out   fp32     19 ms   6.1e-6               1.7e-3
            upcast_fp32_matmul     fp32     77 ms   1.6e-6               1.7e-3

        Batch-invariant mode always upcasts.
        """

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.num_linears * config.out_features,
            bias=config.bias,
        )
        if config.matmul_mode != "default" and type(self)._linear is not Linear._linear:
            raise ValueError(
                f"{type(self).__qualname__} overrides _linear and ignores "
                f"matmul_mode={config.matmul_mode!r}."
            )

        self.matmul_mode = config.matmul_mode
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
        if self.matmul_mode == "default":
            return F.linear(input, weight, bias)

        if self.matmul_mode == "upcast_fp32_matmul":
            bias = None if bias is None else bias.float()
            return F.linear(input.float(), weight.float(), bias)

        if self.matmul_mode == "bf16_matmul_fp32_out":
            # TODO: batch-invariant mode can't use this op (cuBLAS's out_dtype GEMM isn't
            # batch-invariant) and upcasts instead. A bf16-input, fp32-output matmul in
            # batch_invariant_ops would let it use _Fp32OutputLinearFunction too.
            if is_in_batch_invariant_mode():
                bias = None if bias is None else bias.float()
                return F.linear(input.float(), weight.float(), bias)

            # aten::mm.dtype (bf16 inputs, fp32 output) is only implemented for CUDA/ROCm.
            if not (input.is_cuda and input.dtype == weight.dtype == torch.bfloat16):
                raise ValueError(
                    'matmul_mode="bf16_matmul_fp32_out" needs bf16 CUDA operands, got '
                    f"{input.dtype}/{weight.dtype} on {input.device}. "
                    'Use "upcast_fp32_matmul".'
                )

            output = _Fp32OutputLinearFunction.apply(
                input.reshape(-1, input.shape[-1]), weight
            )
            output = output.reshape(*input.shape[:-1], -1)
            return output if bias is None else output + bias.float()

        raise AssertionError(f"unknown matmul_mode {self.matmul_mode!r}")


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


@spmd.register_local_autograd_function
class _RouterGateLinearFunction(torch.autograd.Function):
    """Router projection with FP32 output and backward GEMMs."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, input_TD: torch.Tensor, weight_ED: torch.Tensor
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
        ctx.weight_dtype = weight_ED.dtype
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
            ).to(ctx.weight_dtype)

        return grad_input_TD, grad_weight_ED


class RouterGateLinear(Linear):
    """Router projection with FP32 output and backward compute.

    TODO: fold into ``Linear(matmul_mode="bf16_matmul_fp32_out")``: same forward, and its
    backward matches this one's gradients at ~1/3 the GPU time. That mode raises off CUDA, where
    this class falls back to fp32.

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
        output_TE = _RouterGateLinearFunction.apply(input, weight)
        if bias is not None:
            output_TE = output_TE + bias.float()
        return output_TE


__all__ = [
    "ColumnParallelLinear",
    "Linear",
    "RowParallelLinear",
    "RouterGateLinear",
]
