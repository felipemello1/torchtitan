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
#   T = num tokens, D = model dimension, O = output features


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
        return F.linear(input, weight, bias)


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


class Fp32OutputLinear(Linear):
    """``Linear`` with fp32 output, close to fp32 precision at close to bf16 speed.

    Takes bf16 input and weight as they are (no fp32 copies), multiplies them with bf16 GEMMs that
    accumulate in fp32, and runs a backward that approximates an fp32 one. For projections whose
    output needs fp32 precision, e.g. an LM head (logprobs) or a MoE router gate.
    See ``_Fp32OutputLinearFunction``.
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
        output = _Fp32OutputLinearFunction.apply(
            input.reshape(-1, input.shape[-1]), weight
        )
        output = output.reshape(*input.shape[:-1], -1)
        return output if bias is None else output + bias.float()


@spmd.register_local_autograd_function
class _Fp32OutputLinearFunction(torch.autograd.Function):
    """Linear op with close to fp32 precision at close to bf16 speed.

    Forward: bf16 input and weight, a bf16 GEMM that accumulates in fp32, fp32 output.
    Backward: approximates an fp32 backward with bf16 or fp16 GEMMs; see ``backward``.

    Off CUDA, with non-bf16 operands, or in batch-invariant mode, both passes fall back to fp32
    matmuls, which are slower (on Blackwell, TorchTitan runs them as BF16x9: ~9x the bf16 cost).
    Accuracy and timings: https://github.com/felipemello1/torchtitan/pull/55
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, input_TD: torch.Tensor, weight_OD: torch.Tensor
    ) -> torch.Tensor:
        """``output = input @ weight.T``: a bf16 GEMM that accumulates in fp32 and returns fp32.

        ``torch.mm(..., out_dtype=torch.float32)`` has no autograd formula ("derivative for
        aten::mm is not implemented"), hence this Function:

            input (bf16) --+
                           +--> bf16 GEMM, fp32 accumulate --> output (fp32)
            weight (bf16) -+

        The product of two bf16 numbers is exact in fp32, so upcasting input and weight first
        would add no precision, only an fp32 copy of the weight. Accumulating in fp32 is enough.
        """
        ctx.use_bf16_gemm = (
            # aten::mm.dtype (bf16 inputs, fp32 output) is only implemented for CUDA/ROCm.
            input_TD.is_cuda
            and input_TD.dtype == weight_OD.dtype == torch.bfloat16
            # TODO: batch-invariant mode can't use this op (cuBLAS's out_dtype GEMM isn't
            # batch-invariant), so it takes the slow fallback. A bf16-input, fp32-output matmul
            # in batch_invariant_ops would let it take the optimized path.
            and not is_in_batch_invariant_mode()
        )
        ctx.save_for_backward(input_TD, weight_OD)
        if ctx.use_bf16_gemm:
            return torch.mm(input_TD, weight_OD.T, out_dtype=torch.float32)
        # Slow fallback: upcast the input and weight to fp32.
        return torch.mm(input_TD.float(), weight_OD.float().T)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TO: torch.Tensor):  # pyrefly: ignore[bad-override]
        """``grad_input = grad_output @ weight``, ``grad_weight = grad_output.T @ input``.

        Uses only 16-bit GEMMs, yet the gradients come out close to an fp32 backward's: at ~1.3x the
        cost of a plain bf16 backward for wide outputs (fp16 row scales), ~2x for narrow ones (hi + lo).

        The problem: grad_output is fp32 (the output was fp32), but fast GEMMs run on bf16 inputs
        (bf16 tensor cores), and bf16 keeps only the top 16 bits of an fp32:

            fp32:  [sign | exponent (8 bits) | mantissa (23 bits)]   24 significant bits
            bf16:  [sign | exponent (8 bits) | mantissa  (7 bits)]    8 significant bits

        Ways to multiply the fp32 grad_output by the bf16 input and weight:
        - Naive: round grad_output to bf16, then bf16 GEMMs. Fast, but drops 16 bits.
        - fp32 matmul: precise but slow. On H100 it runs on CUDA cores (~15x lower peak
          throughput than bf16 tensor cores). On Blackwell, TorchTitan emulates it with BF16x9:
          each fp32 operand is split into 3 bf16 pieces (3 x 8 = 24 bits), and all 3 x 3 = 9
          piece pairs are multiplied.
        - hi + lo (narrow outputs, e.g. a router): input and weight are already exactly bf16 (a
          single piece), so only grad_output is split. 2 pieces keep 16 bits, far more than survive
          the final rounding of the gradients to bf16, so a third piece isn't needed. It only uses
          bf16 GEMMs, so it isn't Blackwell-specific.
        - fp16 with one scale per row (wide outputs, e.g. an LM head): fp16 keeps 11 bits to bf16's
          8. Each grad_output row is scaled so its max |value| becomes exactly 2^15, so any loss
          scale fits and the largest entry (usually the label's) is exact; the scales are undone on
          the fp32 GEMM outputs. The weight gets a power-of-two scale to the top of fp16's range;
          for grad_weight the row scales are undone in the input, which is then scaled as a whole.
          fp16 GEMMs run at bf16 speed.

            round grad_output to bf16   1 GEMM    fast, loses precision
            fp32 matmul (BF16x9)        9 GEMMs   precise, ~9x the cost
            hi + lo                     2 GEMMs   precise, ~2x the cost
            fp16 with row scales        1 GEMM    precise; also casts the input and weight to fp16

            grad_output = hi + lo   hi = top 16 bits (exactly a bf16), lo = bf16(grad_output - hi)
            0.1 = 0.099609375 + 0.000391006     off by 4e-7 (bf16(0.1) alone: off by 1e-4)

            grad_input  = hi @ weight  + lo @ weight
            grad_weight = hi.T @ input + lo.T @ input

        For a router (out_features <= tokens) the fp16 casts of the [T, D] input would cost more
        than the extra GEMMs, so it keeps hi + lo, stacked so only the small weight is duplicated:

            grad_input:   [hi | lo] @ [W; W]    =  hi @ W + lo @ W       (summed in the GEMM)
            grad_weight:  hi.T @ x + lo.T @ x                            (two small GEMMs)

        LM head, 27B shape (8192 tokens, 124160 outputs, 5120 inputs, H100): 77.7 ms with hi + lo,
        41.2 ms with fp16 row scales, 31.9 ms rounding grad_output to bf16; gradient error vs fp64
        1.67e-3 vs 1.67e-3 (dh) and 1.72e-3 vs 1.73e-3 (dW), where rounding exact gradients to bf16
        gives 1.66e-3 / 1.72e-3.
        """
        input_TD, weight_OD = ctx.saved_tensors
        # Usually a no-op (the output is fp32); autocast can make the fallback's output bf16.
        grad_output_TO = grad_output_TO.float()
        grad_input_TD = grad_weight_OD = None

        if not ctx.use_bf16_gemm:
            # Slow fallback: fp32 matmuls, gradients returned in each operand's dtype.
            if ctx.needs_input_grad[0]:
                grad_input_TD = torch.mm(grad_output_TO, weight_OD.float())
                grad_input_TD = grad_input_TD.to(input_TD.dtype)
            if ctx.needs_input_grad[1]:
                grad_weight_OD = torch.mm(grad_output_TO.T, input_TD.float())
                grad_weight_OD = grad_weight_OD.to(weight_OD.dtype)
            return grad_input_TD, grad_weight_OD

        num_tokens, out_features = grad_output_TO.shape

        if out_features > num_tokens:
            # Wide output (e.g. an LM head): fp16 with one scale per row, 1 GEMM per gradient.
            # Each row's max |value| maps to exactly 2^15, which fp16 represents exactly (a
            # power-of-two scale would round it: 3.1e-4 vs 2.3e-4 grad_input error before the
            # bf16 cast). The clamp keeps all-zero rows finite. Each torch.mul(..., out=fp16)
            # scales and casts in one kernel.
            row_max_T1 = torch.linalg.vector_norm(
                grad_output_TO, ord=float("inf"), dim=1, keepdim=True
            )
            row_scale_T1 = 2.0**15 / row_max_T1.clamp(2.0**-45, 2.0**75)
            scaled_TO = torch.mul(
                grad_output_TO,
                row_scale_T1,
                out=torch.empty_like(grad_output_TO, dtype=torch.float16),
            )
            if ctx.needs_input_grad[0]:
                weight_scale = _fp16_scale(
                    torch.linalg.vector_norm(weight_OD, ord=float("inf"))
                )
                weight_fp16_OD = torch.mul(
                    weight_OD,
                    weight_scale,
                    out=torch.empty_like(weight_OD, dtype=torch.float16),
                )
                grad_input_TD = torch.mm(
                    scaled_TO, weight_fp16_OD, out_dtype=torch.float32
                )
                grad_input_TD = torch.mul(
                    grad_input_TD,
                    (row_scale_T1 * weight_scale).reciprocal(),
                    out=torch.empty_like(input_TD),
                )
            if ctx.needs_input_grad[1]:
                # grad_weight = scaled.T @ (input / row_scale), with input / row_scale scaled
                # as a whole to the top of fp16's range.
                input_row_max_T1 = torch.linalg.vector_norm(
                    input_TD, ord=float("inf"), dim=1, keepdim=True
                )
                input_scale = _fp16_scale(
                    (input_row_max_T1.float() / row_scale_T1).amax()
                )
                scaled_input_TD = torch.mul(
                    input_TD,
                    input_scale / row_scale_T1,
                    out=torch.empty_like(input_TD, dtype=torch.float16),
                )
                grad_weight_OD = torch.mm(
                    scaled_TO.T, scaled_input_TD, out_dtype=torch.float32
                )
                grad_weight_OD = torch.mul(
                    grad_weight_OD,
                    input_scale.reciprocal(),
                    out=torch.empty_like(weight_OD),
                )
        else:
            # Narrow output (e.g. a router): stack [hi | lo] along O; only the small weight grows.
            hi_TO, lo_TO = _split_into_bf16_hi_lo(grad_output_TO)
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


def _fp16_scale(absmax: torch.Tensor) -> torch.Tensor:
    """Power-of-two scale that moves ``absmax`` into [2^14, 2^15), the top of fp16's range.

    Exact to undo, and values down to ~2^-29 of ``absmax`` stay fp16 normals. Exponents are
    clamped to [-60, 60], so zero ``absmax`` gets a finite scale and a product or quotient of two
    scales stays finite in fp32.

    Example:
        absmax 139.0 (frexp exponent 8) -> scale 2^7, and 139 * 2^7 = 17792
    """
    _, exponent = torch.frexp(absmax.float())
    return torch.exp2((15 - exponent).clamp(-60, 60).float())


def _split_into_bf16_hi_lo(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an fp32 tensor into bf16 ``hi`` and ``lo`` with ``hi + lo`` within 2^-15 of it."""
    # Clearing the low bits makes hi exactly a bf16 value and tensor - hi exact in fp32. A bit
    # mask rather than .to(bf16): torch.compile folds a bf16 round trip away, making lo zero.
    hi = (tensor.view(torch.int32) & _BF16_BITS_OF_FP32).view(torch.float32)
    return hi.to(torch.bfloat16), (tensor - hi).to(torch.bfloat16)


__all__ = [
    "ColumnParallelLinear",
    "Fp32OutputLinear",
    "Linear",
    "RowParallelLinear",
]
