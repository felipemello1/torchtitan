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

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.protocols.module import Module

# Shape suffix legend:
#   T = num tokens, D = model dimension, O = output features, P = grad_output pieces (2 or 3)


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
        return torch._grouped_mm(
            input_RI,
            weight_EOI.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )


class FP32OutputLinear(Linear):
    """``Linear`` with an fp32 output and close to fp32 gradients, at close to bf16 speed. Useful
    for layers that need higher precision, e.g. an LM head or a MoE router gate.

    Forward: bf16 input and weight, a bf16 GEMM that accumulates in fp32, fp32 output.
    Backward: approximates an fp32 backward with bf16 GEMMs, and returns grad_weight in fp32.

    Falls back to slower fp32 matmuls when:
    (a) the input is not on CUDA,
    (b) the input or the weight is not bf16, or
    (c) batch-invariant mode is on.

    Accuracy and timings: https://github.com/pytorch/torchtitan/pull/4923
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        exact_grad_output_split: bool = True
        """Backward splits the fp32 grad_output into bf16 pieces; see ``backward``.
        True: 3 pieces, exact. False: 2 pieces, 16 of fp32's 24 bits.
        The third piece makes the backward 1.4-1.6x slower. It cuts a MoE router's grad_input
        error 7x, but doesn't help an LM head."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.exact_grad_output_split = config.exact_grad_output_split

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # torch.mm is 2D. Linear.forward already flattened a stacked weight ([num_linears, O, D]
        # -> [num_linears * O, D]); here we flatten the input's token dims, [B, S, D] -> [B * S, D].
        output = _FP32OutputLinearFunction.apply(
            input.reshape(-1, input.shape[-1]), weight, self.exact_grad_output_split
        )
        output = output.reshape(*input.shape[:-1], -1)
        return output if bias is None else output + bias.float()


@spmd.register_local_autograd_function
class _FP32OutputLinearFunction(torch.autograd.Function):
    """``output = input @ weight.T`` in fp32, with bf16 GEMMs. See ``FP32OutputLinear``."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        input_TD: torch.Tensor,
        weight_OD: torch.Tensor,
        exact_grad_output_split: bool,
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
        ctx.exact_grad_output_split = exact_grad_output_split
        ctx.save_for_backward(input_TD, weight_OD)
        if ctx.use_bf16_gemm:
            return torch.mm(input_TD, weight_OD.T, out_dtype=torch.float32)
        # Slow fallback: upcast the input and weight to fp32.
        return torch.mm(input_TD.float(), weight_OD.float().T)

    @staticmethod
    # Raise on double backward (create_graph=True, e.g. a gradient penalty), which training never
    # uses: the int32 bit masks below would silently drop those gradients.
    @once_differentiable
    def backward(ctx, grad_output_TO: torch.Tensor):  # pyrefly: ignore[bad-override]
        """``grad_input = grad_output @ weight``, ``grad_weight = grad_output.T @ input``.

        Close to an fp32 backward, at a fraction of its cost (table below).

        The problem: grad_output is fp32, because the output was. Fast GEMMs take bf16 inputs, and
        a bf16 keeps only 8 of an fp32's 24 significant bits:

            fp32:  sign | exponent (8 bits) | mantissa (23 bits)   1 + 23 = 24 significant bits
            bf16:  sign | exponent (8 bits) | mantissa  (7 bits)   1 +  7 =  8 significant bits

        Both have 8 exponent bits, so 3 bf16 pieces hold an fp32 exactly (8 + 8 + 8 = 24 bits),
        and 2 pieces hold 16 of its 24 bits (bit picture in ``_split_into_bf16_pieces``):

            3 pieces, exact:  0.1 = 0.100097656 - 0.000097752 + 0.000000097   (hi + mid + lo)
            2 pieces:         0.1 ~ 0.100097656 - 0.000097752                 (hi + lo, off by 1e-7)

        We want, accumulated in fp32:
            grad_input  = grad_output @ weight     rounded once to the input's dtype
            grad_weight = grad_output.T @ input    returned in fp32; autograd rounds it to the
                                                   weight's grad_dtype (bf16 unless set to fp32)

        Ways to do it:
        - Round grad_output to bf16, then bf16 GEMMs: fast, but keeps 8 of its 24 bits.
        - Upcast input and weight, then fp32 matmuls: precise, but slow on H100 (CUDA cores). On
          Blackwell, cuBLAS can emulate them with BF16x9 (3 pieces per operand, all 9 pairs).
        - Split grad_output only (this): input and weight already are bf16, so only grad_output
          is split. Each piece is one bf16 GEMM, accumulated in fp32:

            grad_input  = hi @ weight  + lo @ weight      (+ mid @ weight with 3 pieces)
            grad_weight = hi.T @ input + lo.T @ input     (+ mid.T @ input)

        2 or 3 pieces: a GEMM loses precision as it accumulates, more for longer sums. A third piece
        only helps where that loss is smaller than what 2 pieces drop:
        - a router's grad_input sums over ~128 experts, so a router would benefit from 3;
        - an LM head's grad_input sums over the ~150k vocab, so a third piece adds very little;
        - grad_weight sums over tokens, so a third piece adds very little to either.

        Relative error vs fp64 (grad_input before its bf16 rounding), and backward time as a
        multiple of a bf16 Linear's (H100; the eager column runs the LM head's split compiled, see
        ``_compiled_split_into_stacked_bf16_pieces``):

                                    relative error             backward time
                                    grad_input  grad_weight    eager   compiled
            LM head (Qwen3-8B, 2048 tokens; bf16 backward: 7.2 ms)
              bf16 grad_output      1.5e-3      1.2e-3          1.0x    1.0x
              2 pieces              2.9e-4      7.6e-6          2.5x    2.5x
              3 pieces              2.9e-4      8.1e-6          3.5x    4.0x
              fp32 matmul (IEEE)    1.2e-4      1.8e-6         13.3x   14.0x
            router (2048 -> 128; errors on 16k tokens, times on 64k; bf16 backward: 0.30 ms)
              bf16 grad_output      1.7e-3      1.4e-3          1.0x    1.0x
              2 pieces              2.5e-6      4.5e-6          2.2x    1.7x
              3 pieces              3.6e-7      4.1e-6          3.3x    2.3x
              fp32 matmul (IEEE)    5.3e-8      3.4e-7          5.5x    5.4x

        Stacking: each piece needs a GEMM against the same weight or input. Stacking the pieces
        into one operand runs one GEMM per gradient instead, but copies the operand they share.
        We stack along the dim that makes that copy small. With 2 pieces (x = input, W = weight):

            LM head (T = 2048 tokens, O = 152k vocab, D = 4096): stack along tokens
                grad_input  = [hi; lo] @ W          then add the two halves
                grad_weight = [hi; lo].T @ [x; x]   copies x: 32 MiB
                (stacking along O instead would copy W: 2.3 GiB)
            router (T = 64k tokens, O = 128 experts, D = 2048): stack along out_features
                grad_input  = [hi | lo] @ [W; W]    copies W: 1 MiB
                (stacking along T instead would copy x: 512 MiB)
                grad_weight = hi.T @ x + lo.T @ x   one small GEMM per piece
        """
        input_TD, weight_OD = ctx.saved_tensors
        needs_grad_input, needs_grad_weight, _ = ctx.needs_input_grad
        # A no-op unless autocast made the fallback's output bf16.
        grad_output_TO = grad_output_TO.float()
        grad_input_TD = grad_weight_OD = None

        # ======== Fallback, cases (a)-(c) in FP32OutputLinear: fp32 matmuls ========
        if not ctx.use_bf16_gemm:
            if needs_grad_input:
                grad_input_TD = torch.mm(grad_output_TO, weight_OD.float())
                grad_input_TD = grad_input_TD.to(input_TD.dtype)
            if needs_grad_weight:
                grad_weight_OD = torch.mm(grad_output_TO.T, input_TD.float())
            return grad_input_TD, grad_weight_OD, None

        num_pieces = 3 if ctx.exact_grad_output_split else 2
        num_tokens, out_features = grad_output_TO.shape

        if out_features > num_tokens:
            # ======== Wide (e.g. an LM head): stack along tokens, copy the input ========
            # Dispatch modes (make_fx, FakeTensorMode, FlopCounterMode) can't see into the compiled
            # kernel, so they take the eager split. is_compiling() goes first: Dynamo graph-breaks
            # on the dispatch-stack check.
            if (
                torch.compiler.is_compiling()
                or torch._C._len_torch_dispatch_stack() > 0
            ):
                stacked_PTO = _split_into_stacked_bf16_pieces(
                    grad_output_TO, exact=ctx.exact_grad_output_split
                )
            else:
                # Chunk token counts change between steps (RL): one dynamic-T graph serves them all.
                torch._dynamo.maybe_mark_dynamic(grad_output_TO, 0)
                stacked_PTO = _compiled_split_into_stacked_bf16_pieces(
                    grad_output_TO, exact=ctx.exact_grad_output_split
                )
            if needs_grad_input:
                # TODO: summing all out_features in one GEMM sets this error. Summing 8192 at a time
                # in fp32 (addmm(out=)): 95% -> 99.5% correctly rounded, +7-13% GEMM time (H100).
                # Compiled, addmm(out_dtype=) fails to lower until pytorch/pytorch#190936.
                grad_input_PTD = torch.mm(
                    stacked_PTO, weight_OD, out_dtype=torch.float32
                )
                grad_input_PTD = grad_input_PTD.unflatten(0, (num_pieces, num_tokens))
                grad_input_TD = grad_input_PTD.sum(dim=0).to(input_TD.dtype)
            if needs_grad_weight:
                # TODO: with ChunkedLossWrapper, autograd adds each chunk's grad_weight into
                # weight.grad in a separate kernel. addmm(out=weight.grad), as MXFP8Linear does,
                # saves 2.9 ms and a 2.3 GiB temporary per Qwen3-8B chunk (H100). Eager only;
                # pytorch/torchtitan#4386 adds the same for plain Linear, which we could reuse.
                grad_weight_OD = torch.mm(
                    stacked_PTO.T,
                    # Copying x (0.02 ms) makes this 1.4x faster than one GEMM per piece + add.
                    torch.cat([input_TD] * num_pieces),
                    out_dtype=torch.float32,
                )
        else:
            # ======== Narrow (e.g. a router): stack along out_features, copy the weight ========
            pieces_TO = _split_into_bf16_pieces(
                grad_output_TO, exact=ctx.exact_grad_output_split
            )
            if needs_grad_input:
                grad_input_TD = torch.mm(
                    torch.cat(pieces_TO, dim=1), torch.cat([weight_OD] * num_pieces)
                )
            if needs_grad_weight:
                grad_weight_OD = torch.mm(
                    pieces_TO[0].T, input_TD, out_dtype=torch.float32
                )
                for piece_TO in pieces_TO[1:]:
                    grad_weight_OD += torch.mm(
                        piece_TO.T, input_TD, out_dtype=torch.float32
                    )

        # TODO: compiled, AOTAutograd still rounds grad_weight to bf16 (pytorch/pytorch#197381),
        # and autograd rounds it for a non-leaf weight, e.g. SimpleFSDP's or num_linears > 1
        # (pytorch/pytorch#189633).
        return grad_input_TD, grad_weight_OD, None


# The fp32 bits that bf16 keeps: sign, exponent and the top 7 mantissa bits (0xFFFF0000).
_BF16_BITS_OF_FP32 = -65536


def _split_into_bf16_pieces(tensor: torch.Tensor, *, exact: bool) -> list[torch.Tensor]:
    """Split an fp32 tensor into bf16 pieces that sum back to it: [hi, mid, lo] if exact, else
    [hi, lo].

    A bf16 keeps only the top 8 of an fp32's 24 significant bits. To keep more, cut the fp32
    into pieces: take the nearest bf16, subtract it, and repeat on what is left. Each
    subtraction is exact in fp32, so only the last piece loses anything.

    Example, x = 0.1:
        hi  = nearest bf16 to x            =  0.100097656   (a bit too big)
        mid = nearest bf16 to x - hi       = -0.000097752   (negative: corrects hi)
        lo  = x - hi - mid                 =  0.000000097

        3 pieces: hi + mid + lo == x exactly.
        2 pieces: stop after the second piece, so hi + lo = 0.100097656 - 0.000097752 is off
        by 1e-7.

    3 pieces are exact for |x| >= 2^-110, the range where bf16 can still hold the last piece.
    Rounding to nearest, as Triton and XLA do, gives later pieces mixed signs, which the GEMM
    sums more accurately than same-sign pieces: LM-head grad_weight error 1.4e-5 (truncating)
    -> 7.6e-6 (H100).
    """
    # TODO: use .to(torch.bfloat16) once Inductor stops dropping bf16 round trips in fused kernels
    # (simpler, and faster in eager). Today x - x.to(bf16).float() compiles to 0, so the pieces
    # use integer bit ops. https://github.com/pytorch/pytorch/issues/179561 was closed, but still
    # reproduces on the 2026-10-02 nightly.
    hi = _round_to_bf16(tensor)
    rest = tensor - hi
    if not exact:
        return [hi.to(torch.bfloat16), rest.to(torch.bfloat16)]
    mid = _round_to_bf16(rest)
    lo = rest - mid
    return [hi.to(torch.bfloat16), mid.to(torch.bfloat16), lo.to(torch.bfloat16)]


def _split_into_stacked_bf16_pieces(
    tensor_TO: torch.Tensor, *, exact: bool
) -> torch.Tensor:
    """``_split_into_bf16_pieces``, stacked along tokens: fp32 [T, O] -> bf16 [P * T, O]."""
    return torch.cat(_split_into_bf16_pieces(tensor_TO, exact=exact))


# One kernel splits grad_output into the stacked pieces: 1.66 vs 6.80 ms eager (Qwen3-8B LM-head
# chunk, 2 pieces, H100). Always compiled, like FlexAttention, so compile-off and RL runs (whose
# loss region is never compiled) get it too; an opt-in compile region would leave them eager.
# - Only the split: compiling the Function rounds grad_weight to bf16 (TODO at the end of backward).
# - No fullgraph: with it, TORCH_COMPILE_DISABLE=1 and the recompile limit raise inside backward.
# - No dynamic=True: a symbolic vocab dim is 15-40% slower; backward marks only the token dim.
# TODO: Inductor's ConcatKernel lowering reads grad_output once, 1.66 -> 1.14 ms (options
# max_pointwise_cat_inputs=1, max_complex_pointwise_cat_inputs=1). Not used: these internal knobs
# change across releases, and an unknown option fails at import.
_compiled_split_into_stacked_bf16_pieces = torch.compile(
    _split_into_stacked_bf16_pieces
)


def _round_to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Nearest bf16 value, ties away from zero, kept in fp32: add half a bf16 ulp, then cut."""
    bits = tensor.view(torch.int32)
    return ((bits + 0x8000) & _BF16_BITS_OF_FP32).view(torch.float32)


__all__ = [
    "ColumnParallelLinear",
    "GroupedLinear",
    "FP32OutputLinear",
    "Linear",
    "RowParallelLinear",
]
