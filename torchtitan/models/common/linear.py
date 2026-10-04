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

import contextlib
import math
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.protocols.module import Module

# Shape suffix legend:
#   T = num tokens, D = model dimension, O = output features, P = grad_output pieces (2 or 3)
#   M, K, N = a generic GEMM's dims: a_MK @ b_KN -> out_MN


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


class SharedExpertRowParallelLinear(Linear):
    """Row-parallel shared-expert projection with a conditional reduction.

    With sequence parallelism, the output is reduce-scattered from Partial to
    Shard(0). Otherwise it remains Partial so the MoE can combine routed and
    shared partials before one all-reduce.
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
        if tp_group is None or not spmd_dense_sp_enabled():
            return output
        return spmd.redistribute(
            output,
            tp_group,
            src=spmd.P,
            dst=spmd.S(0),
            backward_options={"op_dtype": output.dtype},
        )


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
    Under `accumulate_into_weight_grad` (ChunkedLossWrapper), an LM head's eager backward adds
    into an existing weight.grad.

    An fp32 input is rounded to bf16 for the GEMMs and gets an fp32 gradient. The rounding is
    exact for a bf16 upcast to fp32, which TP hands the lm_head so it can sum the partial
    grad_inputs in fp32 (``pre_lm_head_norm_config``). For any other fp32 input it adds ~1.7e-3
    error to the output and grad_weight (fp32 matmuls: ~1.5e-7, H100).

    Falls back to slower fp32 matmuls when:
    (a) the input is not on CUDA,
    (b) the weight is not bf16, or the input is neither bf16 nor fp32, or
    (c) batch-invariant mode is on.

    Accuracy and timings: https://github.com/pytorch/torchtitan/pull/4923
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        grad_output_pieces: int = 3
        """How many bf16 pieces backward splits the fp32 grad_output into (see ``backward``): 3
        is exact, 2 keeps about 16 of its 24 significant bits. The third piece makes the backward
        1.4-1.6x slower; it cuts a MoE router's grad_input error 7x, but doesn't help an LM head."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.grad_output_pieces = config.grad_output_pieces

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # torch.mm takes 2D inputs, so flatten the input: [B, S, D] -> [B * S, D]. The weight is
        # already 2D: Linear.forward flattens a stacked [num_linears, O, D] to [num_linears * O, D].
        output = _FP32OutputLinearFunction.apply(
            input.reshape(-1, input.shape[-1]), weight, self.grad_output_pieces
        )
        output = output.reshape(*input.shape[:-1], -1)
        return output if bias is None else output + bias.float()


_ACCUMULATE_INTO_WEIGHT_GRAD = ContextVar("accumulate_into_weight_grad", default=False)


@contextlib.contextmanager
def accumulate_into_weight_grad() -> Iterator[None]:
    """Let FP32OutputLinear calls made here add grad_weight into an existing weight.grad inside the
    backward GEMM, instead of in autograd's separate add. Only the eager wide backward (an LM head)
    does it; a router, the fp32 fallback and a traced call (torch.compile, make_fx) don't.

    The backward then returns no grad_weight, so enter it only around calls whose backward is a
    plain `.backward()`, as ChunkedLossWrapper's per-chunk one is, and outside compiled code (Dynamo
    can't trace the ContextVar write). Otherwise, once weight.grad exists:
    - `backward(inputs=...)` and zero-bubble PP's input pass still add into weight.grad;
    - `torch.autograd.grad` wrt the weight raises (it returns None with `allow_unused=True`).

    Example:

        with accumulate_into_weight_grad():
            logits = lm_head(hidden_chunk)
        F.cross_entropy(logits, labels).backward()  # adds into lm_head.weight.grad, if it exists
    """
    token = _ACCUMULATE_INTO_WEIGHT_GRAD.set(True)
    try:
        yield
    finally:
        _ACCUMULATE_INTO_WEIGHT_GRAD.reset(token)


def weight_to_accumulate_into(weight: torch.Tensor) -> torch.Tensor | None:
    """`weight`, if a backward may add into its .grad in place; None when:
    (a) outside `accumulate_into_weight_grad`;
    (b) traced: a traced backward can't write into .grad (Dynamo sets is_compiling, graph_trainer's
        make_fx tracer only a proxy mode);
    (c) the weight is not a leaf (SimpleFSDP's, num_linears > 1).
    """
    is_tracing = torch.compiler.is_compiling() or get_proxy_mode() is not None
    accumulate = not is_tracing and _ACCUMULATE_INTO_WEIGHT_GRAD.get()
    return weight if accumulate and weight.is_leaf else None


@spmd.register_local_autograd_function
class _FP32OutputLinearFunction(torch.autograd.Function):
    """``output = input @ weight.T`` in fp32, with bf16 GEMMs. See ``FP32OutputLinear``."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx,
        input_TD: torch.Tensor,
        weight_OD: torch.Tensor,
        grad_output_pieces: int,
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
            and weight_OD.dtype == torch.bfloat16
            and input_TD.dtype in (torch.bfloat16, torch.float32)
            # TODO: batch-invariant mode can't use this op (cuBLAS's out_dtype GEMM isn't
            # batch-invariant), so it takes the slow fallback. A bf16-input, fp32-output matmul
            # in batch_invariant_ops would let it take the optimized path.
            and not is_in_batch_invariant_mode()
        )
        ctx.grad_output_pieces = grad_output_pieces
        # The wide backward may add into this parameter's .grad; see `accumulate_into_weight_grad`.
        # On ctx, not saved: saved-tensor hooks may unpack a copy.
        ctx.weight_param = weight_to_accumulate_into(weight_OD)
        ctx.input_dtype = input_TD.dtype
        if ctx.use_bf16_gemm:
            # A no-op for bf16. TP's fp32 lm_head input is an upcast bf16, so rounding it is exact.
            input_TD = input_TD.to(torch.bfloat16)
            ctx.save_for_backward(input_TD, weight_OD)
            return torch.mm(input_TD, weight_OD.T, out_dtype=torch.float32)
        # Slow fallback: upcast the input and weight to fp32.
        ctx.save_for_backward(input_TD, weight_OD)
        return torch.mm(input_TD.float(), weight_OD.float().T)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TO: torch.Tensor):  # pyrefly: ignore[bad-override]
        """``grad_input = grad_output @ weight``, ``grad_weight = grad_output.T @ input``.

        Close to an fp32 backward, at a fraction of its cost (table below).

        The problem: we have 3 tensors,
            weight:       bf16
            input:        bf16
            grad_output:  fp32, because the output was fp32

        We want, accumulated in fp32:
            grad_input  = grad_output @ weight     rounded once to the input's dtype
            grad_weight = grad_output.T @ input    returned in fp32

        How can we do it fast, with close to fp32 precision?

        Option 1: round grad_output to bf16, then bf16 GEMMs. GPUs run those on tensor cores, ~15x
        faster than fp32 matmuls on H100. Fast, but loses precision: grad_output keeps only 8 of
        its 24 significant bits.

        Option 2: upcast weight and input, then do all matmuls in fp32. Precise, but slow (table
        below).

        Option 3 (this function): split grad_output into 3 (or 2) bf16 pieces that sum to it.
        weight and input already are bf16, so only grad_output needs splitting.

            fp32:  sign | exponent (8 bits) | mantissa (23 bits)   1 + 23 = 24 significant bits
            bf16:  sign | exponent (8 bits) | mantissa  (7 bits)   1 +  7 =  8 significant bits

        bf16 keeps 8 of fp32's 24 significant bits but has the same exponent, so each piece keeps
        its own scale, and 3 pieces hold an fp32 exactly (``_split_into_bf16_pieces``):

            3 pieces, exact:  0.1 = 0.100097656 - 0.000097752 + 0.000000097   (hi + mid + lo)
            2 pieces:         0.1 ~ 0.100097656 - 0.000097752                 (hi + lo, off by 1e-7)

        Substituting grad_output = hi + mid + lo turns each fp32 GEMM into bf16 GEMMs:

            grad_input  = grad_output @ weight
                        = (hi + mid + lo) @ weight
                        = hi @ weight + mid @ weight + lo @ weight      3 bf16 GEMMs (2 without mid)

            grad_weight = grad_output.T @ input
                        = hi.T @ input + mid.T @ input + lo.T @ input   3 bf16 GEMMs (2 without mid)

        2 or 3 pieces: the GEMM also rounds as it accumulates, more for longer sums, so a third
        piece helps most where the sum is short (a router's 128 experts, not a 152k vocab summed
        8192 per GEMM).

        Relative error vs fp64 (grad_input before its bf16 rounding), and backward time as a
        multiple of a bf16 Linear's (H100; the eager column runs the LM head's split compiled, see
        ``_compiled_split_into_stacked_bf16_pieces``):

                                    relative error             backward time
                                    grad_input  grad_weight    eager   compiled
            LM head (Qwen3-8B, 2048 tokens; bf16 backward: 7.2 ms)
              bf16 grad_output      1.5e-3      1.2e-3          1.0x    1.0x
              2 pieces              1.3e-5      7.6e-6          2.4x    2.6x
              3 pieces              1.3e-5      8.0e-6          3.7x    4.0x
              fp32 matmul (IEEE)    1.2e-4      1.8e-6         13.4x   13.5x
            router (2048 -> 128; errors on 16k tokens, times on 64k; bf16 backward: 0.29 ms)
              bf16 grad_output      1.7e-3      1.4e-3          1.0x    1.0x
              2 pieces              2.5e-6      4.5e-6          2.2x    1.7x
              3 pieces              3.6e-7      4.1e-6          3.3x    2.3x
              fp32 matmul (IEEE)    5.3e-8      3.4e-7          5.5x    5.5x

        Stacking: every piece multiplies the same weight (for grad_input) or input (for
        grad_weight). Concatenating the pieces runs one GEMM per gradient instead of one per
        piece (1.4x faster for the LM head's grad_weight), but the shared operand gets
        concatenated too. We concatenate along the dim that makes that copy small.
        With 2 pieces (hi, lo: [T, O]; x = input: [T, D]; W = weight: [O, D]):

            LM head (T = 2048 tokens, O = 152k vocab, D = 4096): concatenate along tokens
                grad_input  = cat([hi, lo], dim=0) @ W               [2T, D]: add its two halves
                grad_weight = cat([hi, lo], dim=0).T @ cat([x, x])   copies x: 32 MiB
                (concatenating along O would copy W instead: 2.3 GiB)
            router (T = 64k tokens, O = 128 experts, D = 2048): concatenate along out_features
                grad_input  = cat([hi, lo], dim=1) @ cat([W, W])     copies W: 1 MiB
                (concatenating along T would copy x instead: 512 MiB)
                grad_weight = hi.T @ x + lo.T @ x                    one small GEMM per piece
        """
        input_TD, weight_OD = ctx.saved_tensors
        needs_grad_input, needs_grad_weight, _ = ctx.needs_input_grad
        # .float() is a no-op unless autocast made the fallback's output bf16.
        grad_output_TO = grad_output_TO.float()
        grad_input_TD = grad_weight_OD = None

        # ==== Fallback, cases (a)-(c) in FP32OutputLinear. Use slower fp32 matmuls instead ====
        if not ctx.use_bf16_gemm:
            if needs_grad_input:
                grad_input_TD = torch.mm(grad_output_TO, weight_OD.float())
                grad_input_TD = grad_input_TD.to(ctx.input_dtype)
            if needs_grad_weight:
                grad_weight_OD = torch.mm(grad_output_TO.T, input_TD.float())
            return grad_input_TD, grad_weight_OD, None

        num_pieces = ctx.grad_output_pieces
        num_tokens, out_features = grad_output_TO.shape

        # For speed, stack along the smaller dim (see "Stacking" in the docstring).
        if out_features > num_tokens:
            # ==== Wide (e.g. an LM head): stack along tokens, copy the input ====
            # Dispatch modes (make_fx, FakeTensorMode, FlopCounterMode) can't see into the compiled
            # kernel, so they take the eager split. is_compiling() goes first: Dynamo graph-breaks
            # on the dispatch-stack check.
            if (
                torch.compiler.is_compiling()
                or torch._C._len_torch_dispatch_stack() > 0
            ):
                stacked_PTO = _split_into_stacked_bf16_pieces(
                    grad_output_TO, grad_output_pieces=ctx.grad_output_pieces
                )
            else:
                # Chunk token counts change between steps (RL): one dynamic-T graph serves them all.
                torch._dynamo.maybe_mark_dynamic(grad_output_TO, 0)
                stacked_PTO = _compiled_split_into_stacked_bf16_pieces(
                    grad_output_TO, grad_output_pieces=ctx.grad_output_pieces
                )
            if needs_grad_input:
                # TODO: a kernel that adds each 64-long partial sum in fp32 outside the tensor core
                # got 5.3e-6 in a Triton prototype (split-K: 1.3e-5), but ran ~33% slower than this
                # path did before split-K, and it's a custom GEMM to maintain. vLLM does this for a
                # one-sided router GEMM: https://github.com/vllm-project/vllm/pull/55899
                grad_input_PTD = _mm_fp32_split_k(stacked_PTO, weight_OD)
                grad_input_PTD = grad_input_PTD.unflatten(0, (num_pieces, num_tokens))
                grad_input_TD = grad_input_PTD.sum(dim=0).to(ctx.input_dtype)
            if needs_grad_weight:
                # Copying x (0.02 ms) makes this 1.4x faster than one GEMM per piece + add.
                input_PTD = torch.cat([input_TD] * num_pieces)
                running_grad_OD = (
                    None if ctx.weight_param is None else ctx.weight_param.grad
                )
                if running_grad_OD is None:
                    grad_weight_OD = torch.mm(
                        stacked_PTO.T, input_PTD, out_dtype=torch.float32
                    )
                else:
                    # A later chunk or microbatch: add into weight.grad inside the GEMM instead of
                    # in a separate autograd kernel, and return no grad_weight. Qwen3-8B chunk, fp32
                    # .grad, H100: backward 19.5 -> 16.6 ms, bitwise the same .grad with one lm_head
                    # call per backward. .grad keeps its buffer, which CUDA graph replays rely on.
                    torch.addmm(
                        running_grad_OD,
                        stacked_PTO.T,
                        input_PTD,
                        out_dtype=running_grad_OD.dtype,
                        out=running_grad_OD,
                    )
        else:
            # ==== Narrow (e.g. a router): stack along out_features, copy the weight ====
            pieces_TO = _split_into_bf16_pieces(
                grad_output_TO, grad_output_pieces=ctx.grad_output_pieces
            )
            if needs_grad_input:
                grad_input_TD = torch.mm(
                    torch.cat(pieces_TO, dim=1),
                    torch.cat([weight_OD] * num_pieces),
                    out_dtype=ctx.input_dtype,
                )
            if needs_grad_weight:
                grad_weight_OD = torch.mm(
                    pieces_TO[0].T, input_TD, out_dtype=torch.float32
                )
                for piece_TO in pieces_TO[1:]:
                    grad_weight_OD += torch.mm(
                        piece_TO.T, input_TD, out_dtype=torch.float32
                    )

        # TODO: autograd keeps this fp32 grad_weight only in eager, for a leaf weight whose
        # grad_dtype is fp32 (FSDP2 sets it: https://github.com/pytorch/pytorch/pull/194434).
        # It still rounds the values to bf16, though .grad stays fp32,
        # - inside a torch.compile region: https://github.com/pytorch/pytorch/pull/197381
        # - for a non-leaf weight, e.g. SimpleFSDP's or num_linears > 1:
        #   https://github.com/pytorch/pytorch/issues/189633
        return grad_input_TD, grad_weight_OD, None


# The fp32 bits that bf16 keeps: sign, exponent and the top 7 mantissa bits (0xFFFF0000).
_BF16_BITS_OF_FP32 = -65536


def _split_into_bf16_pieces(
    grad_output_TO: torch.Tensor, *, grad_output_pieces: int
) -> list[torch.Tensor]:
    """Split an fp32 tensor into bf16 pieces that sum back to it: [hi, lo] or [hi, mid, lo].

    A bf16 keeps only the top 8 of an fp32's 24 significant bits. To keep more, cut the fp32
    into pieces: take the nearest bf16, subtract it, and repeat on what is left. Each
    subtraction is exact in fp32, so only the last piece loses anything.

    Args:
        grad_output_TO: fp32 tensor to split.
        grad_output_pieces: 2 keeps about 16 of the 24 significant bits; 3 is exact.

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
    hi = _round_to_bf16(grad_output_TO)
    rest = grad_output_TO - hi
    if grad_output_pieces == 2:
        return [hi.to(torch.bfloat16), rest.to(torch.bfloat16)]
    mid = _round_to_bf16(rest)
    lo = rest - mid
    return [hi.to(torch.bfloat16), mid.to(torch.bfloat16), lo.to(torch.bfloat16)]


def _split_into_stacked_bf16_pieces(
    grad_output_TO: torch.Tensor, *, grad_output_pieces: int
) -> torch.Tensor:
    """``_split_into_bf16_pieces``, stacked along tokens: fp32 [T, O] -> bf16 [P * T, O]."""
    return torch.cat(
        _split_into_bf16_pieces(grad_output_TO, grad_output_pieces=grad_output_pieces)
    )


# One kernel splits grad_output into the stacked pieces: 1.66 vs 6.80 ms eager (Qwen3-8B LM-head
# chunk, 2 pieces, H100). Always compiled, like FlexAttention, so models with empty
# local_compile_regions get it too; a @local_compile region would leave them eager.
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


# A GEMM's fp32 accumulator truncates as it sums, so its error grows with the length of the sum.
# The wide grad_input sums over out_features (152k for an LM head), so each GEMM sums at most this
# many. Keep it a multiple of 8: K slices that aren't 16-byte aligned make cuBLAS 2-3x slower.
_MAX_K_PER_GEMM = 8192


# A custom op, so torch.compile runs this loop as written. Inductor can't lower addmm(out_dtype=)
# (pytorch/pytorch#190936), and it fuses the adds of out = out + mm(...) into one kernel that holds
# 17 of the 19 Qwen3-8B GEMM results at once (1.0 GiB): compiled backward +6% instead of +4% (H100).
@torch.library.custom_op(
    "torchtitan::mm_fp32_split_k",
    mutates_args=(),
    device_types="cuda",
    # Any strides work. With the default (exact strides), Inductor writes the 2-piece stack twice,
    # once for this op and once for the grad_weight GEMM: +0.5 ms per compiled Qwen3-8B backward.
    tags=torch.Tag.flexible_layout,
)
def _mm_fp32_split_k(a_MK: torch.Tensor, b_KN: torch.Tensor) -> torch.Tensor:
    """``a @ b`` in fp32: one bf16 GEMM per ``_MAX_K_PER_GEMM`` slice of K, added in fp32.

    Example: K = 151936 -> 18 GEMMs over 8192 and one over 4480; K <= 8192 -> one GEMM.

    Qwen3-8B LM head grad_input (2 pieces, 2048 tokens, H100), by K per GEMM:

                        fp32 result      bf16 grad_input     backward time
                        relative error   correctly rounded   eager   compiled
        all of K        2.9e-4           95.2%               1.00x   1.00x
        16384           2.7e-5           99.3%               1.02x   1.03x
        8192            1.3e-5           99.6%               1.05x   1.04x
        4096            6.6e-6           99.7%               1.10x   1.09x
    """
    a_slices = a_MK.split(_MAX_K_PER_GEMM, dim=1)
    b_slices = b_KN.split(_MAX_K_PER_GEMM)
    out_MN = torch.mm(a_slices[0], b_slices[0], out_dtype=torch.float32)
    for a_slice, b_slice in zip(a_slices[1:], b_slices[1:]):
        # out= adds into out_MN inside the GEMM. A functional addmm copies out_MN first, which
        # more than doubles the cost of splitting (Qwen3-8B backward: +7% instead of +3%).
        torch.addmm(out_MN, a_slice, b_slice, out_dtype=torch.float32, out=out_MN)
    return out_MN


@_mm_fp32_split_k.register_fake
def _(a_MK: torch.Tensor, b_KN: torch.Tensor) -> torch.Tensor:
    return a_MK.new_empty(a_MK.shape[0], b_KN.shape[1], dtype=torch.float32)


__all__ = [
    "accumulate_into_weight_grad",
    "weight_to_accumulate_into",
    "ColumnParallelLinear",
    "GroupedLinear",
    "FP32OutputLinear",
    "Linear",
    "RowParallelLinear",
]
