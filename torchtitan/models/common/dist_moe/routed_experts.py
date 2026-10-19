# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan module boundary for Dist-MoE routed experts.

Shape suffixes use ``T`` for local input tokens, ``K`` for selected experts,
``E`` for local experts, ``F`` for expert intermediate dimension, and ``D``
for model dimension.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.distributed.local_compile import local_compile
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.base.module import Module

from torchtitan.models.common.linear import GroupedLinear
from torchtitan.quantization._fsdp_tensor import (
    _ShardedFSDPTensor,
    _UnshardedFSDPTensor,
)

from . import _dist_moe as dist_moe
from .runtime import DistMoeRuntime


# dynamic=True: one graph for any token count. The generator's eager warmup
# compiles it before CUDA graph capture, which cannot compile.
@local_compile("dist_moe_mask_padded_routes", batch_invariant=True, dynamic=True)
def _mask_padded_routes(
    topk_expert_ids_TK: torch.Tensor,
    topk_scores_TK: torch.Tensor,
    padding_mask_T: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route padded rows to expert ``-1`` with score ``0``; one kernel when compiled."""
    padding_mask_TK = padding_mask_T.unsqueeze(-1)
    return (
        topk_expert_ids_TK.masked_fill(padding_mask_TK, -1),
        topk_scores_TK.masked_fill(padding_mask_TK, 0.0),
    )


@dataclass(frozen=True, slots=True)
class _BlockMajorOperands:
    """Dist-MoE's block-major ``[E, N, K]`` copy of one expert weight."""

    weight: torch.Tensor


class _BlockMajorWeight(_ShardedFSDPTensor):
    """W13 or W2 whose FSDP unshard writes Dist-MoE's block-major copy.

    The copy is rebuilt from the freshly gathered row-major weight on every
    unshard that gathers (the initial load and each weight sync) into the same
    tensor, so CUDA graphs keep reading one address. An unshard of a group that
    is already unsharded gathers nothing and does not call this.
    """

    interleaved_gate_up: ClassVar[bool] = False

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _BlockMajorOperands | None = None,
    ) -> _BlockMajorOperands:
        # W2 [E, D, F] is unchanged; W13 [E, 2, F, D] becomes [E, 2F, D].
        weight_ENK = logical_tensor.flatten(1, -2)
        if out is None:
            out = _BlockMajorOperands(torch.empty_like(weight_ENK))
        out.weight.copy_(weight_ENK)
        dist_moe.relayout_to_block_major(
            out.weight, interleave_gate_up=self.interleaved_gate_up
        )
        return out


class _InterleavedBlockMajorWeight(_BlockMajorWeight):
    """W13 with gate and up rows interleaved, so the dispatch GEMM writes SwiGLU."""

    interleaved_gate_up: ClassVar[bool] = True


class DistMoeRoutedExperts(Module):
    """BF16 routed experts executed by the standalone Dist-MoE backend.

    W13 and W2 remain ordinary TorchTitan modules so parameter, FSDP,
    optimizer, and checkpoint ownership does not change. Their module
    ``forward`` methods are not called: Dist-MoE consumes the weights directly
    and owns dispatch, SwiGLU, expert GEMMs, and combine.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Configure BF16-specific Dist-MoE execution.

        Args:
            w13: Grouped gate/up projection with stored shape ``(E, 2, F, D)``.
            w2: Grouped down projection with stored shape ``(E, D, F)``.
            top_k: Number of experts selected for each local input token.
            output_postprocess: Optional module translated into the annex's
                typed or eager post-expert processing policy.
            inplace_wgrad_accum: Whether Dist-MoE writes W13/W2 gradients
                directly into existing standard ``parameter.grad`` buffers.
                This is enabled by default and remains visible to graph tracing.
            bf16_grouped_gemm_preset: Optional expert override for the annex's
                BF16 FPROP/DGRAD grouped-GEMM schedule. ``None`` selects the
                shape-aware production defaults.
            activation: Expert activation implemented by the Dist-MoE kernels.
            swiglu_alpha: Sigmoid multiplier when ``activation`` is clamped
                SwiGLU; otherwise ``None``.
            swiglu_limit: Gate and up-projection bound when ``activation`` is
                clamped SwiGLU; otherwise ``None``.
            block_major_weights: Hand Dist-MoE a block-major copy of W13/W2.
                Inference only; set by the vLLM generator from
                ``DistMoeRuntime.Config``.
            interleaved_gate_up: Also interleave W13's gate and up rows in that
                copy, so the dispatch GEMM writes the SwiGLU output. Needs
                ``block_major_weights``; set by the vLLM generator.
        """

        w13: GroupedLinear.Config
        w2: GroupedLinear.Config
        top_k: int
        output_postprocess: Module.Config | None = None
        inplace_wgrad_accum: bool = True
        bf16_grouped_gemm_preset: dist_moe.Bf16GroupedGemmPreset | None = None
        activation: Literal["swiglu", "swiglu_clamped"] = "swiglu"
        swiglu_alpha: float | None = None
        swiglu_limit: float | None = None
        block_major_weights: bool = False
        interleaved_gate_up: bool = False

        def __post_init__(self) -> None:
            """Validate expert dimensions consumed by the annex kernels."""
            if self.w13.group_size != self.w2.group_size:
                raise ValueError("w13 and w2 must contain the same number of experts")
            if self.w13.in_features != self.w2.out_features:
                raise ValueError("w13 input and w2 output dimensions must match")
            if self.w13.num_linears != 2:
                raise ValueError("w13 output must contain gate and up projections")
            if self.w13.out_features != self.w2.in_features:
                raise ValueError("w13 output and w2 input dimensions must match")
            if self.w2.num_linears != 1:
                raise ValueError("w2 must contain one down projection")
            if self.top_k <= 0:
                raise ValueError("top_k must be positive")
            if self.activation == "swiglu":
                if self.swiglu_alpha is not None or self.swiglu_limit is not None:
                    raise ValueError("plain SwiGLU must not define clamp parameters")
            elif self.activation == "swiglu_clamped":
                if self.swiglu_alpha is None or self.swiglu_limit is None:
                    raise ValueError("clamped SwiGLU requires alpha and limit")
            else:
                raise ValueError(f"unsupported activation {self.activation!r}")
            postprocess_config = self.output_postprocess
            owner = None if postprocess_config is None else postprocess_config._owner
            if postprocess_config is not None and not callable(
                getattr(owner, "to_dist_moe_postprocess", None)
            ):
                raise TypeError(
                    f"{type(postprocess_config).__qualname__} cannot execute "
                    "inside Dist-MoE"
                )

    def __init__(self, config: Config):
        Module.__init__(self)
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.output_postprocess = (
            config.output_postprocess.build()
            if config.output_postprocess is not None
            else None
        )
        self.hidden_dim = config.w13.in_features
        self.intermediate_dim = config.w2.in_features
        self.num_experts = config.w13.group_size
        self.top_k = config.top_k
        self.inplace_wgrad_accum = config.inplace_wgrad_accum
        self.bf16_grouped_gemm_preset = config.bf16_grouped_gemm_preset
        self.block_scaled_config: dist_moe.BlockScaledConfig | None = None
        self.activation = config.activation
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit
        self.block_major_weights = config.block_major_weights
        self.interleaved_gate_up = config.interleaved_gate_up
        self._runtime: DistMoeRuntime | None = None
        if self.block_major_weights:
            self._wrap_block_major_weights()

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Leave communication and activation storage to the shared runtime."""
        del buffer_device

    def _wrap_block_major_weights(self) -> None:
        """Wrap W13/W2 in the FSDP weight class that builds the block-major copy."""
        w13_cls = (
            _InterleavedBlockMajorWeight
            if self.interleaved_gate_up
            else _BlockMajorWeight
        )
        for proj, wrapper_cls in ((self.w13, w13_cls), (self.w2, _BlockMajorWeight)):
            if isinstance(proj.weight, wrapper_cls):
                continue
            weight = proj.weight
            wrapped_weight = torch.nn.Parameter(
                wrapper_cls(weight.data), requires_grad=weight.requires_grad
            )
            spmd.assert_type_like(wrapped_weight, weight)
            proj.weight = wrapped_weight

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        # spmd_types returns a plain tensor when EP/TP shards a weight. Wrap it
        # again before fully_shard() consumes it.
        super()._parallelize(parallelism_context)
        if self.block_major_weights:
            self._wrap_block_major_weights()

    def _weight_operands(
        self,
    ) -> tuple[
        torch.Tensor | dist_moe.PreparedWeight,
        torch.Tensor | dist_moe.PreparedWeight,
    ]:
        """Return W13 and W2 operands for the annex invocation."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        if self.block_major_weights:
            if not isinstance(w13_E2FD, _UnshardedFSDPTensor) or not isinstance(
                w2_EDF, _UnshardedFSDPTensor
            ):
                raise RuntimeError(
                    "Block-major Dist-MoE weights are built by FSDP's unshard; "
                    "unshard the experts' FSDP group before the forward"
                )
            return w13_E2FD.operands.weight, w2_EDF.operands.weight
        w13_EFD = w13_E2FD.flatten(1, 2)
        return w13_EFD, w2_EDF

    def _output_postprocess(
        self,
    ) -> dist_moe.RMSNormPostprocess | Callable[[torch.Tensor], torch.Tensor] | None:
        """Bind the current TorchTitan postprocess parameters to the annex."""
        module = self.output_postprocess
        if module is None:
            return None
        factory = getattr(module, "to_dist_moe_postprocess", None)
        if not callable(factory):
            raise TypeError(
                f"{type(module).__qualname__} cannot execute inside Dist-MoE"
            )
        postprocess = factory()
        if not (
            isinstance(postprocess, dist_moe.RMSNormPostprocess)
            or callable(postprocess)
        ):
            raise TypeError(
                "to_dist_moe_postprocess() must return an RMSNormPostprocess "
                "or callable"
            )
        return postprocess

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run distributed dispatch, expert computation, and combine.

        Args:
            x_TD: Local input tokens with model dimension ``D``.
            topk_scores_TK: Selected routing weights for ``K`` experts.
            topk_expert_ids_TK: Selected global expert IDs.
            num_local_tokens_per_expert_E: Router statistics retained by the
                surrounding MoE module; Dist-MoE derives dispatch metadata from
                the selected IDs.
            padding_mask_T: Optional bool ``(T,)``, true for rows the caller
                padded. Their routes get expert ID ``-1`` and score ``0``, which
                Dist-MoE never dispatches, and their output rows are zero.

        Returns:
            Combined local expert output with shape ``(T, D)``.
        """
        del num_local_tokens_per_expert_E
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Dist-MoE context is not initialized")
        if padding_mask_T is not None:
            topk_expert_ids_TK, topk_scores_TK = _mask_padded_routes(
                topk_expert_ids_TK, topk_scores_TK, padding_mask_T
            )
        # Every EP rank must pass the annex the same row count, and local batches
        # can differ across ranks. Pad to the call's count with rows routed to
        # expert -1, then slice the output.
        num_tokens = x_TD.shape[0]
        num_call_tokens = runtime.num_local_input_tokens_for_call(num_tokens)
        num_padded_tokens = num_call_tokens - num_tokens
        if num_padded_tokens < 0:
            raise ValueError(
                f"{num_tokens} local tokens exceed the Dist-MoE call's {num_call_tokens}"
            )
        if num_padded_tokens > 0:
            # Dist-MoE requires score 0 on -1 routes. The zero input rows are
            # never dispatched.
            x_TD = F.pad(x_TD, (0, 0, 0, num_padded_tokens))
            topk_scores_TK = F.pad(topk_scores_TK, (0, 0, 0, num_padded_tokens))
            topk_expert_ids_TK = F.pad(
                topk_expert_ids_TK, (0, 0, 0, num_padded_tokens), value=-1
            )
        w13_operand, w2_operand = self._weight_operands()
        postprocess = self._output_postprocess()
        execution_options = dist_moe.ExecutionOptions(
            inplace_wgrad_accum=self.inplace_wgrad_accum,
            experts_output_postprocess=postprocess,
            # -1 routes come from the padding mask (rows inside the call) and
            # from the pad above (rows appended up to the call size). A Python
            # callback otherwise reads unspecified storage in those rows, and
            # their stale values can reach its gradients.
            zero_out_padded_callback_inputs=(
                callable(postprocess)
                and (padding_mask_T is not None or num_padded_tokens > 0)
            ),
        )
        out_TD = remat.region(
            dist_moe.routed_experts,
            self.remat_region_name("dist_moe"),
            recompute=False,
        )(
            x_TD.contiguous(),
            topk_expert_ids_TK.contiguous(),
            topk_scores_TK.contiguous(),
            w13_operand,
            w2_operand,
            runtime.context,
            options=execution_options,
        )
        remat.recompute_needs_tensor(out_TD)
        return out_TD[:num_tokens] if num_padded_tokens > 0 else out_TD
