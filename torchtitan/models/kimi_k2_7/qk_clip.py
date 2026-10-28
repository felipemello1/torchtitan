# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from dataclasses import dataclass
from typing import Any, cast, ClassVar

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_remat as remat
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.nn.attention.flex_attention import AuxRequest
from torch.utils.weak import WeakTensorKeyDictionary

from torchtitan.components.optim import OptimizersContainer
from torchtitan.distributed import ParallelismContext
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    FlexInnerAttention,
    InnerAttention,
    MLAAttention,
    VarlenAttentionMetadata,
)
from torchtitan.models.deepseek_v3.attention import DeepSeekMLAAttention

# Shape suffixes:
# T = packed tokens, H = attention heads, D = projection rows per head,
# I = input features.


class QKClipFlexInnerAttention(FlexInnerAttention):
    """FlexInnerAttention that records the maximum score for each query head."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.max_attention_logits_H: list[torch.Tensor] = []

    def _get_aux_request(self, *, return_lse: bool) -> AuxRequest:
        return AuxRequest(lse=return_lse, max_scores=self.training)

    def _process_aux(self, aux: Any) -> None:
        if self.training and not remat.is_recomputing():
            max_scores_1HT = aux.max_scores
            assert max_scores_1HT is not None
            # Record gradient-accumulation and PP microbatches once, excluding
            # activation-checkpoint replay.
            self.max_attention_logits_H.append(max_scores_1HT.amax(dim=(0, 2)).detach())


class QKClipVarlenInnerAttention(InnerAttention):
    """Document-causal varlen attention that records the maximum score for each query head.

    torch's ``varlen_attn`` returns only the LSE, so this calls cudnn-frontend's
    interval-mask attention (``cudnn.flex_attention``, >= 1.31), JIT-compiled CuTe DSL
    kernels that also return the maximum. Their backward needs a dense output gradient
    (an expanded one raises).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        @property
        def is_full_attention(self) -> bool:
            return True

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> VarlenAttentionMetadata:
            """Build the packed-document metadata Varlen attention uses."""
            return create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )

    # One mask plan per microbatch, shared by every layer: a build costs 1-2 ms of host
    # time and a blocking device-to-host copy. Keyed by the metadata's ``cu_seq_q``, so
    # a plan is freed together with its microbatch's metadata.
    _mask_plans: ClassVar[WeakTensorKeyDictionary] = WeakTensorKeyDictionary()

    def __init__(self, config: Config) -> None:
        super().__init__()
        try:
            found = importlib.util.find_spec("cudnn.flex_attention") is not None
        except ModuleNotFoundError:  # Parent package `cudnn` missing.
            found = False
        if not found:
            raise ValueError(
                "Kimi QK clipping on the varlen backend needs nvidia-cudnn-frontend>=1.31, "
                "but 'cudnn.flex_attention' is not installed. Install it or select "
                "attn_backend='flex'."
            )
        self.max_attention_logits_H: list[torch.Tensor] = []

    @classmethod
    def _mask_plan(
        cls,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        attention_metadata: VarlenAttentionMetadata,
    ) -> Any:
        """Build (or reuse) the document-causal mask plan for this microbatch."""
        from cudnn.flex_attention import create_mask_plan

        cu_seq = attention_metadata.cu_seq_q
        needs_backward = torch.is_grad_enabled() and any(
            t.requires_grad for t in (q_THK, k_THK, v_THV)
        )
        key = (q_THK.shape, k_THK.shape, v_THV.shape, q_THK.dtype, needs_backward)
        cached = cls._mask_plans.get(cu_seq)
        if cached is not None and cached[0] == key:
            return cached[1]
        # Each query sees keys up to itself within its own document, given as the end of
        # that range per query: cu_seq [0, 3, 5] -> visible_end [1, 2, 3, 1, 2].
        token_T = torch.arange(q_THK.shape[0], device=q_THK.device, dtype=torch.int32)
        doc_start_T = cu_seq[torch.searchsorted(cu_seq, token_T, right=True) - 1]
        visible_end_11T = (token_T - doc_start_T + 1).view(1, 1, -1)
        plan = create_mask_plan(
            visible_end_11T,
            q_THK,
            k_THK,
            v_THV,
            cu_seqlens_q=cu_seq,
            cu_seqlens_k=attention_metadata.cu_seq_k,
            max_seqlen_q=attention_metadata.max_q,
            max_seqlen_k=attention_metadata.max_k,
            build_backward=needs_backward,
        )
        cls._mask_plans[cu_seq] = (key, plan)
        return plan

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: VarlenAttentionMetadata,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        from cudnn.flex_attention import flex_attn_func

        if kwargs:
            # e.g. out_transform: this kernel has no output epilogue to apply it in.
            raise TypeError(
                f"{type(self).__name__} got unsupported arguments: {sorted(kwargs)}"
            )

        # Like FlexInnerAttention.compiled_flex_attn: the plan build and the kernel run
        # outside SPMD typechecking, and the output takes the query's type.
        with spmd.no_typecheck():
            plan = self._mask_plan(q_THK, k_THK, v_THV, attention_metadata)
            out_THV, max_logit_H = flex_attn_func(
                q_THK,
                k_THK,
                v_THV,
                mask_plan=plan,
                softmax_scale=scale,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                return_max_logit=True,
            )
        if spmd.is_type_checking():
            spmd.assert_type(
                out_THV, spmd.get_local_type(q_THK), spmd.get_partition_spec(q_THK)
            )
        if self.training and not remat.is_recomputing():
            # Same per-head maximum of the scaled logits as QKClipFlexInnerAttention.
            self.max_attention_logits_H.append(max_logit_H)
        return out_THV


# Inner attentions that record per-head maximum logits for QK clipping.
_QKClipAttention = QKClipFlexInnerAttention | QKClipVarlenInnerAttention


def _validate_head_sharding(weight: DTensor) -> None:
    """Require MLA weights to be replicated or contiguously sharded on dim 0."""
    for placement in weight.placements:
        if type(placement) is Replicate:
            continue
        # ``_StridedShard`` subclasses ``Shard``, so an exact type check rejects
        # it: its non-default shard order would pair heads with the wrong rows.
        if type(placement) is not Shard or placement.dim != 0:
            raise ValueError(
                "QK clipping requires MLA weights sharded only on tensor "
                "dimension 0."
            )


def _replicated_scales(scales_H: torch.Tensor, weight: DTensor) -> DTensor:
    """Represent per-head scales on the same distributed mesh as ``weight``.

    The MAX all-reduce already leaves identical scales on every rank.
    ``from_local`` records the scales as replicated without communication,
    allowing DTensor dispatch to align them with sharded heads. Using
    ``distribute_tensor`` would add an unnecessary broadcast.
    """
    return DTensor.from_local(
        scales_H,
        weight.device_mesh,
        tuple(Replicate() for _ in weight.placements),
        run_check=False,
    )


@torch.no_grad()
def _scale_mla_heads(
    weight: DTensor,
    scales_H: torch.Tensor,
    *,
    rows_per_head: int,
    nope_rows_per_head: int,
    nope_scale_exponent: float,
    remaining_scale_exponent: float | None,
) -> None:
    """Scale the NoPE and remaining rows of every MLA head in place.

    ``weight`` is viewed as ``[num_heads, rows_per_head, in_features]``, and
    ``scales_H`` contains one scale per head. The remaining rows are unchanged
    when ``remaining_scale_exponent`` is ``None``.
    """
    num_heads = scales_H.numel()
    if weight.ndim != 2 or weight.shape[0] != num_heads * rows_per_head:
        raise ValueError("QK clip scales do not match the MLA weight shape.")
    _validate_head_sharding(weight)

    scales_H11 = _replicated_scales(scales_H, weight).view(-1, 1, 1)
    heads_HDI = weight.view(num_heads, rows_per_head, weight.shape[1])
    heads_HDI[:, :nope_rows_per_head].mul_(scales_H11.pow(nope_scale_exponent))
    if remaining_scale_exponent is not None:
        heads_HDI[:, nope_rows_per_head:].mul_(scales_H11.pow(remaining_scale_exponent))


@torch.no_grad()
def _clip_mla_weights(
    attention: DeepSeekMLAAttention,
    scales_H: torch.Tensor,
    *,
    alpha: float,
) -> None:
    q_projection = attention.wq if attention.q_lora_rank == 0 else attention.wq_b
    # Query: NoPE rows take ``scale ** alpha``, RoPE rows take the full scale.
    _scale_mla_heads(
        cast(DTensor, q_projection.weight),
        scales_H,
        rows_per_head=attention.qk_head_dim,
        nope_rows_per_head=attention.qk_nope_head_dim,
        nope_scale_exponent=alpha,
        remaining_scale_exponent=1.0,
    )
    # Key/value: the K rows take the remaining ``scale ** (1 - alpha)``, and the
    # V rows stay unchanged.
    _scale_mla_heads(
        cast(DTensor, cast(Any, attention.inner_attention.wkv_b).weight),
        scales_H,
        rows_per_head=attention.qk_nope_head_dim + attention.v_head_dim,
        nope_rows_per_head=attention.qk_nope_head_dim,
        nope_scale_exponent=1.0 - alpha,
        remaining_scale_exponent=None,
    )


@torch.no_grad()
def qk_clip(
    model_parts: list[nn.Module],
    *,
    reduction_mesh: DeviceMesh,
) -> None:
    """Apply the Kimi K2 QK clipping update with the report defaults."""
    threshold = 100.0
    alpha = 0.5
    attention_layers = [
        module
        for model_part in model_parts
        for module in model_part.modules()
        if isinstance(module, DeepSeekMLAAttention)
        and isinstance(module.inner_attention, MLAAttention)
        and isinstance(module.inner_attention.inner_attention, _QKClipAttention)
    ]
    if not attention_layers:
        return

    inner_attentions = [
        cast(_QKClipAttention, layer.inner_attention.inner_attention)
        for layer in attention_layers
    ]
    # Each entry holds one layer's local maximum logit per query head.
    layer_max_logits_H = [
        torch.stack(inner_attention.max_attention_logits_H).amax(dim=0)
        for inner_attention in inner_attentions
    ]
    num_heads_per_layer = [logits.numel() for logits in layer_max_logits_H]
    max_logits_H = torch.cat(layer_max_logits_H)
    # TODO: the loss mesh includes CP. Ulysses CP splits heads across CP ranks, so
    # this MAX would mix different heads: reduce per head first once Kimi K2.x
    # supports CP (KimiK25Model.parallelize rejects it today).
    if reduction_mesh.size() > 1:
        dist.all_reduce(
            max_logits_H,
            op=dist.ReduceOp.MAX,
            group=reduction_mesh.get_group(),
        )
    scales_H = threshold / max_logits_H.clamp_min(threshold)

    for attention, layer_scales_H in zip(
        attention_layers,
        scales_H.split(num_heads_per_layer),
        strict=True,
    ):
        _clip_mla_weights(attention, layer_scales_H, alpha=alpha)
        cast(
            _QKClipAttention, attention.inner_attention.inner_attention
        ).max_attention_logits_H.clear()


def register_qk_clip_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallelism_context: ParallelismContext,
) -> None:
    """Apply QK clipping after each ordinary optimizer step."""
    reduction_mesh = parallelism_context.get_mesh("loss")

    def _qk_clip_hook(
        _optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        qk_clip(
            model_parts,
            reduction_mesh=reduction_mesh,
        )

    optimizers.register_step_post_hook(_qk_clip_hook)


__all__ = [
    "QKClipFlexInnerAttention",
    "QKClipVarlenInnerAttention",
    "register_qk_clip_hook",
]
