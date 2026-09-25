# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM paged-cache adapter for TorchTitan's Gated DeltaNet.

The enclosing Qwen3.5 module owns all parameters. This adapter runs Attention
Gym's paging-aware convolution and GDN kernels.

Batch-invariant execution has two additional requirements:

* The accumulated SSM cache state uses float32. Decode otherwise rounds the state
  through bfloat16 after every token, unlike a single prefill call. The convolution
  cache stays in model dtype because it only stores trailing input columns.
* Batch-invariant recurrence uses the same Attention Gym scan as the trainer.

Decode and prefill update the paged convolution and SSM state pools directly.
"""

import dataclasses
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_gym.linear import (
    causal_conv1d_decode,
    gate_transform,
    l2norm,
    paged_causal_conv1d,
    paged_chunk_gdn,
    recurrent_gdn,
    recurrent_gdn_decode,
)

from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.models.qwen3_5.gdn import GatedDeltaNet
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.rl.model.gdn_backend import (
    GDNExecutionPath,
    TorchTitanGDNAttentionBackend,
    TorchTitanGDNAttentionMetadata,
)
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class VLLMInnerGatedDeltaNet(Module, MambaBase):
    """Paged-cache inner GDN implementation.

    The enclosing ``qwen3_5.gdn.GatedDeltaNet`` owns all parameters. This
    module owns only vLLM cache plumbing and kernel dispatch.

    The enclosing module and vLLM cache are both head-sharded under tensor
    parallelism. Speculative decoding is not supported.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_idx: int
        num_k_heads: int
        num_v_heads: int
        head_k_dim: int
        head_v_dim: int
        conv_kernel_size: int = 4

    def __init__(self, config: Config) -> None:
        super().__init__()

        vllm_config = get_current_vllm_config()
        self.tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        speculative_config = vllm_config.speculative_config
        self.num_speculative_tokens = (
            speculative_config.num_speculative_tokens if speculative_config else 0
        )
        # vLLM speculative decoding retains one state per draft position and
        # commits the accepted state; Attention Gym currently mutates one slot.
        if self.num_speculative_tokens != 0:
            raise ValueError("Attention Gym GDN does not support speculative decoding.")

        self.num_k_heads = config.num_k_heads
        self.num_v_heads = config.num_v_heads
        self.head_k_dim = config.head_k_dim
        self.head_v_dim = config.head_v_dim
        self.conv_kernel_size = config.conv_kernel_size

        # vLLM's state-shape calculator takes global head counts, while the
        # computation and allocated cache use local head counts.
        if (
            self.num_k_heads % self.tensor_parallel_size != 0
            or self.num_v_heads % self.tensor_parallel_size != 0
        ):
            raise ValueError(
                f"num_k_heads ({self.num_k_heads}) and num_v_heads "
                f"({self.num_v_heads}) must both be divisible by "
                f"tensor_parallel_size ({self.tensor_parallel_size})."
            )
        self.local_num_k_heads = self.num_k_heads // self.tensor_parallel_size
        self.local_num_v_heads = self.num_v_heads // self.tensor_parallel_size
        self.local_key_dim = self.local_num_k_heads * self.head_k_dim

        if is_conv_state_dim_first():
            raise ValueError(
                "Attention Gym GDN requires VLLM_SSM_CONV_STATE_LAYOUT=SD so "
                "the paged convolution history has contiguous channels."
            )

        # Attention Gym's paged kernels mutate the SSM state pool directly and
        # require FP32 state in both regular and batch-invariant execution.
        if self.cache_config.mamba_ssm_cache_dtype not in {"auto", "float32"}:
            raise ValueError(
                "Attention Gym GDN requires mamba_ssm_cache_dtype='float32', "
                f"got {self.cache_config.mamba_ssm_cache_dtype!r}."
            )
        self.cache_config.mamba_ssm_cache_dtype = "float32"

        # vLLM populates this via the KV-cache allocator: (conv_state, ssm_state).
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        self.prefix = f"model.layers.{config.layer_idx}.linear_attn"
        compilation_config = vllm_config.compilation_config
        if self.prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate GDN layer name: {self.prefix}")
        compilation_config.static_forward_context[self.prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return TorchTitanGDNAttentionBackend

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        """Return the (conv_state, ssm_state) cache dtypes.

        Required by vLLM's MambaBase interface: the KV-cache allocator calls this
        to allocate the paged conv and SSM state before the model runs.
        """
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        """Return the per-slot (conv_state, ssm_state) cache shapes.

        Required by vLLM's MambaBase interface: the KV-cache allocator calls this
        to size the paged conv and SSM state before the model runs.
        """
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tensor_parallel_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_speculative_tokens,
        )

    def _split_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split local fused channels and add a singleton batch dim."""
        num_tokens = mixed_qkv.shape[0]
        local_key_dim = self.local_key_dim
        query = (
            mixed_qkv[:, :local_key_dim]
            .contiguous()
            .view(1, num_tokens, self.local_num_k_heads, self.head_k_dim)
        )
        key = (
            mixed_qkv[:, local_key_dim : 2 * local_key_dim]
            .contiguous()
            .view(1, num_tokens, self.local_num_k_heads, self.head_k_dim)
        )
        value = (
            mixed_qkv[:, 2 * local_key_dim :]
            .contiguous()
            .view(1, num_tokens, self.local_num_v_heads, self.head_v_dim)
        )
        return query, key, value

    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Run convolution and recurrence against vLLM's paged state in place."""
        assert (
            conv_bias is None
        ), "Attention Gym convolution kernels do not support bias"
        attn_metadata = get_forward_context().attn_metadata
        # vLLM's profiling/warmup runs have no attention metadata; leave the
        # zero-filled output.
        if attn_metadata is None:
            return
        assert isinstance(attn_metadata, dict)
        gdn_metadata = attn_metadata[self.prefix]
        assert isinstance(gdn_metadata, TorchTitanGDNAttentionMetadata)
        assert (
            gdn_metadata.spec_sequence_masks is None
        ), "VLLMInnerGatedDeltaNet does not support speculative decoding"

        num_actual_tokens = gdn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return
        state_indices = gdn_metadata.non_spec_state_indices_tensor
        cu_seqlens = gdn_metadata.non_spec_query_start_loc
        has_initial_state = gdn_metadata.has_initial_state
        assert (
            state_indices is not None
            and cu_seqlens is not None
            and has_initial_state is not None
        )
        # SP/model-input rounding need not pad attention metadata. FULL bucket
        # padding does: its prepared rows and captured slices stay fixed while
        # the builder restages slots and freshness, including null padding.
        if gdn_metadata.execution_path is GDNExecutionPath.SINGLE_TOKEN:
            num_decode_rows = state_indices.numel()
            # [q|k|v] is a column slice of the fused projection; the conv kernels
            # need contiguous rows.
            # TODO: drop the copy once Attention Gym's conv kernels accept a row stride.
            conv_output = causal_conv1d_decode(
                mixed_qkv[:num_decode_rows].contiguous(),
                conv_weight,
                self.kv_cache[0],
                activation="silu",
                state_indices=state_indices,
                has_initial_state=has_initial_state,
            )
            if not is_in_batch_invariant_mode():
                recurrent_gdn_decode(
                    conv_output,
                    a[:num_decode_rows].unsqueeze(0),
                    b[:num_decode_rows].unsqueeze(0),
                    A_log.float(),
                    dt_bias.float(),
                    self.kv_cache[1],
                    state_indices,
                    has_initial_state=has_initial_state,
                    scale=self.head_k_dim**-0.5,
                    out=output[:num_decode_rows].unsqueeze(0),
                )
            else:
                self._forward_gdn(
                    conv_output,
                    a[:num_decode_rows],
                    b[:num_decode_rows],
                    A_log,
                    dt_bias,
                    output[:num_decode_rows],
                    cu_seqlens,
                    state_indices,
                    has_initial_state,
                )
            return

        conv_output = paged_causal_conv1d(
            mixed_qkv[:num_actual_tokens].contiguous().unsqueeze(0),
            conv_weight,
            self.kv_cache[0],
            state_indices,
            activation="silu",
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        ).squeeze(0)
        self._forward_gdn(
            conv_output,
            a[:num_actual_tokens],
            b[:num_actual_tokens],
            A_log,
            dt_bias,
            output[:num_actual_tokens],
            cu_seqlens,
            state_indices,
            has_initial_state,
        )

    def _forward_gdn(
        self,
        conv_output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens: torch.Tensor,
        all_slots: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> None:
        """Map each cu_seqlens interval to its paged SSM slot in all_slots.

        Null slots skip state writes and produce zero output for padding.
        """
        query, key, value = self._split_qkv(conv_output)
        # Keep the existing FP32 arithmetic while reusing the GDN gate contract.
        decay = gate_transform(
            a.unsqueeze(0),
            A_log.float(),
            dt_bias.float(),
            kind="softplus",
            impl="reference",
        )
        update_gate = torch.sigmoid(b).unsqueeze(0)
        query = l2norm(query, cu_seqlens=cu_seqlens)
        key = l2norm(key, cu_seqlens=cu_seqlens)

        if is_in_batch_invariant_mode():
            recurrent_output, _ = recurrent_gdn(
                query,
                key,
                value,
                decay,
                update_gate,
                self.kv_cache[1],
                cu_seqlens=cu_seqlens,
                scale=self.head_k_dim**-0.5,
                state_indices=all_slots,
                has_initial_state=has_initial_state,
                # Triton autotuning breaks batch invariance.
                autotune=False,
            )
        else:
            recurrent_output = paged_chunk_gdn(
                query,
                key,
                value,
                decay,
                update_gate,
                self.kv_cache[1],
                all_slots,
                cu_seqlens=cu_seqlens,
                has_initial_state=has_initial_state,
                scale=self.head_k_dim**-0.5,
            )
        output.copy_(recurrent_output[0].to(output.dtype))

    def forward(
        self,
        mixed_qkv_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_weight_CW: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
    ) -> torch.Tensor:
        """Run the flattened vLLM cache operation on rank-local tensors.

        ``VLLMGatedDeltaNet`` passes its fused rank-local ``[q|k|v]`` projection and
        the matching ``[conv_q|conv_k|conv_v]`` weight.
        """
        assert key_head_dim == self.head_k_dim
        assert value_head_dim == self.head_v_dim
        assert conv_weight_CW.shape[-1] == self.conv_kernel_size

        num_tokens = mixed_qkv_TC.shape[0]
        # Padded rows must remain defined across vLLM graph replays.
        output_THV = mixed_qkv_TC.new_zeros(
            num_tokens, self.local_num_v_heads, self.head_v_dim
        )
        self._forward(
            mixed_qkv_TC,
            a_TH,
            b_TH,
            conv_weight_CW,
            None,
            A_log_H,
            dt_bias_H,
            output_THV,
        )
        return output_THV


class VLLMGatedDeltaNet(GatedDeltaNet):
    """``GatedDeltaNet`` that runs its six input projections as one GEMM and its convs as one.

    ``share_input_storage`` lays each rank's local shards of ``in_proj_{q,k,v,z,a,b}``
    and ``conv_{q,k,v}`` out contiguously and re-points the parameters at row views of
    those buffers. The model definition is unchanged: parameters, state-dict keys and
    TP layouts are the trainer's, and loading or weight sync writes through the views.

    Example, TP=2 on rank r (each projection is head-sharded on rows):

        in_proj_weight = [q_r | k_r | v_r | z_r | a_r | b_r]   # one local GEMM
        conv_weight    = [conv_q_r | conv_k_r | conv_v_r]      # one conv over [q_r|k_r|v_r]

    Decode only: in training, the backward of the fused GEMM concatenates six gradients.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GatedDeltaNet.Config):
        pass

    def share_input_storage(self) -> None:
        """Re-point the projection and conv weights at row slices of two local buffers."""
        projections = (
            self.in_proj_q,
            self.in_proj_k,
            self.in_proj_v,
            self.in_proj_z,
            self.in_proj_a,
            self.in_proj_b,
        )
        self.in_proj_weight, self.in_proj_split_sizes = _share_row_storage(projections)
        convs = (self.conv_q, self.conv_k, self.conv_v)
        conv_weight_C1W, _ = _share_row_storage(convs)
        self.conv_weight = conv_weight_C1W.squeeze(1)

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: VarlenMetadata | None = None,
    ) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            # All six projections read x: gather it once, like GatedDeltaNet.
            x_TD = spmd.redistribute(
                x_TD,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
            )
        num_tokens = x_TD.shape[0]
        # Unused by the vLLM inner GDN, which reads the vLLM GDN metadata instead.
        cu_seqlens = torch.arange(
            0, num_tokens + 1, num_tokens, dtype=torch.int32, device=x_TD.device
        )
        num_qkv_channels = sum(self.in_proj_split_sizes[:3])
        with spmd.local():
            projected_TC = F.linear(x_TD, self.in_proj_weight)
            # q, k and v are adjacent columns, so [q|k|v] is one row-strided view.
            mixed_qkv_TC = projected_TC[:, :num_qkv_channels]
            gate_TC, a_TH, b_TH = projected_TC[:, num_qkv_channels:].split(
                self.in_proj_split_sizes[3:], dim=-1
            )
            gate_THV = gate_TC.view(num_tokens, -1, self.value_head_dim)
        output_THV = self.inner_gated_delta_net(
            mixed_qkv_TC,
            a_TH,
            b_TH,
            self.conv_weight,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
        )
        output_THV = self.norm(output_THV, gate_THV)
        return self.out_proj(output_THV.reshape(num_tokens, -1))


def fused_inner_sharding_config(sharding_config: ShardingConfig) -> ShardingConfig:
    """Rename the inner GDN's per-projection inputs to ``VLLMInnerGatedDeltaNet``'s fused ones.

    Example:

        {"query_TC": P, "key_TC": P, "value_TC": P, "conv_q_weight_C1W": W, ...}
        # -> {"mixed_qkv_TC": P, "conv_weight_CW": W, ...}
    """
    renamed = {"query_TC": "mixed_qkv_TC", "conv_q_weight_C1W": "conv_weight_CW"}
    dropped = {"key_TC", "value_TC", "conv_k_weight_C1W", "conv_v_weight_C1W"}

    def fuse(shardings: dict[str, SpmdType]) -> dict[str, SpmdType]:
        return {
            renamed.get(name, name): layout
            for name, layout in shardings.items()
            if name not in dropped
        }

    return dataclasses.replace(
        sharding_config,
        in_src_shardings=fuse(sharding_config.in_src_shardings),
        in_dst_shardings=fuse(sharding_config.in_dst_shardings),
    )


def _share_row_storage(
    modules: tuple[nn.Module, ...],
) -> tuple[torch.Tensor, list[int]]:
    """Concatenate the modules' local weights on dim 0 and make each weight a row view."""
    buffer = torch.cat([module.weight.detach() for module in modules])
    split_sizes = [module.weight.shape[0] for module in modules]
    for module, view in zip(modules, buffer.split(split_sizes), strict=True):
        shared = nn.Parameter(view, requires_grad=False)
        # Keep the parameter's SPMD layout annotation (TP head sharding).
        spmd.assert_type_like(shared, module.weight)
        module.weight = shared
    return buffer, split_sizes
