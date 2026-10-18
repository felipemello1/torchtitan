# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated DeltaNet modules for Qwen3.5."""

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels,
# H = attention heads, K = query/key head dimension, V = value head dimension,
# S = state slots, W = convolution kernel width.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm, recurrent_gdn
from torch import nn

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.protocols.module import Module


@spmd.local_map(
    in_types=(
        {"dp": spmd.S(0), "tp": spmd.S(1)},
        {"dp": spmd.R, "tp": spmd.S(0)},
        {"dp": spmd.V, "tp": spmd.R},
    ),
    out_types={"dp": spmd.S(0), "tp": spmd.S(1)},
)
def _causal_conv1d_varlen(
    x_TD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Depthwise causal conv with per-document resets (CUDA-only).

    A pure-torch per-document reference lives in
    ``tests/unit_tests/gpu/test_qwen3_5_deltanet.py``.
    """
    out_BTD = causal_conv1d(
        x_TD.unsqueeze(0),
        weight.squeeze(1),
        activation="silu",
        cu_seqlens=cu_seqlens,
    )
    assert isinstance(out_BTD, torch.Tensor)
    return out_BTD.squeeze(0)


class RMSNormGated(Module):
    """Gated RMSNorm: ``silu(gate) * weight * norm(x)``.

    Takes ``(x, gate)`` separately. Weight is ones-initialized.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = (self.weight.float() * x).to(input_dtype)
        x = x * F.silu(gate.float())
        return x.to(input_dtype)


@torch.library.custom_op(
    "torchtitan::recurrent_gdn_fwd", mutates_args=(), device_types="cuda"
)
def _recurrent_gdn_fwd(
    q_BTHK: torch.Tensor,
    k_BTHK: torch.Tensor,
    v_BTHV: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Run the batch-invariant GDN recurrent forward kernel.

    The vLLM generator uses Attention Gym's paging-aware recurrent kernel for
    per-token decode. The trainer uses the same recurrence with a materialized
    float32 initial state and varlen metadata so its forward is bitwise identical
    to generation.
    """
    num_sequences = int(cu_seqlens.numel()) - 1
    # state_cache_SHVK: [num_sequences + 1, H, V, K].
    state_cache_SHVK = q_BTHK.new_empty(
        num_sequences + 1,
        v_BTHV.shape[2],
        v_BTHV.shape[3],
        q_BTHK.shape[3],
        dtype=torch.float32,
    )
    state_indices = torch.arange(
        1,
        num_sequences + 1,
        dtype=torch.int32,
        device=q_BTHK.device,
    )
    has_initial_state = torch.zeros(
        num_sequences,
        dtype=torch.bool,
        device=q_BTHK.device,
    )
    # The recurrent operator consumes normalized Q/K.
    normalized_q_BTHK = l2norm(q_BTHK, cu_seqlens=cu_seqlens)
    normalized_k_BTHK = l2norm(k_BTHK, cu_seqlens=cu_seqlens)
    out_BTHV, _ = recurrent_gdn(
        normalized_q_BTHK,
        normalized_k_BTHK,
        v_BTHV,
        g,
        beta,
        state_cache_SHVK,
        cu_seqlens=cu_seqlens,
        scale=q_BTHK.shape[-1] ** -0.5,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        autotune=False,
    )
    return out_BTHV.to(q_BTHK.dtype)


@_recurrent_gdn_fwd.register_fake
def _recurrent_gdn_fwd_fake(
    q_BTHK: torch.Tensor,
    k_BTHK: torch.Tensor,
    v_BTHV: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(v_BTHV, dtype=q_BTHK.dtype)


def _chunk_gdn_gradients(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute the parallel GDN chunk kernel and return its gradients."""
    with torch.enable_grad():
        inputs = tuple(
            tensor.detach().requires_grad_(True) for tensor in (q, k, v, g, beta)
        )
        normalized_q = l2norm(inputs[0], cu_seqlens=cu_seqlens)
        normalized_k = l2norm(inputs[1], cu_seqlens=cu_seqlens)
        output, _ = chunk_gdn(
            normalized_q,
            normalized_k,
            inputs[2],
            inputs[3],
            inputs[4],
            cu_seqlens=cu_seqlens,
            scale=inputs[0].shape[-1] ** -0.5,
            impl="fused",
        )
        grad_q, grad_k, grad_v, grad_g, grad_beta = torch.autograd.grad(
            output, inputs, grad_output
        )
        return grad_q, grad_k, grad_v, grad_g, grad_beta


def _recurrent_gdn_setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs)


def _recurrent_gdn_backward(ctx, grad_output):
    q, k, v, g, beta, cu_seqlens = ctx.saved_tensors
    grads = _chunk_gdn_gradients(
        grad_output,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
    )
    return (*grads, None)


_recurrent_gdn_fwd.register_autograd(
    _recurrent_gdn_backward, setup_context=_recurrent_gdn_setup_context
)


class GatedDeltaKernel(Module):
    """Run GDN on rank-local tensors.

    This module provides a local SPMD boundary for the sharding code. A
    pure-torch reference implementation lives in
    ``tests/unit_tests/gpu/test_qwen3_5_deltanet.py``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        xv_THV: torch.Tensor,
        g_TH: torch.Tensor,
        beta_TH: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xq_BTHK = xq_THK.unsqueeze(0)
        xk_BTHK = xk_THK.unsqueeze(0)
        xv_BTHV = xv_THV.unsqueeze(0)
        g_BTH = g_TH.unsqueeze(0)
        beta_BTH = beta_TH.unsqueeze(0)

        if is_in_batch_invariant_mode() and cu_seqlens is not None:
            return _recurrent_gdn_fwd(
                xq_BTHK,
                xk_BTHK,
                xv_BTHV,
                g_BTH,
                beta_BTH,
                cu_seqlens,
            ).squeeze(0)

        normalized_q = l2norm(xq_BTHK, cu_seqlens=cu_seqlens)
        normalized_k = l2norm(xk_BTHK, cu_seqlens=cu_seqlens)
        output, _ = chunk_gdn(
            normalized_q,
            normalized_k,
            xv_BTHV,
            g_BTH,
            beta_BTH,
            cu_seqlens=cu_seqlens,
            scale=xq_BTHK.shape[-1] ** -0.5,
            impl="fused",
        )
        return output.squeeze(0)


class InnerGatedDeltaNet(Module):
    """Dense GDN computation behind the vLLM replacement boundary.

    Takes the fused rank-local `[q|k|v]` projection and the matching
    `[conv_q|conv_k|conv_v]` weight. The depthwise causal conv is per channel,
    so one conv over `[q|k|v]` equals three separate convs; q, k and v are then
    column views of its output. The vLLM replacement keeps this signature and
    runs the conv against its paged state.

    Example (local heads: 2 key heads of 4, 4 value heads of 4):

        mixed_qkv_TC.shape == (T, 2*4 + 2*4 + 4*4) == (T, 32)
        # conv -> split [8 | 8 | 16] -> q [T, 2, 4], k [T, 2, 4], v [T, 4, 4]
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        kernel: GatedDeltaKernel.Config

    def __init__(self, config: Config):
        super().__init__()
        self.kernel = config.kernel.build()

    def forward(
        self,
        mixed_qkv_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
    ) -> torch.Tensor:
        """Run one Q/K/V convolution and the recurrence on local heads."""
        num_tokens = mixed_qkv_TC.shape[0]
        use_varlen_kernels = cu_seqlens.numel() > 2 or is_in_batch_invariant_mode()
        value_dim = A_log_H.shape[0] * value_head_dim
        key_dim = (mixed_qkv_TC.shape[-1] - value_dim) // 2

        if use_varlen_kernels:
            # After share_input_storage, [q|k|v] is a row-strided column view;
            # Attention Gym's varlen conv needs contiguous rows (no-op otherwise).
            conv_TC = _causal_conv1d_varlen(
                mixed_qkv_TC.contiguous(), conv_weight_C1W, cu_seqlens
            )
        else:
            x_1CT = F.pad(
                mixed_qkv_TC.transpose(0, 1).unsqueeze(0),
                [conv_weight_C1W.shape[-1] - 1, 0],
            )
            conv_TC = (
                F.silu(
                    F.conv1d(
                        x_1CT, conv_weight_C1W, None, groups=conv_weight_C1W.shape[0]
                    )
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        xq_TC, xk_TC, xv_TC = conv_TC.split([key_dim, key_dim, value_dim], dim=-1)
        g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
        beta_TH = torch.sigmoid(b_TH)
        return self.kernel(
            xq_TC.reshape(num_tokens, -1, key_head_dim),
            xk_TC.reshape(num_tokens, -1, key_head_dim),
            xv_TC.reshape(num_tokens, -1, value_head_dim),
            g_TH,
            beta_TH,
            cu_seqlens=cu_seqlens if use_varlen_kernels else None,
        )


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention.

    Uses recurrent state + gated delta rule instead of softmax attention.
    No RoPE, different head structure from standard attention. Conv and
    recurrent state are reset at document boundaries whenever document
    offsets (``VarlenMetadata``) are provided -- the transformer block picks
    them out of the model's attention-mask dict under the ``"deltanet"`` key
    (both attention backends). With no offsets (``None``) the packed sequence
    is processed as a single continuous stream.

    The six input projections and three convs are separate parameters (the
    checkpoint, weight-sync and optimizer format). The forward fuses each rank's
    local shards: one GEMM for `[q|k|v]` and one conv over it, with z, a and b as
    their own projections (after `share_input_storage`, one GEMM for all six).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        key_head_dim: int
        value_head_dim: int
        conv_kernel_size: int = 4

        # Sub-module configs
        in_proj_q: Linear.Config
        in_proj_k: Linear.Config
        in_proj_v: Linear.Config
        in_proj_z: Linear.Config
        in_proj_a: Linear.Config
        in_proj_b: Linear.Config
        conv_q: Conv1d.Config
        conv_k: Conv1d.Config
        conv_v: Conv1d.Config
        inner_gated_delta_net: Module.Config
        norm: RMSNormGated.Config
        out_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
        value_dim = config.in_proj_v.out_features

        self.in_proj_q = config.in_proj_q.build()
        self.in_proj_k = config.in_proj_k.build()
        self.in_proj_v = config.in_proj_v.build()
        self.in_proj_z = config.in_proj_z.build()
        self.in_proj_a = config.in_proj_a.build()
        self.in_proj_b = config.in_proj_b.build()

        self.conv_q = config.conv_q.build()
        self.conv_k = config.conv_k.build()
        self.conv_v = config.conv_v.build()

        n_value_heads = value_dim // config.value_head_dim
        self.A_log = nn.Parameter(torch.empty(n_value_heads))
        self.dt_bias = nn.Parameter(torch.empty(n_value_heads))

        self.norm = config.norm.build()
        self.out_proj = config.out_proj.build()
        self.inner_gated_delta_net = config.inner_gated_delta_net.build()

        # Quantized or LoRA projections override Linear._linear, so their weights
        # can't be concatenated with the others.
        self.fuse_qkv_projections = all(
            type(projection) is Linear
            for projection in (self.in_proj_q, self.in_proj_k, self.in_proj_v)
        )
        # Set by share_input_storage (inference): the parameters become row views
        # of these buffers, so the fused GEMM and conv read them without a copy.
        self._shared_input_weight_CD: torch.Tensor | None = None
        self._shared_input_split_sizes: list[int] = []
        self._shared_conv_weight_C1W: torch.Tensor | None = None

    def _input_projections(self) -> tuple[Linear, ...]:
        return (
            self.in_proj_q,
            self.in_proj_k,
            self.in_proj_v,
            self.in_proj_z,
            self.in_proj_a,
            self.in_proj_b,
        )

    def share_input_storage(self) -> None:
        """Lay out the projection and conv weights in two buffers, for inference.

        The parameters become row views of the buffers, so the forward runs one
        GEMM and one conv over them without concatenating each step, and loading
        or weight sync writes straight into them. Parameters, state-dict keys
        and SPMD layouts are unchanged. Inference engines that own their
        parameters' storage call this after materializing the model and before
        loading; training does not (FSDP owns each parameter's storage).

        Example, TP=2 on rank r:

            _shared_input_weight_CD = [q_r | k_r | v_r | z_r | a_r | b_r]
            _shared_input_split_sizes = [q_r + k_r + v_r, z_r, a_r, b_r]  # row counts
            _shared_conv_weight_C1W = [conv_q_r | conv_k_r | conv_v_r]
        """
        projections = self._input_projections()
        if any(type(projection) is not Linear for projection in projections):
            raise ValueError(
                "share_input_storage needs plain Linear input projections, got "
                f"{[type(projection).__name__ for projection in projections]}"
            )
        self._shared_input_weight_CD = _share_row_storage(projections)
        row_counts = [projection.weight.shape[0] for projection in projections]
        self._shared_input_split_sizes = [sum(row_counts[:3]), *row_counts[3:]]
        self._shared_conv_weight_C1W = _share_row_storage(
            (self.conv_q, self.conv_k, self.conv_v)
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: VarlenMetadata | None = None,
    ) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            # All six input projections consume x, so gather it once.
            x_TD = spmd.redistribute(
                x_TD,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
                backward_options={"op_dtype": x_TD.dtype},
            )

        num_tokens = x_TD.shape[0]
        if attention_masks is not None:
            cu_seqlens = attention_masks.cu_seq_q
        else:
            cu_seqlens = torch.arange(
                0,
                num_tokens + 1,
                num_tokens,
                dtype=torch.int32,
                device=x_TD.device,
            )

        if self._shared_input_weight_CD is not None:
            # Inference: one GEMM over the buffer from share_input_storage.
            mixed_qkv_TC, gate_THV, a_TH, b_TH = _project_with_one_gemm(
                x_TD,
                self._shared_input_weight_CD,
                self._shared_input_split_sizes,
                self.value_head_dim,
            )
        else:
            # Training: q, k and v as one GEMM; z, a and b as their own projections.
            qkv_projections = (self.in_proj_q, self.in_proj_k, self.in_proj_v)
            if self.fuse_qkv_projections:
                mixed_qkv_TC = _linear_over_concatenated_rows(
                    x_TD, *(projection.weight for projection in qkv_projections)
                )
            else:
                mixed_qkv_TC = _concat_columns(
                    *(projection(x_TD) for projection in qkv_projections)
                )
            gate_THV = _split_heads(self.in_proj_z(x_TD), self.value_head_dim)
            a_TH = self.in_proj_a(x_TD)
            b_TH = self.in_proj_b(x_TD)

        conv_weight_C1W = self._shared_conv_weight_C1W
        if conv_weight_C1W is None:
            conv_weight_C1W = _concat_rows(
                self.conv_q.weight, self.conv_k.weight, self.conv_v.weight
            )
        output_THV = self.inner_gated_delta_net(
            mixed_qkv_TC,
            a_TH,
            b_TH,
            conv_weight_C1W,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
        )
        output_THV = self.norm(output_THV, gate_THV)
        return self.out_proj(output_THV.reshape(num_tokens, -1))


# The fused projections run on rank-local shards. Concatenating TP shards along
# their sharded dim gives [q_r | k_r | v_r], which has no global SPMD meaning, so
# the concat, GEMM and split run under local SPMD types. Each output is
# head-sharded on TP, like the per-projection outputs.
_HEAD_SHARDED = {"dp": spmd.S(0), "tp": spmd.S(1)}
_HEAD_SHARDED_ROWS = {"dp": spmd.R, "tp": spmd.S(0)}


@spmd.local_map(out_types=_HEAD_SHARDED)
def _linear_over_concatenated_rows(
    x_TD: torch.Tensor, *weights_CD: torch.Tensor
) -> torch.Tensor:
    """Run one GEMM over row-concatenated local weight shards, e.g. `[q_r | k_r | v_r]`.

    Training fuses only q, k and v: the conv reads this output contiguously, and its
    backward writes `d[q|k|v]` straight into this GEMM's gradient. Fusing z, a and b
    too would make the backward concatenate their gradients, which inductor lowers as
    a slow masked pointwise kernel unless `max_pointwise_cat_inputs=0`.
    """
    return F.linear(x_TD, torch.cat(weights_CD))


@spmd.local_map(out_types=(_HEAD_SHARDED,) * 4)
def _project_with_one_gemm(
    x_TD: torch.Tensor,
    weight_CD: torch.Tensor,
    split_sizes: list[int],
    value_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run all six projections as one GEMM over the local `[q|k|v|z|a|b]` weight.

    `mixed_qkv_TC` is then a row-strided column view of the GEMM output.

    Args:
        x_TD: rank's input, replicated on TP.
        weight_CD: local `[q|k|v|z|a|b]` weight.
        split_sizes: its `[q+k+v, z, a, b]` row counts.
        value_head_dim: splits the gate into value heads.

    Returns:
        `(mixed_qkv_TC, gate_THV, a_TH, b_TH)`.
    """
    mixed_qkv_TC, gate_TC, a_TH, b_TH = F.linear(x_TD, weight_CD).split(
        split_sizes, dim=-1
    )
    return mixed_qkv_TC, gate_TC.view(x_TD.shape[0], -1, value_head_dim), a_TH, b_TH


@spmd.local_map(out_types=_HEAD_SHARDED)
def _split_heads(x_TC: torch.Tensor, head_dim: int) -> torch.Tensor:
    """View head-sharded columns `[T, H * head_dim]` as `[T, H, head_dim]`.

    Local, because TP splits the columns on head boundaries, which the global SPMD
    type checker cannot see.
    """
    return x_TC.view(x_TC.shape[0], -1, head_dim)


@spmd.local_map(out_types=_HEAD_SHARDED)
def _concat_columns(*outputs_TC: torch.Tensor) -> torch.Tensor:
    """Concatenate local projection outputs by columns, e.g. `[q_r | k_r | v_r]`."""
    return torch.cat(outputs_TC, dim=-1)


@spmd.local_map(out_types=_HEAD_SHARDED_ROWS)
def _concat_rows(*weights: torch.Tensor) -> torch.Tensor:
    """Concatenate local weight shards by rows, e.g. `[conv_q_r | conv_k_r | conv_v_r]`."""
    return torch.cat(weights)


def _share_row_storage(modules: tuple[nn.Module, ...]) -> torch.Tensor:
    """Copy the modules' local weights into one buffer and make each weight a row view of it.

    The views get no gradients (the forward reads the buffer), so this is for inference.
    """
    buffer = torch.cat([module.weight.detach() for module in modules])
    row_counts = [module.weight.shape[0] for module in modules]
    for module, view in zip(modules, buffer.split(row_counts), strict=True):
        shared = nn.Parameter(view, requires_grad=False)
        # Keep the parameter's SPMD layout annotation (TP head sharding).
        spmd.assert_type_like(shared, module.weight)
        module.weight = shared
    spmd.assert_type_like(buffer, modules[0].weight)
    return buffer
