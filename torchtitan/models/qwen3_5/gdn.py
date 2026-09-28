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
from typing import NamedTuple

import spmd_types as spmd
import torch
import torch.nn.functional as F
from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm, recurrent_gdn
from torch import nn

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import get_shared_prefix_starts, VarlenMetadata
from torchtitan.protocols.module import Module


class DeltaNetSharedPrefixMetadata(NamedTuple):
    """GDN sequence layout for completions that share one packed prefix.

    Rows are reordered to [every root, every completion], where a root is a
    document or a shared prefix. Each completion starts from its prefix's final
    convolution and recurrent state, so the prefix is scanned once. Built by
    `create_deltanet_shared_prefix_metadata`.
    """

    permutation: torch.Tensor
    """[T] packed row of each reordered row: roots first, then completions."""
    inverse_permutation: torch.Tensor
    """[T] reordered row of each packed row."""
    cu_seqlens: torch.Tensor
    """[num_roots + num_completions + 1] sequence offsets in reordered rows."""
    num_roots: int
    """Documents and shared prefixes; they come first in the reordered rows."""
    num_root_tokens: int
    """Rows before the first completion in the reordered rows."""
    completion_roots: torch.Tensor
    """[num_completions] index of each completion's prefix among the roots."""
    completion_conv_rows: torch.Tensor
    """[num_completions, W - 1] reordered rows of each prefix's last conv inputs."""
    completion_conv_mask: torch.Tensor
    """[num_completions, W - 1] False where the prefix is shorter than W - 1."""

    _SPMD_TYPE = spmd.SpmdType(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.TP: spmd.R,
        },
        partition_spec=spmd.PartitionSpec(MeshAxisName.DP),
    )
    _CONV_SPMD_TYPE = spmd.SpmdType(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.TP: spmd.R,
        },
        partition_spec=spmd.PartitionSpec(MeshAxisName.DP, None),
    )

    def annotate_spmd_types(self) -> None:
        """Annotate the rank-local index tensors under the dense model-parallel mesh."""
        for indices in (
            self.permutation,
            self.inverse_permutation,
            self.cu_seqlens,
            self.completion_roots,
        ):
            spmd.assert_type(indices, self._SPMD_TYPE)
        spmd.assert_type(self.completion_conv_rows, self._CONV_SPMD_TYPE)
        spmd.assert_type(self.completion_conv_mask, self._CONV_SPMD_TYPE)

    def conv_initial_state(self, x_TC: torch.Tensor) -> torch.Tensor:
        """Per-sequence conv state [S, W - 1, C]: zeros for roots, the prefix's tail for completions."""
        num_completions, width = self.completion_conv_rows.shape
        completion_state = x_TC.index_select(
            0, self.completion_conv_rows.flatten()
        ).view(num_completions, width, -1)
        completion_state = completion_state * self.completion_conv_mask[..., None]
        root_state = completion_state.new_zeros(
            self.num_roots, width, completion_state.shape[-1]
        )
        return torch.cat([root_state, completion_state])


def create_deltanet_shared_prefix_metadata(
    positions: torch.Tensor, *, conv_kernel_size: int
) -> DeltaNetSharedPrefixMetadata:
    """Split packed documents into roots and completions that continue their shared prefix.

    A root starts where `positions` resets to 0. A completion starts where
    positions drop to `p > 0` (see `get_shared_prefix_starts`), and so does the
    first completion's tail at row `document start + p`: every completion then
    continues from the state after its document's first `p` tokens.

    Example:
        # Prompt [P0, P1] with completions [A0, A1] and [B0], then document [D0, D1].
        # rows:      P0 P1 A0 A1 B0 D0 D1
        positions = torch.tensor([0, 1, 2, 3, 2, 0, 1])
        create_deltanet_shared_prefix_metadata(positions, conv_kernel_size=2)
        # -> permutation = [0, 1, 5, 6, 2, 3, 4]  # roots [P0 P1] [D0 D1], then [A0 A1] [B0]
        #    cu_seqlens = [0, 2, 4, 6, 7], num_roots = 2, num_root_tokens = 4
        #    completion_roots = [0, 0], completion_conv_rows = [[1], [1]]  # P1
    """
    num_tokens = positions.shape[0]
    device = positions.device
    token_index = torch.arange(num_tokens, device=device)
    is_doc_start = positions == 0
    shared_prefix_starts = get_shared_prefix_starts(positions)
    doc_starts = torch.cummax(torch.where(is_doc_start, token_index, 0), dim=0).values
    prefix_ends = doc_starts + positions

    # Sequences: documents, completions, and the first completion of each prefix.
    is_sequence_start = is_doc_start | shared_prefix_starts
    is_sequence_start[prefix_ends[shared_prefix_starts]] = True
    sequence_starts = is_sequence_start.nonzero(as_tuple=True)[0]
    sequence_lengths = torch.diff(
        sequence_starts, append=sequence_starts.new_tensor([num_tokens])
    )
    is_root = positions[sequence_starts].eq(0)
    token_is_root = is_root[torch.cumsum(is_sequence_start, dim=0) - 1]

    permutation = torch.cat(
        [
            token_is_root.nonzero(as_tuple=True)[0],
            (~token_is_root).nonzero(as_tuple=True)[0],
        ]
    )
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = token_index
    cu_seqlens = F.pad(
        torch.cat([sequence_lengths[is_root], sequence_lengths[~is_root]]).cumsum(0),
        (1, 0),
    ).to(torch.int32)

    # A completion's prefix is the root that starts at its document start.
    completion_starts = sequence_starts[~is_root]
    root_rank = torch.cumsum(is_root, dim=0) - 1
    root_sequence = torch.searchsorted(sequence_starts, doc_starts[completion_starts])
    completion_roots = root_rank[root_sequence]
    conv_rows = (
        prefix_ends[completion_starts, None]
        - (conv_kernel_size - 1)
        + torch.arange(conv_kernel_size - 1, device=device)
    )
    completion_conv_mask = conv_rows >= doc_starts[completion_starts, None]
    completion_conv_rows = inverse_permutation[conv_rows.clamp_min(0)]

    # host syncs, once per microbatch rather than per layer
    num_roots, num_root_tokens = torch.stack(
        [is_root.sum(), token_is_root.sum()]
    ).tolist()
    if spmd.is_type_checking():
        for indices in (
            permutation,
            inverse_permutation,
            cu_seqlens,
            completion_roots,
            completion_conv_rows,
            completion_conv_mask,
        ):
            spmd.mutate_type(indices, "dp", src=spmd.R, dst=spmd.V)
    return DeltaNetSharedPrefixMetadata(
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        cu_seqlens=cu_seqlens,
        num_roots=num_roots,
        num_root_tokens=num_root_tokens,
        completion_roots=completion_roots,
        completion_conv_rows=completion_conv_rows,
        completion_conv_mask=completion_conv_mask,
    )


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
    *,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise causal conv with per-document resets (CUDA-only).

    ``initial_state`` ([S, W - 1, D]) holds each sequence's preceding inputs;
    None starts every sequence from zeros. A pure-torch per-document reference
    lives in ``tests/unit_tests/gpu/test_qwen3_5_deltanet.py``.
    """
    out_BTD = causal_conv1d(
        x_TD.unsqueeze(0),
        weight.squeeze(1),
        activation="silu",
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
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
        shared_prefix: DeltaNetSharedPrefixMetadata | None = None,
    ) -> torch.Tensor:
        xq_BTHK = xq_THK.unsqueeze(0)
        xk_BTHK = xk_THK.unsqueeze(0)
        xv_BTHV = xv_THV.unsqueeze(0)
        g_BTH = g_TH.unsqueeze(0)
        beta_BTH = beta_TH.unsqueeze(0)

        if is_in_batch_invariant_mode() and cu_seqlens is not None:
            # TODO(rl): pass shared-prefix states through `_recurrent_gdn_fwd` and its
            # backward; the RL controller rejects share_prompt with batch_invariant.
            assert shared_prefix is None
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
        if shared_prefix is not None:
            assert cu_seqlens is not None
            return _chunk_gdn_with_shared_prefixes(
                normalized_q,
                normalized_k,
                xv_BTHV,
                g_BTH,
                beta_BTH,
                cu_seqlens=cu_seqlens,
                shared_prefix=shared_prefix,
                scale=xq_BTHK.shape[-1] ** -0.5,
            ).squeeze(0)
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


def _chunk_gdn_with_shared_prefixes(
    q_BTHK: torch.Tensor,
    k_BTHK: torch.Tensor,
    v_BTHV: torch.Tensor,
    g_BTH: torch.Tensor,
    beta_BTH: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    shared_prefix: DeltaNetSharedPrefixMetadata,
    scale: float,
) -> torch.Tensor:
    """Scan the roots, then start each completion from its prefix's final state.

    Inputs are in reordered rows [roots, completions] (see
    `DeltaNetSharedPrefixMetadata`), with q and k already normalized.
    """
    num_roots = shared_prefix.num_roots
    num_tokens = shared_prefix.num_root_tokens
    root_output, root_state = chunk_gdn(
        q_BTHK[:, :num_tokens],
        k_BTHK[:, :num_tokens],
        v_BTHV[:, :num_tokens],
        g_BTH[:, :num_tokens],
        beta_BTH[:, :num_tokens],
        cu_seqlens=cu_seqlens[: num_roots + 1],
        scale=scale,
        output_final_state=True,
        impl="fused",
    )
    assert root_state is not None
    completion_output, _ = chunk_gdn(
        q_BTHK[:, num_tokens:],
        k_BTHK[:, num_tokens:],
        v_BTHV[:, num_tokens:],
        g_BTH[:, num_tokens:],
        beta_BTH[:, num_tokens:],
        initial_state=root_state.index_select(0, shared_prefix.completion_roots),
        cu_seqlens=cu_seqlens[num_roots:] - num_tokens,
        scale=scale,
        impl="fused",
    )
    return torch.cat([root_output, completion_output], dim=1)


class InnerGatedDeltaNet(Module):
    """Dense GDN computation behind the vLLM replacement boundary.

    The trainer keeps Q, K, and V separate, matching the main-branch GDN flow.
    The vLLM replacement may fuse them internally for its paged convolution
    cache without changing this dense path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        kernel: GatedDeltaKernel.Config

    def __init__(self, config: Config):
        super().__init__()
        self.kernel = config.kernel.build()

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        shared_prefix: DeltaNetSharedPrefixMetadata | None = None,
    ) -> torch.Tensor:
        """Run separate Q/K/V convolutions and recurrence on local heads."""
        num_tokens = query_TC.shape[0]
        use_varlen_kernels = cu_seqlens.numel() > 2 or is_in_batch_invariant_mode()

        def causal_conv(
            x_TC: torch.Tensor,
            weight_C1W: torch.Tensor,
        ) -> torch.Tensor:
            if use_varlen_kernels:
                return _causal_conv1d_varlen(
                    x_TC,
                    weight_C1W,
                    cu_seqlens,
                    initial_state=(
                        None
                        if shared_prefix is None
                        else shared_prefix.conv_initial_state(x_TC)
                    ),
                )

            x_1CT = F.pad(
                x_TC.transpose(0, 1).unsqueeze(0),
                [weight_C1W.shape[-1] - 1, 0],
            )
            return (
                F.silu(
                    F.conv1d(
                        x_1CT,
                        weight_C1W,
                        None,
                        groups=weight_C1W.shape[0],
                    )
                )
                .squeeze(0)
                .transpose(0, 1)
            )

        xq_THK = causal_conv(query_TC, conv_q_weight_C1W).reshape(
            num_tokens, -1, key_head_dim
        )
        xk_THK = causal_conv(key_TC, conv_k_weight_C1W).reshape(
            num_tokens, -1, key_head_dim
        )
        xv_THV = causal_conv(value_TC, conv_v_weight_C1W).reshape(
            num_tokens, -1, value_head_dim
        )
        g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
        beta_TH = torch.sigmoid(b_TH)
        return self.kernel(
            xq_THK,
            xk_THK,
            xv_THV,
            g_TH,
            beta_TH,
            cu_seqlens=cu_seqlens if use_varlen_kernels else None,
            shared_prefix=shared_prefix,
        )


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention.

    Uses recurrent state + gated delta rule instead of softmax attention.
    No RoPE, different head structure from standard attention. Conv and
    recurrent state are reset at document boundaries whenever document
    offsets (``VarlenMetadata``) are provided -- the transformer block picks
    them out of the model's attention-mask dict under the ``"deltanet"`` key
    (both attention backends). With no offsets (``None``) the packed sequence
    is processed as a single continuous stream. With
    ``DeltaNetSharedPrefixMetadata``, completions continue their shared prefix's
    state instead of starting fresh.
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

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: VarlenMetadata | DeltaNetSharedPrefixMetadata | None = None,
    ) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is not None:
            # All six input projections consume x, so gather it once before
            # entering their separate compute paths.
            x_TD = spmd.redistribute(
                x_TD,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
                backward_options={"op_dtype": x_TD.dtype},
            )

        num_tokens = x_TD.shape[0]
        shared_prefix = None
        if isinstance(attention_masks, DeltaNetSharedPrefixMetadata):
            shared_prefix = attention_masks
            cu_seqlens = attention_masks.cu_seqlens
        elif attention_masks is not None:
            cu_seqlens = attention_masks.cu_seq_q
        else:
            cu_seqlens = torch.arange(
                0,
                num_tokens + 1,
                num_tokens,
                dtype=torch.int32,
                device=x_TD.device,
            )

        # Shared prefixes reorder the recurrence inputs to [roots, completions].
        recurrence_x_TD = (
            x_TD
            if shared_prefix is None
            else x_TD.index_select(0, shared_prefix.permutation)
        )
        query_TC = self.in_proj_q(recurrence_x_TD)
        key_TC = self.in_proj_k(recurrence_x_TD)
        value_TC = self.in_proj_v(recurrence_x_TD)
        gate_TC = self.in_proj_z(x_TD)
        a_TH = self.in_proj_a(recurrence_x_TD)
        b_TH = self.in_proj_b(recurrence_x_TD)

        output_THV = self.inner_gated_delta_net(
            query_TC,
            key_TC,
            value_TC,
            a_TH,
            b_TH,
            self.conv_q.weight,
            self.conv_k.weight,
            self.conv_v.weight,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            shared_prefix=shared_prefix,
        )
        if shared_prefix is not None:
            output_THV = output_THV.index_select(0, shared_prefix.inverse_permutation)
        gate_THV = gate_TC.view(num_tokens, -1, self.value_head_dim)
        output_THV = self.norm(output_THV, gate_THV)
        out_TD = output_THV.reshape(num_tokens, -1)
        return self.out_proj(out_TD)
