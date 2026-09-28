# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 GatedDeltaNet fused input projections against the per-projection math.

Shape suffixes: T = tokens, D = model dim, H = heads, K = key head dim,
V = value head dim.
"""

import dataclasses
import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch
import torch.nn.functional as F
from spmd_types.checker import typecheck
from torch import nn
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.transform.base import convert_config_type
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.gdn import GatedDeltaNet
from torchtitan.models.qwen3_5.sharding import set_qwen35_sharding_config


class _ReferenceGatedDeltaKernel(nn.Module):
    """Token-by-token gated delta rule on CPU, standing in for the Triton kernel."""

    def forward(self, xq_THK, xk_THK, xv_THV, g_TH, beta_TH, *, cu_seqlens=None):
        assert cu_seqlens is None
        repeat = xv_THV.shape[1] // xq_THK.shape[1]
        scale = xq_THK.shape[-1] ** -0.5
        q_THK = F.normalize(xq_THK.float(), dim=-1).repeat_interleave(repeat, 1) * scale
        k_THK = F.normalize(xk_THK.float(), dim=-1).repeat_interleave(repeat, 1)
        v_THV = xv_THV.float()
        state_HKV = v_THV.new_zeros(v_THV.shape[1], q_THK.shape[-1], v_THV.shape[-1])
        outputs = []
        for t in range(v_THV.shape[0]):
            state_HKV = state_HKV * g_TH[t].float().exp()[:, None, None]
            error_HV = v_THV[t] - torch.einsum("hk,hkv->hv", k_THK[t], state_HKV)
            update_HV = error_HV * beta_TH[t].float()[:, None]
            state_HKV = state_HKV + torch.einsum("hk,hv->hkv", k_THK[t], update_HV)
            outputs.append(torch.einsum("hk,hkv->hv", q_THK[t], state_HKV))
        return torch.stack(outputs).to(xv_THV.dtype)


class _MarkerLinear(Linear):
    """A Linear subclass, as quantized or LoRA projections are."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass


def _delta_net_config(*, sharded=False, enable_sp=False):
    config = model_registry("debugmodel")
    if sharded:
        set_qwen35_sharding_config(config, enable_sp=enable_sp, enable_ep=False)
    delta_net = next(layer.delta_net for layer in config.layers if layer.delta_net)
    return delta_net, config.dim


def _build(config) -> GatedDeltaNet:
    module = config.build()
    module.inner_gated_delta_net.kernel = _ReferenceGatedDeltaKernel()
    return module


def _init_weights(module: nn.Module, seed: int = 0) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            values = torch.randn(param.shape, generator=generator) * 0.1
            if name == "A_log":
                values = values.abs()
            param.copy_(values)


def _per_projection_forward(module: GatedDeltaNet, x_TD: torch.Tensor):
    """The six-projection, three-conv GatedDeltaNet math (main branch)."""
    num_tokens = x_TD.shape[0]

    def causal_conv(x_TC, conv):
        x_1CT = F.pad(x_TC.transpose(0, 1).unsqueeze(0), [conv.weight.shape[-1] - 1, 0])
        out_1CT = F.conv1d(x_1CT, conv.weight, None, groups=conv.weight.shape[0])
        return F.silu(out_1CT).squeeze(0).transpose(0, 1)

    xq_THK = causal_conv(F.linear(x_TD, module.in_proj_q.weight), module.conv_q)
    xk_THK = causal_conv(F.linear(x_TD, module.in_proj_k.weight), module.conv_k)
    xv_THV = causal_conv(F.linear(x_TD, module.in_proj_v.weight), module.conv_v)
    gate_THV = F.linear(x_TD, module.in_proj_z.weight).view(
        num_tokens, -1, module.value_head_dim
    )
    a_TH = F.linear(x_TD, module.in_proj_a.weight)
    b_TH = F.linear(x_TD, module.in_proj_b.weight)
    g_TH = -torch.exp(module.A_log.float()) * F.softplus(a_TH.float() + module.dt_bias)
    output_THV = module.inner_gated_delta_net.kernel(
        xq_THK.reshape(num_tokens, -1, module.key_head_dim),
        xk_THK.reshape(num_tokens, -1, module.key_head_dim),
        xv_THV.reshape(num_tokens, -1, module.value_head_dim),
        g_TH,
        torch.sigmoid(b_TH),
    )
    output_THV = module.norm(output_THV, gate_THV)
    return F.linear(output_THV.reshape(num_tokens, -1), module.out_proj.weight)


def _forward_and_grads(module, x_TD, forward):
    x_TD = x_TD.detach().clone().requires_grad_()
    module.zero_grad(set_to_none=True)
    output = forward(module, x_TD)
    output.square().sum().backward()
    grads = {name: param.grad for name, param in module.named_parameters()}
    return output.detach(), x_TD.grad, grads


class TestGatedDeltaNetFusedProjections(unittest.TestCase):
    def _reference(self, x_TD):
        config, _ = _delta_net_config()
        reference = _build(config)
        _init_weights(reference)
        return _forward_and_grads(reference, x_TD, _per_projection_forward)

    def test_fused_paths_match_per_projection_math(self):
        _, dim = _delta_net_config()
        x_TD = torch.randn(7, dim, generator=torch.Generator().manual_seed(1))
        expected_out, expected_x_grad, expected_grads = self._reference(x_TD)

        for layout in ("fused_qkv", "subclassed_q", "subclassed_z"):
            with self.subTest(layout=layout):
                config, _ = _delta_net_config()
                if layout == "subclassed_q":
                    config = dataclasses.replace(
                        config,
                        in_proj_q=convert_config_type(config.in_proj_q, _MarkerLinear),
                    )
                if layout == "subclassed_z":
                    config = dataclasses.replace(
                        config,
                        in_proj_z=convert_config_type(config.in_proj_z, _MarkerLinear),
                    )
                module = _build(config)
                _init_weights(module)
                self.assertEqual(module.fuse_qkv_projections, layout != "subclassed_q")
                out, x_grad, grads = _forward_and_grads(
                    module, x_TD, lambda module, x_TD: module(x_TD)
                )
                torch.testing.assert_close(out, expected_out)
                torch.testing.assert_close(x_grad, expected_x_grad)
                for name, grad in expected_grads.items():
                    torch.testing.assert_close(grads[name], grad, msg=name)

    def test_shared_storage_matches_and_keeps_the_state_dict(self):
        _, dim = _delta_net_config()
        x_TD = torch.randn(7, dim, generator=torch.Generator().manual_seed(1))
        expected_out, _, _ = self._reference(x_TD)

        config, _ = _delta_net_config()
        module = _build(config)
        keys = list(module.state_dict())
        module.share_input_storage()
        self.assertEqual(list(module.state_dict()), keys)
        # Loading after sharing (the generator's order) writes into the buffers.
        reference_state = _build(_delta_net_config()[0])
        _init_weights(reference_state)
        module.load_state_dict(reference_state.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(module(x_TD), expected_out)

        buffer_CD = module._shared_input_weight_CD
        q, k, v, z = (
            module.in_proj_q.weight.shape[0],
            module.in_proj_k.weight.shape[0],
            module.in_proj_v.weight.shape[0],
            module.in_proj_z.weight.shape[0],
        )
        new_z = torch.randn_like(module.in_proj_z.weight)
        new_conv_k = torch.randn_like(module.conv_k.weight)
        module.load_state_dict(
            {"in_proj_z.weight": new_z, "conv_k.weight": new_conv_k}, strict=False
        )
        torch.testing.assert_close(buffer_CD[q + k + v : q + k + v + z], new_z)
        conv_q = module.conv_q.weight.shape[0]
        torch.testing.assert_close(
            module._shared_conv_weight_C1W[conv_q : conv_q + new_conv_k.shape[0]],
            new_conv_k,
        )
        self.assertIsInstance(module.in_proj_z.weight, nn.Parameter)

    def test_shared_storage_rejects_subclassed_projections(self):
        config, _ = _delta_net_config()
        config = dataclasses.replace(
            config, in_proj_z=convert_config_type(config.in_proj_z, _MarkerLinear)
        )
        with self.assertRaisesRegex(ValueError, "plain Linear"):
            _build(config).share_input_storage()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA (Attention Gym kernels)")
class TestGatedDeltaNetFusedProjectionsCuda(unittest.TestCase):
    def test_shared_storage_matches_training_path_with_varlen_kernels(self):
        _, dim = _delta_net_config()
        positions = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int32)
        masks = create_varlen_metadata_for_document(positions.cuda())
        x_TD = torch.randn(7, dim, generator=torch.Generator().manual_seed(1))
        x_TD = x_TD.to("cuda", torch.bfloat16)
        outputs = {}
        for shared in (False, True):
            config, _ = _delta_net_config()
            module = config.build()
            _init_weights(module)
            module = module.to("cuda", torch.bfloat16)
            if shared:
                module.share_input_storage()
            with torch.no_grad():
                outputs[shared] = module(x_TD, masks)
        # bf16: the fused GEMMs may round a few elements differently.
        torch.testing.assert_close(outputs[True], outputs[False], rtol=2e-2, atol=2e-2)


class TestGatedDeltaNetFusedProjectionsTensorParallel(DTensorTestBase):
    """TP=2 with the real Qwen3.5 sharding configs, SPMD type checking, and gloo."""

    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_tp2_matches_unsharded_per_projection_math(self):
        _, dim = _delta_net_config()
        num_tokens = 6
        x_full_TD = torch.randn(
            num_tokens, dim, generator=torch.Generator().manual_seed(1)
        )
        reference = _build(_delta_net_config()[0])
        _init_weights(reference)
        expected_out, expected_x_grad, expected_grads = _forward_and_grads(
            reference, x_full_TD, _per_projection_forward
        )

        for enable_sp in (False, True):
            for layout in ("training", "one_gemm_shared"):
                with self.subTest(enable_sp=enable_sp, layout=layout):
                    self._check(
                        reference,
                        x_full_TD,
                        expected_out,
                        expected_x_grad,
                        expected_grads,
                        enable_sp=enable_sp,
                        layout=layout,
                    )

    def _check(
        self,
        reference,
        x_full_TD,
        expected_out,
        expected_x_grad,
        expected_grads,
        *,
        enable_sp,
        layout,
    ):
        config, _ = _delta_net_config(sharded=True, enable_sp=enable_sp)
        parallel = _build(config)
        parallel.load_state_dict(reference.state_dict())
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=enable_sp,
        )
        with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
            parallel_dims.build_mesh()
        parallel._parallelize(parallel_dims)
        if layout == "one_gemm_shared":
            parallel.share_input_storage()

        input_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        )
        x_local_TD = (
            x_full_TD.chunk(self.world_size)[self.rank] if enable_sp else x_full_TD
        ).clone()
        x_local_TD.requires_grad_()
        mesh = parallel_dims.spmd_dense_mesh()
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=None, dense_sp_enabled=enable_sp)
        inference = layout == "one_gemm_shared"
        with set_current_spmd_mesh(mesh), typecheck(local=False):
            spmd.assert_type(x_local_TD, input_layout)
            with torch.set_grad_enabled(not inference):
                out = parallel(x_local_TD, None)
            if not inference:
                out.square().sum().backward()

        expected_local_out = (
            expected_out.chunk(self.world_size)[self.rank]
            if enable_sp
            else expected_out
        )
        torch.testing.assert_close(out.detach(), expected_local_out)
        if inference:
            # Shared storage is for inference: no gradients to compare.
            return
        expected_local_x_grad = (
            expected_x_grad.chunk(self.world_size)[self.rank]
            if enable_sp
            else expected_x_grad
        )
        # TP reduces partial gradients in a different order than one device.
        tolerance = {"rtol": 1e-4, "atol": 1e-4}
        torch.testing.assert_close(x_local_TD.grad, expected_local_x_grad, **tolerance)
        # Each rank holds its heads' rows of every input projection and conv.
        for name in (
            "in_proj_q.weight",
            "in_proj_k.weight",
            "in_proj_v.weight",
            "in_proj_z.weight",
            "in_proj_a.weight",
            "in_proj_b.weight",
            "conv_q.weight",
            "conv_v.weight",
        ):
            local_grad = dict(parallel.named_parameters())[name].grad
            torch.testing.assert_close(
                local_grad,
                expected_grads[name].chunk(self.world_size)[self.rank],
                msg=name,
                **tolerance,
            )


if __name__ == "__main__":
    unittest.main()
