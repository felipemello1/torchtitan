# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.qwen3_5.model import OffsetRMSNorm
from torchtitan.models.qwen3_5.sharding import _qk_norm_sharding


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestOffsetRMSNormCompile(unittest.TestCase):
    def setUp(self):
        LocalCompileConfig(regions=["offset_rmsnorm"]).apply_local_compile()

    def tearDown(self):
        LocalCompileConfig(regions=[]).apply_local_compile()
        torch._dynamo.reset()

    def test_forward_and_backward_emit_triton(self):
        from torch._inductor.utils import run_fw_bw_and_get_code

        module = (
            OffsetRMSNorm.Config(dim=128, eps=1e-6).build().cuda().to(torch.bfloat16)
        )
        with torch.no_grad():
            module.weight.normal_()
        x = torch.randn(
            2048,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        _, codes = run_fw_bw_and_get_code(lambda: module(x))

        self.assertGreaterEqual(sum("triton" in code for code in codes), 2)
        self.assertTrue(any("rsqrt" in code for code in codes))

    def test_forward_and_backward_match_eager(self):
        module = (
            OffsetRMSNorm.Config(dim=128, eps=1e-6).build().cuda().to(torch.bfloat16)
        )
        with torch.no_grad():
            module.weight.normal_()
        x = torch.randn(
            8,
            6,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        grad_output = torch.randn_like(x)

        output = module(x)
        grad_x, grad_weight = torch.autograd.grad(
            output,
            (x, module.weight),
            grad_output,
        )

        reference_x = x.detach().clone().requires_grad_()
        reference_weight = module.weight.detach().clone().requires_grad_()
        reference_x_fp32 = reference_x.float()
        variance = reference_x_fp32.square().mean(-1, keepdim=True)
        reference_output = (
            (1.0 + reference_weight.float())
            * reference_x_fp32
            * torch.rsqrt(variance + module.eps)
        ).to(reference_x.dtype)
        reference_grad_x, reference_grad_weight = torch.autograd.grad(
            reference_output,
            (reference_x, reference_weight),
            grad_output,
        )

        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(grad_x, reference_grad_x)
        torch.testing.assert_close(grad_weight, reference_grad_weight)

    def test_layer_and_qk_norms_fit_recompile_limit(self):
        # Qwen3.5-35B-A3B call sites: [T, D] layer norms, the q norm on a strided
        # chunk of wq's output and the k norm, for several T, then no_grad.
        dim, n_heads, n_kv_heads, head_dim = 2048, 16, 2, 256
        layer_norm = OffsetRMSNorm.Config(dim=dim).build().cuda().to(torch.bfloat16)
        qk_norm = OffsetRMSNorm.Config(dim=head_dim).build().cuda().to(torch.bfloat16)

        def run(num_tokens: int, requires_grad: bool) -> None:
            def rand(*shape: int) -> torch.Tensor:
                return torch.randn(
                    *shape,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=requires_grad,
                )

            q, _ = rand(num_tokens, n_heads, 2 * head_dim).chunk(2, dim=-1)
            outputs = [
                layer_norm(rand(num_tokens, dim)),
                qk_norm(q),
                qk_norm(rand(num_tokens, n_kv_heads, head_dim)),
            ]
            if requires_grad:
                sum(output.float().sum() for output in outputs).backward()

        # One graph per call-site layout and grad mode: 3 x 2, the expected count
        # (the default limit is 8).
        with torch._dynamo.config.patch(recompile_limit=6):
            for num_tokens in (16384, 8192, 2048):
                run(num_tokens, requires_grad=True)
            with torch.no_grad():
                for num_tokens in (16384, 4096, 2048):
                    run(num_tokens, requires_grad=False)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestOffsetRMSNormTensorParallel(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_compiled_qk_norm(self):
        LocalCompileConfig(regions=["offset_rmsnorm"]).apply_local_compile()
        device = self.device_type
        config = OffsetRMSNorm.Config(
            dim=128,
            eps=1e-6,
            param_init={"weight": torch.nn.init.zeros_},
            sharding_config=_qk_norm_sharding(),
        )
        module = config.build().to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            module.weight.zero_()

        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        with patch("torchtitan.distributed.parallelism_context.device_type", device):
            parallelism_context.build_mesh()
        module._parallelize(parallelism_context)

        x_full = torch.randn(16, 4, 128, device=device, dtype=torch.bfloat16)
        x_local = x_full.chunk(self.world_size, 1)[self.rank].contiguous()
        x_local.requires_grad_()
        mesh = parallelism_context.spmd_dense_mesh()
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=None, dense_sp_enabled=False)
        with set_current_spmd_mesh(mesh):
            output = module(x_local)
            output.sum().backward()

        self.assertEqual(output.shape, x_local.shape)
        LocalCompileConfig(regions=[]).apply_local_compile()
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
