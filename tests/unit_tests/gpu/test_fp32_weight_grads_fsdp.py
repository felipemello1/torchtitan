# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.fsdp import enable_fp32_weight_grads
from torchtitan.models.common.linear import Linear


class TestFp32WeightGradsFSDP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 1

    def _weight_grad_errors(self, fp32_weight_grads: bool) -> dict[str, float]:
        """FSDP's final weight gradient vs the exact product of the captured bf16 tensors."""
        torch.manual_seed(0)
        model = nn.Sequential(
            Linear.Config(in_features=256, out_features=512).build(),
            nn.LayerNorm(512),
            Linear.Config(in_features=512, out_features=128, num_linears=2).build(),
        ).cuda()
        mesh = init_device_mesh("cuda", (self.world_size,))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        fully_shard(model[0], mesh=mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy)
        if fp32_weight_grads:
            enable_fp32_weight_grads(model)

        captured = {}

        def capture(name):
            def hook(module, inputs, output):
                x = inputs[0].detach()
                output.register_hook(lambda g: captured.__setitem__(name, (x, g)))

            return hook

        model[0].register_forward_hook(capture("0"))
        model[2].register_forward_hook(capture("2"))
        x = torch.randn(1024, 256, device="cuda")
        model(x).float().sum().backward()

        errors = {}
        for name in ("0", "2"):
            grad = model[int(name)].weight.grad
            grad = grad.to_local() if isinstance(grad, DTensor) else grad
            layer_input, grad_output = captured[name]
            exact = (
                grad_output.reshape(1024, -1).double().T @ layer_input.double()
            ).reshape(grad.shape)
            errors[name] = ((grad.double() - exact).norm() / exact.norm()).item()
        # The LayerNorm shares the root FSDP module with the stacked Linear, so
        # FSDP reduce-scatters both gradients in one dtype.
        assert model[1].weight.grad is not None
        return errors

    @with_comms
    def test_fsdp_reduces_exact_fp32_linear_weight_grads(self):
        bf16_errors = self._weight_grad_errors(fp32_weight_grads=False)
        fp32_errors = self._weight_grad_errors(fp32_weight_grads=True)
        for name in ("0", "2"):
            self.assertGreater(bf16_errors[name], 1e-4)
            self.assertLess(fp32_errors[name], 1e-5)
