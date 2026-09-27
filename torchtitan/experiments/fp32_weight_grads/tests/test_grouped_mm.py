# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp32-output grouped GEMM (``grouped_mm.install()``) vs fp64, on SM90/SM100 GPUs.

Builds the extension on first run (~1.5 min); needs nvcc and CUTLASS (see grouped_mm.py).
"""

import getpass
import os
import pathlib

import pytest
import torch

from torchtitan.experiments.fp32_weight_grads import grouped_mm

_CUTLASS_DIR = pathlib.Path(
    os.environ.get("CUTLASS_DIR", f"/tmp/{getpass.getuser()}/cutlass")
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (9, 10)
    or not (_CUTLASS_DIR / "include").exists(),
    reason="needs an SM90/SM100 GPU and a CUTLASS checkout (CUTLASS_DIR)",
)


# out_features <= 128 takes the kernel's small-tile path, 512 the large one
@pytest.mark.parametrize("out_features", [512, 128])
def test_fp32_output_is_exact_up_to_fp32_accumulation(out_features):
    grouped_mm.install()
    torch.manual_seed(0)
    in_features = 256
    # 4 experts over 1024 routed rows, one of them empty
    offsets_E = torch.tensor([256, 256, 640, 1024], device="cuda", dtype=torch.int32)
    starts = [0, *offsets_E.tolist()[:-1]]
    grad_output_RO = torch.randn(1024, out_features, device="cuda").bfloat16()
    input_RI = torch.randn(1024, in_features, device="cuda").bfloat16()
    weight_EOI = torch.randn(4, out_features, in_features, device="cuda").bfloat16()

    # weight gradient layout (2d x 2d -> [E, O, I]) and forward layout (2d x 3d -> [R, O])
    cases = {
        "weight grad": (
            grad_output_RO.T,
            input_RI,
            torch.stack(
                [
                    grad_output_RO[start:end].double().T @ input_RI[start:end].double()
                    for start, end in zip(starts, offsets_E.tolist())
                ]
            ),
        ),
        "forward": (
            input_RI,
            weight_EOI.transpose(-2, -1),
            torch.cat(
                [
                    input_RI[start:end].double() @ weight_EOI[expert].double().T
                    for expert, (start, end) in enumerate(
                        zip(starts, offsets_E.tolist())
                    )
                ]
            ),
        ),
    }
    for name, (mat_a, mat_b, exact) in cases.items():
        fp32_out = torch._grouped_mm(
            mat_a, mat_b, offs=offsets_E, out_dtype=torch.float32
        )
        bf16_out = torch._grouped_mm(mat_a, mat_b, offs=offsets_E)
        fp32_error = ((fp32_out.double() - exact).norm() / exact.norm()).item()
        bf16_error = ((bf16_out.double() - exact).norm() / exact.norm()).item()
        assert fp32_out.dtype == torch.float32, name
        assert fp32_error < 1e-5 < 1e-4 < bf16_error, (name, fp32_error, bf16_error)
