# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``torchtitan.train`` with the two PyTorch changes fp32 weight gradients pair with.

Installs ``grouped_mm.install()`` (fp32-output grouped GEMM for MoE experts) and
``fsdp_reduce_scatter.install()`` (FSDP2 reduce-scatter without the copy-in), then runs the
regular trainer. All arguments pass through.

Example:

    torchrun --nproc_per_node=8 -m torchtitan.experiments.fp32_weight_grads.train \\
        --module qwen3 --config qwen3_30b_a3b --training.mixed_precision_grad float32
"""

import runpy

from torchtitan.experiments.fp32_weight_grads import fsdp_reduce_scatter, grouped_mm

if __name__ == "__main__":
    grouped_mm.install()
    fsdp_reduce_scatter.install()
    runpy.run_module("torchtitan.train", run_name="__main__", alter_sys=True)
