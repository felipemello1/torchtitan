# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config converter for an fp32-output lm_head."""

from dataclasses import dataclass
from typing import Literal

from torchtitan.models.common.linear import Linear

from .converter import ModelConfigConverter

__all__ = ["LMHeadFp32OutputConverter"]


class LMHeadFp32OutputConverter(ModelConfigConverter):
    """Set an fp32-output ``matmul_mode`` on the decoder's ``lm_head``.

    Only the lm_head changes. The same model config backs the trainer and the vLLM
    generator, so both compute fp32 logits with the same op.
    """

    _TARGET = "lm_head"

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        matmul_mode: Literal[
            "bf16_matmul_fp32_out", "upcast_fp32_matmul"
        ] = "bf16_matmul_fp32_out"
        """How the lm_head matmul treats its operands.
        "bf16_matmul_fp32_out": bf16 operands as given, fp32 accumulation and output;
        backward within ~2% of an fp32 backward's error.
        "upcast_fp32_matmul": copies both operands to fp32 every call, then an fp32 matmul.

        Qwen3.5-27B LM head, 2048 real tokens (2048x5120 input, 248320x5120 weight), bf16
        operands, fwd + cross-entropy + bwd; errors vs fp64:

            mode                   output   time    mean logprob error   grad_input error
            default (bf16 head)    bf16     12 ms   1.2e-2               1.1e-2
            bf16_matmul_fp32_out   fp32     20 ms   6.1e-6               1.7e-3
            upcast_fp32_matmul     fp32     77 ms   1.6e-6               1.7e-3

        Batch-invariant mode always upcasts.
        """

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config):
        found = False
        for fqn, linear_config, _, _ in model_config.traverse(Linear.Config):
            if fqn.rsplit(".", 1)[-1] == self._TARGET:
                linear_config.matmul_mode = self.config.matmul_mode
                found = True
        if not found:
            raise ValueError(
                f"LMHeadFp32OutputConverter found no Linear named {self._TARGET!r} in "
                "the model config. The torchtitan decoder names its output projection "
                f"{self._TARGET!r} (see torchtitan/models/common/decoder.py)."
            )
        return model_config
