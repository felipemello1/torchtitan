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
        bf16 backward.
        "upcast_fp32_matmul": copies both operands to fp32 every call, then an fp32 matmul.

        bf16 operands, 2048x5120 input, 248320x5120 weight (an LM head):

            mode                   output   fwd+bwd   mean logprob error
            default (bf16 head)    bf16     12 ms     8.4e-3
            bf16_matmul_fp32_out   fp32     12 ms     7.5e-5
            upcast_fp32_matmul     fp32     76 ms     1.5e-6

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
