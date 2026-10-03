# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Override: run the decoder-block RMSNorms with quack's fused CuTe-DSL kernels.

Activate with::

    --override torchtitan_recipes.overrides.quack_rmsnorm.quack_rmsnorm

Why
---
Recent PyTorch nightlies already route ``F.rms_norm`` on CUDA to a vendored
copy of quack's kernel, but only when the installed ``nvidia-cutlass-dsl`` is
one of an exact allow-list of versions (check with
``torch._native.cutedsl_utils._version_is_ok()``). Otherwise ``F.rms_norm``
falls back to ATen's layer-norm kernels, whose fused forward plus backward costs
~111 us at T=4096, D=7168 on GB300, vs ~68 us with quack. This override calls
the ``quack`` package directly, so the block norms get the fast kernel
whatever cutlass-dsl version is installed. When the native path is already
active the two run the same kernel and the override changes nothing.

Scope
-----
* Claims only ``*.attention_norm`` and ``*.ffn_norm`` (the two norms every
  decoder block runs per token, including DeepSeek-V3's MTP block). Smaller
  norms (MLA's ``q_norm`` / ``kv_norm``) see no gain and stay stock.
* Eager only. Under ``torch.compile``, Inductor fuses the stock norm with its
  neighbors, so the module keeps the stock path while compiling.
* Falls back to the stock path for inputs quack does not take (CPU, DTensor,
  unsupported dtypes, empty tensors) and while SPMD type checking is active,
  since only stock ops carry SPMD type rules.
* Parameters and the state dict are unchanged. Outputs keep ``x``'s dtype,
  like ``F.rms_norm``; numerics match the stock kernel within bf16 rounding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
from torch.distributed.tensor import DTensor

from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.observability.logging import warn_once

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from quack.rmsnorm import rmsnorm as _quack_rmsnorm

    _QUACK_IMPORT_ERROR: ImportError | None = None
else:
    try:
        from quack.rmsnorm import rmsnorm as _quack_rmsnorm

        _QUACK_IMPORT_ERROR = None
    except ImportError as e:
        _QUACK_IMPORT_ERROR = e

__all__ = ["QuackRMSNorm"]

_QUACK_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class QuackRMSNorm(RMSNorm):
    """``RMSNorm`` whose eager CUDA forward and backward run quack's kernels.

    Example:
        norm = QuackRMSNorm.Config(normalized_shape=7168, eps=1e-6).build().cuda()
        y = norm(x)  # same values and dtype as RMSNorm, quack's fused kernels
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._runs_quack(x):
            return super().forward(x)
        return _quack_rmsnorm(x, self.weight, eps=self.eps)

    def _runs_quack(self, x: torch.Tensor) -> bool:
        if _QUACK_IMPORT_ERROR is not None:
            return False
        if torch.compiler.is_compiling() or spmd.is_type_checking():
            return False
        weight = self.weight
        supported = (
            weight is not None
            and x.is_cuda
            and x.numel() > 0
            and x.dtype in _QUACK_DTYPES
            and weight.dtype in _QUACK_DTYPES
            and weight.device == x.device
            and not isinstance(x, DTensor)
            and not isinstance(weight, DTensor)
        )
        if not supported:
            warn_once(
                logger,
                "QuackRMSNorm: input unsupported by quack (needs a CUDA, non-DTensor, "
                "non-empty fp16/bf16/fp32 tensor and an affine weight); "
                "falling back to the stock RMSNorm.",
            )
        return supported


@override(
    target=RMSNorm.Config,
    exact=True,
    fqns=["*.attention_norm", "*.ffn_norm"],
    description="Decoder-block RMSNorms with quack's fused CuTe-DSL kernels (CUDA, eager).",
)
def quack_rmsnorm(cfg: RMSNorm.Config) -> QuackRMSNorm.Config:
    if _QUACK_IMPORT_ERROR is not None:
        raise ImportError(
            "QuackRMSNorm override was requested but `quack` is not installed; "
            "install quack-kernels to use torchtitan_recipes.overrides.quack_rmsnorm."
        ) from _QUACK_IMPORT_ERROR
    return derive(cfg, QuackRMSNorm.Config)
