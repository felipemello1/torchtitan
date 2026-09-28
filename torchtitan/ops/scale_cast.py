# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``(x * scale).to(dtype)`` in one pass, for the fp16 row-scaled backward of ``Fp32OutputLinear``.

``torch.mul(x, scale, out=...)`` with mixed dtypes runs TensorIterator's dynamic-casting kernel
(~1.3 TB/s on H100); the Triton pass runs at the device's copy bandwidth (~2.2 TB/s).
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # CPU-only installs (e.g. torch's CPU wheels); scale_cast uses torch.mul.
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit  # pyrefly: ignore[unbound-name]
    def _scale_cast_kernel(
        x_ptr,
        scale_ptr,
        out_ptr,
        num_cols,
        stride_row,
        ROW_SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """out[row] = x[row] * scale, with one scale per row if ROW_SCALE, in out's dtype."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols
        if ROW_SCALE:
            scale = tl.load(scale_ptr + row)
        else:
            scale = tl.load(scale_ptr)
        x = tl.load(x_ptr + row * stride_row + cols, mask=mask).to(tl.float32)
        tl.store(
            out_ptr + row * num_cols + cols,
            (x * scale).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def _can_use_triton(tensor: torch.Tensor) -> bool:
    """An eager, plain CUDA tensor: not a DTensor, fake or functional tensor, nor traced."""
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import FunctionalTensor
    from torch.distributed.tensor import DTensor
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return (
        _HAS_TRITON
        and tensor.is_cuda
        and not isinstance(tensor, (DTensor, FakeTensor, FunctionalTensor))
        and not torch.compiler.is_compiling()
        and get_proxy_mode() is None
    )


def scale_cast(
    x_RC: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """``(x * scale).to(dtype)``, where ``scale`` is an fp32 scalar or one fp32 value per row.

    Example:
        weight [124160, 5120] bf16, scale 2^16 (0-dim) -> fp16 [124160, 5120]: 1.2 ms on H100
        (``torch.mul(..., out=fp16)``: 2.0 ms)
    """
    num_rows, num_cols = x_RC.shape
    row_scale = scale.numel() > 1
    if not _can_use_triton(x_RC):
        return torch.mul(
            x_RC,
            scale.reshape(num_rows, 1) if row_scale else scale,
            out=torch.empty(num_rows, num_cols, device=x_RC.device, dtype=dtype),
        )
    if x_RC.stride(1) != 1:
        x_RC = x_RC.contiguous()
    out_RC = torch.empty(num_rows, num_cols, device=x_RC.device, dtype=dtype)
    block = 4096
    torch.library.wrap_triton(_scale_cast_kernel)[
        (num_rows, triton.cdiv(num_cols, block))
    ](
        x_RC,
        scale.reshape(-1).contiguous(),
        out_RC,
        num_cols,
        x_RC.stride(0),
        ROW_SCALE=row_scale,
        BLOCK=block,
        num_warps=8,
    )
    return out_RC
