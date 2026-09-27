# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``torch._grouped_mm(..., out_dtype=torch.float32)`` for bf16 inputs, via a CUDA extension.

``torch._grouped_mm`` rejects an fp32 output for bf16 inputs (its CUTLASS kernel hardcodes bf16).
``GroupedLinear`` with fp32 weight gradients calls it with ``out_dtype=torch.float32``, which the
proposed PyTorch change allows. Until then, ``install()`` builds that kernel change as an extension
(``grouped_mm_fp32.cu``, JIT, cached) and wraps ``torch._grouped_mm`` so exactly those calls use it.

Build requirements: a CUDA toolkit matching torch's CUDA (``CUDA_HOME``; defaults to the pip
``nvidia-cuda-nvcc`` install for CUDA 13) and a CUTLASS checkout at PyTorch's pin (``CUTLASS_DIR``).

Example:

    install()
    torch._grouped_mm(grad_RO.T, input_RI, offs=offsets_E, out_dtype=torch.float32)  # [E, O, I] fp32
"""

import getpass
import os
import pathlib
import sys
import tempfile

import torch

_SOURCE = pathlib.Path(__file__).resolve().parent / "grouped_mm_fp32.cu"


def install() -> None:
    """Route ``torch._grouped_mm(bf16, bf16, out_dtype=torch.float32)`` to the extension."""
    pytorch_grouped_mm = torch._grouped_mm

    def grouped_mm(mat_a, mat_b, offs=None, bias=None, out_dtype=None):
        if (
            out_dtype == torch.float32
            and mat_a.dtype == mat_b.dtype == torch.bfloat16
            and bias is None
        ):
            return torch.ops.grouped_mm_fp32.grouped_mm(mat_a, mat_b, offs)
        return pytorch_grouped_mm(
            mat_a, mat_b, offs=offs, bias=bias, out_dtype=out_dtype
        )

    load_extension()
    torch._grouped_mm = grouped_mm


def load_extension(verbose: bool = False) -> None:
    """Build for the current GPU (sm_90a / sm_100a / sm_103a) if needed, then load."""
    import nvidia

    # cpp_extension reads CUDA_HOME when first imported, and needs pip's ninja on PATH.
    cuda_home = pathlib.Path(
        os.environ.setdefault(
            "CUDA_HOME", str(pathlib.Path(nvidia.__path__[0]) / "cu13")
        )
    )
    os.environ["PATH"] = f"{pathlib.Path(sys.executable).parent}:{os.environ['PATH']}"
    from torch.utils.cpp_extension import load

    # cpp_extension links -lcudart, but pip's CUDA only ships libcudart.so.13; keep the symlink
    # machine-local, since the repo may live on a filesystem shared across machines.
    libdir = (
        pathlib.Path(tempfile.gettempdir())
        / f"grouped_mm_fp32_cudart_{getpass.getuser()}"
    )
    libdir.mkdir(exist_ok=True)
    cudart = libdir / "libcudart.so"
    cudart_target = cuda_home / "lib" / "libcudart.so.13"
    if not cudart.is_symlink() or cudart.readlink() != cudart_target:
        cudart.unlink(missing_ok=True)
        cudart.symlink_to(cudart_target)

    cutlass = pathlib.Path(
        os.environ.get("CUTLASS_DIR", f"/tmp/{getpass.getuser()}/cutlass")
    )
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    load(
        name=f"grouped_mm_fp32_sm{arch}",
        sources=[str(_SOURCE)],
        extra_include_paths=[
            str(cutlass / "include"),
            str(cutlass / "tools/util/include"),
        ],
        extra_ldflags=[f"-L{libdir}"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DNDEBUG",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ],
        is_python_module=False,
        verbose=verbose,
    )
