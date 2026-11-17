# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared process environment defaults for TorchTitan RL jobs."""

from __future__ import annotations

import os
import sys
import warnings


RL_ENV_DEFAULTS: dict[str, str] = {
    # Set both names because supported torch and vLLM versions may read different
    # allocator configuration variables. Expandable segments prevent reserved but
    # unallocated memory from blocking DeepEP buffers and CUDA graph pools.
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_DEBUG": "WARN",
}


def apply_env_defaults() -> None:
    """Apply TorchTitan RL defaults without replacing existing values."""
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and "torch" in sys.modules:
        warnings.warn(
            "The 'torch' module has already been imported. Setting "
            "PYTORCH_CUDA_ALLOC_CONF may not have an effect. For best results, "
            "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before "
            "importing torch.",
            stacklevel=2,
        )
    for key, value in RL_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


def apply_rollout_worker_env_defaults() -> None:
    """`bootstrap` setup callable for a rollout worker process: torch and numpy math run on 1 CPU
    thread, unless the host's env sets `OMP_NUM_THREADS`."""
    # Torch and numpy each start a pool of threads for their math: numpy at import (up to 64 threads),
    # torch at its first parallel op (one per CPU). If worker code runs such math on a generator's
    # host, those threads compete with the inference engine for the CPUs. The cost: that math runs on 1 core.
    # Setting it works only before torch or numpy loads. Monarch imports a bootstrap callable's
    # module before it calls it, so this lives here: this module imports neither.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
