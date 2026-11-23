# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared process environment defaults for TorchTitan RL jobs."""

from __future__ import annotations

import os
import socket
import stat
import sys
import warnings


RL_ENV_DEFAULTS: dict[str, str] = {
    # Expandable segments prevent reserved but unallocated memory from blocking
    # DeepEP buffers and CUDA graph pools.
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "NCCL_DEBUG": "WARN",
}

# Set by the SLURM and PBS launchers: a directory on each node's local disk for JIT caches.
NODE_CACHE_ROOT_ENV = "TORCHTITAN_RL_NODE_CACHE_ROOT"
_NODE_CACHE_SUBDIRS = {
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "VLLM_CACHE_ROOT": "vllm",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
}


def set_node_cache_env() -> None:
    """Point the JIT caches at a root on each node's local disk, keeping values the user set.

    Triton, vLLM and FlashInfer default to the home directory, often a shared network filesystem,
    where nodes compiling into one cache can hang or crash. TorchInductor defaults to a shared
    /tmp path that another user can create first.

    Example (uid 1000, TRITON_CACHE_DIR set in the submit shell):
        TORCHTITAN_RL_NODE_CACHE_ROOT    /tmp/torchtitan-1000
        TRITON_CACHE_DIR                 unchanged
        TORCHINDUCTOR_CACHE_DIR          /tmp/torchtitan-1000/inductor, also over torch's own default
        VLLM_CACHE_ROOT                  /tmp/torchtitan-1000/vllm
    """
    from torch._inductor.runtime.cache_dir_utils import default_cache_dir

    # Importing Inductor writes torch's shared /tmp default into the environment; treat it as unset.
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR") == default_cache_dir():
        del os.environ["TORCHINDUCTOR_CACHE_DIR"]

    root = os.environ.setdefault(NODE_CACHE_ROOT_ENV, f"/tmp/torchtitan-{os.getuid()}")
    for name, subdir in _NODE_CACHE_SUBDIRS.items():
        os.environ.setdefault(name, os.path.join(root, subdir))


def apply_env_defaults() -> None:
    """Apply TorchTitan RL defaults without replacing existing values.

    If a launcher set `TORCHTITAN_RL_NODE_CACHE_ROOT`, also make sure only this user can write it:
        missing                                                 -> created, mode 0700
        a directory this user owns that no one else can write   -> used as is
        another user's, a symlink, or group/other-writable      -> RuntimeError
    """
    allocator_conf = os.environ.get("PYTORCH_ALLOC_CONF")
    legacy_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if legacy_conf is not None and legacy_conf != allocator_conf:
        raise ValueError(
            "Use PYTORCH_ALLOC_CONF and unset PYTORCH_CUDA_ALLOC_CONF; "
            "the legacy variable is only supported as an identical compatibility copy."
        )
    if allocator_conf is None and "torch" in sys.modules:
        warnings.warn(
            "The 'torch' module has already been imported. Setting "
            "the default PYTORCH_ALLOC_CONF may not have an effect. "
            "Set PYTORCH_ALLOC_CONF before importing torch.",
            stacklevel=2,
        )
    for key, value in RL_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    # TODO: remove this mirror after updating pinned vLLM and its legacy readers.
    # 9f03b510c restores max_split_size_mb from the legacy name; fixed upstream:
    # https://github.com/vllm-project/vllm/pull/58430
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        os.environ["PYTORCH_ALLOC_CONF"],
    )

    # The root sits in shared /tmp: another user could create it first and plant files the job loads.
    node_cache_root = os.environ.get(NODE_CACHE_ROOT_ENV)
    if node_cache_root is not None:
        try:
            os.mkdir(node_cache_root, mode=0o700)
        except FileExistsError:
            info = os.lstat(node_cache_root)
            group_or_others_can_write = info.st_mode & 0o022
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.geteuid()
                or group_or_others_can_write
            ):
                raise RuntimeError(
                    f"{node_cache_root} on {socket.gethostname()} must be a directory that only "
                    f"you can write (owner uid {info.st_uid}, mode {oct(info.st_mode & 0o777)}). "
                    f"Remove it, or set {NODE_CACHE_ROOT_ENV} to another directory."
                ) from None
