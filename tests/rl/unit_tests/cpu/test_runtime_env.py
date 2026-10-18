# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

from torchtitan.rl._env import apply_env_defaults, RL_ENV_DEFAULTS


def test_apply_env_defaults() -> None:
    with patch.dict(os.environ, {}, clear=True):
        apply_env_defaults()

        assert dict(os.environ) == RL_ENV_DEFAULTS


def test_apply_env_defaults_preserves_existing_values() -> None:
    with patch.dict(
        os.environ,
        {"NCCL_DEBUG": "INFO", "UNRELATED": "value"},
        clear=True,
    ):
        apply_env_defaults()

        assert os.environ["NCCL_DEBUG"] == "INFO"
        assert os.environ["UNRELATED"] == "value"


# 32 threads each malloc once while all are alive, then glibc's malloc_info prints one
# <heap> per arena.
_MALLOC_INFO_AFTER_32_THREADS = """
import ctypes, threading
from torchtitan.rl._env import apply_env_defaults

apply_env_defaults()
libc = ctypes.CDLL("libc.so.6")
barrier = threading.Barrier(32)

def malloc_once():
    libc.malloc(1024)
    barrier.wait()

threads = [threading.Thread(target=malloc_once) for _ in range(32)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
libc.malloc_info(0, ctypes.c_void_p.in_dll(libc, "stdout"))
libc.fflush(None)
"""


def test_apply_env_defaults_caps_malloc_arenas_of_this_process(monkeypatch) -> None:
    """glibc ignores MALLOC_ARENA_MAX set after exec: mallopt must cap 32 threads to 16 arenas."""
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)

    result = subprocess.run(
        [sys.executable, "-c", _MALLOC_INFO_AFTER_32_THREADS],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.count('<heap nr="') == 16
