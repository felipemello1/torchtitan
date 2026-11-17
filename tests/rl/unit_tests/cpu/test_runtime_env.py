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


def test_the_env_module_does_not_import_torch() -> None:
    """`apply_rollout_worker_env_defaults` sets `OMP_NUM_THREADS` before torch loads only if its
    module doesn't import torch."""
    imports_torch = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torchtitan.rl._env; print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert imports_torch == "False"
