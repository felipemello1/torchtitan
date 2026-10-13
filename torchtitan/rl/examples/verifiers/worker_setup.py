# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch-free on purpose: env-server workers import this module to unpickle
`setup_env_server_worker`."""

import sys

from verifiers.v1.utils.logging import setup_logging


def setup_env_server_worker(level: str) -> None:
    """Block torch, then route Verifiers logs to stderr at `level`.

    Verifiers calls it in the env server and in each pool worker, before any renderer exists:
        env server   torch already imported through torchtitan -> left as is
        pool worker  `import torch` raises ImportError, so the renderer's
                     `transformers.AutoTokenizer` loads without torch
                     (Qwen3.5 renderer: 776 -> 341 MB anonymous per worker)
    """
    if "torch" not in sys.modules:
        sys.modules["torch"] = None
    setup_logging(level)
