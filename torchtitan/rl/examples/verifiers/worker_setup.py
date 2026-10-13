# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Env-server worker setup. Workers import this module to unpickle it: it must not import torch."""

import sys

from verifiers.v1.utils.logging import setup_logging


def setup_env_server_worker(level: str) -> None:
    """Block torch, then route Verifiers logs to stderr at `level`.

    Verifiers runs this first in the env server and in each pool worker it spawns.
    The renderer's `transformers.AutoTokenizer` imports torch: +411 MB anonymous memory
    per worker. With torch blocked, transformers loads the tokenizer without it.
    """
    # The env server process already imported torch through torchtitan; keep it there.
    if "torch" not in sys.modules:
        sys.modules["torch"] = None
    setup_logging(level)
