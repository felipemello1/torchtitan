# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend B: vLLM `AsyncLLM` under a Monarch executor.

vLLM owns the continuous-batching loop and TP; this backend has no hand-driven
engine loop. The cost is a Monarch executor so `AsyncLLM`'s `EngineCore` runs on
the Monarch-spawned workers. See `design.md` and `options_and_consequences.md
§9.3` for the tradeoff vs Backend A.
"""

from __future__ import annotations

from torchtitan.experiments.rl.actors.generators.base import VLLMGeneratorBase


class AsyncLLMGenerator(VLLMGeneratorBase):
    """`AsyncLLM` + Monarch-executor generator (Backend B)."""

    def __init__(self, config, **kwargs):
        # TODO(backend-b): build AsyncLLM with a Monarch executor (in-process
        # EngineCore to avoid forge's WorkerRegistry + cloudpickle-env hops) and
        # implement generate / pull_model_state_dict / close against it. Until
        # then select backend=llm_engine (the default).
        raise NotImplementedError(
            "AsyncLLM backend (B) is not implemented yet; set "
            "generator.backend=llm_engine. See design.md / options §9.3."
        )
