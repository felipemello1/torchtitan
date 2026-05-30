# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchtitan.experiments.rl.actors.generators.types import (
    GenerateFn,
    GeneratorBackend,
    SamplingConfig,
    VLLMCudagraphConfig,
)


def build_generator(backend: GeneratorBackend):
    """Return the generator actor class the controller spawns for `backend`.

    Imports the backend lazily so loading the shared types (e.g. in
    `config_registry`) does not import vLLM / the engine.
    """
    from torchtitan.experiments.rl.actors.generators.async_llm.generator import (
        AsyncLLMGenerator,
    )
    from torchtitan.experiments.rl.actors.generators.llm_engine.generator import (
        LLMEngineGenerator,
    )

    return {
        GeneratorBackend.LLM_ENGINE: LLMEngineGenerator,
        GeneratorBackend.ASYNC_LLM: AsyncLLMGenerator,
    }[backend]


__all__ = [
    "build_generator",
    "GenerateFn",
    "GeneratorBackend",
    "SamplingConfig",
    "VLLMCudagraphConfig",
]
