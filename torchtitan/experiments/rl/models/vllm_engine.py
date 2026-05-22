# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM engine construction helpers for TorchTitan RL."""

from __future__ import annotations

import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import CompileConfig
from torchtitan.experiments.rl.models.vllm_registry import (
    registry_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
)
from torchtitan.experiments.rl.sampling import TRAINING_VLLM_LOGPROBS_MODE
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.utils import has_cuda_capability
from vllm import EngineArgs
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def build_torchtitan_vllm_engine_args(
    *,
    config,
    model_spec: ModelSpec,
    model_path: str,
    compile_config: CompileConfig,
    checkpoint_config: CheckpointManager.Config,
    max_num_seqs: int,
) -> EngineArgs:
    """Register TorchTitan's vLLM wrappers and assemble ``EngineArgs``.

    Sets ``VLLM_ATTENTION_BACKEND`` / ``VLLM_USE_V2_MODEL_RUNNER`` and calls
    :func:`registry_to_vllm` before returning, so the engine is built
    against TorchTitan's parallelism layout and config parser instead of
    vLLM's defaults.
    """
    os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    registry_to_vllm(
        model_spec,
        parallelism=config.parallelism,
        compile_config=compile_config,
        checkpoint_config=checkpoint_config,
    )
    engine_kwargs = dict(
        # ``model`` is the path to the HF checkpoint directory. The
        # config is sourced from torchtitan's ModelSpec via
        # ``config_format=TORCHTITAN_CONFIG_FORMAT`` (no config.json
        # read), but vLLM still uses this path to locate the
        # tokenizer assets and the safetensors weight shards.
        model=model_path,
        trust_remote_code=True,
        config_format=TORCHTITAN_CONFIG_FORMAT,
        dtype=config.model_dtype,
        tensor_parallel_size=config.parallelism.tensor_parallel_degree,
        # Monarch already spawned TP workers via proc mesh. ``external_launcher``
        # tells vLLM to run one worker per process (no subprocess spawning).
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=config.gpu_memory_limit,
        enforce_eager=not config.cudagraph.enable,
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum.CUSTOM,
        ),
        # Enables RequestOutput.metrics, so generator metrics can be returned.
        disable_log_stats=False,
        # Return logprobs after sampling-temperature processing so trainer and
        # behavior logprobs are computed against the same distribution.
        logprobs_mode=TRAINING_VLLM_LOGPROBS_MODE,
    )
    engine_kwargs["max_num_seqs"] = max_num_seqs
    # FA2 requires block_size to be a multiple of 256.
    if not has_cuda_capability(9, 0):
        engine_kwargs["block_size"] = 256
    vllm_compilation_config = config.cudagraph.get_vllm_compilation_config(
        max_num_seqs=max_num_seqs,
    )
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if config.debug.seed is not None:
        engine_kwargs["seed"] = config.debug.seed
    return EngineArgs(**engine_kwargs)
