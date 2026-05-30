# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared base for the vLLM generator backends.

`VLLMGeneratorBase` holds everything that does not depend on which vLLM engine
runs: config, the engine-agnostic init (model registration, determinism), the
structured-logger step sync, and the metrics buffer + drain. Each backend
subclass builds its own engine and defines `generate` / `pull_model_state_dict`
/ `close`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import torch
from monarch.actor import Actor, current_rank, endpoint
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import (
    CompileConfig,
    Configurable,
    DebugConfig,
    ParallelismConfig,
)
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.actors.generators.types import (
    GeneratorBackend,
    SamplingConfig,
    VLLMCudagraphConfig,
)
from torchtitan.experiments.rl.models.vllm_registry import registry_to_vllm
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import init_logger
from vllm.outputs import RequestOutput

logger = logging.getLogger(__name__)


def _prepare_generation_request_metrics(
    output: RequestOutput, *, prefix: str
) -> list[m.Metric]:
    """Prepare vLLM metrics from a RequestOutput.

    For `[num_prompts]` submitted prompts, vLLM returns `[num_prompts]`
    per parent `RequestOutput`s (one per `add_request` call), each carrying
    a single `RequestStateStats` on `.metrics`.

    Caveat under `SamplingParams.n > 1`: vLLM stores one `RequestStateStats`
    per child request; the parent output exposes the **last-finishing**
    child's timeline. `arrival_time` is shared across siblings, but
    [`queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts`,
    `num_generation_tokens`] describe one specific child — not an aggregate,
    not the first sibling's. The other `n-1` siblings' stats are dropped by
    vLLM at ``output_processor._finish_request``.
    """

    # TODO: Per-request fields here come from RequestOutput.metrics
    # (RequestStateStats). Engine-aggregate stats, such as KV-cache usage,
    # prefix-cache hit rate, preemptions, and batch occupancy, live in
    # SchedulerStats / IterationStats and require registering a
    # vllm.v1.metrics.loggers.StatLoggerBase via
    # LLMEngine.from_engine_args(..., stat_loggers=[...]).

    metric_values: dict[str, float] = {}
    if output.num_cached_tokens is not None:
        metric_values[f"{prefix}/num_cached_tokens"] = output.num_cached_tokens

    stats = output.metrics
    if stats is not None:
        metric_values[f"{prefix}/queue_time_ms"] = (
            stats.scheduled_ts - stats.queued_ts
        ) * 1000

        if stats.num_generation_tokens > 0:
            metric_values[f"{prefix}/time_to_first_token_ms"] = (
                stats.first_token_latency * 1000
            )
            metric_values[f"{prefix}/prefill_time_ms"] = (
                stats.first_token_ts - stats.scheduled_ts
            ) * 1000

        if stats.num_generation_tokens > 1:
            first_to_last_token_ms = (stats.last_token_ts - stats.first_token_ts) * 1000
            metric_values[f"{prefix}/decode_time_ms"] = first_to_last_token_ms
            metric_values[
                f"{prefix}/inter_token_latency_ms"
            ] = first_to_last_token_ms / (stats.num_generation_tokens - 1)

    # Emit each value with both Mean and Max aggregators.
    return [
        metric
        for key, value in metric_values.items()
        for metric in (m.Metric(key, m.Mean(value)), m.Metric(key, m.Max(value)))
    ]


class VLLMGeneratorBase(Actor, Configurable):
    """Base for vLLM generator actors.

    Subclasses build the engine and define `generate(prompt_token_ids, *,
    request_id, sampling_config, metrics_prefix) -> Completion`,
    `pull_model_state_dict(version)`, and `close()`. This base owns the shared
    config, the engine-agnostic init, `sync_log_step`, and the per-request
    metrics buffer drained by `pop_metrics`.

    Args:
        config: Generator-specific configuration.
        model_spec: TorchTitan model specification.
        model_path: Path to the HF model checkpoint.
        compile_config: Per-layer torch.compile config shared with the trainer
            so both sides compile identically.
        max_num_seqs: vLLM batch dim (max concurrent sequences), sized by the
            controller from per-step generation capacity.
        output_dir: Structured-logger output directory.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration."""

        backend: GeneratorBackend = GeneratorBackend.LLM_ENGINE
        """Which vLLM integration to spawn (both share this Config)."""

        num_generators: int = 1
        """How many independent generator engines to spawn. Each owns a TP group
        and its own KV/prefix cache; the controller routes a group to one by
        `hash(group_id) % num_generators` for prefix-cache affinity."""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        """Default sampling parameters for generation."""

        model_dtype: str = "bfloat16"
        """Data type for model weights, passed directly to vLLM (auto, float16, bfloat16, float32)."""

        gpu_memory_limit: float = 0.9
        """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

        cudagraph: VLLMCudagraphConfig = field(default_factory=VLLMCudagraphConfig)
        """CUDA graph capture settings for the vLLM engine."""

        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        """Controls whether the vLLM wrapper loads initial HF weights.
        In the RL loop this should stay disabled (default ``enable=False``)
        because weights arrive from TorchStore. For standalone inference,
        set ``enable=True`` and ``initial_load_in_hf=True``."""

        debug: DebugConfig = field(default_factory=DebugConfig)
        """Debug and determinism settings."""

        def __post_init__(self):
            # VLLMGenerator only supports TP. vLLM handles its own parallelism;
            # we only apply TP via the core parallelize function.
            p = self.parallelism
            if p.data_parallel_replicate_degree != 1:
                raise ValueError(
                    f"Generator does not support data parallel replication, "
                    f"got dp_replicate={p.data_parallel_replicate_degree}"
                )
            if p.pipeline_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support pipeline parallelism, "
                    f"got pp={p.pipeline_parallel_degree}"
                )
            if p.context_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support context parallelism, "
                    f"got cp={p.context_parallel_degree}"
                )
            if p.expert_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support expert parallelism, "
                    f"got ep={p.expert_parallel_degree}"
                )
            if p.enable_sequence_parallel:
                raise ValueError(
                    "Generator does not support sequence parallelism: "
                    "spmd_types erasure mode requires sequence length to be "
                    "evenly divisible by TP, which doesn't hold for inference "
                    "(uneven batches). Set enable_sequence_parallel=False."
                )
            if not p.disable_loss_parallel:
                raise ValueError(
                    "Generator requires disable_loss_parallel=True, "
                    f"got disable_loss_parallel={p.disable_loss_parallel}"
                )

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        model_path: str,
        compile_config: CompileConfig,
        max_num_seqs: int,
        output_dir: str,
    ):
        init_logger()
        sl.init_structured_logger(
            source="rl_generator",
            output_dir=output_dir,
            rank=current_rank().rank,
            enable=config.debug.enable_structured_logging,
        )
        sl.log_trace_instant("structured_logger_started")

        self.config = config
        self.model_spec = model_spec
        self.model_path = model_path
        # max_num_seqs controls vLLM's maximum batch dimension: it sets the upper
        # bound for concurrent sequences, determines KV-cache block allocation
        # (and therefore GPU memory usage), and bounds CUDA graph capture sizes.
        self._max_num_seqs = max_num_seqs

        # Register the TorchTitan model + parser with vLLM, then set the engine
        # env vars. The engine itself is built by the backend subclass.
        registry_to_vllm(
            model_spec,
            parallelism=config.parallelism,
            compile_config=compile_config,
            checkpoint_config=config.checkpoint,
        )
        os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

        set_batch_invariance(config.debug.batch_invariant)
        self._set_determinism(config.debug)

        self.policy_version = 0
        self._metrics: list[m.Metric] = []

    @staticmethod
    def _set_determinism(debug: DebugConfig) -> None:
        """Apply deterministic flags for the generator.

        The generator doesn't use torchtitan's ParallelDims, so we apply
        the deterministic flags directly instead of using set_determinism().
        """
        if debug.deterministic:
            torch.use_deterministic_algorithms(
                True, warn_only=debug.deterministic_warn_only
            )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        if debug.seed is not None:
            torch.manual_seed(debug.seed)

    @endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        """Sync the structured-logger step counter from the controller."""
        sl.set_step(step, relative_step=relative_step)

    @endpoint
    async def pop_metrics(self) -> list[m.Metric]:
        """Return and clear the per-request metrics buffered since the last call.

        Per-request `generate` returns a bare `Completion`; backends buffer
        `_prepare_generation_request_metrics` output here and the controller
        drains it once per collection round.
        """
        metrics = self._metrics
        self._metrics = []
        return metrics
