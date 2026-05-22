# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with VLLMGenerator (vLLM) and PolicyTrainer (TorchTitan)
   running on separate GPU meshes
2. Weight synchronization across meshes: trainer gathers full (unsharded) weights,
   generator reshards to match its own parallelism layout via distribute_tensor
3. Envs driven rollouts; reward and advantage computation live inline
   in the controller.

Command to run:
python3 torchtitan/experiments/rl/grpo.py \
    --module rl --config rl_grpo_qwen3_0_6b \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import hashlib
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace

# must run before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.components.dataloading.utils import pack
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import BatchConfig, CompileConfig, ConfigManager, Configurable
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.config_derivation import (
    AsyncPipelineConfig,
    compute_generator_max_num_seqs,
    compute_world_size,
    derived_capacity,
    DerivedRLConfig,
)
from torchtitan.experiments.rl.envs import (
    EnvBuilder,
    EnvDataset,
    EnvExample,
    TokenEnvConfig,
)
from torchtitan.experiments.rl.generation_scheduler import GenerationScheduler
from torchtitan.experiments.rl.loss import DAPOLoss
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.observability.metrics.rl import (
    _TrainStepTimings,
    _WeightSyncTimings,
    _zero_weight_sync_timings,
    build_rollout_metrics,
    build_train_step_metrics,
    rename_metric_prefix,
    REQUIRED_TRAIN_STEP_HEALTH_KEYS,
    validate_train_step_fwd_bwd_metrics,
)
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.replay import (
    has_advantage_signal,
    ReplayBatch,
    ReplayBuffer,
    rollouts_to_replay_samples,
)
from torchtitan.experiments.rl.rollout_logging import RolloutSampleLogger
from torchtitan.experiments.rl.rollouts import run_rollout_group
from torchtitan.experiments.rl.sampling import SamplingConfig, TrainingLogprobConfig
from torchtitan.experiments.rl.types import (
    Completion,
    ReplaySample,
    RolloutOutput,
    TrainingBatch,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)
_ZERO_ADVANTAGE_EPS = 1e-12


def _generator_session_key(request_id: str) -> str:
    """Return the stable rollout-session key encoded in a request ID."""
    return request_id.rsplit(":turn=", 1)[0]


def _generator_index_for_request_id(request_id: str, num_generators: int) -> int:
    """Map a rollout session to a generator instance."""
    if num_generators <= 0:
        raise ValueError(f"num_generators must be positive, got {num_generators}")
    digest = hashlib.blake2b(
        _generator_session_key(request_id).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big") % num_generators


def _group_request_positions_by_generator(
    request_ids: Sequence[str],
    *,
    num_generators: int,
) -> dict[int, list[int]]:
    """Group batch positions by sticky generator assignment."""
    by_generator: dict[int, list[int]] = {}
    for position, request_id in enumerate(request_ids):
        generator_idx = _generator_index_for_request_id(
            request_id,
            num_generators,
        )
        by_generator.setdefault(generator_idx, []).append(position)
    return by_generator


class Provisioner:
    """Allocates non-overlapping GPU ranges for Monarch proc meshes.

    In non-colocated mode, the trainer and generator run on separate GPU
    meshes (e.g. GPUs 0-3 for training, GPUs 4-7 for generation). Each
    call to `allocate(n)` reserves the next *n* GPUs and returns a
    bootstrap callable that sets `CUDA_VISIBLE_DEVICES` before CUDA
    initializes in the spawned process, ensuring each mesh only sees its
    own devices.
    """

    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.next_gpu = 0

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> Callable[[], None]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            # TODO: Remove once Monarch/PyTorch fixes concurrent import during unpickling.
            import torch  # noqa: F401

        return _bootstrap


def _log_samples(rollouts: list[RolloutOutput]) -> None:
    """Log the first rollout per group for debugging."""
    seen_groups: set[str] = set()
    for rollout in rollouts:
        if rollout.group_id in seen_groups:
            continue
        seen_groups.add(rollout.group_id)
        text = ""
        if rollout.turns and rollout.turns[0].response_messages:
            text = str(rollout.turns[0].response_messages[0].get("content") or "")
        reward = rollout.reward if rollout.reward is not None else float("nan")
        logger.info(
            "  [%s sample=%d status=%s reward=%+.3f]",
            rollout.group_id,
            rollout.sample_idx,
            rollout.status,
            reward,
        )
        logger.info("       A: %s", text[:300].replace("\n", " ").strip())


async def _raise_rollout_task_errors(
    tasks: list[asyncio.Task[None]],
    *,
    timeout_s: float = 0.0,
) -> None:
    """Raise the first background rollout producer exception, if any."""
    if timeout_s > 0.0:
        await asyncio.wait(
            tasks,
            timeout=timeout_s,
            return_when=asyncio.FIRST_EXCEPTION,
        )
    else:
        await asyncio.sleep(0)

    for task in tasks:
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                raise exc


@dataclass(slots=True)
class _RolloutDropStats:
    """Rollout drops accumulated between optimizer steps."""

    empty_groups: int
    zero_advantage_groups: int
    metrics: list[m.Metric]


@dataclass(slots=True)
class _RolloutDropCounters:
    """Producer-side rollout drops accumulated between optimizer steps."""

    max_no_signal_groups: int | None
    empty_groups: int = 0
    zero_advantage_groups: int = 0
    consecutive_dropped_groups: int = 0
    consecutive_empty_groups: int = 0
    consecutive_zero_advantage_groups: int = 0
    zero_advantage_rewards: list[float] = field(default_factory=list)

    def pop(self) -> _RolloutDropStats:
        metrics: list[m.Metric] = []
        if self.zero_advantage_rewards:
            metrics.append(
                m.Metric(
                    "rollout/dropped_zero_advantage_reward",
                    m.SummaryStats.from_list(self.zero_advantage_rewards),
                )
            )
        values = _RolloutDropStats(
            empty_groups=self.empty_groups,
            zero_advantage_groups=self.zero_advantage_groups,
            metrics=metrics,
        )
        self.empty_groups = 0
        self.zero_advantage_groups = 0
        self.zero_advantage_rewards.clear()
        return values

    def record_empty(self) -> None:
        self.empty_groups += 1
        self.consecutive_empty_groups += 1
        self._record_drop()

    def record_zero_advantage(self, rewards: list[float]) -> None:
        self.zero_advantage_groups += 1
        self.consecutive_zero_advantage_groups += 1
        self.zero_advantage_rewards.extend(rewards)
        self._record_drop()

    def record_admitted(self) -> None:
        self.consecutive_dropped_groups = 0
        self.consecutive_empty_groups = 0
        self.consecutive_zero_advantage_groups = 0

    def _record_drop(self) -> None:
        self.consecutive_dropped_groups += 1
        if (
            self.max_no_signal_groups is not None
            and self.consecutive_dropped_groups >= self.max_no_signal_groups
        ):
            raise RuntimeError(
                "no trainable rollout groups admitted after "
                f"{self.consecutive_dropped_groups} consecutive drops "
                f"({self.consecutive_empty_groups} empty, "
                f"{self.consecutive_zero_advantage_groups} zero-advantage)"
            )


def _build_train_step_trace_scalars(
    *,
    samples: list[ReplaySample],
    replay_batch: ReplayBatch,
    fwd_bwd_metrics: dict[str, float],
    optimizer_metrics: dict[str, float],
    checkpoint_saved: bool,
    timings: _TrainStepTimings,
    dropped_empty_groups: int,
    dropped_zero_advantage_groups: int,
    train_version: int,
) -> dict[str, float]:
    """Build structured-logger scalars for one train step."""
    validate_train_step_fwd_bwd_metrics(fwd_bwd_metrics)
    behavior_versions = [sample.behavior_version for sample in samples]
    buffer_depth_samples = 0.0
    dropped_stale_samples = 0.0
    for metric in replay_batch.metrics:
        if metric.key == "replay/buffer/depth_samples_post_pull":
            buffer_depth_samples = float(metric.value.value)
        elif metric.key == "replay/buffer/dropped_stale_samples":
            dropped_stale_samples = float(metric.value.value)

    trace_scalars = {
        "replay.buffer_depth_samples": buffer_depth_samples,
        "replay.dropped_stale_samples": dropped_stale_samples,
        "rollout.dropped_empty_groups": dropped_empty_groups,
        "rollout.dropped_zero_advantage_groups": dropped_zero_advantage_groups,
        "replay.train_version": train_version,
        "replay.behavior_version_min": (
            min(behavior_versions) if behavior_versions else 0
        ),
        "replay.behavior_version_max": (
            max(behavior_versions) if behavior_versions else 0
        ),
        "timing.replay_wait_ms": timings.replay_wait_s * 1000,
        "timing.weight_sync_admission_drain_ms": (
            timings.weight_sync.admission_drain_s * 1000
        ),
        "timing.weight_sync_push_ms": timings.weight_sync.push_s * 1000,
        "timing.weight_sync_pull_ms": timings.weight_sync.pull_s * 1000,
        "timing.checkpoint_ms": timings.checkpoint_s * 1000,
        "checkpoint.saved": float(checkpoint_saved),
    }
    for key in REQUIRED_TRAIN_STEP_HEALTH_KEYS:
        trace_scalars[key.replace("/", ".")] = fwd_bwd_metrics[key]
    key = "health/train/skipped_nonfinite_loss"
    trace_scalars[key.replace("/", ".")] = optimizer_metrics.get(key, 0.0)
    return trace_scalars


class Batcher(Configurable):
    """Packs training samples into ``[B, seq_len]`` batches for the trainer.

    The controller collects rollouts until the total response tokens reach
    ``num_tokens_target`` (= ``global_batch_size * seq_len``), then
    packs all collected samples into fixed-length rows, truncates to
    ``global_batch_size``, and splits into
    ``[grad_accum_steps][dp_degree]`` microbatches.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        batch: BatchConfig = field(default_factory=BatchConfig)

    def __init__(
        self,
        config: Config,
        *,
        pad_id: int,
        global_batch_size: int | None = None,
    ):
        self.local_batch_size = config.batch.local_batch_size
        self.global_batch_size = (
            global_batch_size
            if global_batch_size is not None
            else config.batch.global_batch_size
        )
        self.seq_len = config.batch.seq_len
        self.pad_id = pad_id

    @property
    def num_tokens_target(self) -> int:
        return self.global_batch_size * self.seq_len

    def batch(
        self,
        samples: list[ReplaySample],
        *,
        dp_degree: int,
    ) -> tuple[list[list[TrainingBatch]], int, dict[str, float]]:
        """Pack samples into `[B, seq_len]` microbatches.

        Returns:
            microbatches: shape `[gradient_accumulation_steps][dp_degree]`,
                each entry is a `TrainingBatch` with `local_batch_size` rows.
            num_global_valid_tokens: total response tokens across the batch
                (excludes padding). Used to normalize the loss so that
                gradient accumulation matches a single large-batch step.
            packing_metrics: dict of packing efficiency metrics for logging.

        Example::

            batcher = Batcher(Batcher.Config(batch=BatchConfig(
                local_batch_size=2, global_batch_size=8, seq_len=2048,
            )), pad_id=0)
            mb, num_tok, stats = batcher.batch(samples, dp_degree=2)
            # mb is [grad_accum=2][dp=2], each TrainingBatch holds 2 rows.
            # num_tok = total response tokens across all 8 packed rows.
        """
        packed_rows = list(self._pack_samples(samples))

        num_rows_before_truncate = len(packed_rows)
        if len(packed_rows) > self.global_batch_size:
            packed_rows = packed_rows[: self.global_batch_size]

        gradient_accumulation_steps = self.global_batch_size // (
            self.local_batch_size * dp_degree
        )

        num_global_valid_tokens = sum(
            int(row["loss_mask"].sum().item()) for row in packed_rows
        )

        microbatches: list[list[TrainingBatch]] = []
        for step in range(gradient_accumulation_steps):
            step_batches: list[TrainingBatch] = []
            for rank in range(dp_degree):
                start = (step * dp_degree + rank) * self.local_batch_size
                end = start + self.local_batch_size
                step_batches.append(self.collate(packed_rows[start:end]))
            microbatches.append(step_batches)

        # TODO: Optimize rollout collection to reduce wasted samples.
        # Currently the controller estimates token counts without padded
        # tokens, which can overshoot because packing adds prompt tokens
        # and padding. Track packing metrics to monitor waste.
        total_token_slots = len(packed_rows) * self.seq_len
        packing_metrics = {
            "batcher/packing_efficiency": (
                num_global_valid_tokens / total_token_slots
                if total_token_slots > 0
                else 0.0
            ),
            "batcher/num_packed_rows": float(len(packed_rows)),
            "batcher/num_rows_wasted": float(
                max(0, num_rows_before_truncate - len(packed_rows))
            ),
        }

        return microbatches, num_global_valid_tokens, packing_metrics

    def _iter_training_samples(self, items: Sequence[ReplaySample]) -> Iterator[dict]:
        """Yield one packing-ready dict per item.

        `ReplaySample` carries token-aligned masks/logprobs, so this method
        only broadcasts the scalar advantage over masked response tokens.

        Example::

            sample = ReplaySample(
                token_ids=[5, 7, 8],
                loss_mask=[0, 1, 1],
                ref_logprobs=[0.0, -0.3, -0.4],
                advantage=0.5,
                group_id="g0",
                sample_idx=0,
                behavior_version=1,
                reward=1.0,
            )
            list(batcher._iter_training_samples([sample]))[0]
            # {"input_ids":[5,7,8], "ref_logprobs":[0.0,-0.3,-0.4],
            #  "loss_mask":[0.0,1.0,1.0], "advantages":[0.0,0.5,0.5]}
        """
        for item in items:
            yield {
                "input_ids": item.token_ids,
                "ref_logprobs": item.ref_logprobs,
                "loss_mask": [float(mask) for mask in item.loss_mask],
                "advantages": [item.advantage * float(mask) for mask in item.loss_mask],
            }

    def _pack_samples(self, samples: list[ReplaySample]) -> Iterator[dict]:
        """Pack all samples into [1, seq_len] rows."""
        yield from pack(
            self._iter_training_samples(samples),
            max_seq_length=self.seq_len,
            pad_values={
                "input_ids": self.pad_id,
                "ref_logprobs": 0.0,
                "loss_mask": 0.0,
                "advantages": 0.0,
            },
        )

    @staticmethod
    def collate(rows: list[dict]) -> TrainingBatch:
        """Concatenate packed rows into a single [B, L] TrainingBatch."""
        return TrainingBatch(
            token_ids=torch.cat([r["input_ids"] for r in rows]),
            positions=torch.cat([r["positions"] for r in rows]),
            ref_logprobs=torch.cat([r["ref_logprobs"] for r in rows]),
            loss_mask=torch.cat([r["loss_mask"] for r in rows]),
            advantages=torch.cat([r["advantages"] for r in rows]),
        )


class RLTrainer(Configurable):
    """Top-level RL training orchestrator."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training."""

        model_spec: ModelSpec | None = None
        """Model specification shared by trainer and generator.
        Set programmatically via config_registry (not from CLI)."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets folder (model weights, tokenizer, config files)."""

        num_steps: int = 10
        """Number of RL training steps."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        num_prompts_per_step: int = 5
        """Number of prompt groups per dataset sample step."""

        group_size: int = 8
        """Number of sampled completions per prompt group."""

        max_rollout_turns: int = 1
        """Maximum assistant turns per rollout."""

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily per validation pass."""

        train_env_builder: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Builds single-use training envs from sampled dataset rows."""

        train_dataset: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Dataset used to sample training rollout groups."""

        validation_env_builder: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Builds single-use validation envs from sampled dataset rows."""

        validation_dataset: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Dataset used to sample validation rollout groups."""

        renderer: RendererConfig = field(default_factory=RendererConfig)
        """Renderer used for message <-> token conversion."""

        log_samples: bool = False
        """Log first rollout per group during training and validation."""

        save_rollout_samples: bool = False
        """Write a bounded rollout JSONL sample for smoke/debug runs."""

        max_rollout_sample_groups: int = 2
        """Maximum groups per step/phase written by `RolloutSampleLogger`."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        batcher: Batcher.Config = field(default_factory=Batcher.Config)
        """Batcher config: local_batch_size, seq_len."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=DAPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

        num_generator_instances: int = 1
        """Number of independent generator actor meshes to spawn."""

        max_offpolicy_steps: int = 1
        """Drop replay samples older than this many policy-version steps."""

        async_pipeline: AsyncPipelineConfig = field(default_factory=AsyncPipelineConfig)
        """Capacity overrides for async rollout producers and replay."""

        drop_zero_advantage_groups: bool = True
        """Drop rollout groups whose normalized advantages are all zero."""

        max_no_signal_groups: int | None = 100
        """Fail after this many consecutive empty or zero-advantage groups."""

        metrics: m.MetricsProcessor.Config = field(
            default_factory=m.MetricsProcessor.Config
        )

        def __post_init__(self):
            if self.generator.checkpoint.enable:
                raise ValueError(
                    "Generator checkpoint must be disabled in the RL loop "
                    "(weights are synced from the trainer via TorchStore). "
                    "Set generator.checkpoint.enable=False."
                )

            if self.trainer.debug.batch_invariant:
                if not self.trainer.debug.deterministic:
                    raise ValueError("batch_invariant requires deterministic=True")
                # TODO: Replace trainer dtype constraint to use mixed
                #  training enabled by FSDP.
                if self.trainer.training.dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 training dtype, "
                        f"got {self.trainer.training.dtype!r}"
                    )
                if self.generator.model_dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 generator dtype, "
                        f"got {self.generator.model_dtype!r}"
                    )
                if self.trainer.parallelism.enable_sequence_parallel:
                    raise ValueError(
                        "batch_invariant mode doesn't support SP now. "
                        "SP uses reduce-scatter which only supports Ring in NCCL "
                        "and has not been validated for determinism."
                    )
            if self.group_size <= 0:
                raise ValueError(f"group_size must be positive, got {self.group_size}")
            if self.max_rollout_turns <= 0:
                raise ValueError(
                    f"max_rollout_turns must be positive, got {self.max_rollout_turns}"
                )
            if self.max_rollout_sample_groups < 0:
                raise ValueError(
                    "max_rollout_sample_groups must be non-negative, "
                    f"got {self.max_rollout_sample_groups}"
                )
            if self.num_prompts_per_step <= 0:
                raise ValueError(
                    "num_prompts_per_step must be positive, "
                    f"got {self.num_prompts_per_step}"
                )
            if self.num_validation_samples <= 0:
                raise ValueError(
                    "num_validation_samples must be positive, "
                    f"got {self.num_validation_samples}"
                )
            if self.max_offpolicy_steps < 0:
                raise ValueError(
                    "max_offpolicy_steps must be non-negative, "
                    f"got {self.max_offpolicy_steps}"
                )
            if self.max_no_signal_groups is not None and self.max_no_signal_groups <= 0:
                raise ValueError(
                    "max_no_signal_groups must be positive or None, "
                    f"got {self.max_no_signal_groups}"
                )
            if self.num_generator_instances <= 0:
                raise ValueError(
                    "num_generator_instances must be positive, "
                    f"got {self.num_generator_instances}"
                )
            pipeline = self.async_pipeline
            if (
                pipeline.rollout_concurrency is not None
                and pipeline.rollout_concurrency <= 0
            ):
                raise ValueError(
                    "async_pipeline.rollout_concurrency must be positive "
                    f"or None, got {pipeline.rollout_concurrency}"
                )
            if (
                pipeline.replay_buffer_samples is not None
                and pipeline.replay_buffer_samples <= 0
            ):
                raise ValueError(
                    "async_pipeline.replay_buffer_samples must be positive "
                    f"or None, got {pipeline.replay_buffer_samples}"
                )
            if (
                pipeline.max_admitted_generation_prompts is not None
                and pipeline.max_admitted_generation_prompts <= 0
            ):
                raise ValueError(
                    "async_pipeline.max_admitted_generation_prompts must be "
                    "positive or None, got "
                    f"{pipeline.max_admitted_generation_prompts}"
                )
            TrainingLogprobConfig.from_sampling(self.generator.sampling)
            _ = self.derived

        @property
        def derived(self) -> DerivedRLConfig:
            """Resolved async replay capacity view."""
            return derived_capacity(self)

    def __init__(self, config: Config):
        self.config = config
        self.trainer = None
        self.generator = None
        self.generators: list = []
        self._proc_meshes = []
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        self.rollout_sample_logger = (
            RolloutSampleLogger(
                config.dump_folder,
                max_groups_per_step=config.max_rollout_sample_groups,
            )
            if config.save_rollout_samples
            else None
        )
        self.train_dataset: EnvDataset = config.train_dataset.build()
        self.train_env_builder: EnvBuilder = config.train_env_builder.build()
        self.validation_dataset: EnvDataset = config.validation_dataset.build()
        self.validation_env_builder: EnvBuilder = config.validation_env_builder.build()
        self.renderer = config.renderer.build(model_path=config.hf_assets_path)
        self._stop_token_ids = list(self.renderer.get_stop_token_ids())
        tokenizer = HuggingFaceTokenizer(tokenizer_path=config.hf_assets_path)
        self.batcher = Batcher(
            config.batcher,
            pad_id=tokenizer.eos_id,
            global_batch_size=config.derived.global_batch_rows,
        )

    async def close(self):
        """Best-effort: tear down actors, close metric backends, then stop proc meshes."""
        logger.info("Closing: tearing down actors and process meshes.")
        actors_to_close = [("trainer", self.trainer)]
        generators = self._generator_actors()
        if generators:
            actors_to_close.extend(
                (f"generator_{idx}", generator)
                for idx, generator in enumerate(generators)
            )
        else:
            actors_to_close.append(("generator", self.generator))

        for actor_name, actor in actors_to_close:
            if actor is None:
                continue
            try:
                await actor.close.call()
            except Exception:
                logger.exception("%s.close failed", actor_name)

        try:
            self.metrics_processor.close()
        except Exception:
            logger.exception("metrics_processor close failed")

        for i, mesh in enumerate(self._proc_meshes):
            try:
                await mesh.stop()
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)
        self._proc_meshes = []

    def _generator_actors(self) -> list:
        if getattr(self, "generators", []):
            return list(self.generators)
        generator = getattr(self, "generator", None)
        return [] if generator is None else [generator]

    async def _sync_actor_log_step(self, step: int) -> None:
        await self._await_call(self.trainer.sync_log_step.call(step))
        for generator in self._generator_actors():
            await self._await_call(generator.sync_log_step.call(step))

    def _get_rank_0_value(self, result, has_gpus: bool = True):
        """Extract rank 0 result, handling both single and multi-node meshes.

        Monarch actor endpoints return results from all ranks in the mesh.
        This picks out rank 0's result by indexing into the host and GPU
        dimensions as needed (multi-node meshes have an extra host dimension).
        This should be used in cases where all ranks return the same result.
        """
        kwargs = {}
        if self._multi_node:
            kwargs["hosts"] = 0
        if has_gpus:
            kwargs["gpus"] = 0
        return result.item(**kwargs)

    def _spawn_role_meshes(
        self,
        *,
        host_mesh,
        trainer_nodes: int | None,
        generator_nodes: int | None,
        gpus_per_node: int | None,
        total_gpus: int,
    ):
        num_generators = self.config.num_generator_instances
        if host_mesh is None:
            provisioner = Provisioner(total_gpus=total_gpus)
            trainer_mesh = this_host().spawn_procs(
                per_host={"gpus": self.trainer_world_size},
                bootstrap=provisioner.allocate(self.trainer_world_size),
            )
            generator_meshes = [
                this_host().spawn_procs(
                    per_host={"gpus": self.generator_world_size},
                    bootstrap=provisioner.allocate(self.generator_world_size),
                    name=(
                        "generator"
                        if num_generators == 1
                        else f"generator_{generator_idx}"
                    ),
                )
                for generator_idx in range(num_generators)
            ]
            return trainer_mesh, generator_meshes

        if num_generators != 1:
            raise ValueError(
                "num_generator_instances > 1 is only supported when host_mesh is None"
            )
        if trainer_nodes is None or generator_nodes is None or gpus_per_node is None:
            raise ValueError(
                "trainer_nodes, generator_nodes, and gpus_per_node are "
                "required when host_mesh is provided"
            )
        if self.trainer_world_size % trainer_nodes != 0:
            raise ValueError(
                f"trainer_world_size ({self.trainer_world_size}) must be "
                f"evenly divisible by trainer_nodes ({trainer_nodes})"
            )
        if self.generator_world_size % generator_nodes != 0:
            raise ValueError(
                f"generator_world_size ({self.generator_world_size}) must be "
                f"evenly divisible by generator_nodes ({generator_nodes})"
            )

        trainer_gpus_per_node = self.trainer_world_size // trainer_nodes
        generator_gpus_per_node = self.generator_world_size // generator_nodes
        trainer_host_mesh = host_mesh.slice(hosts=slice(0, trainer_nodes))
        generator_host_mesh = host_mesh.slice(
            hosts=slice(trainer_nodes, trainer_nodes + generator_nodes)
        )
        trainer_provisioner = Provisioner(total_gpus=gpus_per_node)
        generator_provisioner = Provisioner(total_gpus=gpus_per_node)
        trainer_mesh = trainer_host_mesh.spawn_procs(
            per_host={"gpus": trainer_gpus_per_node},
            bootstrap=trainer_provisioner.allocate(trainer_gpus_per_node),
        )
        generator_mesh = generator_host_mesh.spawn_procs(
            per_host={"gpus": generator_gpus_per_node},
            bootstrap=generator_provisioner.allocate(generator_gpus_per_node),
        )
        return trainer_mesh, [generator_mesh]

    @sl.log_trace_span("setup_async")
    async def setup_async(
        self,
        *,
        host_mesh=None,
        trainer_nodes: int | None = None,
        generator_nodes: int | None = None,
        gpus_per_node: int | None = None,
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Kept separate from ``__init__`` because actor spawning, torch
        elastic env setup, TorchStore initialization, and the initial
        weight push/pull are all ``await``-based runtime side effects
        that cannot run in a synchronous constructor.

        Creates separate GPU meshes for trainer and generator and
        synchronizes initial weights from trainer to generator. Must be
        called before :meth:`train`.

        Args:
            host_mesh: Optional multi-node HostMesh. When provided,
                whole nodes are dedicated to trainer vs generator
                roles instead of partitioning GPUs on a single host.
            trainer_nodes: Number of nodes for the trainer (required when
                host_mesh is provided).
            generator_nodes: Number of nodes for the generator (required when
                host_mesh is provided).
            gpus_per_node: GPUs per node, assumed to be the same across all
                nodes (no heterogeneous node configurations). Required when
                host_mesh is provided.
        """
        config = self.config

        self.trainer_world_size = compute_world_size(config.trainer.parallelism)
        self.generator_world_size = compute_world_size(config.generator.parallelism)
        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        num_generators = config.num_generator_instances
        total_generator_gpus = self.generator_world_size * num_generators
        total_gpus = self.trainer_world_size + total_generator_gpus
        if num_generators == 1:
            logger.info(
                f"{self.generator_world_size} generator GPUs + "
                f"{self.trainer_world_size} trainer GPUs = {total_gpus} total"
            )
        else:
            logger.info(
                f"{num_generators} generator instances x "
                f"{self.generator_world_size} GPUs = "
                f"{total_generator_gpus} generator GPUs + "
                f"{self.trainer_world_size} trainer GPUs = {total_gpus} total"
            )

        self._multi_node = host_mesh is not None

        with sl.log_trace_span("mesh_spawn"):
            trainer_mesh, generator_meshes = self._spawn_role_meshes(
                host_mesh=host_mesh,
                trainer_nodes=trainer_nodes,
                generator_nodes=generator_nodes,
                gpus_per_node=gpus_per_node,
                total_gpus=total_gpus,
            )

            # Store proc meshes for cleanup
            self._proc_meshes = [trainer_mesh, *generator_meshes]

            await setup_torch_elastic_env_async(trainer_mesh)
            for generator_mesh in generator_meshes:
                await setup_torch_elastic_env_async(generator_mesh)

            # Spawn actors on their respective meshes
            self.trainer = trainer_mesh.spawn(
                "trainer",
                PolicyTrainer,
                config.trainer,
                model_spec=config.model_spec,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                compile_config=config.compile,
                output_dir=config.dump_folder,
            )

            self.generators = [
                generator_mesh.spawn(
                    "generator"
                    if len(generator_meshes) == 1
                    else f"generator_{generator_idx}",
                    VLLMGenerator,
                    config.generator,
                    model_spec=config.model_spec,
                    model_path=config.hf_assets_path,
                    compile_config=config.compile,
                    max_num_seqs=compute_generator_max_num_seqs(config),
                    output_dir=config.dump_folder,
                )
                for generator_idx, generator_mesh in enumerate(generator_meshes)
            ]
            self.generator = self.generators[0]

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        with sl.log_trace_span("torchstore_init"):
            await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # Initial weight sync from trainer to every generator.
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self._await_call(self.trainer.push_model_state_dict.call())
        with sl.log_trace_span("generator_pull_model_state_dict"):
            for generator in self.generators:
                await self._await_call(generator.pull_model_state_dict.call(0))

    @staticmethod
    async def _await_call(actor_call):
        """Await a Monarch endpoint call without blocking the event loop."""
        if hasattr(actor_call, "get"):
            return await asyncio.to_thread(actor_call.get)
        return await actor_call

    async def _await_rank_0(self, actor_call, has_gpus: bool = True):
        """Await an actor call and unwrap the rank-0 value."""
        return self._get_rank_0_value(
            await self._await_call(actor_call),
            has_gpus=has_gpus,
        )

    def _sampling_for_rollout(self, sampling: SamplingConfig) -> SamplingConfig:
        """Attach renderer stop tokens to a per-call sampling config."""
        return replace(sampling, stop_token_ids=self._stop_token_ids)

    def _make_generation_scheduler(
        self,
        *,
        metrics_prefix: str,
    ) -> GenerationScheduler:
        async def generate_batch(
            prompt_token_ids_batch: list[list[int]],
            request_ids: list[str],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            generators = self._generator_actors()
            if not generators:
                raise RuntimeError("generation scheduler has no generator actors")
            if len(generators) == 1:
                completions, metrics = await self._await_rank_0(
                    generators[0].generate.call(
                        prompt_token_ids_batch,
                        request_ids=request_ids,
                        sampling_config=sampling,
                        metrics_prefix=metrics_prefix,
                    )
                )
                return completions, metrics

            by_generator = _group_request_positions_by_generator(
                request_ids,
                num_generators=len(generators),
            )
            merged: list[Completion | None] = [None] * len(prompt_token_ids_batch)
            merged_metrics: list[m.Metric] = []
            for generator_idx in range(len(generators)):
                positions = by_generator.get(generator_idx, [])
                queue_depth_key = f"{metrics_prefix}/{generator_idx}/queue_depth"
                merged_metrics.extend(
                    [
                        m.Metric(queue_depth_key, m.Mean(float(len(positions)))),
                        m.Metric(queue_depth_key, m.Max(float(len(positions)))),
                    ]
                )
            for generator_idx, positions in by_generator.items():
                completions, metrics = await self._await_rank_0(
                    generators[generator_idx].generate.call(
                        [prompt_token_ids_batch[position] for position in positions],
                        request_ids=[request_ids[position] for position in positions],
                        sampling_config=sampling,
                        metrics_prefix=f"{metrics_prefix}/{generator_idx}",
                    )
                )
                for position, completion in zip(positions, completions, strict=True):
                    merged[position] = completion
                merged_metrics.extend(metrics)

            if any(completion is None for completion in merged):
                raise RuntimeError("generator routing failed to fill all completions")
            return (
                [completion for completion in merged if completion is not None],
                merged_metrics,
            )

        return GenerationScheduler(
            generate_batch,
            max_admitted_prompts=self.config.derived.max_admitted_generation_prompts,
        )

    async def _sync_generator_weights(
        self,
        *,
        generation_scheduler: GenerationScheduler,
        policy_version: int,
    ) -> _WeightSyncTimings:
        t_weight_sync_start = time.perf_counter()
        with sl.log_trace_span("weight_sync_admission_drain"):
            await generation_scheduler.pause_for_weight_sync()
        t_weight_sync_drain_s = time.perf_counter() - t_weight_sync_start

        try:
            t_push_start = time.perf_counter()
            with sl.log_trace_span("trainer_push_model_state_dict"):
                await self._await_call(self.trainer.push_model_state_dict.call())
            t_weight_sync_push_s = time.perf_counter() - t_push_start
            t_pull_start = time.perf_counter()
            with sl.log_trace_span("generator_pull_model_state_dict"):
                for generator in self._generator_actors():
                    await self._await_call(
                        generator.pull_model_state_dict.call(policy_version)
                    )
            t_weight_sync_pull_s = time.perf_counter() - t_pull_start
        finally:
            await generation_scheduler.resume_after_weight_sync()

        return _WeightSyncTimings(
            admission_drain_s=t_weight_sync_drain_s,
            push_s=t_weight_sync_push_s,
            pull_s=t_weight_sync_pull_s,
            total_s=time.perf_counter() - t_weight_sync_start,
        )

    @staticmethod
    def _forward_backward_skip_metrics(
        fwd_bwd_metrics: dict[str, float],
        *,
        policy_version: int,
    ) -> dict[str, float] | None:
        """Return optimizer metrics for a forward/backward result to skip."""
        validate_train_step_fwd_bwd_metrics(fwd_bwd_metrics)
        if math.isfinite(fwd_bwd_metrics.get("loss/mean", float("nan"))):
            return None
        return {
            "train/policy_version": float(policy_version),
            "health/train/skipped_nonfinite_loss": 1.0,
        }

    def _log_train_step(
        self,
        *,
        step: int,
        samples: list[ReplaySample],
        replay_batch: ReplayBatch,
        rollouts: list[RolloutOutput],
        generation_scheduler: GenerationScheduler,
        fwd_bwd_metrics: dict[str, float],
        optimizer_metrics: dict[str, float],
        packing_metrics: dict[str, float],
        checkpoint_saved: bool,
        timings: _TrainStepTimings,
        dropped_empty_groups: int,
        dropped_zero_advantage_groups: int,
        drop_metrics: list[m.Metric],
        train_version: int,
    ) -> None:
        """Build, trace, and emit train-step metrics."""
        live_generation_metrics = [
            rename_metric_prefix(
                [metric],
                old_prefix="generator",
                new_prefix="generator/live",
            )[0]
            for metric in generation_scheduler.pop_metrics()
        ]
        step_metrics = build_train_step_metrics(
            samples=samples,
            replay_batch=replay_batch,
            rollouts=rollouts,
            live_generation_metrics=live_generation_metrics,
            fwd_bwd_metrics=fwd_bwd_metrics,
            optimizer_metrics=optimizer_metrics,
            packing_metrics=packing_metrics,
            checkpoint_saved=checkpoint_saved,
            timings=timings,
            dropped_empty_groups=dropped_empty_groups,
            dropped_zero_advantage_groups=dropped_zero_advantage_groups,
            drop_metrics=drop_metrics,
            train_version=train_version,
        )
        sl.log_trace_scalar(
            _build_train_step_trace_scalars(
                samples=samples,
                replay_batch=replay_batch,
                fwd_bwd_metrics=fwd_bwd_metrics,
                optimizer_metrics=optimizer_metrics,
                checkpoint_saved=checkpoint_saved,
                timings=timings,
                dropped_empty_groups=dropped_empty_groups,
                dropped_zero_advantage_groups=dropped_zero_advantage_groups,
                train_version=train_version,
            )
        )
        self.metrics_processor.log(
            step=step,
            metrics=step_metrics,
            is_validation=False,
        )

    async def _collect_finite_rollouts(
        self,
        *,
        env_dataset: EnvDataset,
        env_builder: EnvBuilder,
        num_groups: int,
        group_size: int,
        sample_step: int,
        sampling: SamplingConfig,
        metrics_prefix: str,
    ) -> tuple[list[RolloutOutput], list[m.Metric]]:
        """Collect a bounded set of rollout groups."""
        sampling = self._sampling_for_rollout(sampling)
        generation_scheduler = self._make_generation_scheduler(
            metrics_prefix=metrics_prefix,
        )
        token_env_config = TokenEnvConfig(
            max_trajectory_tokens=self.config.batcher.batch.seq_len,
            max_generation_tokens=sampling.max_tokens,
        )
        examples = [
            env_dataset.sample_group(sample_step=sample_step, group_idx=idx)
            for idx in range(num_groups)
        ]
        pending: set[asyncio.Task[list[RolloutOutput]]] = set()
        rollouts: list[RolloutOutput] = []
        next_idx = 0
        max_inflight = max(self.config.derived.rollout_concurrency, 1)

        try:
            while next_idx < len(examples) or pending:
                while next_idx < len(examples) and len(pending) < max_inflight:
                    example = examples[next_idx]
                    pending.add(
                        asyncio.create_task(
                            run_rollout_group(
                                env_builder=env_builder,
                                example=example,
                                group_size=group_size,
                                renderer=self.renderer,
                                completion_fn=generation_scheduler.submit,
                                sampling=sampling,
                                max_turns=self.config.max_rollout_turns,
                                token_env_config=token_env_config,
                            )
                        )
                    )
                    next_idx += 1

                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    rollouts.extend(task.result())
        except BaseException:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
        finally:
            await generation_scheduler.close()

        metrics = build_rollout_metrics(
            "rollout",
            rollouts,
            generation_scheduler.pop_metrics(),
        )
        return rollouts, metrics

    async def _continuous_rollouts(
        self,
        *,
        worker_idx: int,
        replay_buffer: ReplayBuffer,
        generation_scheduler: GenerationScheduler,
        drop_counters: _RolloutDropCounters,
        producer_rollouts: dict[tuple[str, int], RolloutOutput],
        shutdown: asyncio.Event,
        next_example: Callable[[], Awaitable[EnvExample]],
    ) -> None:
        """Produce rollout groups until training finishes or a producer fails."""
        sampling = self._sampling_for_rollout(self.config.generator.sampling)
        token_env_config = TokenEnvConfig(
            max_trajectory_tokens=self.config.batcher.batch.seq_len,
            max_generation_tokens=sampling.max_tokens,
        )
        try:
            while not shutdown.is_set():
                example = await next_example()
                with sl.log_trace_span("rollout_group"):
                    group_rollouts = await run_rollout_group(
                        env_builder=self.train_env_builder,
                        example=example,
                        group_size=self.config.group_size,
                        renderer=self.renderer,
                        completion_fn=generation_scheduler.submit,
                        sampling=sampling,
                        max_turns=self.config.max_rollout_turns,
                        token_env_config=token_env_config,
                    )

                if self.rollout_sample_logger is not None:
                    self.rollout_sample_logger.write(
                        step=example.sample_step,
                        phase="train_rollout",
                        rollouts=group_rollouts,
                    )

                samples = rollouts_to_replay_samples(group_rollouts)
                if shutdown.is_set():
                    return
                if not samples:
                    drop_counters.record_empty()
                    sl.log_trace_scalar({"rollout.dropped_empty_groups": 1})
                    continue
                if self.config.drop_zero_advantage_groups and not has_advantage_signal(
                    samples, eps=_ZERO_ADVANTAGE_EPS
                ):
                    rewards = [
                        float(rollout.reward)
                        for rollout in group_rollouts
                        if rollout.reward is not None
                    ]
                    drop_counters.record_zero_advantage(rewards)
                    sl.log_trace_scalar({"rollout.dropped_zero_advantage_groups": 1})
                    continue

                drop_counters.record_admitted()
                for rollout in group_rollouts:
                    producer_rollouts[(rollout.group_id, rollout.sample_idx)] = rollout
                with sl.log_trace_span("replay_buffer_put"):
                    await replay_buffer.put(samples)
        except asyncio.CancelledError:
            raise
        except RuntimeError:
            if shutdown.is_set():
                return
            shutdown.set()
            await replay_buffer.close()
            logger.exception("rollout producer %d failed", worker_idx)
            raise
        except BaseException:
            shutdown.set()
            await replay_buffer.close()
            logger.exception("rollout producer %d failed", worker_idx)
            raise

    @sl.log_trace_span("validate")
    async def validate(self) -> list[m.Metric]:
        """Run validation on held-out prompts using greedy sampling."""
        t_validate_start = time.perf_counter()
        greedy = SamplingConfig(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.config.generator.sampling.max_tokens,
        )
        rollouts, validation_metrics = await self._collect_finite_rollouts(
            env_dataset=self.validation_dataset,
            env_builder=self.validation_env_builder,
            num_groups=self.config.num_validation_samples,
            group_size=1,
            sample_step=0,
            sampling=greedy,
            metrics_prefix="validation_generator",
        )

        if self.config.log_samples:
            _log_samples(rollouts)
        if self.rollout_sample_logger is not None:
            self.rollout_sample_logger.write(
                step=0,
                phase="validation",
                rollouts=rollouts,
            )

        validation_metrics = rename_metric_prefix(
            validation_metrics,
            old_prefix="rollout",
            new_prefix="validation/rollout",
        )
        validation_metrics = rename_metric_prefix(
            validation_metrics,
            old_prefix="reward",
            new_prefix="validation/reward",
        )
        t_validate_s = time.perf_counter() - t_validate_start
        validation_metrics += [
            m.Metric("validation/num_samples", m.NoReduce(float(len(rollouts)))),
            m.Metric("timing/validate", m.NoReduce(t_validate_s)),
        ]
        return validation_metrics

    async def train(self):
        num_steps = self.config.num_steps
        logprob_config = TrainingLogprobConfig.from_sampling(
            self.config.generator.sampling
        )
        logger.info(f"Pre-training validation; then {num_steps} steps of RL training")

        pre_validation_metrics = await self.validate()
        self.metrics_processor.log(
            step=0,
            metrics=pre_validation_metrics,
            is_validation=True,
        )
        pre_validation_agg = m.MetricsProcessor._aggregate_metrics(
            pre_validation_metrics
        )

        sl.log_trace_instant("training_start")

        derived = self.config.derived
        replay_buffer = ReplayBuffer(
            max_samples=max(derived.replay_buffer_samples, 1),
            max_age_steps=self.config.max_offpolicy_steps,
        )
        generation_scheduler = self._make_generation_scheduler(
            metrics_prefix="generator"
        )
        drop_counters = _RolloutDropCounters(
            max_no_signal_groups=self.config.max_no_signal_groups,
        )
        producer_rollouts: dict[tuple[str, int], RolloutOutput] = {}
        shutdown = asyncio.Event()
        next_group_idx = 0
        next_group_lock = asyncio.Lock()

        async def next_example() -> EnvExample:
            nonlocal next_group_idx
            async with next_group_lock:
                absolute_group_idx = next_group_idx
                next_group_idx += 1
            sample_step, group_idx = divmod(
                absolute_group_idx,
                max(self.config.num_prompts_per_step, 1),
            )
            return self.train_dataset.sample_group(
                sample_step=sample_step,
                group_idx=group_idx,
            )

        rollout_tasks = [
            asyncio.create_task(
                self._continuous_rollouts(
                    worker_idx=worker_idx,
                    replay_buffer=replay_buffer,
                    generation_scheduler=generation_scheduler,
                    drop_counters=drop_counters,
                    producer_rollouts=producer_rollouts,
                    shutdown=shutdown,
                    next_example=next_example,
                )
            )
            for worker_idx in range(max(derived.rollout_concurrency, 1))
        ]

        policy_version = 0
        try:
            for step in range(1, num_steps + 1):
                sl.set_step(step)
                await self._sync_actor_log_step(step)

                t_step_start = time.perf_counter()
                t_replay_start = time.perf_counter()
                train_version = policy_version
                with sl.log_trace_span("replay_buffer_get_batch"):
                    replay_batch = await replay_buffer.get_batch(
                        min_loss_tokens=self.batcher.num_tokens_target,
                        train_version=train_version,
                    )
                t_replay_wait_s = time.perf_counter() - t_replay_start
                await _raise_rollout_task_errors(rollout_tasks)

                drop_stats = drop_counters.pop()
                samples = replay_batch.samples
                rollouts: list[RolloutOutput] = []
                rollout_keys: set[tuple[str, int]] = set()
                for sample in samples:
                    key = (sample.group_id, sample.sample_idx)
                    if key not in rollout_keys and key in producer_rollouts:
                        rollouts.append(producer_rollouts[key])
                    rollout_keys.add(key)
                dropped_keys = {
                    (sample.group_id, sample.sample_idx)
                    for sample in replay_batch.dropped_samples
                }
                for key in rollout_keys | dropped_keys:
                    producer_rollouts.pop(key, None)
                if self.config.log_samples:
                    _log_samples(rollouts)
                if not samples:
                    await _raise_rollout_task_errors(rollout_tasks, timeout_s=1.0)
                    raise RuntimeError(
                        "replay buffer closed before producing trainable samples"
                    )

                t_train_start = time.perf_counter()
                with sl.log_trace_span("batcher_batch"):
                    (
                        microbatches,
                        num_global_valid_tokens,
                        packing_metrics,
                    ) = self.batcher.batch(samples, dp_degree=self.trainer_dp_degree)

                with sl.log_trace_span("trainer_forward_backward_call"):
                    fwd_bwd_metrics = await self._await_rank_0(
                        self.trainer.forward_backward.call(
                            microbatches,
                            num_global_valid_tokens=num_global_valid_tokens,
                            logprob_config=logprob_config,
                        )
                    )
                skip_metrics = self._forward_backward_skip_metrics(
                    fwd_bwd_metrics,
                    policy_version=policy_version,
                )
                if skip_metrics is not None:
                    t_train_s = time.perf_counter() - t_train_start
                    t_step_s = time.perf_counter() - t_step_start
                    logger.warning(
                        "Step %d skipped optimizer step because loss was not finite",
                        step,
                    )
                    self._log_train_step(
                        step=step,
                        samples=samples,
                        replay_batch=replay_batch,
                        rollouts=rollouts,
                        generation_scheduler=generation_scheduler,
                        fwd_bwd_metrics=fwd_bwd_metrics,
                        optimizer_metrics=skip_metrics,
                        packing_metrics=packing_metrics,
                        checkpoint_saved=False,
                        timings=_TrainStepTimings(
                            step_s=t_step_s,
                            replay_wait_s=t_replay_wait_s,
                            rollout_s=t_replay_wait_s,
                            train_s=t_train_s,
                            checkpoint_s=0.0,
                            weight_sync=_zero_weight_sync_timings(),
                        ),
                        dropped_empty_groups=drop_stats.empty_groups,
                        dropped_zero_advantage_groups=drop_stats.zero_advantage_groups,
                        drop_metrics=drop_stats.metrics,
                        train_version=train_version,
                    )
                    continue

                with sl.log_trace_span("trainer_optim_step_call"):
                    optim_output = await self._await_rank_0(
                        self.trainer.optim_step.call()
                    )
                trainer_policy_version = optim_output.policy_version
                optimizer_metrics = {
                    **optim_output.metrics,
                    "health/train/skipped_nonfinite_loss": 0.0,
                }
                t_train_s = time.perf_counter() - t_train_start

                weight_sync_timings = await self._sync_generator_weights(
                    generation_scheduler=generation_scheduler,
                    policy_version=trainer_policy_version,
                )
                policy_version = trainer_policy_version

                t_checkpoint_start = time.perf_counter()
                with sl.log_trace_span("trainer_save_checkpoint_call"):
                    checkpoint_saved = await self._await_rank_0(
                        self.trainer.save_checkpoint.call(
                            step,
                            last_step=step == num_steps,
                        )
                    )
                t_checkpoint_s = time.perf_counter() - t_checkpoint_start
                t_step_s = time.perf_counter() - t_step_start

                self._log_train_step(
                    step=step,
                    samples=samples,
                    replay_batch=replay_batch,
                    rollouts=rollouts,
                    generation_scheduler=generation_scheduler,
                    fwd_bwd_metrics=fwd_bwd_metrics,
                    optimizer_metrics=optimizer_metrics,
                    packing_metrics=packing_metrics,
                    checkpoint_saved=checkpoint_saved,
                    timings=_TrainStepTimings(
                        step_s=t_step_s,
                        replay_wait_s=t_replay_wait_s,
                        rollout_s=t_replay_wait_s,
                        train_s=t_train_s,
                        checkpoint_s=t_checkpoint_s,
                        weight_sync=weight_sync_timings,
                    ),
                    dropped_empty_groups=drop_stats.empty_groups,
                    dropped_zero_advantage_groups=drop_stats.zero_advantage_groups,
                    drop_metrics=drop_stats.metrics,
                    train_version=train_version,
                )
        finally:
            shutdown.set()
            await replay_buffer.close()
            for task in rollout_tasks:
                task.cancel()
            await generation_scheduler.close()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        post_validation_metrics = await self.validate()
        self.metrics_processor.log(
            step=num_steps,
            metrics=post_validation_metrics,
            is_validation=True,
        )
        post_validation_agg = m.MetricsProcessor._aggregate_metrics(
            post_validation_metrics
        )

        reward_keys = sorted(
            k
            for k in set(pre_validation_agg) | set(post_validation_agg)
            if "reward" in k
        )
        logger.info("=" * 60)
        logger.info("Validation summary (pre / post):")
        for key in reward_keys:
            pre = pre_validation_agg.get(key, float("nan"))
            post = post_validation_agg.get(key, float("nan"))
            logger.info(f"  {key}:  {pre:+.3f}  /  {post:+.3f}")
        logger.info("=" * 60)


async def main():
    config = ConfigManager().parse_args()
    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        # pyrefly: ignore [missing-attribute]
        enable=config.trainer.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    rl_trainer = config.build()
    try:
        await rl_trainer.setup_async()
        await rl_trainer.train()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await rl_trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
