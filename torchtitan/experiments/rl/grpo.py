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
3. Async env rollouts feed a bounded replay FIFO while the trainer consumes
   completed groups.

Command to run:
python3 torchtitan/experiments/rl/grpo.py \
    --module rl --config rl_grpo_qwen3_0_6b \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import dataclasses
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

# must run before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import (
    CompileConfig,
    ConfigManager,
    Configurable,
    ParallelismConfig,
)
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.envs import (
    EnvBuilder,
    EnvDataset,
    EnvExample,
    TokenEnvConfig,
)
from torchtitan.experiments.rl.generation_scheduler import GenerationScheduler
from torchtitan.experiments.rl.loss import GRPOLoss
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.observability.metrics.rl import (
    _TrainStepTimings,
    _WeightSyncTimings,
    _zero_weight_sync_timings,
    build_rollout_metrics,
    build_train_step_metrics,
    REQUIRED_TRAIN_STEP_HEALTH_KEYS,
    rename_metric_prefix,
    validate_train_step_fwd_bwd_metrics,
)
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.replay import (
    has_advantage_signal,
    ReplayBatch,
    ReplayBuffer,
    ReplayGroup,
    rollouts_to_replay_samples,
)
from torchtitan.experiments.rl.rollout_logging import RolloutSampleLogger
from torchtitan.experiments.rl.rollouts import run_rollout_group
from torchtitan.experiments.rl.sampling import SamplingConfig, TrainingLogprobConfig
from torchtitan.experiments.rl.types import (
    Completion,
    OptimStepOutput,
    ReplaySample,
    RolloutOutput,
    TrainingBatch,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)

_ZERO_ADVANTAGE_EPS = 1e-12


def _build_train_step_trace_scalars(
    *,
    replay_batch: ReplayBatch,
    fwd_bwd_metrics: dict[str, float],
    optimizer_metrics: dict[str, float],
    checkpoint_saved: bool,
    timings: _TrainStepTimings,
    dropped_empty_groups: int,
    dropped_zero_advantage_groups: int,
    train_version: int,
) -> dict[str, float]:
    """Build structured-logger scalar breadcrumbs for one train step."""
    validate_train_step_fwd_bwd_metrics(fwd_bwd_metrics)
    behavior_versions = [group.behavior_version for group in replay_batch.groups]
    max_behavior_versions = [
        group.max_behavior_version for group in replay_batch.groups
    ]
    trace_scalars = {
        "replay.buffer_depth_groups": replay_batch.stats.depth_groups,
        "replay.dropped_stale_groups": replay_batch.stats.num_dropped_stale_groups,
        "rollout.dropped_empty_groups": dropped_empty_groups,
        "rollout.dropped_zero_advantage_groups": dropped_zero_advantage_groups,
        "replay.train_version": train_version,
        "replay.behavior_version_min": (
            min(behavior_versions) if behavior_versions else 0
        ),
        "replay.behavior_version_max": (
            max(max_behavior_versions) if max_behavior_versions else 0
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
    for key in (
        "train/skipped_nonfinite_loss",
        "train/skipped_nonfinite_grad_norm",
    ):
        trace_scalars[key.replace("/", ".")] = optimizer_metrics.get(key, 0.0)
    return trace_scalars


class Provisioner:
    """Allocates non-overlapping GPU ranges for Monarch proc meshes.

    In non-colocated mode, the trainer and generator run on separate GPU
    meshes (e.g. GPUs 0-3 for training, GPUs 4-7 for generation). Each
    call to `allocate(n)` reserves the next *n* GPUs and returns a
    bootstrap callable that sets `CUDA_VISIBLE_DEVICES` before CUDA
    initializes in the spawned process, ensuring each mesh only sees its
    own devices. If the parent process already has `CUDA_VISIBLE_DEVICES`
    set, allocation is from that visible list rather than physical GPU 0.
    """

    def __init__(self, total_gpus: int = 8):
        visible_devices = [
            item.strip()
            for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if item.strip()
        ]
        self._gpu_ids = (
            visible_devices if visible_devices else [str(i) for i in range(total_gpus)]
        )
        if total_gpus > len(self._gpu_ids):
            raise RuntimeError(
                f"Requested {total_gpus} GPUs but CUDA_VISIBLE_DEVICES exposes "
                f"only {len(self._gpu_ids)}: {self._gpu_ids}"
            )
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
        gpu_ids = self._gpu_ids[self.next_gpu : self.next_gpu + num_gpus]
        self.next_gpu += num_gpus

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            # Import torch after CUDA_VISIBLE_DEVICES is scoped to this actor.
            # This avoids concurrent import during Monarch worker unpickling.
            import torch  # noqa: F401

        return _bootstrap


def _log_samples(rollouts: list[RolloutOutput]) -> None:
    """Log the first response per rollout group."""
    seen_groups: set[str] = set()
    for rollout in rollouts:
        if rollout.group_id in seen_groups:
            continue
        seen_groups.add(rollout.group_id)
        reward_str = (
            f" reward={rollout.reward:+.3f}" if rollout.reward is not None else ""
        )
        logger.info(f"  [group {rollout.group_id}]{reward_str}")
        for turn in rollout.turns[:1]:
            text = next(
                (
                    str(message.get("content") or "")
                    for message in turn.response_messages
                    if message.get("role") == "assistant"
                ),
                "",
            )
            logger.info(f"       A: {text[:300].replace(chr(10), ' ').strip()}")


async def _raise_rollout_task_errors(
    tasks: list[asyncio.Task[None]],
    *,
    timeout_s: float = 0.0,
) -> None:
    """Surface background rollout producer failures on the trainer path."""
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
    """Rollout drops and associated metrics accumulated between train steps."""

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

    def record_zero_advantage(self, rewards: list[float] | None = None) -> None:
        self.zero_advantage_groups += 1
        self.consecutive_zero_advantage_groups += 1
        if rewards:
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
                f"{self.consecutive_zero_advantage_groups} zero-advantage). "
                "The task may have no reward variation for the current model; "
                "try a harder task, higher sampling temperature, or "
                "--no-drop-zero-advantage-groups for debugging."
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
        """Completed GRPO groups consumed by each optimizer step.

        The total samples per step is ``num_prompts_per_step * rollout_group_size``.
        Async producers may generate these groups before the optimizer step
        that consumes them.
        """

        rollout_group_size: int = 8
        """Number of sampled siblings per prompt for GRPO advantage centering."""

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

        train_dataset: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Dataset config for training rollout groups."""

        train_env_builder: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env builder config for training rollout groups."""

        validation_dataset: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Dataset config for validation rollout groups."""

        validation_env_builder: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env builder config for validation rollout groups."""

        log_samples: bool = False
        """Log first completion per episode during training and validation."""

        save_rollout_samples: bool = False
        """Write bounded rollout conversation samples under ``dump_folder``."""

        max_rollout_sample_groups: int = 2
        """Maximum rollout groups to save per step and phase."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

        renderer: RendererConfig = field(default_factory=RendererConfig)
        """Renderer used for message/token conversion on the controller."""

        max_rollout_turns: int = 1
        """Maximum assistant turns per rollout."""

        step_timeout_s: float | None = 1800.0
        """Timeout for one env step. ``None`` disables timeout."""

        max_trajectory_tokens: int | None = None
        """Optional prompt plus generation-token cap enforced by the controller."""

        async_rollout_groups: int = 1
        """Number of long-lived async rollout producer tasks."""

        max_admitted_generation_prompts: int | None = None
        """Maximum prompts admitted to the generator and awaiting completion.

        ``None`` uses twice the vLLM running-sequence cap, which keeps the
        waiting queue bounded while still letting multi-turn rollouts enqueue
        their next turns before the previous wave fully drains.
        """

        replay_buffer_groups: int = 2
        """Completed-group FIFO capacity for async rollout producers.

        The trainer still consumes ``num_prompts_per_step`` groups per
        optimizer step. This value only controls how far completed rollouts can
        queue ahead before producers block.
        """

        max_offpolicy_steps: int | None = 1
        """Drop replay groups older than this policy-version lag. ``None`` keeps all."""

        drop_zero_advantage_groups: bool = True
        """Drop constant-reward rollout groups before replay admission.

        With this enabled, ``num_steps`` counts optimizer updates. Debug smokes
        on random checkpoints may need ``--no-drop-zero-advantage-groups`` if
        the model produces no reward variation.
        """

        max_no_signal_groups: int | None = 100
        """Fail after this many consecutive producer-side dropped groups.

        Prevents saturated or broken tasks from waiting forever on replay
        admission. ``None`` disables the guard.
        """

        actor_close_timeout_s: float = 30.0
        """Best-effort timeout for actor close endpoints and mesh shutdown."""

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
            for name in (
                "train_dataset",
                "train_env_builder",
                "validation_dataset",
                "validation_env_builder",
            ):
                if getattr(self, name) is None:
                    raise ValueError(f"{name} must be set")
            if self.max_rollout_turns <= 0:
                raise ValueError(
                    f"max_rollout_turns must be positive, got {self.max_rollout_turns}"
                )
            if self.async_rollout_groups <= 0:
                raise ValueError(
                    "async_rollout_groups must be positive, "
                    f"got {self.async_rollout_groups}"
                )
            if (
                self.max_admitted_generation_prompts is not None
                and self.max_admitted_generation_prompts <= 0
            ):
                raise ValueError(
                    "max_admitted_generation_prompts must be positive or None, "
                    f"got {self.max_admitted_generation_prompts}"
                )
            if self.rollout_group_size <= 0:
                raise ValueError(
                    "rollout_group_size must be positive, "
                    f"got {self.rollout_group_size}"
                )
            if self.generator.sampling.n != 1:
                raise ValueError(
                    "RLTrainer uses rollout_group_size for GRPO sibling fanout; "
                    "generator.sampling.n must stay 1 so each rollout turn "
                    f"produces exactly one completion, got {self.generator.sampling.n}"
                )
            TrainingLogprobConfig.from_sampling(self.generator.sampling)
            if self.replay_buffer_groups <= 0:
                raise ValueError(
                    "replay_buffer_groups must be positive, "
                    f"got {self.replay_buffer_groups}"
                )
            if self.max_rollout_sample_groups < 0:
                raise ValueError(
                    "max_rollout_sample_groups must be non-negative, "
                    f"got {self.max_rollout_sample_groups}"
                )
            if self.max_no_signal_groups is not None and self.max_no_signal_groups <= 0:
                raise ValueError(
                    "max_no_signal_groups must be positive or None, got "
                    f"{self.max_no_signal_groups}"
                )
            if self.actor_close_timeout_s <= 0:
                raise ValueError(
                    "actor_close_timeout_s must be positive, "
                    f"got {self.actor_close_timeout_s}"
                )

    def __init__(self, config: Config):
        self.config = config
        self.trainer = None
        self.generator = None
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

    async def close(self):
        """Best-effort: tear down actors, close metric backends, then stop proc meshes."""
        logger.info("Closing: tearing down actors and process meshes.")
        close_timeout_s = getattr(
            getattr(self, "config", None),
            "actor_close_timeout_s",
            30.0,
        )
        for actor_name, actor in (
            ("trainer", self.trainer),
            ("generator", self.generator),
        ):
            if actor is None:
                continue
            try:
                await asyncio.wait_for(
                    actor.close.call(),
                    timeout=close_timeout_s,
                )
            except TimeoutError:
                logger.warning(
                    "%s.close timed out after %.1fs; stopping proc meshes",
                    actor_name,
                    close_timeout_s,
                )
            except Exception:
                logger.exception("%s.close failed", actor_name)

        try:
            self.metrics_processor.close()
        except Exception:
            logger.exception("metrics_processor close failed")

        for i, mesh in enumerate(self._proc_meshes):
            try:
                await asyncio.wait_for(
                    mesh.stop(),
                    timeout=close_timeout_s,
                )
            except TimeoutError:
                logger.warning(
                    "mesh.stop[%d] timed out after %.1fs",
                    i,
                    close_timeout_s,
                )
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)
        self._proc_meshes = []

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

    @staticmethod
    def _compute_world_size(p: ParallelismConfig) -> int:
        """Compute world size from all parallel dimensions."""
        dp_shard = max(p.data_parallel_shard_degree, 1)
        return (
            p.data_parallel_replicate_degree
            * dp_shard
            * p.tensor_parallel_degree
            * p.pipeline_parallel_degree
            * p.context_parallel_degree
        )

    @staticmethod
    def _max_generator_num_seqs(config: Config) -> int:
        """Cap vLLM running sequences, not total admitted scheduler prompts."""
        train_fanout = max(config.async_rollout_groups, 1) * config.rollout_group_size
        validation_fanout = min(
            max(config.num_validation_samples, 1),
            max(config.async_rollout_groups, 1),
        )
        return max(train_fanout, validation_fanout, 1)

    def _spawn_role_meshes(
        self,
        *,
        host_mesh,
        trainer_nodes: int | None,
        generator_nodes: int | None,
        gpus_per_node: int | None,
        total_gpus: int,
    ):
        if host_mesh is None:
            provisioner = Provisioner(total_gpus=total_gpus)
            trainer_mesh = this_host().spawn_procs(
                per_host={"gpus": self.trainer_world_size},
                bootstrap=provisioner.allocate(self.trainer_world_size),
            )
            generator_mesh = this_host().spawn_procs(
                per_host={"gpus": self.generator_world_size},
                bootstrap=provisioner.allocate(self.generator_world_size),
            )
            return trainer_mesh, generator_mesh

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
        return trainer_mesh, generator_mesh

    def _spawn_actors(self, *, trainer_mesh, generator_mesh) -> None:
        config = self.config
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
        self.generator = generator_mesh.spawn(
            "generator",
            VLLMGenerator,
            config.generator,
            model_spec=config.model_spec,
            model_path=config.hf_assets_path,
            compile_config=config.compile,
            max_num_seqs=self._max_generator_num_seqs(config),
            output_dir=config.dump_folder,
        )

    def _shard_samples(self, samples: list[ReplaySample]) -> list[list[ReplaySample]]:
        """Round-robin partition replay samples across DP ranks."""
        return [
            [samples[i] for i in range(rank, len(samples), self.trainer_dp_degree)]
            for rank in range(self.trainer_dp_degree)
        ]

    @staticmethod
    @sl.log_trace_span("_collate_samples")
    def _collate_samples(samples: list[ReplaySample]) -> TrainingBatch:
        """Pack replay samples into a varlen token batch.

        Example::

            sample = ReplaySample(
                token_ids=[10, 11, 20],
                loss_mask=[0, 0, 1],
                behavior_logprobs=[0.0, 0.0, -0.4],
                advantage=0.7,
                group_id="g0",
                sample_idx=0,
                behavior_version=3,
                reward=1.0,
            )
            batch = RLTrainer._collate_samples([sample])
            # batch.seq_lens == [3]
        """
        all_ids: list[int] = []
        all_masks: list[int] = []
        all_behavior_logprobs: list[float] = []
        all_advantages: list[float] = []

        if not samples:
            return TrainingBatch(
                token_ids=torch.zeros((1, 1), dtype=torch.long),
                seq_lens=[1],
                loss_mask=torch.zeros((1, 1), dtype=torch.bool),
                behavior_logprobs=torch.zeros((1, 1), dtype=torch.float32),
                advantages=torch.zeros((1, 1), dtype=torch.float32),
            )

        for sample in samples:
            all_ids.extend(sample.token_ids)
            all_masks.extend(sample.loss_mask)
            all_behavior_logprobs.extend(sample.behavior_logprobs)
            all_advantages.extend(
                sample.advantage if mask else 0.0 for mask in sample.loss_mask
            )

        return TrainingBatch(
            token_ids=torch.tensor([all_ids], dtype=torch.long),
            seq_lens=[len(sample.token_ids) for sample in samples],
            loss_mask=torch.tensor([all_masks], dtype=torch.bool),
            behavior_logprobs=torch.tensor(
                [all_behavior_logprobs], dtype=torch.float32
            ),
            advantages=torch.tensor([all_advantages], dtype=torch.float32),
        )

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

        self.trainer_world_size = self._compute_world_size(config.trainer.parallelism)
        self.generator_world_size = self._compute_world_size(
            config.generator.parallelism
        )
        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        total_gpus = self.trainer_world_size + self.generator_world_size
        logger.info(
            f"{self.generator_world_size} generator GPUs + "
            f"{self.trainer_world_size} trainer GPUs = {total_gpus} total"
        )

        self._multi_node = host_mesh is not None

        with sl.log_trace_span("mesh_spawn"):
            trainer_mesh, generator_mesh = self._spawn_role_meshes(
                host_mesh=host_mesh,
                trainer_nodes=trainer_nodes,
                generator_nodes=generator_nodes,
                gpus_per_node=gpus_per_node,
                total_gpus=total_gpus,
            )
            self._proc_meshes = [trainer_mesh, generator_mesh]

            await setup_torch_elastic_env_async(trainer_mesh)
            await setup_torch_elastic_env_async(generator_mesh)
            self._spawn_actors(trainer_mesh=trainer_mesh, generator_mesh=generator_mesh)

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        with sl.log_trace_span("torchstore_init"):
            await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # Initial weight sync from trainer to generator
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self._await_call(self.trainer.push_model_state_dict.call())
        with sl.log_trace_span("generator_pull_model_state_dict"):
            await self._await_call(self.generator.pull_model_state_dict.call(0))

    async def _await_rank_0(self, actor_call):
        """Await a Monarch call without blocking the controller event loop."""
        result = await self._await_call(actor_call)
        return self._get_rank_0_value(result)

    @staticmethod
    async def _await_call(actor_call):
        """Await a Monarch call without blocking the controller event loop."""
        if hasattr(actor_call, "get"):
            return await asyncio.to_thread(actor_call.get)
        return await actor_call

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
            completions, metrics = await self._await_rank_0(
                self.generator.generate.call(
                    prompt_token_ids_batch,
                    request_ids=request_ids,
                    sampling_config=sampling,
                    metrics_prefix=metrics_prefix,
                )
            )
            return completions, metrics

        max_admitted_prompts = self.config.max_admitted_generation_prompts
        if max_admitted_prompts is None:
            max_admitted_prompts = 2 * self._max_generator_num_seqs(self.config)

        return GenerationScheduler(
            generate_batch,
            max_active_prompts=max_admitted_prompts,
        )

    def _sampling_with_stop_token_ids(self, sampling: SamplingConfig) -> SamplingConfig:
        stop_token_ids = list(self._stop_token_ids)
        if list(sampling.stop_token_ids) == stop_token_ids:
            return sampling
        return dataclasses.replace(sampling, stop_token_ids=stop_token_ids)

    def _token_env_config_for(self, sampling: SamplingConfig) -> TokenEnvConfig:
        return TokenEnvConfig(
            max_trajectory_tokens=self.config.max_trajectory_tokens,
            max_generation_tokens=sampling.max_tokens,
            step_timeout_s=self.config.step_timeout_s,
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
            t_weight_sync_push_start = time.perf_counter()
            with sl.log_trace_span("trainer_push_model_state_dict"):
                await self._await_call(self.trainer.push_model_state_dict.call())
            t_weight_sync_push_s = time.perf_counter() - t_weight_sync_push_start

            t_weight_sync_pull_start = time.perf_counter()
            with sl.log_trace_span("generator_pull_model_state_dict"):
                await self._await_call(
                    self.generator.pull_model_state_dict.call(policy_version)
                )
            t_weight_sync_pull_s = time.perf_counter() - t_weight_sync_pull_start
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
        loss_mean = fwd_bwd_metrics.get("loss/mean", float("nan"))
        nonfinite_log_ratio_frac = fwd_bwd_metrics["loss/ratio/nonfinite_frac"]
        if math.isfinite(loss_mean) and nonfinite_log_ratio_frac <= 0.0:
            return None
        return {
            "train/policy_version": float(policy_version),
            "train/skipped_nonfinite_loss": 1.0,
            "train/skipped_nonfinite_grad_norm": 0.0,
        }

    @staticmethod
    def _optimizer_step_skipped(
        optim_output: OptimStepOutput,
        *,
        previous_policy_version: int,
    ) -> bool:
        """Return whether ``optim_step`` declined to update weights."""
        return (
            optim_output.policy_version == previous_policy_version
            or optim_output.metrics.get("train/skipped_nonfinite_grad_norm", 0.0) > 0.0
        )

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
                metric,
                old_prefix="generator/",
                new_prefix="generator/live/",
            )
            for metric in generation_scheduler.pop_metrics()
        ]
        step_metrics = build_train_step_metrics(
            samples=samples,
            replay_batch=replay_batch,
            rollouts=rollouts,
            live_generation_metrics=live_generation_metrics,
            fwd_bwd_metrics=fwd_bwd_metrics,
            optimizer_metrics=optimizer_metrics,
            checkpoint_saved=checkpoint_saved,
            timings=timings,
            dropped_empty_groups=dropped_empty_groups,
            dropped_zero_advantage_groups=dropped_zero_advantage_groups,
            drop_metrics=drop_metrics,
            train_version=train_version,
        )
        trace_scalars = _build_train_step_trace_scalars(
            replay_batch=replay_batch,
            fwd_bwd_metrics=fwd_bwd_metrics,
            optimizer_metrics=optimizer_metrics,
            checkpoint_saved=checkpoint_saved,
            timings=timings,
            dropped_empty_groups=dropped_empty_groups,
            dropped_zero_advantage_groups=dropped_zero_advantage_groups,
            train_version=train_version,
        )
        sl.log_trace_scalar(trace_scalars)
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
        metrics_prefix: str = "generator",
    ) -> tuple[list[RolloutOutput], list[m.Metric]]:
        """Collect a finite set of rollout groups."""
        sampling = self._sampling_with_stop_token_ids(sampling)
        generation_scheduler = self._make_generation_scheduler(
            metrics_prefix=metrics_prefix,
        )
        token_env_config = self._token_env_config_for(sampling)
        examples = [
            env_dataset.sample_group(sample_step=sample_step, group_idx=group_idx)
            for group_idx in range(num_groups)
        ]
        pending: set[asyncio.Task[list[RolloutOutput]]] = set()
        rollouts: list[RolloutOutput] = []
        next_idx = 0
        max_inflight = max(self.config.async_rollout_groups, 1)

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

        rollout_metrics = build_rollout_metrics(
            rollouts,
            generation_metrics=generation_scheduler.pop_metrics(),
            prefix="rollout",
        )
        return rollouts, rollout_metrics

    async def _continuous_rollouts(
        self,
        *,
        worker_idx: int,
        replay_buffer: ReplayBuffer,
        generation_scheduler: GenerationScheduler,
        drop_counters: _RolloutDropCounters,
        shutdown: asyncio.Event,
        next_example: Callable[[], Awaitable[EnvExample]],
    ) -> None:
        """Produce rollout groups until training finishes or a producer fails."""
        sampling = self._sampling_with_stop_token_ids(self.config.generator.sampling)
        group_size = self.config.rollout_group_size
        token_env_config = self._token_env_config_for(sampling)

        try:
            while not shutdown.is_set():
                example = await next_example()
                with sl.log_trace_span("rollout_group"):
                    group_rollouts = await run_rollout_group(
                        env_builder=self.train_env_builder,
                        example=example,
                        group_size=group_size,
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
                    sl.log_trace_scalar(
                        {
                            "rollout.dropped_empty_groups": 1,
                            "rollout.sample_step": example.sample_step,
                        }
                    )
                    continue
                has_signal = has_advantage_signal(samples, eps=_ZERO_ADVANTAGE_EPS)
                if self.config.drop_zero_advantage_groups and not has_signal:
                    dropped_rewards = [
                        float(rollout.reward)
                        for rollout in group_rollouts
                        if rollout.reward is not None
                    ]
                    drop_counters.record_zero_advantage(dropped_rewards)
                    sl.log_trace_scalar(
                        {
                            "rollout.dropped_zero_advantage_groups": 1,
                            "rollout.sample_step": example.sample_step,
                        }
                    )
                    continue

                replay_group = ReplayGroup.from_rollouts(
                    samples=samples,
                    rollouts=group_rollouts,
                )
                sl.log_trace_scalar(
                    {
                        "rollout.behavior_version": replay_group.behavior_version,
                        "rollout.max_behavior_version": (
                            replay_group.max_behavior_version
                        ),
                        "rollout.sample_step": example.sample_step,
                        "rollout.num_samples": len(samples),
                    }
                )
                drop_counters.record_admitted()
                with sl.log_trace_span("replay_buffer_put"):
                    await replay_buffer.put(replay_group)
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
    async def validate(
        self,
        *,
        log_step: int = 0,
        phase: str = "validation",
    ) -> list[m.Metric]:
        """Run finite greedy rollout collection for validation."""
        t_validate_start = time.perf_counter()
        num_samples = self.config.num_validation_samples
        validation_sample_step = 0
        greedy = SamplingConfig(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.config.generator.sampling.max_tokens,
            stop_token_ids=list(self._stop_token_ids),
        )
        rollouts, validation_metrics = await self._collect_finite_rollouts(
            env_dataset=self.validation_dataset,
            env_builder=self.validation_env_builder,
            num_groups=num_samples,
            group_size=1,
            sample_step=validation_sample_step,
            sampling=greedy,
            metrics_prefix="validation/generator",
        )

        if self.config.log_samples:
            _log_samples(rollouts)
        if self.rollout_sample_logger is not None:
            self.rollout_sample_logger.write(
                step=log_step,
                phase=phase,
                rollouts=rollouts,
            )

        t_validate_s = time.perf_counter() - t_validate_start
        return [
            rename_metric_prefix(
                metric,
                old_prefix="rollout/",
                new_prefix="validation/",
            )
            for metric in validation_metrics
        ] + [
            m.Metric("validation/num_samples", m.NoReduce(float(len(rollouts)))),
            m.Metric("timing/validate", m.NoReduce(t_validate_s)),
        ]

    async def train(self):
        num_steps = self.config.num_steps
        logprob_config = TrainingLogprobConfig.from_sampling(
            self.config.generator.sampling
        )
        logger.info(f"Pre-training validation; then {num_steps} steps of RL training")

        pre_validation_metrics = await self.validate(
            log_step=0,
            phase="pre_validation",
        )
        self.metrics_processor.log(
            step=0,
            metrics=pre_validation_metrics,
            is_validation=True,
        )
        pre_validation_agg = m.MetricsProcessor._aggregate_metrics(
            pre_validation_metrics
        )

        sl.log_trace_instant("training_start")

        replay_buffer = ReplayBuffer(
            max_groups=max(self.config.replay_buffer_groups, 1),
            max_age_steps=self.config.max_offpolicy_steps,
        )
        generation_scheduler = self._make_generation_scheduler(
            metrics_prefix="generator",
        )
        drop_counters = _RolloutDropCounters(
            max_no_signal_groups=self.config.max_no_signal_groups,
        )
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
                    shutdown=shutdown,
                    next_example=next_example,
                )
            )
            for worker_idx in range(max(self.config.async_rollout_groups, 1))
        ]

        policy_version = 0
        try:
            for step in range(1, num_steps + 1):
                sl.set_step(step)
                await self._await_call(self.trainer.sync_log_step.call(step))
                await self._await_call(self.generator.sync_log_step.call(step))

                t_step_start = time.perf_counter()
                t_buffer_start = time.perf_counter()
                batch_train_version = policy_version
                with sl.log_trace_span("replay_buffer_get_batch"):
                    replay_batch = await replay_buffer.get_batch(
                        min_groups=self.config.num_prompts_per_step,
                        train_version=batch_train_version,
                    )
                t_buffer_wait_s = time.perf_counter() - t_buffer_start
                await _raise_rollout_task_errors(rollout_tasks)

                drop_stats = drop_counters.pop()
                samples = replay_batch.samples
                rollouts = [
                    rollout
                    for group in replay_batch.groups
                    for rollout in group.rollouts
                ]
                if self.config.log_samples:
                    _log_samples(rollouts)
                if not samples:
                    await _raise_rollout_task_errors(
                        rollout_tasks,
                        timeout_s=1.0,
                    )
                    raise RuntimeError(
                        "replay buffer closed before producing trainable samples"
                    )

                t_train_start = time.perf_counter()
                batches = [
                    self._collate_samples(per_rank_samples)
                    for per_rank_samples in self._shard_samples(samples)
                ]
                num_global_valid_tokens = sum(
                    sample.num_loss_tokens for sample in samples
                )
                with sl.log_trace_span("trainer_forward_backward_call"):
                    fwd_bwd_metrics = await self._await_rank_0(
                        self.trainer.forward_backward.call(
                            batches,
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
                        "Step %d skipped optimizer step because "
                        "forward/backward metrics were not usable: %s",
                        step,
                        fwd_bwd_metrics,
                    )
                    self._log_train_step(
                        step=step,
                        samples=samples,
                        replay_batch=replay_batch,
                        rollouts=rollouts,
                        fwd_bwd_metrics=fwd_bwd_metrics,
                        optimizer_metrics=skip_metrics,
                        checkpoint_saved=False,
                        generation_scheduler=generation_scheduler,
                        timings=_TrainStepTimings(
                            step_s=t_step_s,
                            replay_wait_s=t_buffer_wait_s,
                            train_s=t_train_s,
                            checkpoint_s=0.0,
                            weight_sync=_zero_weight_sync_timings(),
                        ),
                        dropped_empty_groups=drop_stats.empty_groups,
                        dropped_zero_advantage_groups=drop_stats.zero_advantage_groups,
                        drop_metrics=drop_stats.metrics,
                        train_version=batch_train_version,
                    )
                    continue
                with sl.log_trace_span("trainer_optim_step_call"):
                    optim_output = await self._await_rank_0(
                        self.trainer.optim_step.call()
                    )
                trainer_policy_version = optim_output.policy_version
                optimizer_metrics = {
                    **optim_output.metrics,
                    "train/skipped_nonfinite_loss": 0.0,
                }
                t_train_s = time.perf_counter() - t_train_start

                if self._optimizer_step_skipped(
                    optim_output,
                    previous_policy_version=policy_version,
                ):
                    t_step_s = time.perf_counter() - t_step_start
                    logger.warning(
                        "Step %d skipped weight sync and checkpoint because "
                        "optimizer step did not publish a new policy version",
                        step,
                    )
                    self._log_train_step(
                        step=step,
                        samples=samples,
                        replay_batch=replay_batch,
                        rollouts=rollouts,
                        generation_scheduler=generation_scheduler,
                        fwd_bwd_metrics=fwd_bwd_metrics,
                        optimizer_metrics=optimizer_metrics,
                        checkpoint_saved=False,
                        timings=_TrainStepTimings(
                            step_s=t_step_s,
                            replay_wait_s=t_buffer_wait_s,
                            train_s=t_train_s,
                            checkpoint_s=0.0,
                            weight_sync=_zero_weight_sync_timings(),
                        ),
                        dropped_empty_groups=drop_stats.empty_groups,
                        dropped_zero_advantage_groups=drop_stats.zero_advantage_groups,
                        drop_metrics=drop_stats.metrics,
                        train_version=batch_train_version,
                    )
                    continue

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
                    fwd_bwd_metrics=fwd_bwd_metrics,
                    optimizer_metrics=optimizer_metrics,
                    checkpoint_saved=checkpoint_saved,
                    generation_scheduler=generation_scheduler,
                    timings=_TrainStepTimings(
                        step_s=t_step_s,
                        replay_wait_s=t_buffer_wait_s,
                        train_s=t_train_s,
                        checkpoint_s=t_checkpoint_s,
                        weight_sync=weight_sync_timings,
                    ),
                    dropped_empty_groups=drop_stats.empty_groups,
                    dropped_zero_advantage_groups=drop_stats.zero_advantage_groups,
                    drop_metrics=drop_stats.metrics,
                    train_version=batch_train_version,
                )
        finally:
            shutdown.set()
            await replay_buffer.close()
            for task in rollout_tasks:
                task.cancel()
            await generation_scheduler.close()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        post_validation_metrics = await self.validate(
            log_step=num_steps,
            phase="post_validation",
        )
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

    rl_trainer = RLTrainer(config)
    try:
        await rl_trainer.setup_async()
        await rl_trainer.train()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await rl_trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
