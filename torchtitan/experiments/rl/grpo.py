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
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.observability.metrics.rl import (
    build_replay_metrics,
    build_rollout_metrics,
    rename_metric_prefix,
)
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.replay import (
    has_advantage_signal,
    ReplayBuffer,
    ReplayGroup,
    rollouts_to_replay_samples,
)
from torchtitan.experiments.rl.rollout_logging import RolloutSampleLogger
from torchtitan.experiments.rl.rollout_runner import (
    RolloutGroupResult,
    run_rollout_group,
)
from torchtitan.experiments.rl.sampling import SamplingConfig
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


class GRPOLoss(Configurable):
    """Clipped GRPO surrogate loss.

    Takes token-selected policy logprobs, behavior logprobs, and advantages.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_eps: float = 0.2
        """PPO clipping epsilon for the probability ratio."""

    def __init__(self, config: Config):
        self.clip_eps = config.clip_eps

    def __call__(
        self,
        policy_logprobs: torch.Tensor,
        behavior_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        num_global_valid_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ratio = torch.exp(policy_logprobs - behavior_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        # pg = policy gradient.
        token_pg_losses = -torch.min(ratio * advantages, clipped_ratio * advantages)
        pg_loss = token_pg_losses.sum() / num_global_valid_tokens

        with torch.no_grad():
            clipped_frac = (torch.abs(ratio - clipped_ratio) > 1e-6).to(ratio.dtype)
            loss_metrics = {
                "loss/mean": pg_loss.detach(),
                "loss/ratio/mean": ratio.sum() / num_global_valid_tokens,
                "loss/ratio/clipped_frac": clipped_frac.sum() / num_global_valid_tokens,
            }

        return pg_loss, loss_metrics


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
            # TODO: Remove once Monarch/PyTorch fixes concurrent import during unpickling.
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


def _raise_rollout_task_errors(tasks: list[asyncio.Task[None]]) -> None:
    """Surface background rollout producer failures on the trainer path."""
    for task in tasks:
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                raise exc


@dataclass(slots=True)
class _RolloutDropCounters:
    """Producer-side rollout drops accumulated between optimizer steps."""

    empty_groups: int = 0
    zero_advantage_groups: int = 0

    def pop(self) -> tuple[int, int]:
        values = (self.empty_groups, self.zero_advantage_groups)
        self.empty_groups = 0
        self.zero_advantage_groups = 0
        return values


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
        """Number of distinct prompts (= GRPO groups) drawn per training step.

        The total samples per step is ``num_prompts_per_step * rollout_group_size``.
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
        """Number of rollout groups kept in flight by the async producer."""

        replay_buffer_groups: int = 2
        """Maximum completed rollout groups buffered for async training."""

        max_offpolicy_steps: int | None = 1
        """Drop queued groups older than this many policy versions. ``None`` keeps all."""

        drop_zero_advantage_groups: bool = True
        """Drop constant-reward rollout groups before replay admission.

        With this enabled, ``num_steps`` counts optimizer updates. Debug smokes
        on random checkpoints may need ``--no-drop-zero-advantage-groups`` if
        the model produces no reward variation.
        """

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
        for actor_name, actor in (
            ("trainer", self.trainer),
            ("generator", self.generator),
        ):
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
        """Upper-bound one controller scheduler flush into the vLLM engine."""
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
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            completions, metrics = await self._await_rank_0(
                self.generator.generate.call(
                    prompt_token_ids_batch,
                    sampling_config=sampling,
                    metrics_prefix=metrics_prefix,
                )
            )
            return completions, metrics

        return GenerationScheduler(generate_batch)

    def _make_completion_fn(
        self,
        *,
        generation_scheduler: GenerationScheduler,
    ):
        async def completion_fn(
            prompt_token_ids: list[int],
            turn_sampling: SamplingConfig,
            request_id: str,
        ) -> Completion:
            return await generation_scheduler.submit(
                prompt_token_ids=prompt_token_ids,
                sampling=turn_sampling,
                request_id=request_id,
            )

        return completion_fn

    def _sampling_with_stop_token_ids(self, sampling: SamplingConfig) -> SamplingConfig:
        stop_token_ids = list(self._stop_token_ids)
        if list(sampling.stop_token_ids) == stop_token_ids:
            return sampling
        return dataclasses.replace(sampling, stop_token_ids=stop_token_ids)

    async def _collect_validation_rollouts(
        self,
        *,
        env_dataset: EnvDataset,
        env_builder: EnvBuilder,
        num_groups: int,
        group_size: int,
        step: int,
        sampling: SamplingConfig,
        metrics_prefix: str = "generator",
    ) -> tuple[list[RolloutOutput], list[m.Metric]]:
        """Collect a finite set of validation rollout groups."""
        sampling = self._sampling_with_stop_token_ids(sampling)
        generation_scheduler = self._make_generation_scheduler(
            metrics_prefix=metrics_prefix,
        )
        completion_fn = self._make_completion_fn(
            generation_scheduler=generation_scheduler,
        )
        token_env_config = TokenEnvConfig(
            max_trajectory_tokens=self.config.max_trajectory_tokens,
            max_generation_tokens=sampling.max_tokens,
            step_timeout_s=self.config.step_timeout_s,
        )
        examples = [
            env_dataset.sample_group(step=step, group_idx=group_idx)
            for group_idx in range(num_groups)
        ]
        pending: set[asyncio.Task[RolloutGroupResult]] = set()
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
                                completion_fn=completion_fn,
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
                    rollouts.extend(task.result().rollouts)
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
        completion_fn = self._make_completion_fn(
            generation_scheduler=generation_scheduler,
        )
        token_env_config = TokenEnvConfig(
            max_trajectory_tokens=self.config.max_trajectory_tokens,
            max_generation_tokens=sampling.max_tokens,
            step_timeout_s=self.config.step_timeout_s,
        )

        try:
            while not shutdown.is_set():
                example = await next_example()
                with sl.log_trace_span("rollout_group"):
                    result = await run_rollout_group(
                        env_builder=self.train_env_builder,
                        example=example,
                        group_size=group_size,
                        renderer=self.renderer,
                        completion_fn=completion_fn,
                        sampling=sampling,
                        max_turns=self.config.max_rollout_turns,
                        token_env_config=token_env_config,
                    )
                group_rollouts = result.rollouts

                if self.rollout_sample_logger is not None:
                    self.rollout_sample_logger.write(
                        step=example.step,
                        phase="train_rollout",
                        rollouts=group_rollouts,
                    )

                samples = rollouts_to_replay_samples(group_rollouts)
                if shutdown.is_set():
                    return
                if not samples:
                    drop_counters.empty_groups += 1
                    sl.log_trace_scalar(
                        {
                            "rollout.dropped_empty_groups": 1,
                            "rollout.sample_step": example.step,
                        }
                    )
                    continue
                has_signal = has_advantage_signal(samples, eps=_ZERO_ADVANTAGE_EPS)
                if self.config.drop_zero_advantage_groups and not has_signal:
                    drop_counters.zero_advantage_groups += 1
                    sl.log_trace_scalar(
                        {
                            "rollout.dropped_zero_advantage_groups": 1,
                            "rollout.sample_step": example.step,
                        }
                    )
                    continue

                if (
                    result.behavior_version is None
                    or result.max_behavior_version is None
                ):
                    drop_counters.empty_groups += 1
                    sl.log_trace_scalar(
                        {
                            "rollout.dropped_empty_groups": 1,
                            "rollout.sample_step": example.step,
                        }
                    )
                    continue
                behavior_version = result.behavior_version
                max_behavior_version = result.max_behavior_version
                sl.log_trace_scalar(
                    {
                        "rollout.behavior_version": behavior_version,
                        "rollout.max_behavior_version": max_behavior_version,
                        "rollout.sample_step": example.step,
                        "rollout.num_samples": len(samples),
                    }
                )
                with sl.log_trace_span("replay_buffer_put"):
                    await replay_buffer.put(
                        ReplayGroup(
                            group_id=example.group_id,
                            samples=samples,
                            rollouts=group_rollouts,
                            behavior_version=behavior_version,
                            max_behavior_version=max_behavior_version,
                        )
                    )
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
        step: int = 0,
        phase: str = "validation",
    ) -> list[m.Metric]:
        """Run finite greedy rollout collection for validation."""
        t_validate_start = time.perf_counter()
        num_samples = self.config.num_validation_samples
        greedy = SamplingConfig(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.config.generator.sampling.max_tokens,
            stop_token_ids=list(self._stop_token_ids),
        )
        rollouts, validation_metrics = await self._collect_validation_rollouts(
            env_dataset=self.validation_dataset,
            env_builder=self.validation_env_builder,
            num_groups=num_samples,
            group_size=1,
            step=0,
            sampling=greedy,
            metrics_prefix="validation/generator",
        )

        if self.config.log_samples:
            _log_samples(rollouts)
        if self.rollout_sample_logger is not None:
            self.rollout_sample_logger.write(
                step=step,
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
        logger.info(f"Pre-training validation; then {num_steps} steps of RL training")

        pre_validation_metrics = await self.validate(step=0, phase="pre_validation")
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
        drop_counters = _RolloutDropCounters()
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
                step=sample_step,
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
                batch_min_samples = (
                    self.config.num_prompts_per_step * self.config.rollout_group_size
                )
                with sl.log_trace_span("replay_buffer_get_batch"):
                    replay_batch = await replay_buffer.get_batch(
                        min_samples=batch_min_samples,
                        train_version=batch_train_version,
                    )
                t_buffer_wait_s = time.perf_counter() - t_buffer_start
                _raise_rollout_task_errors(rollout_tasks)

                (
                    dropped_empty_groups,
                    dropped_zero_advantage_groups,
                ) = drop_counters.pop()
                samples = replay_batch.samples
                rollouts = [
                    rollout
                    for group in replay_batch.groups
                    for rollout in group.rollouts
                ]
                if self.config.log_samples:
                    _log_samples(rollouts)
                if not samples:
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
                        )
                    )
                with sl.log_trace_span("trainer_optim_step_call"):
                    optim_output = await self._await_rank_0(
                        self.trainer.optim_step.call()
                    )
                trainer_policy_version = optim_output.policy_version
                optimizer_metrics = optim_output.metrics
                t_train_s = time.perf_counter() - t_train_start

                t_weight_sync_start = time.perf_counter()
                with sl.log_trace_span("weight_sync_admission_drain"):
                    await generation_scheduler.pause_for_weight_sync()
                t_weight_sync_drain_s = time.perf_counter() - t_weight_sync_start
                try:
                    t_weight_sync_push_start = time.perf_counter()
                    with sl.log_trace_span("trainer_push_model_state_dict"):
                        await self._await_call(
                            self.trainer.push_model_state_dict.call()
                        )
                    t_weight_sync_push_s = (
                        time.perf_counter() - t_weight_sync_push_start
                    )
                    t_weight_sync_pull_start = time.perf_counter()
                    with sl.log_trace_span("generator_pull_model_state_dict"):
                        await self._await_call(
                            self.generator.pull_model_state_dict.call(
                                trainer_policy_version
                            )
                        )
                    t_weight_sync_pull_s = (
                        time.perf_counter() - t_weight_sync_pull_start
                    )
                finally:
                    await generation_scheduler.resume_after_weight_sync()
                policy_version = trainer_policy_version
                t_weight_sync_total_s = time.perf_counter() - t_weight_sync_start

                if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                    logger.error("Loss is NaN/Inf; training diverged")
                    break

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

                total_tokens = sum(len(sample.token_ids) for sample in samples)
                live_generation_metrics = [
                    rename_metric_prefix(
                        metric,
                        old_prefix="generator/",
                        new_prefix="generator/live/",
                    )
                    for metric in generation_scheduler.pop_metrics()
                ]
                rollout_metrics = build_rollout_metrics(
                    rollouts,
                    generation_metrics=[],
                    prefix="rollout",
                )
                behavior_versions = [
                    group.behavior_version for group in replay_batch.groups
                ]
                max_behavior_versions = [
                    group.max_behavior_version for group in replay_batch.groups
                ]
                behavior_version_min = (
                    min(behavior_versions) if behavior_versions else 0
                )
                behavior_version_max = (
                    max(max_behavior_versions) if max_behavior_versions else 0
                )

                step_metrics: list[m.Metric] = []
                step_metrics += rollout_metrics
                step_metrics += live_generation_metrics
                step_metrics += build_replay_metrics(
                    samples,
                    replay_batch.stats,
                    dropped_empty_groups=dropped_empty_groups,
                    dropped_zero_advantage_groups=dropped_zero_advantage_groups,
                )
                step_metrics += [
                    m.Metric(
                        "replay/policy_version/train",
                        m.NoReduce(float(batch_train_version)),
                    ),
                    m.Metric(
                        "replay/policy_version/behavior_min",
                        m.NoReduce(float(behavior_version_min)),
                    ),
                    m.Metric(
                        "replay/policy_version/behavior_max",
                        m.NoReduce(float(behavior_version_max)),
                    ),
                ]
                step_metrics += [
                    m.Metric(k, m.NoReduce(v)) for k, v in fwd_bwd_metrics.items()
                ]
                step_metrics += [
                    m.Metric(k, m.NoReduce(v)) for k, v in optimizer_metrics.items()
                ]
                step_metrics.append(
                    m.Metric("checkpoint/saved", m.NoReduce(float(checkpoint_saved)))
                )
                for key, value in [
                    ("timing/step", t_step_s),
                    ("timing/replay_wait", t_buffer_wait_s),
                    ("timing/train", t_train_s),
                    ("timing/weight_sync/admission_drain", t_weight_sync_drain_s),
                    ("timing/weight_sync/push", t_weight_sync_push_s),
                    ("timing/weight_sync/pull", t_weight_sync_pull_s),
                    ("timing/weight_sync/total", t_weight_sync_total_s),
                    ("timing/checkpoint", t_checkpoint_s),
                ]:
                    step_metrics.append(m.Metric(key, m.NoReduce(value)))
                step_metrics.append(
                    m.Metric(
                        "perf/tokens_per_second",
                        m.NoReduce(total_tokens / t_step_s),
                    )
                )
                sl.log_trace_scalar(
                    {
                        "replay.buffer_depth_groups": replay_batch.stats.depth_groups,
                        "replay.dropped_stale_groups": (
                            replay_batch.stats.num_dropped_stale_groups
                        ),
                        "rollout.dropped_empty_groups": dropped_empty_groups,
                        "rollout.dropped_zero_advantage_groups": (
                            dropped_zero_advantage_groups
                        ),
                        "replay.train_version": batch_train_version,
                        "replay.behavior_version_min": behavior_version_min,
                        "replay.behavior_version_max": behavior_version_max,
                        "timing.replay_wait_ms": t_buffer_wait_s * 1000,
                        "timing.weight_sync_admission_drain_ms": (
                            t_weight_sync_drain_s * 1000
                        ),
                        "timing.weight_sync_push_ms": t_weight_sync_push_s * 1000,
                        "timing.weight_sync_pull_ms": t_weight_sync_pull_s * 1000,
                        "timing.checkpoint_ms": t_checkpoint_s * 1000,
                        "checkpoint.saved": float(checkpoint_saved),
                    }
                )

                self.metrics_processor.log(
                    step=step,
                    metrics=step_metrics,
                    is_validation=False,
                )
        finally:
            shutdown.set()
            await replay_buffer.close()
            for task in rollout_tasks:
                task.cancel()
            await generation_scheduler.close()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        post_validation_metrics = await self.validate(
            step=num_steps,
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
