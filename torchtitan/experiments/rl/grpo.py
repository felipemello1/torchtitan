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
import statistics
import time
from collections import defaultdict
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
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.envs import EnvExample, TokenEnvConfig
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.replay import (
    QueueStats,
    ReplayGroup,
    RolloutQueue,
    rollouts_to_replay_samples,
)
from torchtitan.experiments.rl.rollouts import do_rollout_group
from torchtitan.experiments.rl.types import (
    Completion,
    ReplaySample,
    RolloutOutput,
    RolloutStatus,
    TrainingBatch,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


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


def _prepare_reward_metrics(
    prefix: str,
    rollouts: list[RolloutOutput],
) -> list[m.Metric]:
    """One ``Mean`` metric per observed reward component across rollouts.

    Example::

        rollouts = [
            RolloutOutput(group_id="g0", sample_idx=0, reward=1.5,
                          reward_components={"correctness": 1.0, "format": 0.5}),
            RolloutOutput(group_id="g1", sample_idx=0, reward=0.0,
                          reward_components={"correctness": 0.0}),
        ]
        _prepare_reward_metrics("reward/component", rollouts)
        # -> [
        #      Metric("reward/component/correctness", Mean(sum=1.0, count=2)),
        #      Metric("reward/component/format",      Mean(sum=0.5, count=1)),
        #    ]
    """
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        for name, value in rollout.reward_components.items():
            values_by_name[name].append(float(value))
    return [
        m.Metric(f"{prefix}/{name}", m.Mean.from_list(values))
        for name, values in sorted(values_by_name.items())
    ]


def _rename_metric(metric: m.Metric, *, old_prefix: str, new_prefix: str) -> m.Metric:
    """Replace a metric key prefix while preserving its value object."""
    if metric.key.startswith(old_prefix):
        return m.Metric(new_prefix + metric.key[len(old_prefix) :], metric.value)
    if metric.key == "reward":
        return m.Metric(f"{new_prefix.rstrip('/')}/reward", metric.value)
    if metric.key.startswith("reward/"):
        return m.Metric(f"{new_prefix.rstrip('/')}/{metric.key}", metric.value)
    return metric


@dataclass(slots=True)
class _PendingCompletion:
    """One controller-side generation request awaiting a batched flush."""

    prompt_token_ids: list[int]
    sampling: SamplingConfig
    request_id: str
    future: asyncio.Future[Completion]


def _sampling_key(
    sampling: SamplingConfig,
) -> tuple[float, float, int, tuple[int, ...]]:
    return (
        sampling.temperature,
        sampling.top_p,
        sampling.max_tokens,
        tuple(sampling.stop_token_ids),
    )


class _CompletionBatcher:
    """Coalesce concurrent rollout requests into batched generator calls.

    The controller owns batching so the generator actor receives one ordered
    ``generate`` call per flush on every TP rank. That avoids relying on
    concurrent Monarch endpoint delivery order while still letting rollout
    tasks await completions independently.
    """

    def __init__(
        self,
        generate_batch: Callable[
            [list[list[int]], SamplingConfig],
            Awaitable[list[Completion]],
        ],
    ):
        self._generate_batch = generate_batch
        self._pending: list[_PendingCompletion] = []
        self._flush_task: asyncio.Task[None] | None = None

    async def submit(
        self,
        *,
        prompt_token_ids: list[int],
        sampling: SamplingConfig,
        request_id: str,
    ) -> Completion:
        if sampling.n != 1:
            raise ValueError(f"_CompletionBatcher requires n=1, got {sampling.n}")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Completion] = loop.create_future()
        self._pending.append(
            _PendingCompletion(
                prompt_token_ids=list(prompt_token_ids),
                sampling=sampling,
                request_id=request_id,
                future=future,
            )
        )
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_loop())
        return await future

    async def _flush_loop(self) -> None:
        await asyncio.sleep(0)
        while self._pending:
            batch = self._pending
            self._pending = []
            pending_by_sampling: dict[
                tuple[float, float, int, tuple[int, ...]],
                list[_PendingCompletion],
            ] = defaultdict(list)
            for pending in batch:
                pending_by_sampling[_sampling_key(pending.sampling)].append(pending)

            for pending_group in pending_by_sampling.values():
                await self._flush_group(pending_group)
            await asyncio.sleep(0)

    async def _flush_group(self, pending_group: list[_PendingCompletion]) -> None:
        sampling = pending_group[0].sampling
        try:
            completions = await self._generate_batch(
                [pending.prompt_token_ids for pending in pending_group],
                sampling,
            )
            if len(completions) != len(pending_group):
                raise RuntimeError(
                    "generator returned "
                    f"{len(completions)} completions for "
                    f"{len(pending_group)} requests"
                )
            for pending, completion in zip(
                pending_group,
                completions,
                strict=True,
            ):
                if not pending.future.done():
                    pending.future.set_result(completion)
        except Exception as exc:
            for pending in pending_group:
                if not pending.future.done():
                    pending.future.set_exception(exc)


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

        The total samples per step is `num_prompts_per_step` * `group_size`,
        where `group_size` is `generator.sampling.n` (completions per prompt).
        """

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

        env: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env config for training rollouts."""

        validation_env: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env config for validation rollouts."""

        log_samples: bool = False
        """Log first completion per episode during training and validation."""

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

        rollout_queue_groups: int = 2
        """Maximum completed rollout groups buffered before training consumes them."""

        max_offpolicy_steps: int | None = 1
        """Drop queued groups older than this many policy versions. ``None`` keeps all."""

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
            if self.max_rollout_turns <= 0:
                raise ValueError(
                    f"max_rollout_turns must be positive, got {self.max_rollout_turns}"
                )
            if self.async_rollout_groups <= 0:
                raise ValueError(
                    "async_rollout_groups must be positive, "
                    f"got {self.async_rollout_groups}"
                )
            if self.rollout_queue_groups <= 0:
                raise ValueError(
                    "rollout_queue_groups must be positive, "
                    f"got {self.rollout_queue_groups}"
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
                advantages=[0.0, 0.0, 0.7],
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
                behavior_versions=torch.zeros((0,), dtype=torch.int32),
                rewards=torch.zeros((0,), dtype=torch.float32),
            )

        for sample in samples:
            all_ids.extend(sample.token_ids)
            all_masks.extend(sample.loss_mask)
            all_behavior_logprobs.extend(sample.behavior_logprobs)
            all_advantages.extend(sample.advantages)

        return TrainingBatch(
            token_ids=torch.tensor([all_ids], dtype=torch.long),
            seq_lens=[len(sample.token_ids) for sample in samples],
            loss_mask=torch.tensor([all_masks], dtype=torch.bool),
            behavior_logprobs=torch.tensor(
                [all_behavior_logprobs], dtype=torch.float32
            ),
            advantages=torch.tensor([all_advantages], dtype=torch.float32),
            behavior_versions=torch.tensor(
                [sample.behavior_version for sample in samples],
                dtype=torch.int32,
            ),
            rewards=torch.tensor(
                [sample.reward for sample in samples], dtype=torch.float32
            ),
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

        # TODO(observability): the mesh_spawn span wraps ~80 LoC of branching
        # provisioner logic. Pull a Provisioner.spawn_meshes(...) helper and
        # shrink this span to a single call.
        with sl.log_trace_span("mesh_spawn"):
            if host_mesh is not None:
                # Multi-node mode: dedicate whole nodes to trainer vs generator
                if (
                    trainer_nodes is None
                    or generator_nodes is None
                    or gpus_per_node is None
                ):
                    raise ValueError(
                        "trainer_nodes, generator_nodes, and gpus_per_node are "
                        "required when host_mesh is provided"
                    )
                # Validate that world sizes are evenly divisible by node counts
                assert self.trainer_world_size % trainer_nodes == 0, (
                    f"trainer_world_size ({self.trainer_world_size}) must be "
                    f"evenly divisible by trainer_nodes ({trainer_nodes})"
                )
                assert self.generator_world_size % generator_nodes == 0, (
                    f"generator_world_size ({self.generator_world_size}) must be "
                    f"evenly divisible by generator_nodes ({generator_nodes})"
                )

                # Compute GPUs per node for each role based on the config's
                # world size and number of nodes allocated to that role
                trainer_gpus_per_node = self.trainer_world_size // trainer_nodes
                generator_gpus_per_node = self.generator_world_size // generator_nodes

                trainer_host_mesh = host_mesh.slice(hosts=slice(0, trainer_nodes))
                generator_host_mesh = host_mesh.slice(
                    hosts=slice(trainer_nodes, trainer_nodes + generator_nodes)
                )

                # Use Provisioner to set CUDA_VISIBLE_DEVICES so each role
                # only sees its own GPUs and doesn't conflict with other
                # processes on the node
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
            else:
                # Single-node mode: partition GPUs on this_host() via
                # CUDA_VISIBLE_DEVICES
                provisioner = Provisioner(total_gpus=total_gpus)
                trainer_mesh = this_host().spawn_procs(
                    per_host={"gpus": self.trainer_world_size},
                    bootstrap=provisioner.allocate(self.trainer_world_size),
                )
                generator_mesh = this_host().spawn_procs(
                    per_host={"gpus": self.generator_world_size},
                    bootstrap=provisioner.allocate(self.generator_world_size),
                )

            # Store proc meshes for cleanup
            self._proc_meshes = [trainer_mesh, generator_mesh]

            await setup_torch_elastic_env_async(trainer_mesh)
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

            self.generator = generator_mesh.spawn(
                "generator",
                VLLMGenerator,
                config.generator,
                model_spec=config.model_spec,
                model_path=config.hf_assets_path,
                compile_config=config.compile,
                max_num_seqs=max(
                    config.num_prompts_per_step * config.generator.sampling.n,
                    config.num_validation_samples,
                ),
                output_dir=config.dump_folder,
            )

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
            self.trainer.push_model_state_dict.call().get()
        with sl.log_trace_span("generator_pull_model_state_dict"):
            self.generator.pull_model_state_dict.call(0).get()

    async def _await_rank_0(self, actor_call):
        """Await a Monarch call without blocking the controller event loop."""
        if hasattr(actor_call, "get"):
            result = await asyncio.to_thread(actor_call.get)
        else:
            result = await actor_call
        return self._get_rank_0_value(result)

    def _make_completion_batcher(
        self,
        *,
        metrics_prefix: str,
        generation_metrics: list[m.Metric],
    ) -> _CompletionBatcher:
        async def generate_batch(
            tokenized_prompts: list[list[int]],
            sampling: SamplingConfig,
        ) -> list[Completion]:
            completions, metrics = await self._await_rank_0(
                self.generator.generate.call(
                    tokenized_prompts,
                    sampling_config=sampling,
                    metrics_prefix=metrics_prefix,
                )
            )
            generation_metrics.extend(metrics)
            return completions

        return _CompletionBatcher(generate_batch)

    def _make_completion_fn(
        self,
        *,
        completion_batcher: _CompletionBatcher,
    ):
        async def completion_fn(
            prompt_token_ids: list[int],
            turn_sampling: SamplingConfig,
            request_id: str,
        ) -> Completion:
            per_turn_sampling = dataclasses.replace(
                turn_sampling,
                n=1,
                stop_token_ids=list(self._stop_token_ids),
            )
            return await completion_batcher.submit(
                prompt_token_ids=prompt_token_ids,
                sampling=per_turn_sampling,
                request_id=request_id,
            )

        return completion_fn

    def _build_envs(
        self,
        *,
        env_config: Configurable.Config,
        example: EnvExample,
        group_size: int,
    ):
        return [
            env_config.build(example=example, sample_idx=sample_idx)
            for sample_idx in range(group_size)
        ]

    async def _rollout_one_group(
        self,
        *,
        env_config: Configurable.Config,
        example: EnvExample,
        group_size: int,
        sampling: SamplingConfig,
        completion_batcher: _CompletionBatcher,
    ) -> list[RolloutOutput]:
        envs = self._build_envs(
            env_config=env_config,
            example=example,
            group_size=group_size,
        )
        completion_fn = self._make_completion_fn(
            completion_batcher=completion_batcher,
        )
        return await do_rollout_group(
            envs=envs,
            renderer=self.renderer,
            completion_fn=completion_fn,
            sampling=sampling,
            example=example,
            max_turns=self.config.max_rollout_turns,
            token_env_config=TokenEnvConfig(
                max_trajectory_tokens=self.config.max_trajectory_tokens,
                max_generation_tokens=sampling.max_tokens,
                step_timeout_s=self.config.step_timeout_s,
            ),
        )

    def _make_examples(self, *, step: int, num_groups: int) -> list[EnvExample]:
        return [
            EnvExample(
                group_id=f"step={step}/group={group_idx}",
                step=step,
                group_idx=group_idx,
            )
            for group_idx in range(num_groups)
        ]

    async def _collect_rollouts(
        self,
        *,
        env_config: Configurable.Config,
        num_groups: int,
        group_size: int,
        step: int,
        sampling: SamplingConfig,
        metrics_prefix: str = "generator",
    ) -> tuple[list[RolloutOutput], list[m.Metric]]:
        """Collect rollout groups with bounded async fanout."""
        generation_metrics: list[m.Metric] = []
        completion_batcher = self._make_completion_batcher(
            metrics_prefix=metrics_prefix,
            generation_metrics=generation_metrics,
        )
        examples = self._make_examples(step=step, num_groups=num_groups)
        pending: set[asyncio.Task[list[RolloutOutput]]] = set()
        rollouts: list[RolloutOutput] = []
        next_idx = 0
        max_inflight = max(self.config.async_rollout_groups, 1)

        try:
            while next_idx < len(examples) or pending:
                while next_idx < len(examples) and len(pending) < max_inflight:
                    pending.add(
                        asyncio.create_task(
                            self._rollout_one_group(
                                env_config=env_config,
                                example=examples[next_idx],
                                group_size=group_size,
                                sampling=sampling,
                                completion_batcher=completion_batcher,
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

        rollout_metrics = self._build_rollout_metrics(
            rollouts,
            generation_metrics=generation_metrics,
            prefix="rollout",
        )
        return rollouts, rollout_metrics

    async def _collect_replay_samples(
        self,
        *,
        num_groups: int,
        step: int,
        train_version: int,
    ) -> tuple[list[RolloutOutput], list[ReplaySample], list[m.Metric]]:
        """Collect rollouts into a FIFO queue, then consume replay samples."""
        sampling = self.config.generator.sampling
        group_size = sampling.n
        generation_metrics: list[m.Metric] = []
        completion_batcher = self._make_completion_batcher(
            metrics_prefix="generator",
            generation_metrics=generation_metrics,
        )
        queue = RolloutQueue(
            max_groups=max(self.config.rollout_queue_groups, 1),
            max_age_steps=self.config.max_offpolicy_steps,
        )
        examples = self._make_examples(step=step, num_groups=num_groups)
        pending: set[asyncio.Task[tuple[list[RolloutOutput], ReplayGroup]]] = set()
        rollouts: list[RolloutOutput] = []
        next_idx = 0
        max_inflight = max(self.config.async_rollout_groups, 1)

        async def produce(
            example: EnvExample,
        ) -> tuple[list[RolloutOutput], ReplayGroup]:
            group_rollouts = await self._rollout_one_group(
                env_config=self.config.env,
                example=example,
                group_size=group_size,
                sampling=sampling,
                completion_batcher=completion_batcher,
            )
            samples = rollouts_to_replay_samples(group_rollouts)
            return group_rollouts, ReplayGroup(
                group_id=example.group_id,
                samples=samples,
                behavior_version=min(
                    (r.behavior_version for r in group_rollouts),
                    default=0,
                ),
                train_step=step,
            )

        async def produce_all() -> None:
            nonlocal next_idx, pending
            try:
                while next_idx < len(examples) or pending:
                    while next_idx < len(examples) and len(pending) < max_inflight:
                        pending.add(asyncio.create_task(produce(examples[next_idx])))
                        next_idx += 1
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        group_rollouts, replay_group = task.result()
                        rollouts.extend(group_rollouts)
                        await queue.put(replay_group)
            except BaseException:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise
            finally:
                await queue.close()

        producer_task = asyncio.create_task(produce_all())
        try:
            samples, queue_stats = await queue.get_all(train_version=train_version)
            await producer_task
        except BaseException:
            producer_task.cancel()
            await queue.close()
            await asyncio.gather(producer_task, return_exceptions=True)
            raise

        metrics = self._build_rollout_metrics(
            rollouts,
            generation_metrics=generation_metrics,
            prefix="rollout",
        )
        metrics += self._build_replay_metrics(samples, queue_stats)
        return rollouts, samples, metrics

    @staticmethod
    def _build_replay_metrics(
        samples: list[ReplaySample],
        queue_stats: QueueStats,
    ) -> list[m.Metric]:
        advantages = [
            value
            for sample in samples
            for mask, value in zip(sample.loss_mask, sample.advantages, strict=True)
            if mask
        ]
        return [
            m.Metric("replay/num_samples", m.NoReduce(float(len(samples)))),
            m.Metric(
                "replay/num_loss_tokens", m.NoReduce(float(queue_stats.num_loss_tokens))
            ),
            m.Metric("replay/queue/groups", m.NoReduce(float(queue_stats.num_groups))),
            m.Metric(
                "replay/queue/dropped_stale_groups",
                m.NoReduce(float(queue_stats.num_dropped_stale_groups)),
            ),
            m.Metric(
                "replay/queue/max_age_steps",
                m.NoReduce(float(queue_stats.max_age_steps)),
            ),
            m.Metric("advantage", m.SummaryStats.from_list(advantages)),
        ]

    @staticmethod
    def _build_rollout_metrics(
        rollouts: list[RolloutOutput],
        *,
        generation_metrics: list[m.Metric],
        prefix: str,
    ) -> list[m.Metric]:
        response_lens = [
            len(turn.response_token_ids)
            for rollout in rollouts
            for turn in rollout.turns
        ]
        prompt_lens = [
            len(turn.prompt_token_ids) for rollout in rollouts for turn in rollout.turns
        ]
        total_lens = [
            len(turn.prompt_token_ids) + len(turn.response_token_ids)
            for rollout in rollouts
            for turn in rollout.turns
        ]
        rewards = [
            float(rollout.reward) for rollout in rollouts if rollout.reward is not None
        ]
        truncated = [rollout.status == RolloutStatus.TRUNCATED for rollout in rollouts]
        errored = [rollout.status == RolloutStatus.ERROR for rollout in rollouts]

        metrics: list[m.Metric] = [
            m.Metric(f"{prefix}/response_length", m.Mean.from_list(response_lens)),
            m.Metric(f"{prefix}/response_length", m.Max.from_list(response_lens)),
            m.Metric(f"{prefix}/prompt_length", m.Mean.from_list(prompt_lens)),
            m.Metric(f"{prefix}/prompt_length", m.Max.from_list(prompt_lens)),
            m.Metric(f"{prefix}/total_length", m.Max.from_list(total_lens)),
            m.Metric(f"{prefix}/truncation_rate", m.Mean.from_list(truncated)),
            m.Metric(f"{prefix}/error_rate", m.Mean.from_list(errored)),
            m.Metric("reward", m.SummaryStats.from_list(rewards)),
        ]
        if rewards:
            by_group: dict[str, list[float]] = defaultdict(list)
            for rollout in rollouts:
                if rollout.reward is not None:
                    by_group[rollout.group_id].append(float(rollout.reward))
            group_stds = [
                statistics.pstdev(group_rewards)
                for group_rewards in by_group.values()
                if group_rewards
            ]
            metrics.extend(
                [
                    m.Metric("reward/group_std", m.Mean.from_list(group_stds)),
                    m.Metric("reward/group_std", m.Max.from_list(group_stds)),
                    m.Metric(
                        "reward/zero_std_frac",
                        m.NoReduce(
                            sum(1 for value in group_stds if value == 0.0)
                            / len(group_stds)
                            if group_stds
                            else 0.0
                        ),
                    ),
                ]
            )
        metrics += generation_metrics
        metrics += _prepare_reward_metrics(
            prefix=f"{prefix}/reward/component",
            rollouts=rollouts,
        )
        return metrics

    @sl.log_trace_span("validate")
    async def validate(self) -> list[m.Metric]:
        """Run validation through the same rollout loop as training."""
        t_validate_start = time.perf_counter()
        num_samples = self.config.num_validation_samples
        greedy = SamplingConfig(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.config.generator.sampling.max_tokens,
            stop_token_ids=list(self._stop_token_ids),
        )
        rollouts, validation_metrics = await self._collect_rollouts(
            env_config=self.config.validation_env,
            num_groups=num_samples,
            group_size=1,
            step=0,
            sampling=greedy,
            metrics_prefix="validation/generator",
        )

        if self.config.log_samples:
            _log_samples(rollouts)

        t_validate_s = time.perf_counter() - t_validate_start
        return [
            _rename_metric(metric, old_prefix="rollout/", new_prefix="validation/")
            for metric in validation_metrics
        ] + [
            m.Metric("validation/num_samples", m.NoReduce(float(len(rollouts)))),
            m.Metric("timing/validate", m.NoReduce(t_validate_s)),
        ]

    async def train(self):
        num_steps = self.config.num_steps
        num_groups = self.config.num_prompts_per_step
        logger.info(f"Pre-training validation; then {num_steps} steps of RL training")

        # collect validation metrics before training
        # so we can compare before/after
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

        for step in range(1, num_steps + 1):
            sl.set_step(step)
            # Propagate the step counter to actors for structured logging.
            self.trainer.sync_log_step.call(step)
            self.generator.sync_log_step.call(step)
            # Cancellation point for Ctrl-C (KeyboardInterrupt) handling.
            # This yields to the event loop to check for cancellation, which
            # doesn't happen with `.get` calls.
            # TODO: investigate replacing `.get()` with `await
            await asyncio.sleep(0)

            t_step_start = time.perf_counter()

            # --- rollouts ---
            t_rollout_start = time.perf_counter()
            rollouts, samples, rollout_metrics = await self._collect_replay_samples(
                num_groups=num_groups,
                step=step,
                train_version=step - 1,
            )
            t_rollout_s = time.perf_counter() - t_rollout_start

            if self.config.log_samples:
                _log_samples(rollouts)
            if not samples:
                raise RuntimeError(
                    "rollout collection produced no trainable replay samples"
                )

            # --- train ---
            t_train_start = time.perf_counter()
            batches = [
                self._collate_samples(per_rank_samples)
                for per_rank_samples in self._shard_samples(samples)
            ]
            # Controller has all replay rows pre-shard, so it computes the
            # global trainable-token count instead of an all-reduce.
            num_global_valid_tokens = sum(sample.num_loss_tokens for sample in samples)
            with sl.log_trace_span("trainer_forward_backward_call"):
                fwd_bwd_metrics = self._get_rank_0_value(
                    self.trainer.forward_backward.call(
                        batches,
                        num_global_valid_tokens=num_global_valid_tokens,
                    ).get()
                )
            with sl.log_trace_span("trainer_optim_step_call"):
                optim_output = self._get_rank_0_value(
                    self.trainer.optim_step.call().get()
                )
            trainer_policy_version = optim_output.policy_version
            optimizer_metrics = optim_output.metrics
            t_train_s = time.perf_counter() - t_train_start

            # --- weight sync ---
            # TODO: we should have `push_model_state_dict` return `trainer_policy_version`
            # instead of having `trainer.optim_step` return it
            t_push_start = time.perf_counter()
            with sl.log_trace_span("trainer_push_model_state_dict"):
                self.trainer.push_model_state_dict.call().get()
            t_weight_sync_push_s = time.perf_counter() - t_push_start
            with sl.log_trace_span("generator_pull_model_state_dict"):
                self.generator.pull_model_state_dict.call(trainer_policy_version).get()
            t_weight_sync_total_s = time.perf_counter() - t_push_start
            t_step_s = time.perf_counter() - t_step_start
            # --- divergence check before any logging ---
            if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break

            # --- Prepare metrics ---
            total_tokens = sum(len(sample.token_ids) for sample in samples)

            step_metrics: list[m.Metric] = []

            step_metrics += rollout_metrics

            # Actor metrics are already globally reduced; NoReduce passes them through.
            step_metrics += [
                m.Metric(k, m.NoReduce(v)) for k, v in fwd_bwd_metrics.items()
            ]
            step_metrics += [
                m.Metric(k, m.NoReduce(v)) for k, v in optimizer_metrics.items()
            ]

            # timing metrics
            for key, value in [
                ("timing/step", t_step_s),
                ("timing/rollout", t_rollout_s),
                ("timing/train", t_train_s),
                ("timing/weight_sync/push", t_weight_sync_push_s),
                ("timing/weight_sync/total", t_weight_sync_total_s),
            ]:
                step_metrics.append(m.Metric(key, m.NoReduce(value)))

            step_metrics.append(
                m.Metric("perf/tokens_per_second", m.NoReduce(total_tokens / t_step_s))
            )

            self.metrics_processor.log(
                step=step, metrics=step_metrics, is_validation=False
            )

        post_validation_metrics = await self.validate()
        self.metrics_processor.log(
            step=num_steps,
            metrics=post_validation_metrics,
            is_validation=True,
        )
        post_validation_agg = m.MetricsProcessor._aggregate_metrics(
            post_validation_metrics
        )

        # Side-by-side pre/post summary so the before/after improvement is
        # visible without scrolling back through the train loop.
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
