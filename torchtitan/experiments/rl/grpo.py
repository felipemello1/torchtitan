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
import logging
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace

# must run before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.components.dataloading.utils import pack
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import (
    BatchConfig,
    CompileConfig,
    ConfigManager,
    Configurable,
    ParallelismConfig,
)
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.envs import EnvBuilder, EnvDataset, TokenEnvConfig
from torchtitan.experiments.rl.generation_scheduler import GenerationScheduler
from torchtitan.experiments.rl.loss import DAPOLoss
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.observability.metrics.rl import (
    build_rollout_metrics,
    rename_metric_prefix,
)
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.replay import rollouts_to_replay_samples
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

    def __init__(self, config: Config, *, pad_id: int):
        self.local_batch_size = config.batch.local_batch_size
        self.global_batch_size = config.batch.global_batch_size
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
        """Number of distinct prompts (= GRPO groups) drawn per training step.

        The total rollouts per wave is `num_prompts_per_step * group_size`.
        """

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
            TrainingLogprobConfig.from_sampling(self.generator.sampling)

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
        tokenizer = HuggingFaceTokenizer(tokenizer_path=config.hf_assets_path)
        self.batcher = Batcher(config.batcher, pad_id=tokenizer.eos_id)

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
                    config.num_prompts_per_step * config.group_size,
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
            completions, metrics = await self._await_rank_0(
                self.generator.generate.call(
                    prompt_token_ids_batch,
                    request_ids=request_ids,
                    sampling_config=sampling,
                    metrics_prefix=metrics_prefix,
                )
            )
            return completions, metrics

        return GenerationScheduler(generate_batch)

    async def _sync_generator_weights(
        self,
        *,
        generation_scheduler: GenerationScheduler,
        policy_version: int,
    ) -> dict[str, float]:
        t_weight_sync_start = time.perf_counter()
        with sl.log_trace_span("weight_sync_admission_drain"):
            await generation_scheduler.pause_for_weight_sync()
        t_weight_sync_drain_s = time.perf_counter() - t_weight_sync_start

        try:
            t_push_start = time.perf_counter()
            with sl.log_trace_span("trainer_push_model_state_dict"):
                await self._await_call(self.trainer.push_model_state_dict.call())
            t_weight_sync_push_s = time.perf_counter() - t_push_start
            with sl.log_trace_span("generator_pull_model_state_dict"):
                await self._await_call(
                    self.generator.pull_model_state_dict.call(policy_version)
                )
        finally:
            await generation_scheduler.resume_after_weight_sync()

        return {
            "timing/weight_sync/drain": t_weight_sync_drain_s,
            "timing/weight_sync/push": t_weight_sync_push_s,
            "timing/weight_sync/total": time.perf_counter() - t_weight_sync_start,
        }

    @sl.log_trace_span("_collect_rollouts")
    async def _collect_rollouts(
        self,
        num_groups: int,
        step: int,
        group_offset: int,
        generation_scheduler: GenerationScheduler,
    ) -> tuple[list[RolloutOutput], list[m.Metric]]:
        """Collect rollout groups through the token-env driver."""
        sampling = self._sampling_for_rollout(self.config.generator.sampling)

        rollouts: list[RolloutOutput] = []
        token_env_config = TokenEnvConfig(
            max_trajectory_tokens=self.config.batcher.batch.seq_len,
            max_generation_tokens=sampling.max_tokens,
        )
        for idx in range(num_groups):
            example = self.train_dataset.sample_group(
                sample_step=step,
                group_idx=group_offset + idx,
            )
            rollouts.extend(
                await run_rollout_group(
                    env_builder=self.train_env_builder,
                    example=example,
                    group_size=self.config.group_size,
                    renderer=self.renderer,
                    completion_fn=generation_scheduler.submit,
                    sampling=sampling,
                    max_turns=self.config.max_rollout_turns,
                    token_env_config=token_env_config,
                )
            )

        return rollouts, build_rollout_metrics(
            "rollout",
            rollouts,
            generation_scheduler.pop_metrics(),
        )

    @sl.log_trace_span("validate")
    async def validate(self) -> list[m.Metric]:
        """Run validation on held-out prompts using greedy sampling.

        TODO: investigate using pass@k.
        """
        t_validate_start = time.perf_counter()
        num_samples = self.config.num_validation_samples
        greedy = self._sampling_for_rollout(
            SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                max_tokens=self.config.generator.sampling.max_tokens,
            )
        )
        generation_scheduler = self._make_generation_scheduler(
            metrics_prefix="validation_generator"
        )

        rollouts: list[RolloutOutput] = []
        token_env_config = TokenEnvConfig(
            max_trajectory_tokens=self.config.batcher.batch.seq_len,
            max_generation_tokens=greedy.max_tokens,
        )
        try:
            for idx in range(num_samples):
                example = self.validation_dataset.sample_group(
                    sample_step=0,
                    group_idx=idx,
                )
                rollouts.extend(
                    await run_rollout_group(
                        env_builder=self.validation_env_builder,
                        example=example,
                        group_size=1,
                        renderer=self.renderer,
                        completion_fn=generation_scheduler.submit,
                        sampling=greedy,
                        max_turns=self.config.max_rollout_turns,
                        token_env_config=token_env_config,
                    )
                )
        finally:
            await generation_scheduler.close()

        generation_metrics = generation_scheduler.pop_metrics()
        if self.config.log_samples:
            _log_samples(rollouts)
        if self.rollout_sample_logger is not None:
            self.rollout_sample_logger.write(
                step=0,
                phase="validation",
                rollouts=rollouts,
            )

        validation_metrics = build_rollout_metrics(
            "validation/rollout",
            rollouts,
            generation_metrics,
        )
        validation_metrics = rename_metric_prefix(
            validation_metrics,
            old_prefix="reward",
            new_prefix="validation/reward",
        )
        validation_metrics += [
            m.Metric("validation/num_samples", m.NoReduce(float(len(rollouts)))),
        ]

        t_validate_s = time.perf_counter() - t_validate_start
        validation_metrics.append(m.Metric("timing/validate", m.NoReduce(t_validate_s)))
        return validation_metrics

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

        generation_scheduler = self._make_generation_scheduler(
            metrics_prefix="generator"
        )
        try:
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
                # Collect rollouts until total response tokens reach the
                # token budget. The Batcher then packs, truncates to
                # global_batch_size rows, and splits into microbatches.
                t_rollout_start = time.perf_counter()
                rollouts: list[RolloutOutput] = []
                rollout_metrics: list[m.Metric] = []
                samples: list[ReplaySample] = []
                collected_tokens = 0
                group_offset = 0
                while collected_tokens < self.batcher.num_tokens_target:
                    new_rollouts, new_metrics = await self._collect_rollouts(
                        num_groups,
                        step=step,
                        group_offset=group_offset,
                        generation_scheduler=generation_scheduler,
                    )
                    new_samples = rollouts_to_replay_samples(new_rollouts)
                    if not new_samples:
                        raise RuntimeError(
                            "rollout wave produced no trainable replay samples"
                        )
                    rollouts.extend(new_rollouts)
                    samples.extend(new_samples)
                    rollout_metrics.extend(new_metrics)
                    collected_tokens += sum(
                        sample.num_loss_tokens for sample in new_samples
                    )
                    group_offset += num_groups

                if self.rollout_sample_logger is not None:
                    self.rollout_sample_logger.write(
                        step=step,
                        phase="train",
                        rollouts=rollouts,
                    )
                sample_metrics: list[m.Metric] = [
                    m.Metric(
                        "advantage",
                        m.SummaryStats.from_list(
                            [
                                sample.advantage
                                for sample in samples
                                for _ in range(sample.num_loss_tokens)
                            ]
                        ),
                    )
                ]
                t_rollout_s = time.perf_counter() - t_rollout_start

                if self.config.log_samples:
                    _log_samples(rollouts)

                # --- train ---
                t_train_start = time.perf_counter()
                with sl.log_trace_span("batcher_batch"):
                    (
                        microbatches,
                        num_global_valid_tokens,
                        packing_metrics,
                    ) = self.batcher.batch(samples, dp_degree=self.trainer_dp_degree)

                with sl.log_trace_span("trainer_forward_backward_call"):
                    fwd_bwd_metrics = self._get_rank_0_value(
                        self.trainer.forward_backward.call(
                            microbatches,
                            num_global_valid_tokens=num_global_valid_tokens,
                            logprob_config=TrainingLogprobConfig.from_sampling(
                                self.config.generator.sampling
                            ),
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
                weight_sync_metrics = await self._sync_generator_weights(
                    generation_scheduler=generation_scheduler,
                    policy_version=trainer_policy_version,
                )
                t_step_s = time.perf_counter() - t_step_start
                # --- divergence check before any logging ---
                if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                    logger.error("Loss is NaN/Inf; training diverged")
                    break

                # --- Prepare metrics ---
                total_tokens = sum(len(sample.token_ids) for sample in samples)

                step_metrics: list[m.Metric] = []

                step_metrics += rollout_metrics
                step_metrics += sample_metrics

                # Actor metrics are already globally reduced; NoReduce passes them through.
                step_metrics += [
                    m.Metric(k, m.NoReduce(v)) for k, v in fwd_bwd_metrics.items()
                ]
                step_metrics += [
                    m.Metric(k, m.NoReduce(v)) for k, v in optimizer_metrics.items()
                ]
                step_metrics += [
                    m.Metric(k, m.NoReduce(v)) for k, v in packing_metrics.items()
                ]

                # timing metrics
                for key, value in [
                    ("timing/step", t_step_s),
                    ("timing/rollout", t_rollout_s),
                    ("timing/train", t_train_s),
                    *weight_sync_metrics.items(),
                ]:
                    step_metrics.append(m.Metric(key, m.NoReduce(value)))

                step_metrics.append(
                    m.Metric(
                        "perf/tokens_per_second",
                        m.NoReduce(total_tokens / t_step_s),
                    )
                )

                self.metrics_processor.log(
                    step=step, metrics=step_metrics, is_validation=False
                )
        finally:
            await generation_scheduler.close()

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
