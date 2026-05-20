# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    Configurable,
    DebugConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.actors.utils import (
    compute_logprobs,
    cuda_memory_stats,
    PartialLogprobDrift,
    reset_cuda_peak_memory_stats,
    verify_logprob_identity,
)
from torchtitan.experiments.rl.sampling import TrainingLogprobConfig
from torchtitan.experiments.rl.trainer_microbatch import (
    MetricAccumulator,
    schedule_training_microbatches,
)
from torchtitan.experiments.rl.types import OptimStepOutput, TrainingBatch
from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger

logger = logging.getLogger(__name__)


_TIED_WEIGHT_ALIASES = (
    # (preferred_key, alias_to_drop) — when both are in the state_dict
    # and share the same backing storage, drop the alias. The receiver
    # re-ties via ``init_weights``, so dropping is semantics-safe.
    ("tok_embeddings.weight", "lm_head.weight"),
)


def _dedup_tied_tensors(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop known tied-weight aliases when they actually share storage.

    Qwen3 ties ``tok_embeddings.weight`` and ``lm_head.weight`` when
    ``enable_weight_tying=True``; both are the same buffer. Emitting
    both into the RDMA transfer plan registers the same memory twice
    and roughly doubles the per-rank read budget.

    We check ``data_ptr`` equality on the specific pair rather than
    iterating and dropping every shared-storage entry, because
    DTensor/FSDP-managed tensors can legitimately share a flat backing
    buffer across distinct params; a generic dedup would silently strip
    the state dict.
    """
    out = dict(state_dict)
    for preferred, alias in _TIED_WEIGHT_ALIASES:
        if preferred in out and alias in out:
            try:
                same_storage = out[preferred].data_ptr() == out[alias].data_ptr()
            except Exception:
                same_storage = False
            if same_storage:
                logger.debug(
                    "Dropping tied-weight alias %s (shares storage with %s)",
                    alias,
                    preferred,
                )
                del out[alias]
    return out


class PolicyTrainer(Actor, Configurable):
    """Updates policy from token-aligned RL replay batches.

    Exposes separate `forward_backward` and `optim_step` endpoints, called
    explicitly by the controller.

    Args:
        config: PolicyTrainer.Config with all model/optimizer/parallelism settings.
        model_spec: TorchTitan model specification.
        hf_assets_path: Path to HF assets folder for checkpoint loading.
            Shared with the generator (both load from the same HF checkpoint).
            generator_dtype: Generator dtype used when publishing weights.
            if generator dtype differs from training dtype. If None, no cast is performed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """PolicyTrainer configuration for optimizer, training, and parallelism."""

        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        debug: DebugConfig = field(default_factory=DebugConfig)
        loss: Configurable.Config = field(default_factory=Configurable.Config)
        ac_config: ActivationCheckpointConfig = field(
            default_factory=lambda: ActivationCheckpointConfig(mode="none")
        )
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        max_microbatch_samples: int | None = None
        """Maximum replay samples per forward/backward microbatch."""

        max_microbatch_tokens: int | None = None
        """Target packed tokens per microbatch; individual samples stay intact."""

        dump_folder: str = ""
        """Folder for AC debug dumps when using memory_budget mode."""

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        compile_config: CompileConfig,
        hf_assets_path: str = "",
        generator_dtype: str = "",
        output_dir: str,
    ):
        init_logger()
        sl.init_structured_logger(
            source="rl_trainer",
            output_dir=output_dir,
            rank=current_rank().rank,
            enable=config.debug.enable_structured_logging,
        )
        sl.log_trace_instant("structured_logger_started")

        self.config = config
        self.compile_config = compile_config
        if (
            config.max_microbatch_samples is not None
            and config.max_microbatch_samples <= 0
        ):
            raise ValueError(
                "max_microbatch_samples must be positive or None, got "
                f"{config.max_microbatch_samples}"
            )
        if (
            config.max_microbatch_tokens is not None
            and config.max_microbatch_tokens <= 0
        ):
            raise ValueError(
                "max_microbatch_tokens must be positive or None, got "
                f"{config.max_microbatch_tokens}"
            )
        self.loss_fn = config.loss.build()

        # Only cast if generator dtype differs from training dtype, otherwise
        # staging buffers would be allocated for a no-op cast.
        training_dtype = TORCH_DTYPE_MAP[config.training.dtype]
        gen_dtype = TORCH_DTYPE_MAP[generator_dtype] if generator_dtype else None
        self._transfer_dtype = gen_dtype if gen_dtype != training_dtype else None

        # Device setup
        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # Enable batch-invariant mode BEFORE init_distributed
        set_batch_invariance(config.debug.batch_invariant)

        with sl.log_trace_span("torch_distributed_init"):
            world_size = dist_utils.init_distributed(config.comm)

        self.parallel_dims = ParallelDims.from_config(config.parallelism, world_size)

        # Set determinism flags and seed via core torchtitan utility
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

        # Initialize state dict adapter for HF checkpoint loading
        if model_spec.state_dict_adapter is not None:
            self.sd_adapter = model_spec.state_dict_adapter(
                model_spec.model, hf_assets_path
            )
        else:
            self.sd_adapter = None

        # Create training policy model
        model = self._build_model(model_spec, config, device_type)
        model.train()
        self.model = model
        self.model_parts = [model]

        # Build optimizer and LR scheduler
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )

        self.policy_version = 0

        # Always build CheckpointManager; enable is a field on the config.
        # When enable=False (CI/debug), load() is a no-op and random init stands.
        self.checkpointer = config.checkpoint.build(
            dataloader=None,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=self.sd_adapter,
            base_folder=config.dump_folder,
        )
        self.checkpointer.load()
        if not self.checkpointer.enable:
            logger.warning(
                "Checkpoint disabled, skip weight loading and use random-initialized weights. "
                "Set checkpoint.enable=True to load from a checkpoint."
            )

        self.generator: Any | None = None

        # Data parallelism: mesh is available after _build_model triggers build_mesh
        self.dp_enabled = self.parallel_dims.dp_enabled
        batch_mesh = self.parallel_dims.get_optional_mesh("batch")
        if batch_mesh is not None:
            self.dp_size = batch_mesh.size()
            self.dp_rank = batch_mesh.get_local_rank()
        else:
            self.dp_size = 1
            self.dp_rank = 0

        logger.debug(
            f"PolicyTrainer initialized (dp_rank={self.dp_rank}, dp_size={self.dp_size})"
        )

    def state_dict(self) -> dict[str, Any]:
        return {"policy_version": self.policy_version}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.policy_version = state_dict["policy_version"]

    @endpoint
    async def close(self) -> None:
        """Destroy the worker's torch.distributed process group."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @sl.log_trace_span("build_model")
    def _build_model(
        self,
        model_spec: ModelSpec,
        config: Config,
        device_type: str,
    ):
        """Build, parallelize, and initialize a model with random weights.

        Checkpoint loading (e.g. from HF) is handled separately by
        CheckpointManager after model and optimizer construction.

        Args:
            model_spec: Model specification for building and parallelizing.
            config: Trainer config (used for dtype, parallelism, etc.).
            device_type: Device type string (e.g. "cuda").

        Returns:
            Model with random-initialized weights.
        """

        # TODO: Also support flex attention backend later.
        from torchtitan.models.common.attention import VarlenAttention

        assert isinstance(
            model_spec.model.layers[0].attention.inner_attention, VarlenAttention.Config
        ), "Only varlen attention backend is allowed."

        # Fill sharding configs on the config BEFORE build via the
        # model-agnostic `update_from_config` hook (RL's trainer bypasses
        # `torchtitan.Trainer's` call, so we invoke it directly).
        model_spec.model.update_from_config(trainer_config=config)

        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                model = model_spec.model.build()

        model = model_spec.parallelize_fn(
            model,
            parallel_dims=self.parallel_dims,
            training=config.training,
            parallelism=config.parallelism,
            compile_config=self.compile_config,
            ac_config=config.ac_config,
            dump_folder=config.dump_folder,
        )

        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        return model

    @endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        """Sync the structured-logger step counter from the controller."""
        sl.set_step(step, relative_step=relative_step)

    def _reduce_forward_backward_metrics(
        self,
        *,
        sum_reduced_metrics: dict[str, torch.Tensor],
        max_reduced_metrics: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Reduce forward/backward metrics across the loss mesh.

        Args:
            sum_reduced_metrics: Per-rank shares to be SUM-reduced. Each
                value must be pre-normalized so that summing across ranks
                reconstructs the global metric.
            max_reduced_metrics: Per-rank values to be MAX-reduced.

        Returns:
            {key: float} after collective reduction.
        """
        # TODO: switch from plain tensors to DTensor / spmd_types so the
        # reduction op is encoded in the placement instead of split across
        # `sum_reduced_metrics` / `max_reduced_metrics` dicts.
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")

        out: dict[str, float] = {}
        for values_by_key, op in [
            (sum_reduced_metrics, c10d.ReduceOp.SUM),
            (max_reduced_metrics, c10d.ReduceOp.MAX),
        ]:
            if not values_by_key:
                continue
            keys = list(values_by_key)
            stacked = torch.stack([values_by_key[key].detach() for key in keys])
            if loss_mesh is not None:
                stacked = funcol.all_reduce(stacked, reduceOp=op.name, group=loss_mesh)
            for key, value in zip(keys, stacked.cpu().tolist(), strict=True):
                out[key] = float(value)
        return out

    @endpoint
    @sl.log_trace_span("forward_backward")
    async def forward_backward(
        self,
        train_data: list[TrainingBatch],
        *,
        num_global_valid_tokens: int,
        logprob_config: TrainingLogprobConfig,
    ) -> dict[str, float]:
        """Run forward pass, compute loss, call backward, and reduce metrics.

        Args:
            train_data: List of TrainingBatch, one per DP rank. Local rank
                picks train_data[self.dp_rank].
            num_global_valid_tokens: Total trainable response tokens across all DP
                ranks for this step. The controller computes this before
                sharding replay samples.
            logprob_config: Validated behavior-logprob contract. Trainer
                logprobs are computed under the same sampling-temperature
                transform as the generator behavior logprobs.

        Returns:
            dict[str, float]: Globally-reduced metrics.
        """
        logger.debug(
            f"{os.getpid()=} PolicyTrainer forward_backward "
            f"step {self.policy_version}"
        )

        # RL does not support pipeline parallelism yet, so the trainer
        # owns one model part.
        if len(self.model_parts) != 1:
            raise ValueError(
                f"PolicyTrainer expects exactly one model part, got "
                f"{len(self.model_parts)} (pipeline parallelism is not yet "
                "supported in RL)."
            )
        model = self.model_parts[0]

        device = self.device
        reset_cuda_peak_memory_stats(device)

        schedule = schedule_training_microbatches(
            train_data,
            dp_rank=self.dp_rank,
            max_samples=self.config.max_microbatch_samples,
            max_tokens=self.config.max_microbatch_tokens,
        )

        rope_cache_len = self.model.freqs_cis.shape[0]
        if schedule.max_seq_len > rope_cache_len:
            raise ValueError(
                f"Replay sample length {schedule.max_seq_len} exceeds rope "
                f"cache size {rope_cache_len}. Increase model max_seq_len or "
                "reduce generation max_tokens."
            )

        num_global_valid_tokens: torch.Tensor = torch.tensor(
            float(max(num_global_valid_tokens, 1)),
            device=device,
            dtype=torch.float32,
        )

        self.optimizers.zero_grad()

        metric_accumulator = MetricAccumulator()
        metric_accumulator.add_max(
            {
                "train/microbatches/max": torch.tensor(
                    float(schedule.max_microbatches),
                    device=device,
                    dtype=torch.float32,
                ),
            }
        )

        # Keep synchronization explicit for each microbatch; this path must work
        # across the trainer parallelisms used by the RL experiment.
        for scheduled_microbatch in schedule.microbatches:
            microbatch = scheduled_microbatch.batch
            is_real = scheduled_microbatch.is_real

            with sl.log_trace_span("forward_backward_microbatch"):
                token_ids = microbatch.token_ids.to(device)  # [1, T]
                seq_lens = microbatch.seq_lens
                # The batch's loss tensors are length T (sample-aligned for the
                # microbatch slicer); shift by one to align with the [1, T-1]
                # predictions emitted by compute_logprobs. Cross-sample-boundary
                # positions are always masked off because every sample's first
                # token is a prompt token (mask=0).
                loss_mask = microbatch.loss_mask.to(device)[:, 1:]  # [1, T-1]
                behavior_logprobs = microbatch.behavior_logprobs.to(device)[
                    :, 1:
                ]  # [1, T-1]
                advantages = microbatch.advantages.to(device)[:, 1:]  # [1, T-1]

                positions = torch.cat(
                    [torch.arange(l, device=device) for l in seq_lens]
                ).unsqueeze(0)
                attention_masks = create_varlen_metadata_for_document(positions)

                with sl.log_trace_span("model_forward"):
                    logits = model(
                        token_ids,
                        attention_masks=attention_masks,
                        positions=positions,
                    )
                policy_logprobs = compute_logprobs(  # [1, T-1]
                    logits,
                    token_ids,
                    temperature=logprob_config.temperature,
                )

                with sl.log_trace_span("loss_fn"):
                    loss_out = self.loss_fn(
                        policy_logprobs=policy_logprobs,
                        behavior_logprobs=behavior_logprobs,
                        advantages_per_token=advantages,
                        loss_mask=loss_mask,
                        num_global_valid_tokens=num_global_valid_tokens,
                    )

                with sl.log_trace_span("model_backward"):
                    loss_out.loss.backward()

                drift: PartialLogprobDrift = verify_logprob_identity(
                    behavior_logprobs=behavior_logprobs,
                    policy_logprobs=policy_logprobs,
                    loss_mask=loss_mask,
                    num_global_valid_tokens=num_global_valid_tokens,
                )

                metric_accumulator.add_sum(
                    {
                        **loss_out.sum_metrics,
                        "bit_wise/logprob_diff/mean": drift.logprob_diff_mean,
                        "bit_wise/ratio_tokens_different/mean": (
                            drift.ratio_tokens_different
                        ),
                    },
                    active=is_real,
                )
                metric_accumulator.add_max(
                    {
                        **loss_out.max_metrics,
                        "bit_wise/logprob_diff/max": drift.logprob_diff_max,
                        "train/microbatch_tokens/max": torch.tensor(
                            float(sum(seq_lens)),
                            device=device,
                            dtype=torch.float32,
                        ),
                        "train/microbatch_samples/max": torch.tensor(
                            float(len(seq_lens)),
                            device=device,
                            dtype=torch.float32,
                        ),
                    },
                    active=is_real,
                )

                del (
                    attention_masks,
                    logits,
                    loss_out,
                    policy_logprobs,
                    positions,
                )

        memory_stats = cuda_memory_stats(device)
        if memory_stats:
            metric_accumulator.add_max(
                {
                    f"train/cuda_memory/fwd_bwd/{key}": torch.tensor(
                        value,
                        device=device,
                        dtype=torch.float32,
                    )
                    for key, value in memory_stats.items()
                }
            )

        return self._reduce_forward_backward_metrics(
            sum_reduced_metrics=metric_accumulator.sum_reduced_metrics,
            max_reduced_metrics=metric_accumulator.max_reduced_metrics,
        )

    @endpoint
    @sl.log_trace_span("optim_step")
    async def optim_step(self) -> OptimStepOutput:
        """Clip gradients, step optimizer + LR scheduler, return updated state."""
        # TODO: Accept optional optimizer params (e.g. learning rate)
        # to allow controller-owned schedules (see Tinker API).

        # capture LR before step
        device = getattr(self, "device", None)
        if device is not None:
            reset_cuda_peak_memory_stats(device)
        current_lrs = self.lr_schedulers.schedulers[0].get_last_lr()
        if len(current_lrs) != 1:
            raise ValueError(
                "RL metrics only support a single optimizer LR for "
                f"train/lr; got {current_lrs}"
            )
        current_lr = float(current_lrs[0])

        with sl.log_trace_span("grad_clip"):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            )
        grad_norm_value = float(grad_norm.item())

        if not bool(torch.isfinite(grad_norm).item()):
            logger.warning(
                "Skipping optimizer step because gradient norm is non-finite: %s",
                grad_norm_value,
            )
            self.optimizers.zero_grad()
            memory_metrics = {
                f"train/cuda_memory/optim/{key}": value
                for key, value in (
                    cuda_memory_stats(device).items() if device is not None else ()
                )
            }
            return OptimStepOutput(
                policy_version=self.policy_version,
                metrics={
                    "train/grad_norm/mean": grad_norm_value,
                    "train/lr": current_lr,
                    "train/policy_version": float(self.policy_version),
                    "train/skipped_nonfinite_grad_norm": 1.0,
                    **memory_metrics,
                },
            )

        with sl.log_trace_span("optim"):
            self.optimizers.step()
            self.lr_schedulers.step()

        self.policy_version += 1

        logger.debug(
            f"{os.getpid()=} PolicyTrainer optim_step done, "
            f"policy_version={self.policy_version}"
        )
        memory_metrics = {
            f"train/cuda_memory/optim/{key}": value
            for key, value in (
                cuda_memory_stats(device).items() if device is not None else ()
            )
        }

        return OptimStepOutput(
            policy_version=self.policy_version,
            metrics={
                "train/grad_norm/mean": grad_norm_value,
                "train/lr": current_lr,
                "train/policy_version": float(self.policy_version),
                "train/skipped_nonfinite_grad_norm": 0.0,
                **memory_metrics,
            },
        )

    @endpoint
    @sl.log_trace_span("save_checkpoint")
    async def save_checkpoint(self, step: int, last_step: bool = False) -> bool:
        """Save checkpoint via CheckpointManager.

        Args:
            step: Current training step number.
            last_step: Whether this is the final step of training.

        Returns:
            True if a checkpoint was saved.
        """
        return self.checkpointer.save(step, last_step=last_step)

    @endpoint
    @sl.log_trace_span("push_model_state_dict")
    async def push_model_state_dict(self) -> None:
        """Publish model weights for generator consumption via TorchStore.

        When ``direct_rdma=True``, weights are transferred directly from
        GPU to GPU via one-sided RDMA reads, bypassing StorageVolumes
        entirely. When ``False``, data goes through StorageVolumes
        (which may themselves use RDMA as a transport internally).

        Note: we couple ``is_rdma_available()`` with ``direct_rdma`` here,
        but the two concepts are not identical -- StorageVolumes can also
        use RDMA as their transport layer. ``direct_rdma`` specifically
        means "skip StorageVolumes and let the destination read directly
        from the source's GPU memory".

        """
        from monarch.rdma import is_rdma_available

        # Push the full state dict (matches upstream main behavior). The
        # previous v7 wrapper ``_dedup_tied_tensors`` dropped
        # ``lm_head.weight`` because it shares storage with
        # ``tok_embeddings.weight`` under Qwen3 weight tying on the
        # trainer side; that's safe at TP=1 generator (tying preserved
        # there too), but at TP>1 the generator's parallelized DTensors
        # don't share storage, so ``lm_head.weight`` stayed at random
        # init -> LM head produced ~uniform logits -> gibberish output.
        # See ``discussions/37_multiturn_v7/tbr_refactor/v7_vs_main_tp_diff.md``.
        await ts.put_state_dict(
            self.model.state_dict(),
            "model_state_dict",
            direct_rdma=is_rdma_available(),
            transfer_dtype=self._transfer_dtype,
        )
