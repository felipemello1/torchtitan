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
import torch.nn.functional as F
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
from torchtitan.experiments.rl.loss import LossOutput
from torchtitan.experiments.rl.sampling import TrainingLogprobConfig
from torchtitan.experiments.rl.types import OptimStepOutput, TrainingBatch
from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger

logger = logging.getLogger(__name__)


@sl.log_trace_span("compute_logprobs")
def compute_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute per-token logprobs from logits.

    The generator emits reference logprobs after vLLM's sampling-temperature
    transform (`processed_logprobs` mode). To form correct importance
    ratios on the trainer side, divide logits by the same temperature
    before `log_softmax` so both probability spaces line up.

    Returns logprobs for positions 1..N (the predicted tokens). Output
    shape is `[batch, seq_len - 1]`.

    Example::

        # B=1, L=3, vocab=4. The returned columns score tokens 2 and 1.
        logits = torch.tensor([[
            [0.0, 0.0, 0.0, 0.0],  # position 0 (prompt)
            [1.0, 0.5, 2.0, 0.5],  # position 1
            [0.0, 0.0, 0.0, 0.0],  # position 2
        ]])
        token_ids = torch.tensor([[0, 2, 1]])
        # temperature=1.0, only positions 1..N=2 are returned.
        compute_logprobs(logits, token_ids, temperature=1.0)
        # tensor([[-1.3863, -2.0957]])
    """
    if temperature <= 0.0:
        raise ValueError(f"logprob temperature must be positive, got {temperature}")

    from torch.distributed.tensor import DTensor

    if isinstance(logits, DTensor):
        # TODO: pass `grad_placements=[Replicate(), ...]` to make the autograd
        # contract explicit (see .claude/rules/distributed.md).
        logits = logits.to_local()
    shift_logits = logits[:, :-1, :].float() / temperature
    shift_targets = token_ids[:, 1:]
    B, S = shift_targets.shape
    return -F.cross_entropy(
        shift_logits.reshape(B * S, -1), shift_targets.reshape(B * S), reduction="none"
    ).reshape(B, S)


@dataclass(frozen=True, slots=True)
class PartialLogprobDrift:
    """Per-rank generator-vs-trainer logprob drift awaiting reduction across the loss-mesh."""

    logprob_diff_mean: torch.Tensor
    logprob_diff_max: torch.Tensor
    ratio_tokens_different: torch.Tensor


@torch.no_grad()
@sl.log_trace_span("verify_logprob_identity")
def verify_logprob_identity(
    ref_logprobs: torch.Tensor,
    policy_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    num_global_valid_tokens: torch.Tensor,
) -> PartialLogprobDrift:
    """Compute per-rank drift between generator and trainer logprobs.

    Args:
        ref_logprobs: [B, L] logprobs the generator recorded at sampling
            time (`pi_old`).
        policy_logprobs: [B, L] trainer-computed logprobs (`pi_theta`).
        loss_mask: [B, L] binary mask; 1.0 for trainable (response) tokens.
        num_global_valid_tokens: scalar tensor; total trainable tokens across
            all DP ranks for this optimizer step.

    Returns:
        PartialLogprobDrift with the per-rank shares ready to be all-reduced.

    Example::

        # 2 tokens trainable; drift is small but non-zero on one of them.
        ref    = torch.tensor([[0.0, -0.50, -0.30]])
        policy = torch.tensor([[0.0, -0.52, -0.28]])
        mask   = torch.tensor([[0.0,  1.00,  1.00]])
        N = torch.tensor(2.0)

        drift = verify_logprob_identity(ref, policy, mask,
                                        num_global_valid_tokens=N)
        # diff = [-0.02, 0.02]
        # drift.logprob_diff_mean = (-0.02 + 0.02) / 2 = 0.0
        # drift.logprob_diff_max  = 0.02
        # drift.ratio_tokens_different = 2 / 2 = 1.0   # both > 1e-6
    """
    mask = loss_mask.bool()
    ref_flat = ref_logprobs[mask].float()
    policy_flat = policy_logprobs[mask].float()

    if ref_flat.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=ref_logprobs.device)
        return PartialLogprobDrift(zero, zero, zero)

    denom = num_global_valid_tokens.clamp(min=1.0)
    diff = policy_flat - ref_flat
    return PartialLogprobDrift(
        logprob_diff_mean=diff.sum() / denom,
        logprob_diff_max=diff.abs().max(),
        ratio_tokens_different=(diff.abs() > 1e-6).sum() / denom,
    )


class MetricAccumulator:
    """Combine per-microbatch metrics across grad-accumulation steps.

    `add_sum` builds running sums (so a SUM all-reduce across DP ranks
    reproduces the global metric); `add_max` builds running element-wise
    maxima. Both sides feed the trainer's loss-mesh all-reduce.

    Invariant for `add_sum` values: each must be either an additive count
    or a global-normalized contribution:
    `local_masked_sum / num_global_valid_tokens`. Passing per-microbatch
    local means here would yield `num_microbatches * local_mean` after the
    cross-microbatch sum, not the global mean.

    Example::

        # Two microbatches contribute to the same step's metrics.
        acc = MetricAccumulator()
        acc.add_sum({"loss/mean": torch.tensor(0.3)})
        acc.add_sum({"loss/mean": torch.tensor(0.2)})
        acc.add_max({"loss/ratio/max_abs": torch.tensor(1.10)})
        acc.add_max({"loss/ratio/max_abs": torch.tensor(1.25)})

        acc.sum_reduced_metrics["loss/mean"]          # tensor(0.5)
        acc.max_reduced_metrics["loss/ratio/max_abs"] # tensor(1.25)
    """

    def __init__(self) -> None:
        self.sum_reduced_metrics: dict[str, torch.Tensor] = {}
        self.max_reduced_metrics: dict[str, torch.Tensor] = {}

    def add_sum(self, metrics: dict[str, torch.Tensor]) -> None:
        for key, value in metrics.items():
            previous = self.sum_reduced_metrics.get(key)
            self.sum_reduced_metrics[key] = (
                value if previous is None else previous + value
            )

    def add_max(self, metrics: dict[str, torch.Tensor]) -> None:
        for key, value in metrics.items():
            previous = self.max_reduced_metrics.get(key)
            self.max_reduced_metrics[key] = (
                value if previous is None else torch.maximum(previous, value)
            )


class PolicyTrainer(Actor, Configurable):
    """Updates policy based on collected Episode using TorchTitan components.

    Exposes separate `forward_backward` and `optim_step` endpoints, called
    explicitly by the controller.

    Args:
        config: PolicyTrainer.Config with all model/optimizer/parallelism settings.
        model_spec: TorchTitan model specification.
        hf_assets_path: Path to HF assets folder for checkpoint loading.
            Shared with the generator (both load from the same HF checkpoint).
        generator_dtype: Generator dtype (e.g. "bfloat16"). Needed to cast weights to generator dtype
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

    def reduce_forward_backward_metrics(
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

    def _forward_one_microbatch(
        self,
        microbatch: TrainingBatch,
        *,
        num_global_valid_tokens: torch.Tensor,
        logprob_config: TrainingLogprobConfig,
    ) -> tuple[LossOutput, PartialLogprobDrift]:
        """Forward one packed microbatch and return loss + per-rank drift.

        Caller owns `backward()` and metric accumulation; this method only
        computes the loss tensor and the verification artifacts.
        """
        model = self.model_parts[0]
        device = self.device
        token_ids = microbatch.token_ids.to(device)
        positions = microbatch.positions.to(device)
        loss_mask = microbatch.loss_mask.to(device)
        ref_logprobs = microbatch.ref_logprobs.to(device)
        advantages = microbatch.advantages.to(device)

        attention_masks = create_varlen_metadata_for_document(positions)

        with sl.log_trace_span("model_forward"):
            logits = model(
                token_ids, attention_masks=attention_masks, positions=positions
            )
        # compute_logprobs returns [B, L-1]; pad to [B, L] so positions align
        # with the loss_mask / ref_logprobs / advantages tensors.
        policy_logprobs = torch.nn.functional.pad(
            compute_logprobs(logits, token_ids, temperature=logprob_config.temperature),
            (1, 0),
            value=0.0,
        )

        with sl.log_trace_span("loss_fn"):
            loss_out: LossOutput = self.loss_fn(
                policy_logprobs=policy_logprobs,
                ref_logprobs=ref_logprobs,
                loss_mask=loss_mask,
                advantages=advantages,
                num_global_valid_tokens=num_global_valid_tokens,
            )

        drift = verify_logprob_identity(
            ref_logprobs=ref_logprobs,
            policy_logprobs=policy_logprobs,
            loss_mask=loss_mask,
            num_global_valid_tokens=num_global_valid_tokens,
        )
        return loss_out, drift

    @endpoint
    @sl.log_trace_span("forward_backward")
    async def forward_backward(
        self,
        training_steps: list[list[TrainingBatch]],
        *,
        num_global_valid_tokens: int,
        logprob_config: TrainingLogprobConfig,
    ) -> dict[str, float]:
        """Run one optimizer step's worth of forward+backward across all microsteps.

        Accumulates gradients across grad-accumulation microsteps and
        reduces SUM/MAX metrics over the loss mesh. The optimizer step is
        a separate endpoint call.

        Args:
            training_steps: Per-step microbatches, shape
                `[grad_accum_steps][dp_degree]`. The local rank picks
                `training_steps[step][self.dp_rank]` at each microstep.
            num_global_valid_tokens: Total trainable response tokens across
                all DP ranks for the full optimizer step. The controller
                computes this before sharding.
            logprob_config: Validated reference-logprob contract; trainer
                logprobs are computed under the same sampling-temperature
                transform as the generator's reference logprobs.

        Returns:
            dict[str, float] of globally-reduced metrics for this step.
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
        if not training_steps:
            raise ValueError("training_steps must contain at least one microstep")

        device = self.device
        global_tokens_tensor = torch.tensor(
            float(max(num_global_valid_tokens, 1)),
            device=device,
            dtype=torch.float32,
        )

        # Zero gradients once at the top so the per-microstep backward calls
        # accumulate into a single optimizer step.
        self.optimizers.zero_grad()
        metric_accumulator = MetricAccumulator()

        for step_batches in training_steps:
            if self.dp_rank >= len(step_batches):
                raise ValueError(
                    f"forward_backward got {len(step_batches)} DP shards but "
                    f"dp_rank={self.dp_rank} requires at least "
                    f"{self.dp_rank + 1}"
                )
            local_batch = step_batches[self.dp_rank]

            with sl.log_trace_span("forward_backward_microbatch"):
                loss_out, drift = self._forward_one_microbatch(
                    local_batch,
                    num_global_valid_tokens=global_tokens_tensor,
                    logprob_config=logprob_config,
                )
                with sl.log_trace_span("model_backward"):
                    loss_out.loss.backward()

            metric_accumulator.add_sum(
                {
                    **loss_out.sum_metrics,
                    "bit_wise/logprob_diff/mean": drift.logprob_diff_mean,
                    "bit_wise/ratio_tokens_different/mean": (
                        drift.ratio_tokens_different
                    ),
                }
            )
            metric_accumulator.add_max(
                {
                    **loss_out.max_metrics,
                    "bit_wise/logprob_diff/max": drift.logprob_diff_max,
                }
            )

        return self.reduce_forward_backward_metrics(
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

        with sl.log_trace_span("optim"):
            self.optimizers.step()
            self.lr_schedulers.step()

        self.policy_version += 1

        logger.debug(
            f"{os.getpid()=} PolicyTrainer optim_step done, "
            f"policy_version={self.policy_version}"
        )

        return OptimStepOutput(
            policy_version=self.policy_version,
            metrics={
                "train/grad_norm/mean": float(grad_norm.item()),
                "train/lr": current_lr,
                "train/policy_version": float(self.policy_version),
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

        When `direct_rdma=True`, weights are transferred directly from
        GPU to GPU via one-sided RDMA reads, bypassing StorageVolumes
        entirely. When `False`, data goes through StorageVolumes
        (which may themselves use RDMA as a transport internally).

        Note: we couple `is_rdma_available()` with `direct_rdma` here,
        but the two concepts are not identical -- StorageVolumes can also
        use RDMA as their transport layer. `direct_rdma` specifically
        means "skip StorageVolumes and let the destination read directly
        from the source's GPU memory".

        """
        from monarch.rdma import is_rdma_available

        await ts.put_state_dict(
            self.model.state_dict(),
            "model_state_dict",
            direct_rdma=is_rdma_available(),
            transfer_dtype=self._transfer_dtype,
        )
