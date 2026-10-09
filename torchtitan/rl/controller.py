# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TLDR:
_data_input_loop -> _rollout_loop -> _batcher_loop -> training_batch_queue -> _trainer_loop
           |               ^                    ^
           v               |                    |
           +-------RolloutGroupWorkBuffer-------+

Detailed diagram:

_data_input_loop                                      _rollout_loop[N] (group workers)
+--------------------------------------------------+  +--------------------------------------------------+
| group_buffer.wait_for_slot()                     |  | work = group_buffer.claim_next()                  |
| sample = rollouter.get_training_sample()         |  | group = rollouter.run_group_rollouts(work.sample) |
| work = RolloutGroupWork(group_id, sample)        |  | group_buffer.finalize_work(group)                 |
| group_buffer.add_work(work)                      |  +-----------------------+--------------------------+
+-----------------------+--------------------------+                          ^ |
                        |                                                     | |
                        | adds work entry                                     | | updates same entry
                        v                                                     | v
RolloutGroupWorkBuffer
+---------------------------------------------------------------------------------------------------------------------+
| active slots = (target_offpolicy_steps + 1) * num_prompts_per_train_step, or the adaptive buffer's demand           |
|                                                                                                                     |
| caller            group_buffer call                                            state / active slot                  |
| _data_input_loop  add_work(RolloutGroupWork)                                   WAITING; slot acquired               |
| _rollout_loop[N]  claim_next()                                                 WAITING -> INFLIGHT                  |
| _rollout_loop[N]  finalize_work(RolloutGroup)                                  INFLIGHT -> FINALIZED                |
| _batcher_loop     RolloutGroup = take_finalized()                              FINALIZED -> taken (slot still held) |
| _batcher_loop     release_active_groups(1, "untrainable_group")                slot released                        |
| _trainer_loop     record_step_start(trainer_policy_version)                    adaptive: demand updated             |
| _trainer_loop     release_active_groups(num_prompts_per_train_step, "trained")  slots released after weight pull     |
+---------------------------------------------------------------------------------------------------------------------+
                                                  |
                                                  | group = group_buffer.take_finalized()
                                                  v
_batcher_loop
+----------------------------------------------------------------------------------------+
| training_sample_group = training_sample_builder.build_from_group(rollout_group=group)  |
| if no trainable samples: group_buffer.release_active_groups(1, "untrainable_group")    |
| batch, trainable = batcher.add_training_samples(training_sample_group)                  |
| training_batch_queue.put(TrainerStepBatch)                                             |
+-----------------------+------------------------------+---------------------------------+
                        |  ^                           |
          add group     |  | maybe_training_batch      | put batch
                        v  |                           v
Batcher                                              training_batch_queue
+-------------------------------------------------+   +----------------------------------------------+
| accumulated TrainingSampleGroups                |   | size 1; holds TrainerStepBatch | None        |
| pack at num_prompts_per_train_step               |   +---------------------+------------------------+
+-------------------------------------------------+                         |
                                                                         | packed = training_batch_queue.get()
                                                                         v
_trainer_loop
+----------------------------------------------------------------------------------------------------------+
| train batch -> optim -> push/pull weights -> buffer.release_active_groups(num_prompts_per_train_step)     |
+----------------------------------------------------------------------------------------------------------+

Backpressure (each loop: what it consumes/produces, and what gates each side):
_data_input_loop
  produces: RolloutGroupWork into group_buffer
    waits for:    a free active slot (group_buffer.wait_for_slot)
    unblocked by: _trainer_loop release_active_groups(num_prompts_per_train_step, "trained") after the pull
                  (and _batcher_loop release_active_groups(1,"untrainable_group"))
_rollout_loop[N]
  consumes: a WAITING RolloutGroupWork (group_buffer.claim_next)
    waits for:    a claimable WAITING entry
    unblocked by: _data_input_loop group_buffer.add_work()
  produces: RolloutGroup (group_buffer.finalize_work)
    waits for:    nothing (admits its own claimed slot)
    unblocked by: n/a

_batcher_loop
  consumes: the oldest FINALIZED group inside the window (group_buffer.take_finalized)
    waits for:    a group inside the window becoming FINALIZED (any group when windowed_fifo_batches is None)
    unblocked by: _rollout_loop[N] group_buffer.finalize_work()
  produces: TrainerStepBatch (training_batch_queue.put)
    waits for:    a free training_batch_queue slot (maxsize=1)
    unblocked by: _trainer_loop training_batch_queue.get()
_trainer_loop
  consumes: a TrainerStepBatch (training_batch_queue.get)
    waits for:    a TrainerStepBatch in the queue
    unblocked by: _batcher_loop training_batch_queue.put()
"""

import asyncio
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field, replace
from enum import StrEnum

# PYTORCH_CUDA_ALLOC_CONF is set in torchtitan/rl/__init__.py (before torch is imported)
# and in train.py; see the note there.
import torch  # noqa: F401
import torchstore as ts

from monarch.actor import ProcMesh, this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable
from torchtitan.config.transform import LMHeadFP32OutputConverter
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.components.batcher import Batcher
from torchtitan.rl.components.data_stream_state import DataStreamState
from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.components.work_buffer import (
    AdaptiveRolloutGroupWorkBuffer,
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.rl.distributed.actors.generator import VLLMGeneratorActor
from torchtitan.rl.distributed.actors.trainer import TrainerActor
from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.distributed.weight_sync import WeightSyncManager
from torchtitan.rl.generator import SamplingConfig, VLLMGenerator
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.observability.controller import (
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    compute_rollout_metrics,
    MetricsTimer,
)
from torchtitan.rl.observability.rollout_recorder import RolloutSampleRecorder
from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan.rl.rollout.types import GenerateFn, ReleaseSessionFn
from torchtitan.rl.trainer import Trainer
from torchtitan.rl.types import Completion, TrainerStepBatch

logger = logging.getLogger(__name__)


class ValidationLoopMode(StrEnum):
    """What the trainer does while the step-0 or a periodic validation pass runs. The final pass
    always runs after training.

    Example, OVERLAP_TRAINING with `interval_steps=25` and 100 steps:
        step 0:   a pass starts on policy 0; training starts at once.
        step 7:   the pass ends and is logged at step 7, with `validation/launch_step` 0. Its slowest
                  rollouts sampled policies 0 to 6: `validation/min_policy_version/min` 0,
                  `validation/max_policy_version/max` 6.
        step 25:  the next pass starts on policy 25 or later. If the step-0 pass were still
                  running, this one would start at the end of the step where the step-0 pass ends.
        step 100: a pass still running is logged at step 100; then the final pass runs alone and
                  is also logged at step 100, with `validation/launch_step` 100.
    """

    PAUSE_TRAINER = "pause_trainer"
    """The trainer waits for each pass, so every rollout in it samples one policy."""

    OVERLAP_TRAINING = "overlap_training"
    """The trainer keeps stepping during each pass, so a long rollout's later turns can sample
    newer policies, also in the step-0 pass. For a clean step-0 score, use PAUSE_TRAINER or
    evaluate the checkpoint offline."""

    # TODO: DRAIN_TRAINER, one policy per pass while the trainer trains its backlog: fork branch 61-periodic-validation.


@dataclass(kw_only=True, slots=True)
class ValidationConfig:
    """Held-out validation that runs before training, every `interval_steps`, and after the last step."""

    num_samples: int = 20
    """Held-out prompts per validation pass, one rollout each. 0 skips validation."""

    interval_steps: int | None = None
    """Also validate after every `interval_steps` train steps; None validates only before and after."""

    greedy: bool = True
    """Sample at temperature 0; False samples like training (the generator's sampling config)."""

    loop_mode: ValidationLoopMode = ValidationLoopMode.PAUSE_TRAINER
    """What training does while a pass runs; see `ValidationLoopMode`."""


@dataclass(kw_only=True, slots=True)
class RLModelDefaults:
    """Model changes every RL run needs, applied to the shared model config before the trainer
    and generators copy it."""

    fp32_lm_head: bool = True
    """Swap the lm_head to `HiMidLoLinear`, so the trainer and generator compute fp32 logits with
    the same op. Turn off for a head `LMHeadFP32OutputConverter` cannot convert."""

    freeze_expert_bias: bool = True
    """Keep every MoE layer's expert bias at its loaded value. A moving bias flips more expert
    choices between the trainer and generator each step, so their logprob gap keeps growing.
    No-op on dense models."""

    # TODO: decide an RL aux-loss default once Qwen3 or GPT-OSS MoE trains with one
    #   (https://github.com/pytorch/torchtitan/pull/4772).

    def apply_(self, model: Decoder.Config) -> Decoder.Config:
        """Rewrite `model` in place with these defaults and return its root. Idempotent.

        Example:
            config = rl_grpo_qwen3_30b_a3b_varlen()
            config.model = config.model_defaults.apply_(config.model)
            # lm_head: Linear.Config -> HiMidLoLinear.Config
            # layers[i].moe.freeze_expert_bias: False -> True, for all 48 layers
        """
        if self.fp32_lm_head:
            model = LMHeadFP32OutputConverter.Config().build().convert(model)
        if self.freeze_expert_bias:
            for _fqn, moe_config, _parent, _attr in model.traverse(MoE.Config):
                moe_config.freeze_expert_bias = True
        return model


@dataclass(kw_only=True, slots=True)
class AsyncLoopConfig(Configurable.Config):
    num_training_steps: int = 10
    """Optimizer steps to run."""

    num_prompts_per_train_step: int = 8
    """Global number of prompt groups, across all DPs, whose surviving rollouts compose
    one train step (the global_batch_size, in groups)."""

    num_samples_per_prompt: int = 8
    """Sibling rollouts sampled per prompt (the GRPO group)."""

    target_offpolicy_steps: int = 3
    """Target steady-state offpolicy steps used to set the active buffer size to
    `(S + 1) * P`. Observed offpolicy steps are not guaranteed to equal this
    target: when rollout generation is the bottleneck, the buffer may not fill
    and observed offpolicy steps will be lower. A finite `windowed_fifo_batches` bounds
    how far a slow group may exceed this target; None leaves it unbounded. See
    ``torchtitan/rl/docs/windowed_fifo.md`` for details."""

    windowed_fifo_batches: int | None = None
    """FIFO look-ahead window in train batches.

    None (the default) is greedy: the batcher takes the oldest finished group
    anywhere in the buffer. An unfinished older group does not block a younger
    finished group. Maximum policy age is unbounded.

    Set to `n >= 1` to limit consumption to `n * P` group ids from the oldest
    group in the buffer. Maximum policy age is bounded by
    `target_offpolicy_steps + n`. A value of 1 is FIFO by batch. See
    ``torchtitan/rl/docs/windowed_fifo.md``."""

    group_buffer: RolloutGroupWorkBuffer.Config | AdaptiveRolloutGroupWorkBuffer.Config = field(
        default_factory=RolloutGroupWorkBuffer.Config
    )
    """Which buffer paces generation:
    - `RolloutGroupWorkBuffer` (default): `(target_offpolicy_steps + 1) * P` slots, windowed by
      `windowed_fifo_batches`.
    - `AdaptiveRolloutGroupWorkBuffer`: slots learned from the run, under its own age knobs. It ignores
      `windowed_fifo_batches`, and `target_offpolicy_steps` only sets
      `train_batch/pct_samples_over_target_age`."""
    training_sample_builder: TrainingSampleBuilder.Config = field(
        default_factory=TrainingSampleBuilder.Config
    )
    batcher: Batcher.Config = field(default_factory=Batcher.Config)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self) -> None:
        if self.num_prompts_per_train_step < 1:
            raise ValueError(
                "num_prompts_per_train_step must be >= 1, got "
                f"{self.num_prompts_per_train_step}"
            )
        if self.target_offpolicy_steps < 0:
            raise ValueError(
                f"target_offpolicy_steps must be >= 0, got {self.target_offpolicy_steps}"
            )
        if self.windowed_fifo_batches is not None and self.windowed_fifo_batches < 1:
            raise ValueError(
                f"windowed_fifo_batches must be None or >= 1, got {self.windowed_fifo_batches}"
            )

    @property
    def max_active_rollout_groups(self) -> int:
        return (self.target_offpolicy_steps + 1) * self.num_prompts_per_train_step

    @property
    def window_size(self) -> int | None:
        """FIFO look-ahead window in group ids, `windowed_fifo_batches * P`; None means no window."""
        if self.windowed_fifo_batches is None:
            return None
        return self.windowed_fifo_batches * self.num_prompts_per_train_step

    @property
    def max_concurrent_rollout_groups(self) -> int:
        """Most groups generating at once: every active slot of the fixed buffer, or the adaptive
        buffer's `generation_capacity`. Sizes the rollout workers and vLLM's `max_num_seqs`."""
        if isinstance(self.group_buffer, AdaptiveRolloutGroupWorkBuffer.Config):
            return self.group_buffer.generation_capacity
        return self.max_active_rollout_groups

    @property
    def max_offpolicy_steps(self) -> int | None:
        """Return the worst case consume-time offpolicy bound, or None without a window.

        For active buffer size `B`, window size `W`, and prompts per train step
        `P`, the bound is `(B + W - 2) // P`, which is `S + windowed_fifo_batches`.
        The adaptive buffer drops groups past its own `max_offpolicy_steps` instead.
        """
        if isinstance(self.group_buffer, AdaptiveRolloutGroupWorkBuffer.Config):
            return self.group_buffer.max_offpolicy_steps
        if self.window_size is None:
            return None
        return (
            self.max_active_rollout_groups + self.window_size - 2
        ) // self.num_prompts_per_train_step


def _log_slow_controller_gc(threshold_s: float = 1.0) -> None:
    """Experiment-only: log every full GC in this process slower than ``threshold_s``.

    The controller froze for 10-30 s every ~160 groups in a GB300 run (likely full GC over per-turn
    token lists); this confirms or rules that out.
    """
    import gc

    start = {}

    def callback(phase: str, info: dict) -> None:
        if info["generation"] != 2:
            return
        if phase == "start":
            start["t"] = time.monotonic()
        elif time.monotonic() - start.get("t", time.monotonic()) > threshold_s:
            logger.warning(
                "controller full GC took %.1f s (collected %d)",
                time.monotonic() - start["t"],
                info["collected"],
            )

    gc.callbacks.append(callback)


class Controller(Configurable):
    """Top-level RL async training orchestrator.

    Owns a `Trainer` actor (gradient updates), a `VLLMGenerator` actor
    (sampling), and a `Rollouter` (datasets + rubric + env construction).

    Check the docstring at the top of the file for more details.

    Example:

        config = recipes.rl_grpo_qwen3_0_6b_varlen()
        controller = config.build()
        trainer_mesh = ...        # provisioned by the caller (see train.py)
        generator_meshes = ...
        await controller.setup_async(
            trainer_mesh=trainer_mesh, generator_meshes=generator_meshes
        )
        await controller.run()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training."""

        model: Decoder.Config | None = None
        """Model config shared by the trainer and generator. `model_defaults` rewrites it before
        either copies it."""

        model_defaults: RLModelDefaults = field(default_factory=RLModelDefaults)
        """RL changes to `model`: fp32 logits and a frozen MoE expert bias."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets folder (model weights, tokenizer, config files)."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        async_loop: AsyncLoopConfig = field(default_factory=AsyncLoopConfig)
        """How the data->rollout->batch->train loop is sized and coordinated."""

        rollouter: Rollouter.Config
        """The rollouter: its datasets, envs, and rubric."""
        # TODO: support multiple rollouters for data mixing.

        tokenizer: HuggingFaceTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        """Tokenizer loaded from `hf_assets_path`."""

        renderer: RendererConfig
        """The model's chat template; renders messages to token ids and parses completions
        back. E.g. `from_renderers(Qwen3RendererConfig(enable_thinking=False))`."""

        rollout_recorder: RolloutSampleRecorder.Config = field(
            default_factory=RolloutSampleRecorder.Config
        )
        """JSONL recorder to save sampled rollouts to disk for further inspection and debugging."""

        trainer: Trainer.Config
        """Trainer config. Controls optimizer, training, parallelism."""

        # TODO: put generator, num generators and generator router in a separate config
        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

        num_generators: int = 1
        """Number of generator replicas to spawn as separate proc meshes.

        This is distinct from intra-generator parallelism controlled by
        ``generator.parallelism``. Total generator GPU/process usage is
        ``num_generators * generator_world_size``.
        """

        generator_router: InterGeneratorRouter.Config = field(
            default_factory=InterGeneratorRouter.Config
        )
        """Generator routing strategy configuration."""

        # TODO: rename it to metrics_processor
        metrics: m.MetricsProcessor.Config = field(
            default_factory=m.MetricsProcessor.Config
        )

        def maybe_log(self) -> None:
            debug = self.trainer.debug
            config_dict = self.to_dict()
            if debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(config_dict, indent=2, ensure_ascii=False)}"
                )

            if debug.save_config_file is not None:
                config_file = os.path.join(self.dump_folder, debug.save_config_file)
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, "w") as file:
                    json.dump(config_dict, file, indent=2)
                logger.info(f"Saved job configs to {config_file}")

        def __post_init__(self):
            if self.num_generators < 1:
                raise ValueError(
                    f"num_generators must be at least 1, got {self.num_generators}"
                )
            if self.generator.checkpointer is not None:
                raise ValueError(
                    "Generator checkpoint must be disabled in the RL loop "
                    "(weights are synced from the trainer via TorchStore). "
                    "Set generator.checkpointer=None."
                )
            if self.trainer.training.num_tokens_per_train_step != -1:
                warnings.warn(
                    "trainer.training.num_tokens_per_train_step is ignored by "
                    "the RL loop; configure async_loop.num_prompts_per_train_step "
                    "to control optimizer-step boundaries.",
                    stacklevel=2,
                )
            if self.trainer.parallelism.enable_sequence_parallel:
                sp_degree = self.trainer.parallelism.tensor_parallel_degree
                max_context_length = self.trainer.training.max_context_length
                if sp_degree > 1 and max_context_length % sp_degree != 0:
                    raise ValueError(
                        "training.max_context_length "
                        f"({max_context_length}) must be divisible "
                        f"by sequence parallel degree ({sp_degree})."
                    )

            # TODO: add a check so that all seq_len related variables make sense
            # e.g. rollout max length cannot be larger than the model max_seq_len
            # or the packing len, etc.

            if self.trainer.debug.batch_invariant:
                if torch.version.hip is not None:
                    raise ValueError(
                        "batch_invariant mode is not supported on ROCm: the varlen "
                        "attention path cannot force num_splits=1 (rejected by ROCm), "
                        "so split-k reductions are non-deterministic."
                    )
                if not self.trainer.debug.deterministic:
                    raise ValueError("batch_invariant requires deterministic=True")
                # The trainer forward must compute in bf16 to match the bf16
                # generator, via FSDP mixed precision. The trainer always wraps
                # the model in FSDP (even at data_parallel_shard_degree=1, where
                # FSDP acts purely as a mixed-precision boundary), so
                # mixed_precision_param == "bfloat16" casts the fp32 master
                # weights to bf16 for the forward before any matmul.
                if self.trainer.training.mixed_precision_param != "bfloat16":
                    raise ValueError(
                        "batch_invariant requires the trainer forward to compute "
                        "in bfloat16 to match the generator. Set "
                        "training.mixed_precision_param='bfloat16' (fp32 master "
                        "weights, bf16-cast forward via FSDP mixed precision). "
                        "Got mixed_precision_param="
                        f"{self.trainer.training.mixed_precision_param!r}."
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

    def __init__(self, config: Config):
        # Here, not in `Config.__post_init__`, which also runs when a recipe constructs the
        # config: the lm_head swap cannot be undone, so a later `model_defaults` opt-out would be lost.
        config.model = config.model_defaults.apply_(config.model)
        self.config = config
        config.maybe_log()
        self.trainer: Trainer | None = None
        self.generator_router: InterGeneratorRouter | None = None
        # Resume step (0 = fresh); set in setup_async from the loaded checkpoint.
        self.start_step = 0
        self._data_stream = DataStreamState()
        self._proc_meshes = []
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
        self.renderer = config.renderer.build(tokenizer=self.tokenizer)

        # Carry the base seed and renderer stop tokens on the sampling config so
        # the generator reads them off each request; the rollouter offsets the
        # seed per sample. Avoids the generator depending on request_id format.
        self._sampling = replace(
            config.generator.sampling,
            seed=config.generator.debug.seed,
            stop_token_ids=list(self.renderer.get_stop_token_ids()),
        )
        self._rollouter: Rollouter = config.rollouter.build()
        self.rollout_recorder = config.rollout_recorder.build(
            dump_dir=config.dump_folder
        )
        # With `ValidationLoopMode.OVERLAP_TRAINING`: the pass running beside training.
        self._validation_task: asyncio.Task[list[m.Metric]] | None = None

    async def close(self):
        """Best-effort: tear down actors, close metric backends, then stop proc meshes."""
        logger.info("Closing: tearing down actors and process meshes.")

        # Still set only if run() raised or was cancelled. Wait for the cancel to finish before
        # the rollouter it uses is closed below, and log an error nothing re-raised.
        if self._validation_task is not None:
            self._validation_task.cancel()
            (result,) = await asyncio.gather(
                self._validation_task, return_exceptions=True
            )
            if isinstance(result, Exception):
                logger.error(
                    f"{self._validation_task.get_name()} failed", exc_info=result
                )

        if self.trainer is not None:
            try:
                await self.trainer.close.call()
            except Exception:
                logger.exception("trainer.close failed")

        try:
            await self._rollouter.close()
        except Exception:
            logger.exception("rollouter.close failed")

        if self.generator_router is not None:
            try:
                close_results = await self.generator_router.close_generators.call_one()
                for idx, result in enumerate(close_results):
                    if isinstance(result, BaseException):
                        actor_name = (
                            "generator"
                            if len(close_results) == 1
                            else f"generator[{idx}]"
                        )
                        logger.error(
                            "%s.close failed",
                            actor_name,
                            exc_info=(type(result), result, result.__traceback__),
                        )
            except Exception:
                logger.exception("generator_router.close_generators failed")

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

    def _checkpoint_step_dir(self, step: int) -> str | None:
        """The trainer's checkpoint folder for ``step``, or None without a checkpointer.

        Example: dump_folder="outputs/rl", checkpointer.folder="checkpoint", step=10
            -> "outputs/rl/checkpoint/step-10"
        """
        checkpointer = self.config.trainer.checkpointer
        if checkpointer is None:
            return None
        return os.path.join(
            self.config.dump_folder, checkpointer.folder, f"step-{step}"
        )

    def _get_rank_0_value(self, result):
        """Extract rank 0 result from a Monarch ValueMesh.

        Monarch actor endpoints return results from all ranks in the mesh.
        This method picks out rank 0's result. This should be used in cases
        where all ranks return the same result.
        """
        return result.get(0)

    def _make_generate_fn(self, metrics_prefix: str) -> GenerateFn:
        """Build the rollouter's `GenerateFn`: route a completion via the generator router, namespacing
        generation metrics with `metrics_prefix` and pinning sticky routing on `routing_session_id` (a sample's
        turns reuse one generator's prefix KV)."""
        # TODO: make this a pluggable config (a GenerateFn factory) so non-router generate backends can be swapped in.
        # Bind the router handle to a local so the closure captures it instead of
        # `self`. A GenerateFn may be shipped to another process, where an actor
        # handle serializes cheaply and the whole controller does not.
        generator_router = self.generator_router

        @sl.log_trace_span("generate")
        async def generate(
            prompt_token_ids: list[int],
            *,
            request_id: str,
            group_id: int,
            routing_session_id: str | None = None,
            sampling_config: SamplingConfig | None = None,
        ) -> Completion | None:
            return await generator_router.generate.call_one(
                prompt_token_ids,
                request_id=request_id,
                group_id=group_id,
                routing_session_id=routing_session_id,
                sampling_config=sampling_config,
                metrics_prefix=metrics_prefix,
            )

        return generate

    def _make_release_session_fn(self) -> ReleaseSessionFn:
        """Build the rollouter's `ReleaseSessionFn`: tell the generator router a rollout ended."""
        generator_router = self.generator_router

        async def release_session(*, group_id: int, routing_session_id: str) -> None:
            await generator_router.release_session.call_one(
                group_id=group_id, routing_session_id=routing_session_id
            )

        return release_session

    @sl.log_trace_span("setup_async")
    async def setup_async(
        self,
        *,
        trainer_mesh: ProcMesh,
        generator_meshes: list[ProcMesh],
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Kept separate from ``__init__`` because actor spawning, torch
        elastic env setup, TorchStore initialization, and the initial
        weight push/pull are all ``await``-based runtime side effects
        that cannot run in a synchronous constructor.

        The trainer and generator meshes are provisioned by the caller (see
        ``spawn_proc_mesh``). The router and rollout worker meshes are created
        on the controller host. This method spawns the actors and synchronizes
        initial weights from trainer to generator. Must be called before
        :meth:`run`.

        Args:
            trainer_mesh: ProcMesh the trainer actor is spawned on.
            generator_meshes: ProcMesh objects the generator actors are spawned on.
        """
        _log_slow_controller_gc()
        # Peak concurrent rollout sequences; sizes max_num_seqs below. An overlapped pass adds its
        # rollouts to training's.
        # TODO: training rollouts also keep generating during a paused periodic pass, so max()
        #   under-sizes that case too; kept to leave the default mode's max_num_seqs unchanged.
        async_loop = self.config.async_loop
        num_training_rollouts = (
            async_loop.max_concurrent_rollout_groups * async_loop.num_samples_per_prompt
        )
        validation = async_loop.validation
        rollout_concurrency = (
            num_training_rollouts + validation.num_samples
            if validation.loop_mode == ValidationLoopMode.OVERLAP_TRAINING
            else max(num_training_rollouts, validation.num_samples)
        )
        config = self.config
        if not generator_meshes:
            raise ValueError("setup_async requires at least one generator mesh")

        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        generator_dp_degree = max(config.generator.parallelism.data_parallel_degree, 1)
        num_generator_dp_shards = len(generator_meshes) * generator_dp_degree

        # Ceiling (not target) for the generator's max_num_seqs: the per-generator
        # upper bound on concurrently scheduled sequences. vLLM may admit fewer if KV
        # is tight; this also sets CUDA-graph capture sizes.
        max_num_seqs = min(
            math.ceil(rollout_concurrency / num_generator_dp_shards), 512
        )

        logger.info(
            "max_num_seqs=%d per generator (rollout_concurrency=%d / generator_dp_shards=%d)",
            max_num_seqs,
            rollout_concurrency,
            num_generator_dp_shards,
        )

        # TODO(observability): the mesh_spawn span wraps ~80 LoC of branching
        # provisioner logic. Pull a PerHostProvisioner.spawn_meshes(...) helper and
        # shrink this span to a single call.
        with sl.log_trace_span("mesh_spawn"):
            # One process, so the router is a singleton and every caller reaches
            # it with `call_one`. It gets its own mesh rather than sharing the
            # controller's process so routing does not contend with the training
            # loop for the controller's GIL.
            router_mesh = this_host().spawn_procs(per_host={"cpus": 1})
            # Store proc meshes for cleanup
            self._proc_meshes = [router_mesh, trainer_mesh, *generator_meshes]

            await setup_torch_elastic_env_async(trainer_mesh)
            for generator_mesh in generator_meshes:
                await setup_torch_elastic_env_async(generator_mesh)

            # Spawn actors on their respective meshes
            self.trainer = trainer_mesh.spawn(
                "trainer",
                TrainerActor,
                config.trainer,
                model_config=config.model,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                max_num_documents=config.async_loop.batcher.max_num_documents,
                output_dir=config.dump_folder,
            )

            # TODO: torch.compile with aot_eager backend (inductor crashes the vLLM engine on the shared model path).
            generators = []
            for idx, generator_mesh in enumerate(generator_meshes):
                actor_name = (
                    "generator" if len(generator_meshes) == 1 else f"generator_{idx}"
                )
                generator = generator_mesh.spawn(
                    actor_name,
                    VLLMGeneratorActor,
                    config.generator,
                    model_config=config.model,
                    model_path=config.hf_assets_path,
                    max_num_seqs=max_num_seqs,
                    output_dir=config.dump_folder,
                )
                generators.append(generator)
            self.generator_router = router_mesh.spawn(
                "generator_router",
                InterGeneratorRouter,
                config.generator_router,
                generators=generators,
                enable_cpu_weight_prefetch=config.generator.enable_cpu_weight_prefetch,
                forward_session_releases=config.generator.hold_session_kv,
                group_size=config.async_loop.num_samples_per_prompt,
            )

            await self._rollouter.setup_async(
                tokenizer_config=config.tokenizer,
                renderer_config=config.renderer,
                hf_assets_path=config.hf_assets_path,
                release_session_fn=self._make_release_session_fn(),
            )

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        with sl.log_trace_span("torchstore_init"):
            await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # Resume: __init__ ran CheckpointManager.load(); read back the restored policy_version
        # (0 if fresh) so the loop resumes at the right step and generators pull at that version.
        # The data stream (dataset positions, untrained prompts, group ids) is restored from the
        # same step-N folder. Partially generated rollouts are not; their prompts are regenerated.
        self.start_step = self._get_rank_0_value(
            await self.trainer.get_policy_version.call()
        )
        if self.start_step > 0:
            logger.info(f"Resuming RL training from step {self.start_step}")
            step_dir = self._checkpoint_step_dir(self.start_step)
            if step_dir is not None:
                self._data_stream.load(step_dir, self._rollouter)

        # Start each generator's engine loop on all ranks once, before any
        # rank-0-only generate / pull (rank 0 drives the followers through this
        # loop, so every rank must be running it first).
        with sl.log_trace_span("generator_start_engine_loop"):
            await self.generator_router.start_engine_loop.call_one()

        # Initial weight sync: only the trainer loads weights; generators pull at start_step.
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self.trainer.push_model_state_dict.call()
        with sl.log_trace_span("generator_pull_model_state_dict"):
            await self.generator_router.pull_model_state_dict.call_one(self.start_step)

    # TODO: fold validation into a Validator(Configurable) the controller attaches, instead of these methods.
    @sl.log_trace_span("_collect_validation_rollouts")
    async def _collect_validation_rollouts(
        self, *, num_groups: int, sampling: SamplingConfig, step: int
    ) -> tuple[list[RolloutGroup], list[m.Metric]]:
        """Sample held-out prompts, run each once (n=1) concurrently, and emit validation metrics."""
        # TODO: group_size=1 (best-of-1) only. Support best-of-N.
        generate = self._make_generate_fn(metrics_prefix="validation_generator")
        # TODO(naming): reserve "sample" for TrainingSample; rename the rollouter's raw-prompt "sample" -> "prompt"/"data_input".
        samples = [self._rollouter.get_validation_sample() for _ in range(num_groups)]
        group_results = await asyncio.gather(
            *(
                self._rollouter.run_group_rollouts(
                    generate_fn=generate,
                    sample=sample,
                    # Negative ids keep validation disjoint from training group ids, so their
                    # request_ids can't collide in the shared engine (e.g. post-validation).
                    group_id=-(i + 1),
                    group_size=1,
                    sampling=sampling,
                )
                for i, sample in enumerate(samples)
            ),
            return_exceptions=True,
        )
        # Validation group ids are reused every validation, so their cache salts must not outlive it.
        await self.generator_router.release_groups.call_one(
            [-(i + 1) for i in range(num_groups)]
        )

        # Keep the groups that succeeded; log + count the ones that raised.
        rollout_groups: list[RolloutGroup] = []
        num_failed_groups = 0
        for i, result in enumerate(group_results):
            if isinstance(result, BaseException):
                logger.error(
                    f"validation group {-(i + 1)} (step={step}) failed; dropping",
                    exc_info=(type(result), result, result.__traceback__),
                )
                num_failed_groups += 1
                continue
            rollout_groups.append(result)

        rollouts = [rollout for group in rollout_groups for rollout in group.rollouts]
        metrics = compute_rollout_metrics(prefix="validation", rollouts=rollouts)
        metrics.append(
            m.Metric("validation/group_failures", m.Sum(float(num_failed_groups)))
        )
        # An overlapped pass is logged at the step it ends; this keeps the step it started at.
        metrics.append(m.Metric("validation/launch_step", m.NoReduce(step)))
        # Policies the pass sampled. One, unless the pass overlaps training: then a weight sync
        # during a rollout makes its later tokens sample a newer policy.
        turns = [turn for rollout in rollouts for turn in rollout.turns]
        is_mixed_policy = [
            rollout.turns[-1].max_policy_version > rollout.turns[0].min_policy_version
            for rollout in rollouts
            if rollout.turns
        ]
        metrics += [
            m.Metric(
                "validation/min_policy_version",
                m.Min.from_list([turn.min_policy_version for turn in turns]),
            ),
            m.Metric(
                "validation/max_policy_version",
                m.Max.from_list([turn.max_policy_version for turn in turns]),
            ),
            m.Metric(
                "validation/mixed_policy_rollouts", m.Mean.from_list(is_mixed_policy)
            ),
        ]
        return rollout_groups, metrics

    # TODO: we currently determine validation.num_samples
    # but what if i want to run the entire dataset?
    @sl.log_trace_span("validate")
    async def validate(self, *, step: int) -> list[m.Metric]:
        """Run one rollout per held-out prompt.

        Args:
            step: Training step this validation pass belongs to (0 for the
                pre-training pass); tagged into logged rollout samples.

        Returns:
            Validation rollout metrics, generation metrics, and validation
            timing.
        """
        # TODO: investigate using pass@k for validation.
        t_validate_start = time.perf_counter()
        validation = self.config.async_loop.validation
        if validation.num_samples == 0:  # skip validation (e.g. loss guard CI)
            return []
        sampling = (
            replace(self._sampling, temperature=0.0, top_p=1.0)
            if validation.greedy
            else self._sampling
        )

        rollout_groups, validation_metrics = await self._collect_validation_rollouts(
            num_groups=validation.num_samples, sampling=sampling, step=step
        )

        self.rollout_recorder.record(is_validation=True, rollout_groups=rollout_groups)

        t_validate_s = time.perf_counter() - t_validate_start
        validation_metrics.append(m.Metric("timing/validate", m.NoReduce(t_validate_s)))
        return validation_metrics

    async def run(self) -> None:
        """Start every async loop and run until training completes or a stage crashes.

        Producers (_data_input_loop, _rollout_loop[N], _batcher_loop) loop forever; _trainer_loop is the
        only finite loop -- it runs num_training_steps, then returns, which drives shutdown.

        Shutdown (healthy):  _trainer_loop finishes N steps -> run() finally ->
          group_buffer.close()  (wakes _data_input_loop / _rollout_loop / _batcher_loop blocked on the buffer)
          -> task.cancel()      (wakes anything blocked on training_batch_queue.put/get; close does NOT wake these)
          -> gather(..., return_exceptions=True)
        Shutdown (crash):    any loop raises -> appears in `done` -> run() re-raises -> same finally.
        """
        async_loop = self.config.async_loop
        num_training_steps = async_loop.num_training_steps
        pre_validation_task: asyncio.Task[list[m.Metric]] | None = None
        if self.start_step == 0:
            logger.info(
                f"Running pre-training validation; then {num_training_steps} steps of async RL training"
            )
            sl.log_trace_instant("validation_start")
            if async_loop.validation.loop_mode == ValidationLoopMode.OVERLAP_TRAINING:
                # Generators already hold policy 0 (setup_async), so training can start right away.
                self._start_validation(step=0)
                pre_validation_task = self._validation_task
                pre_validation = {}  # read from the task after training
            else:
                pre_validation = await self._validate_and_log(step=0)
        else:
            # The pre-training baseline was measured by the first run. Re-validating on every
            # restart would cost a full validation pass each time a preempted job resumes.
            logger.info(
                f"Resuming at step {self.start_step}: skipping pre-training validation; "
                f"training through step {num_training_steps}"
            )
            pre_validation = {}
        sl.log_trace_instant("training_start")

        # Trainer policy version, seeded from the resumed step; advances at each optimizer step.
        self._trainer_policy_version = self.start_step

        if isinstance(async_loop.group_buffer, AdaptiveRolloutGroupWorkBuffer.Config):
            logger.info(f"Adaptive rollout buffer: {async_loop.group_buffer}")
            self._group_buffer = async_loop.group_buffer.build(
                num_prompts_per_train_step=async_loop.num_prompts_per_train_step,
                policy_version=self.start_step,
            )
        else:
            # Depth (S + 1) * P targets the mean policy age; the window caps the max age.
            max_active_rollout_groups = async_loop.max_active_rollout_groups
            window_size = async_loop.window_size
            logger.info(
                f"max_active_rollout_groups={max_active_rollout_groups}, "
                f"target_offpolicy_steps={async_loop.target_offpolicy_steps}, "
                f"windowed_fifo_batches={async_loop.windowed_fifo_batches}, "
                f"max_offpolicy_steps={async_loop.max_offpolicy_steps}"
            )

            self._group_buffer = async_loop.group_buffer.build(
                max_active_rollout_groups=max_active_rollout_groups,
                window_size=window_size,
            )

        # Overlaps each step's weight handoff (push -> pull -> buffer-slot release) with the next step
        self._weight_sync = WeightSyncManager(
            trainer=self.trainer,
            generator_router=self.generator_router,
            group_buffer=self._group_buffer,
            num_prompts_per_train_step=async_loop.num_prompts_per_train_step,
        )

        # training_sample_builder
        training_sample_builder = async_loop.training_sample_builder.build()

        # batcher
        batcher = async_loop.batcher.build(
            num_tokens_per_microbatch_per_dp_rank=(
                self.config.trainer.training.num_tokens_per_microbatch_per_dp_rank
            ),
            max_context_length=self.config.trainer.training.max_context_length,
            num_prompts_per_train_step=async_loop.num_prompts_per_train_step,
            dp_degree=self.trainer_dp_degree,
            pad_id=self.tokenizer.eos_id,
            temperature=self._sampling.temperature,
        )

        # training_batch_queue
        training_batch_queue: asyncio.Queue[TrainerStepBatch | None] = asyncio.Queue(
            maxsize=1
        )

        # rollout_loop
        generate_fn = self._make_generate_fn(metrics_prefix="generator")

        # One rollout worker per group that may generate at once: lets generation fill every active slot,
        # including the cold start (step 0 fills every active slot, not just num_prompts_per_train_step per wave).
        # TODO: support warm start
        rollout_tasks = [
            asyncio.create_task(
                self._rollout_loop(
                    group_buffer=self._group_buffer,
                    generate_fn=generate_fn,
                ),
                name=f"rollout_worker_{group_worker_id}",
            )
            for group_worker_id in range(async_loop.max_concurrent_rollout_groups)
        ]

        # data_input_loop
        data_input_task = asyncio.create_task(
            self._data_input_loop(self._group_buffer), name="data_input"
        )

        # training_sample_batcher_loop
        batcher_task = asyncio.create_task(
            self._batcher_loop(
                group_buffer=self._group_buffer,
                training_sample_builder=training_sample_builder,
                batcher=batcher,
                training_batch_queue=training_batch_queue,
            ),
            name="batcher",
        )

        # trainer_loop
        trainer_task = asyncio.create_task(
            self._trainer_loop(
                training_batch_queue, num_training_steps=num_training_steps
            ),
            name="trainer",
        )

        # run everything until trainer finishes its number of steps
        # or some other loop breaks
        background_tasks = [
            data_input_task,
            *rollout_tasks,
            batcher_task,
        ]
        try:
            done, _ = await asyncio.wait(
                [trainer_task, *background_tasks], return_when=asyncio.FIRST_COMPLETED
            )
            # The trainer is the finite clock: it runs num_training_steps then returns -> training is done.
            # Producers loop forever, so a producer in `done` means it crashed (await re-raises) or wrongly
            # returned cleanly (the RuntimeError). Check producers even when the trainer also finished this
            # wakeup, so a simultaneous producer crash isn't hidden behind the finished trainer.
            for task in done:
                if task is trainer_task:
                    continue
                await task  # raises if task crashed; returns if task ended cleanly
                raise RuntimeError(f"{task.get_name()} exited unexpectedly")
            if trainer_task in done:
                await trainer_task
        finally:
            # Graceful first: buffer.close() (awaited) wakes loops blocked on the buffer so they return.
            # Then cancel covers anything blocked on the queue (which close does not wake); gather awaits all.
            await self._group_buffer.close()
            for task in (*background_tasks, trainer_task):
                task.cancel()
            await asyncio.gather(
                *background_tasks, trainer_task, return_exceptions=True
            )

        # A pass that overlapped training may still be running; no new training rollouts start now.
        # Awaited, not cancelled: the generators would keep serving its requests, and the final
        # pass reuses their request ids.
        if self._validation_task is not None:
            logger.info(
                f"Training done; waiting for {self._validation_task.get_name()} before the final pass"
            )
            # Wait without raising; `_log_finished_validation` re-raises a failed pass's error.
            await asyncio.wait([self._validation_task])
            self._log_finished_validation(step=num_training_steps)
        if pre_validation_task is not None:
            pre_validation = m.MetricsProcessor._aggregate_metrics(
                pre_validation_task.result()
            )

        # Post-training validation (held-out eval after the final step).
        post_validation = await self._validate_and_log(step=num_training_steps)
        self._log_reward_delta(pre_validation, post_validation)

    async def _validate_and_log(self, *, step: int) -> dict[str, float]:
        """Run one validation pass, log it, and return its aggregated values for the pre/post delta."""
        metrics = await self.validate(step=step)
        self.metrics_processor.log(step=step, metrics=metrics, is_validation=True)
        return m.MetricsProcessor._aggregate_metrics(metrics)

    def _start_validation(self, *, step: int) -> None:
        """Start a validation pass beside training; `_log_finished_validation` logs it once it ends.

        The caller makes sure the generators hold at least policy `step` and no pass is running.
        """
        # TODO: validation rollouts queue behind training rollouts for generator slots and
        # env-server sandboxes, so a periodic pass can take longer than alone. prime-rl dispatches
        # eval before new training rollouts (PREFER_EVAL); here that needs a request priority in
        # the env server and the generator router.
        logger.info(f"Starting validation at step {step}, beside training")
        self._validation_task = asyncio.create_task(
            self.validate(step=step), name=f"validation_step_{step}"
        )

    def _log_finished_validation(self, *, step: int) -> None:
        """If the pass started by `_start_validation` has ended, log it at `step`; re-raise its error.

        Logged at the current step, not its launch step: metric backends need steps that never go back.
        """
        task = self._validation_task
        if task is None or not task.done():
            return
        self._validation_task = None
        metrics = task.result()
        logger.info(f"{task.get_name()} ended; logging it at step {step}")
        self.metrics_processor.log(step=step, metrics=metrics, is_validation=True)

    def _log_reward_delta(self, pre: dict[str, float], post: dict[str, float]) -> None:
        """Console pre/post reward summary, visible without scrolling back through the loop."""
        reward_keys = sorted(key for key in set(pre) | set(post) if "reward" in key)
        logger.info("=" * 60)
        logger.info("Validation reward (pre / post):")
        # With `ValidationLoopMode.OVERLAP_TRAINING`, "pre" is the step-0 pass, which ran beside training.
        newest_pre_policy = pre.get("validation/max_policy_version/max", 0)
        if newest_pre_policy > 0:
            logger.info(
                f"  pre sampled policies 0 to {newest_pre_policy:.0f}: not a clean pre-training score"
            )
        for key in reward_keys:
            logger.info(
                f"  {key}:  {pre.get(key, float('nan')):+.3f}  /  {post.get(key, float('nan')):+.3f}"
            )
        logger.info("=" * 60)

    async def _data_input_loop(self, group_buffer: RolloutGroupWorkBuffer) -> None:
        """produces a RolloutGroupWork into group_buffer.
        waits for:    a free active slot (group_buffer.wait_for_slot)
        unblocked by: _trainer_loop release_active_groups(num_prompts_per_train_step, "trained")
            after the pull (and _batcher_loop release_active_groups(1,"untrainable_group"))

        Separate from `_rollout_loop`, so slow data prep (e.g. on-the-fly question generation) overlaps
        generation instead of serializing in front of it.
        """

        async def read_training_sample() -> object:
            with sl.log_trace_span("get_training_sample"):
                # to_thread: Dont block on dataset reads
                return await asyncio.to_thread(self._rollouter.get_training_sample)

        # TODO(perf): Slots are current released in batches, while this loop is a single producer.
        # we could a) increase the number of threads; b) revisit how we release slots and see if
        # we can release them on the batcher while still preserving max offpolicy steps.
        # finally, c) we need to check how will this data input loop truly overlaps with the rollout loop.
        while await group_buffer.wait_for_slot():
            # After a resume, prompts that were admitted but not trained come back first.
            group_id, sample = await self._data_stream.admit_next(read_training_sample)
            await group_buffer.add_work(
                RolloutGroupWork(
                    group_id=group_id,
                    sample=sample,
                )
            )
        logger.info("Buffer closed; data input loop stopping")

    async def _rollout_loop(
        self,
        *,
        group_buffer: RolloutGroupWorkBuffer,
        generate_fn: GenerateFn,
    ) -> None:
        """Generate + score one group at a time; a failed group becomes an empty group + a failure metric.

        Staleness is bounded by the buffer's active-slot budget. Raw rollouts are recorded before any drop,
        so dropped groups stay inspectable on disk.

        consumes: a WAITING RolloutGroupWork (group_buffer.claim_next)
            waits for:    a claimable WAITING entry
            unblocked by: _data_input_loop group_buffer.add_work()
        produces: RolloutGroup (group_buffer.finalize_work)
            waits for:    nothing (admits its own claimed slot)
            unblocked by: n/a
        """
        while True:
            work = await group_buffer.claim_next()
            if work is None:  # group_buffer closed/shutdown signal
                logger.info("Buffer closed; rollout worker stopping")
                return
            try:
                with sl.log_trace_span("rollout_group"):
                    group = await self._rollouter.run_group_rollouts(
                        generate_fn=generate_fn,
                        sample=work.sample,
                        group_id=work.group_id,
                        group_size=self.config.async_loop.num_samples_per_prompt,
                        sampling=self._sampling,
                    )
                group.metrics = compute_rollout_metrics(
                    prefix="rollout", rollouts=group.rollouts
                )

                # save rollout for inspection
                self.rollout_recorder.record(
                    is_validation=False,
                    rollout_groups=[group],
                )
            except Exception:
                logger.exception(f"rollout group {work.group_id} failed; dropping")
                group = RolloutGroup(
                    group_id=work.group_id,
                    rollouts=[],
                    metrics=[m.Metric("rollout/group_failures", m.Sum(1.0))],
                )
            # The group makes no more generation calls, so its cache salts can go.
            await self.generator_router.release_groups.call_one([work.group_id])
            await group_buffer.finalize_work(group)
            # Drop the finished group now: this task waits on its next group for up to hours, and
            # 1,000+ held groups (every turn's full prompt as Python lists) made each full GC
            # freeze the controller for 10-30+ s.
            del group, work

    async def _batcher_loop(
        self,
        *,
        group_buffer: RolloutGroupWorkBuffer,
        training_sample_builder: TrainingSampleBuilder,
        batcher: Batcher,
        training_batch_queue: "asyncio.Queue[TrainerStepBatch | None]",
    ) -> None:
        """Take finalized groups, build training_samples, accumulate them, and queue each ready training batch.

        On a clean close/shutdown the group_buffer drains and returns None; we forward a `None` sentinel
        so the trainer stops.

        consumes: the oldest FINALIZED group inside the window (group_buffer.take_finalized)
            waits for:    a group inside the window becoming FINALIZED (any group when windowed_fifo_batches is None)
            unblocked by: _rollout_loop[N] group_buffer.finalize_work()
        produces: TrainerStepBatch (training_batch_queue.put)
            waits for:    a free training_batch_queue slot (maxsize=1)
            unblocked by: _trainer_loop training_batch_queue.get()
        """
        # Policy version that will train the batch being assembled: batches train in order, one per version.
        consuming_policy_version = self.start_step
        while True:
            rollout_group = await group_buffer.take_finalized(
                consuming_policy_version=consuming_policy_version
            )
            if rollout_group is None:  # closed and drained
                logger.info("Buffer drained; batcher loop stopping")
                break
            # In a thread: turning a group's samples into tensors walks every token
            with sl.log_trace_span("training_sample_builder"):
                training_sample_group = await asyncio.to_thread(
                    training_sample_builder.build_from_group,
                    rollout_group=rollout_group,
                )

            # We put a group in. We may get a batch back
            # if there are enough accumulated trainable groups to return one.
            with sl.log_trace_span("batcher_pack"):
                maybe_training_batch, group_is_trainable = await asyncio.to_thread(
                    batcher.add_training_samples,
                    training_sample_group=training_sample_group,
                )
            if not group_is_trainable:
                await group_buffer.release_active_groups(1, reason="untrainable_group")
            if maybe_training_batch is not None:
                await training_batch_queue.put(maybe_training_batch)
                consuming_policy_version += 1
        await training_batch_queue.put(None)
        # TODO(async-rl): if finite datasets are supported, drain a final partial batch here.

    async def _trainer_loop(
        self,
        training_batch_queue: "asyncio.Queue[TrainerStepBatch | None]",
        *,
        num_training_steps: int,
    ) -> None:
        """Run num_training_steps optimizer steps: train one packed batch, publish trainer weights,
        then pull them into generators, log metrics.

        NOTE: Weight sync is overlapped with the training step.
        Trainer push:
            - Called after optimizer.step()
            - Awaited before the next forward/backward (see WeightSyncManager)
        Generator pull:
            - Called after push completes.
            - Awaited before next push (weights changes then)

        Impact on off-policiness: The buffer guarantees that no sample will be born stale,
        as long as we call `self._group_buffer.release_active_groups` after the pull.

        consumes: a TrainerStepBatch (training_batch_queue.get)
            waits for:    a TrainerStepBatch in the queue
            unblocked by: _batcher_loop training_batch_queue.put()
        """
        # With `ValidationLoopMode.OVERLAP_TRAINING`: a validation step asked for a pass, which
        # waits while another pass runs.
        validation_requested = False
        for step in range(self.start_step + 1, num_training_steps + 1):
            sl.set_step(step)  # propagate the step counter to the actors
            with sl.log_trace_span("sync_log_step"):
                await self.trainer.sync_log_step.call(step)
                await self.generator_router.sync_log_step.call_one(step)
                await self._rollouter.sync_log_step(step)
            # timing/step/* splits timing/step/total into phases. A wait_for_* phase is how long the loop
            # idled on background work, not how long that work took.
            step_timer = MetricsTimer()

            with sl.log_trace_span("train_step"), step_timer.record(
                "timing/step/total"
            ):
                # The adaptive buffer sizes its demand from what is ready at this moment.
                await self._group_buffer.record_step_start(
                    trainer_policy_version=self._trainer_policy_version
                )
                # Waits for a TrainerStepBatch to be ready (or None on shutdown).
                with sl.log_trace_span("wait_for_training_batch"), step_timer.record(
                    "timing/step/wait_for_training_batch"
                ):
                    packed = await training_batch_queue.get()

                if packed is None:
                    logger.info("Batcher closed and drained; stopping training")
                    break

                # Policy age is computed HERE, at consumption time, against the live trainer version, so it is
                # faithful to what this step trains on -- not the version when the batch was packed.
                policy_age_panel = compute_policy_age_metrics(
                    trainer_policy_version=self._trainer_policy_version,
                    min_policy_versions=packed.min_policy_versions,
                    target_offpolicy_steps=(
                        self.config.async_loop.target_offpolicy_steps
                    ),
                    max_offpolicy_steps=self.config.async_loop.max_offpolicy_steps,
                )

                # Forward/backward blocks the trainer's loop and would stall a pending push; see WeightSyncManager.
                # TODO(perf): run forward_backward_steps off the trainer's event loop (asyncio.to_thread) so the
                #   push overlaps it again, taking its 0.2-0.4 s off the critical path. Not done: the push's GPU
                #   copy would queue behind forward/backward kernels on the shared default stream, its bf16 copy
                #   would live through forward/backward again, and CUDA/NCCL per-thread state needs a review.
                with (
                    sl.log_trace_span("wait_for_push"),
                    step_timer.record("timing/step/wait_for_push"),
                ):
                    push_metrics = await self._weight_sync.wait_prev_push()

                # TODO(async): can't stream microbatches (interleave pack->train) -- the loss is normalized by
                #   global counts over ALL microbatches, needed before any fwd/bwd. To
                #   support streaming, accumulate raw loss/token counts across microbatches and scale before optimizer.
                with sl.log_trace_span("forward_backward_steps"), step_timer.record(
                    "timing/step/forward_backward"
                ):
                    fwd_bwd_metrics = self._get_rank_0_value(
                        await self.trainer.forward_backward_steps.call(
                            packed.microbatches,
                            packed.global_loss_token_counts,
                            packed.global_routing_token_counts,
                        )
                    )

                    if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                        logger.error("Loss is NaN/Inf; training diverged")
                        break

                with (
                    sl.log_trace_span("optimizer_step"),
                    step_timer.record("timing/step/optimizer"),
                ):
                    optimizer_result = self._get_rank_0_value(
                        await self.trainer.optimizer_step.call(
                            last_step=(step == num_training_steps)
                        )
                    )
                self._trainer_policy_version = optimizer_result.policy_version
                self._data_stream.consume(packed.group_ids)
                if optimizer_result.checkpoint_saved:
                    step_dir = self._checkpoint_step_dir(step)
                    assert (
                        step_dir is not None
                    ), "a checkpoint was saved without a checkpointer"
                    with sl.log_trace_span("save_data_stream_state"):
                        await self._data_stream.save(step_dir, self._rollouter)

                # Await generator weight pull to finish before the trainer's next push.
                with (
                    sl.log_trace_span("wait_for_pull"),
                    step_timer.record("timing/step/wait_for_pull"),
                ):
                    pull_metrics = await self._weight_sync.wait_prev_pull()

                # Overlap this step's push with the next batch wait, and its pull + slot release with the next fwd/bwd.
                self._weight_sync.start_async_push_pull(
                    version=optimizer_result.policy_version
                )

            # TODO(metrics): See if metrics are being computed at the right place. E.g. should we put all
            # rollout related metrics here, or move all of them to the rollouter.
            time_metrics = step_timer.flush()
            with sl.log_trace_span("metrics_log"):
                self.metrics_processor.log(
                    step=step,
                    is_validation=False,
                    metrics=[
                        *packed.metrics,
                        *[
                            m.Metric(key, m.NoReduce(value))
                            for key, value in fwd_bwd_metrics.items()
                        ],
                        *[
                            m.Metric(key, m.NoReduce(value))
                            for key, value in optimizer_result.metrics.items()
                        ],
                        *self._group_buffer.metrics(),
                        *time_metrics,
                        *policy_age_panel,
                        # Push/pull start to done in the background; the loop's waits are timing/step/wait_for_*.
                        *push_metrics,
                        *pull_metrics,
                        *compute_perf_ratio_metrics(
                            num_global_valid_tokens=int(
                                packed.global_loss_token_counts[0]
                            ),
                            time_metrics=time_metrics,
                        ),
                    ],
                )

            validation = self.config.async_loop.validation
            is_validation_step = (
                validation.interval_steps
                and step % validation.interval_steps == 0
                and step < num_training_steps
            )
            if validation.loop_mode == ValidationLoopMode.OVERLAP_TRAINING:
                self._log_finished_validation(step=step)
                if is_validation_step:
                    validation_requested = True
                    if self._validation_task is not None:
                        logger.info(
                            f"Validation step {step}: {self._validation_task.get_name()} is still "
                            "running; the next pass starts when it ends"
                        )
                # No pass starts at the last step: the final pass after training covers it.
                if (
                    validation_requested
                    and self._validation_task is None
                    and step < num_training_steps
                ):
                    # Wait for this step's weight pull (not the pass), so the pass starts on at
                    # least policy `step`.
                    await self._weight_sync.wait_inflight_push_pull()
                    self._start_validation(step=step)
                    validation_requested = False
            elif is_validation_step:
                # Pause the trainer until validation ends. It is the only weight syncer, so
                # every validation rollout samples this step's policy; training rollouts
                # keep generating meanwhile.
                await self._weight_sync.wait_inflight_push_pull()
                await self._validate_and_log(step=step)

        # Finish the last in-flight sync so generators hold the final weights for post-validation.
        await self._weight_sync.wait_inflight_push_pull()
