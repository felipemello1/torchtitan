# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Generator

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.datasets.hf_datasets import build_hf_validation_dataloader
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.observability import (
    CompositeSummaryWriter,
    DefaultAggregator,
    MaxMetric,
    MeanMetric,
    record_metric,
)
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


class BaseValidator:
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config

    def validate(self, model_parts: list[nn.Module]) -> dict[str, float]:
        raise NotImplementedError("validate method not implemented")

    def should_validate(self, step: int) -> bool:
        return step == 1 or step % self.job_config.validation.freq == 0


class Validator(BaseValidator):
    """
    Simple validator focused on correctness and integration.

    Args:
        job_config: Job configuration
        validation_dataloader: The validation dataloader
        loss_fn: Loss function to use for validation
        model: The model to validate (single model, no parallelism)
    """

    validation_dataloader: BaseDataLoader

    def __init__(
        self,
        job_config: JobConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: Generator[None, None, None],
        maybe_enable_amp: Generator[None, None, None],
        metrics_processor: MetricsProcessor,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
        summary_writer: CompositeSummaryWriter | None = None,
        experiment_log_dir: str | None = None,
    ):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.validation_dataloader = build_hf_validation_dataloader(
            job_config=job_config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            infinite=self.job_config.validation.steps != -1,
        )
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        self.metrics_processor = metrics_processor
        self.pp_schedule = pp_schedule
        self.pp_has_first_stage = pp_has_first_stage
        self.pp_has_last_stage = pp_has_last_stage
        self._summary_writer = summary_writer
        self._val_aggregator = (
            DefaultAggregator(experiment_log_dir=experiment_log_dir)
            if experiment_log_dir
            else None
        )
        self._ntokens_since_last_log = 0
        self._time_last_log = time.perf_counter()

        if self.job_config.validation.steps == -1:
            logger.warning(
                "Setting validation steps to -1 might cause hangs because of "
                "unequal sample counts across ranks when dataset is exhausted."
            )

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        # Set model to eval mode
        for model in model_parts:
            model.eval()

        parallel_dims = self.parallel_dims

        accumulated_losses = []
        device_type = utils.device_type
        num_steps = 0

        for input_dict, labels in self.validation_dataloader:
            if (
                self.job_config.validation.steps != -1
                and num_steps >= self.job_config.validation.steps
            ):
                break

            self._ntokens_since_last_log += labels.numel()
            for k, v in input_dict.items():
                input_dict[k] = v.to(device_type)
            inputs = input_dict["input"]
            labels = labels.to(device_type)

            optional_context_parallel_ctx = (
                dist_utils.create_context_parallel_ctx(
                    cp_mesh=parallel_dims.world_mesh["cp"],
                    cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                    cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                    cp_no_restore_buffers={inputs, labels},
                    cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            if parallel_dims.pp_enabled:
                assert self.pp_schedule is not None
                assert self.pp_has_first_stage is not None
                assert self.pp_has_last_stage is not None
                # Pipeline Parallel forward inside eval() call
                with self.validation_context(optional_context_parallel_ctx):
                    targets, losses = (
                        (labels, []) if self.pp_has_last_stage else (None, None)
                    )
                    if self.pp_has_first_stage:
                        self.pp_schedule.eval(
                            inputs,
                            target=targets,
                            losses=losses,
                            input_batch=inputs,
                        )
                    else:
                        self.pp_schedule.eval(
                            target=targets, losses=losses, input_batch=inputs
                        )

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                loss = (
                    # using sum instead of mean because we already rescale the
                    # loss_fn down by a factor of n_microbatches in
                    # torchtitan/distributed/pipeline_parallel.py
                    torch.sum(torch.stack(losses)).to(device_type)
                    if self.pp_has_last_stage
                    else torch.tensor([-1.0], device=device_type)
                )
            else:
                with self.validation_context(optional_context_parallel_ctx):
                    assert len(model_parts) == 1
                    with self.maybe_enable_amp:
                        predictions = model_parts[0](inputs)
                        loss = self.loss_fn(predictions, labels)

            accumulated_losses.append(loss.detach())

            num_steps += 1

        # Compute average loss
        loss = torch.sum(torch.stack(accumulated_losses))
        loss /= num_steps
        if parallel_dims.dp_cp_enabled:
            global_avg_loss = dist_utils.dist_mean(
                loss, parallel_dims.world_mesh["dp_cp"]
            )
        else:
            global_avg_loss = loss.item()

        # Compute validation metrics
        time_delta = time.perf_counter() - self._time_last_log
        tps = self._ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )
        device_mem_stats = self.metrics_processor.device_memory_monitor.get_peak_stats()

        # Record to experiment JSONL
        record_metric("validation/loss", MeanMetric(value=float(global_avg_loss)))
        record_metric("validation/throughput_tps", MeanMetric(value=tps))
        record_metric(
            "validation/memory/max_active_gib",
            MaxMetric(value=device_mem_stats.max_active_gib),
        )
        record_metric(
            "validation/memory/max_reserved_gib",
            MaxMetric(value=device_mem_stats.max_reserved_gib),
        )

        # Aggregate + write to backends (rank 0 only)
        if self._val_aggregator and self._summary_writer:
            if torch.distributed.get_rank() == 0:
                aggregated = self._val_aggregator.collect_and_aggregate(step=step)
                self._summary_writer(step=step, values=aggregated)

        # Console output (same format as MetricsProcessor.log_validation)
        color = self.metrics_processor.color
        logger.info(
            f"{color.yellow}validate step: {step:2}  "
            f"{color.green}loss: {float(global_avg_loss):7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}{color.reset}"
        )

        # Reset
        self._ntokens_since_last_log = 0
        self._time_last_log = time.perf_counter()
        self.metrics_processor.device_memory_monitor.reset_peak_stats()

        # Set model back to train mode
        for model in model_parts:
            model.train()


def build_validator(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    parallel_dims: ParallelDims,
    loss_fn: LossFunction,
    validation_context: Generator[None, None, None],
    maybe_enable_amp: Generator[None, None, None],
    metrics_processor: MetricsProcessor | None = None,
    pp_schedule: _PipelineSchedule | None = None,
    pp_has_first_stage: bool | None = None,
    pp_has_last_stage: bool | None = None,
    summary_writer: CompositeSummaryWriter | None = None,
    experiment_log_dir: str | None = None,
) -> BaseValidator:
    """Build a simple validator focused on correctness."""
    return Validator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
        summary_writer=summary_writer,
        experiment_log_dir=experiment_log_dir,
    )
