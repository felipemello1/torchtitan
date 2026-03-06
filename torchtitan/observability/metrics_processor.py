# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MetricsProcessor — observability pipeline for training metrics.

Manages step context, metric schedules, derived metrics (throughput, memory),
and flushing aggregated metrics to TB/WandB + console.

Caller must call init_observability() before constructing MetricsProcessor.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch.distributed as dist

from torchtitan.components.metrics import (
    BaseLogger,
    build_device_memory_monitor,
    LoggerContainer,
    TensorBoardLogger,
    WandBLogger,
)
from torchtitan.observability.aggregation import aggregate, FileWatcher
from torchtitan.observability.common import clear_step_tags, set_step
from torchtitan.observability.logging_boundary import EveryNSteps
from torchtitan.observability.metrics import (
    MaxMetric,
    MeanMetric,
    record_metric,
)
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, get_peak_flops, NoColor


class MetricsProcessor:
    """Processes and logs training metrics.

    Provides:
    - Step context management (set_step)
    - Schedule queries (should_log, should_log_tensors)
    - Derived metrics (record_throughput, record_memory)
    - Flush: drain JSONL -> aggregate -> TB/WandB + console (log)
    """

    @dataclass
    class Config:
        log_freq: int = 10
        """How often to flush metrics to TB/WandB and print console."""

        log_tensor_metrics_freq: int = 10
        """How often to reduce tensor metrics (expensive all-reduce).
        Must be a multiple of log_freq."""

        enable_tensorboard: bool = False
        save_tb_folder: str = "tb"
        enable_wandb: bool = False
        disable_color_printing: bool = False

        save_for_all_ranks: bool = False
        """Raises NotImplementedError if True. All ranks write to JSONL;
        rank 0 aggregates and writes to TB/WandB."""

        def __post_init__(self):
            if self.save_for_all_ranks:
                raise NotImplementedError(
                    "save_for_all_ranks is not supported. "
                    "All ranks write to JSONL; rank 0 aggregates and writes to TB/WandB."
                )
            if self.log_tensor_metrics_freq % self.log_freq != 0:
                old = self.log_tensor_metrics_freq
                self.log_tensor_metrics_freq = (
                    (self.log_tensor_metrics_freq + self.log_freq - 1)
                    // self.log_freq
                    * self.log_freq
                )
                logger.warning(
                    f"log_tensor_metrics_freq ({old}) is not a multiple of "
                    f"log_freq ({self.log_freq}). Snapped to {self.log_tensor_metrics_freq}."
                )

    def __init__(
        self,
        config: "MetricsProcessor.Config",
        *,
        dump_folder: str = "./outputs",
        rank: int = 0,
        pp_schedule: str = "1F1B",
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        # Assumes init_observability() was already called by the caller.
        self._config = config
        self._rank = rank

        # Logging backends (TB/WandB).
        self._logger = self._build_metric_logger(
            config=config,
            dump_folder=dump_folder,
            rank=rank,
            config_dict=config_dict,
            tag=tag,
        )

        # Background thread tails experiment JSONL files.
        exp_log_dir = os.path.join(dump_folder, "experiment_logs")
        self._watcher = FileWatcher(log_dir=exp_log_dir)

        # Schedules.
        self._tensor_metrics_schedule = EveryNSteps(
            every_n=config.log_tensor_metrics_freq, additional_steps={1}
        )
        self._log_schedule = EveryNSteps(every_n=config.log_freq, additional_steps={1})

        # Training state.
        self.device_memory_monitor = build_device_memory_monitor()
        self.color = NoColor() if config.disable_color_printing else Color()
        self._gpu_peak_flops = get_peak_flops(self.device_memory_monitor.device_name)
        self._time_last_log = time.perf_counter()
        self.ntokens_since_last_log = 0
        self.num_flops_per_token = -1

    # ----------------------------------------------------------------
    # Step management
    # ----------------------------------------------------------------

    def set_step(self, step: int) -> None:
        """Set current step. Call before train_step()."""
        self._step = step
        set_step(step)
        clear_step_tags()

    # ----------------------------------------------------------------
    # Schedule queries
    # ----------------------------------------------------------------

    def should_log(self, step: int) -> bool:
        """Returns True on log steps. Caller uses this to gate log()."""
        return self._log_schedule(step)

    def should_log_tensors(self, step: int) -> bool:
        """Returns True on tensor reduction steps."""
        return self._tensor_metrics_schedule(step)

    # ----------------------------------------------------------------
    # Derived metrics (called every step, outside should_log gate)
    # ----------------------------------------------------------------

    def record_throughput(self) -> None:
        """Compute and record throughput from accumulated tokens since last log reset."""
        time_delta = time.perf_counter() - self._time_last_log
        non_dp = 1  # No pipeline or expert parallelism in the toy.
        tps = self.ntokens_since_last_log / (time_delta * non_dp) if time_delta > 0 else 0
        tflops = self.num_flops_per_token * tps / 1e12 if self.num_flops_per_token > 0 else 0
        mfu = (
            100 * self.num_flops_per_token * tps / self._gpu_peak_flops
            if self._gpu_peak_flops > 0 and self.num_flops_per_token > 0
            else 0
        )
        record_metric("trainer/throughput/tps_mean", MeanMetric(sum=tps))
        record_metric("trainer/throughput/tflops_mean", MeanMetric(sum=tflops))
        record_metric("trainer/throughput/mfu_pct_mean", MeanMetric(sum=mfu))

    def record_memory(self) -> None:
        """Record device memory peak stats since last log reset."""
        mem = self.device_memory_monitor.get_peak_stats()
        record_metric("trainer/memory/max_reserved_gib_max", MaxMetric(value=mem.max_reserved_gib))
        record_metric("trainer/memory/max_active_gib_max", MaxMetric(value=mem.max_active_gib))

    # ----------------------------------------------------------------
    # Flush (called by train() on log steps)
    # ----------------------------------------------------------------

    def log(self, step: int) -> None:
        """Pure flush: barrier -> drain -> aggregate -> write -> console -> reset."""
        dist.barrier()
        if self._rank == 0:
            entries = self._watcher.drain(step)
            aggregated = aggregate(entries)
            if aggregated:
                self._logger.log(aggregated, step)
                self._log_to_console(step, aggregated)

        # Reset training accumulators.
        self.ntokens_since_last_log = 0
        self._time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()

    # ----------------------------------------------------------------
    # Console output
    # ----------------------------------------------------------------

    def _log_to_console(self, step: int, aggregated: dict[str, float]) -> None:
        c = self.color
        loss = aggregated.get("trainer/loss_mean", "--")
        grad_norm = aggregated.get("trainer/grad_norm_max", "--")
        mem_gib = aggregated.get("trainer/memory/max_reserved_gib_max", "--")
        tps = aggregated.get("trainer/throughput/tps_mean", "--")
        mfu = aggregated.get("trainer/throughput/mfu_pct_mean", "--")

        def fmt(val, spec):
            return f"{val:{spec}}" if isinstance(val, (int, float)) else f"{val:>8}"

        logger.info(
            f"{c.red}step: {step:2}  "
            f"{c.green}loss: {fmt(loss, '8.5f')}  "
            f"{c.orange}grad_norm: {fmt(grad_norm, '7.4f')}  "
            f"{c.turquoise}memory: {fmt(mem_gib, '5.2f')}GiB  "
            f"{c.blue}tps: {fmt(tps, ',')}  "
            f"{c.magenta}mfu: {fmt(mfu, '.2f')}%{c.reset}"
        )

    # ----------------------------------------------------------------
    # Logger setup
    # ----------------------------------------------------------------

    def _build_metric_logger(
        self,
        *,
        config: "MetricsProcessor.Config",
        dump_folder: str,
        rank: int,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> BaseLogger:
        """Build TB/WandB loggers. Only rank 0 gets real loggers."""
        has_logging_enabled = config.enable_tensorboard or config.enable_wandb

        if not has_logging_enabled or rank != 0:
            return BaseLogger()

        base_log_dir = os.path.join(
            dump_folder,
            config.save_tb_folder,
            datetime.now().strftime("%Y%m%d-%H%M"),
        )

        logger_container = LoggerContainer()

        if config.enable_wandb:
            try:
                wandb_logger = WandBLogger(base_log_dir, config_dict=config_dict, tag=tag)
                logger_container.add_logger(wandb_logger)
            except Exception as e:
                logger.error(f"Failed to create WandB logger: {e}")

        if config.enable_tensorboard:
            tb_logger = TensorBoardLogger(base_log_dir, tag)
            logger_container.add_logger(tb_logger)

        return logger_container

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def close(self) -> None:
        self._watcher.close()
        self._logger.close()
