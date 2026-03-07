# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.observability.aggregation import aggregate, FileWatcher
from torchtitan.observability.common import clear_step_tags, set_step
from torchtitan.observability.logging_boundary import EveryNSteps
from torchtitan.observability.metrics import (
    MaxMetric,
    MeanMetric,
    record_metric,
    SumMetric,
)
from torchtitan.tools.utils import Color, device_module, device_type, NoColor


# named tuple for passing device memory stats for logging
DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class DeviceMemoryMonitor:
    def __init__(self, device: str = f"{device_type}:0"):
        # pyrefly: ignore [read-only]
        self.device = torch.device(device)  # device object
        self.device_name = device_module.get_device_name(self.device)
        self.device_index = device_module.current_device()
        self.device_capacity = device_module.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        device_module.reset_peak_memory_stats()
        device_module.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        device_info = device_module.memory_stats(self.device)

        max_active = device_info.get("active_bytes.all.peak", -1)
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = device_info.get("reserved_bytes.all.peak", -1)
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = device_info.get("num_alloc_retries", -1)
        num_ooms = device_info.get("num_ooms", -1)

        if num_retries > 0:
            logger.warning(
                f"{num_retries} {device_type.upper()} memory allocation retries."
            )
        if num_ooms > 0:
            logger.warning(f"{num_ooms} {device_type.upper()} OOM errors thrown.")

        return DeviceMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        device_module.reset_peak_memory_stats()


def build_device_memory_monitor():
    device_memory_monitor = DeviceMemoryMonitor(device_type)
    logger.info(
        f"{device_type.upper()} capacity: {device_memory_monitor.device_name} "
        f"with {device_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )
    return device_memory_monitor


class BaseLogger:
    """Logger that does nothing, used when logging is disabled."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    """Logger implementation for TensorBoard."""

    def __init__(self, log_dir: str, tag: str | None = None):
        self.tag = tag
        self.writer = SummaryWriter(log_dir, max_queue=1000)
        logger.info(f"TensorBoard logging enabled. Logs will be saved at {log_dir}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for k, v in metrics.items():
            tag = k if self.tag is None else f"{self.tag}/{k}"
            self.writer.add_scalar(tag, v, step)

    def close(self) -> None:
        self.writer.close()


class WandBLogger(BaseLogger):
    """Logger implementation for Weights & Biases."""

    def __init__(
        self,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        # Import wandb here to avoid startup import
        import wandb

        self.wandb = wandb
        self.tag = tag

        # Create logging directory
        os.makedirs(log_dir, exist_ok=True)

        self.wandb.init(
            entity=os.getenv("WANDB_TEAM", None),
            project=os.getenv("WANDB_PROJECT", "torchtitan"),
            name=os.getenv("WANDB_RUN_NAME", None),
            id=os.getenv("WANDB_RUN_ID", None),
            notes=os.getenv("WANDB_RUN_NOTES", None),
            tags=os.getenv("WANDB_RUN_TAGS", None),
            group=os.getenv("WANDB_RUN_GROUP", None),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE", None),
            resume_from=os.getenv("WANDB_RESUME_FROM", None),
            fork_from=os.getenv("WANDB_FORK_FROM", None),
            dir=log_dir,
            config=config_dict,
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        wandb_metrics = {
            (k if self.tag is None else f"{self.tag}/{k}"): v
            for k, v in metrics.items()
        }
        self.wandb.log(wandb_metrics, step=step)

    def close(self) -> None:
        if self.wandb.run is not None:
            self.wandb.finish()


class LoggerContainer(BaseLogger):
    """Container to call all loggers enabled in the job config."""

    def __init__(self) -> None:
        self._loggers: list[BaseLogger] = []

    def add_logger(self, logger_instance: BaseLogger) -> None:
        self._loggers.append(logger_instance)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for logger_instance in self._loggers:
            logger_instance.log(metrics, step)

    @property
    def number_of_loggers(self) -> int:
        return len(self._loggers)

    def close(self) -> None:
        for logger_instance in self._loggers:
            logger_instance.close()


def ensure_pp_loss_visible(
    *, parallel_dims: ParallelDims, pp_schedule: str, color: Color | NoColor
) -> None:
    """
    Ensures that the loss is visible on the console for pipeline-parallel training.

    For pipeline-parallel training, the loss is only visible on the last pipeline stage.
    This function checks if the appropriate rank is included in the LOG_RANK environment
    variable and warns if it's not.
    """

    # V Block Schedules return loss on rank 0
    if pp_schedule == "ZBVZeroBubble":
        return

    # Calculate the rank where loss is visible (first rank of the last pipeline stage)
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    loss_visible_rank = (world_size // pp_size) * (pp_size - 1)

    # Check if the loss-visible rank is included in LOG_RANK environment variable
    env_logged_ranks = os.environ.get("LOG_RANK", "").split(",")
    if env_logged_ranks == [""]:
        env_logged_ranks = []

    if str(loss_visible_rank) not in env_logged_ranks:
        logger.warning(
            f"{color.red}Pipeline Parallel loss is not visible. "
            f"Please add {color.yellow}rank {loss_visible_rank}{color.red} "
            f"to LOG_RANK environment variable in run_train.sh.{color.reset}"
        )


def _get_metrics_rank(
    *,
    parallel_dims: ParallelDims,
    pp_schedule: str,
) -> int:
    """
    Determines which rank should log metrics.

    Returns:
       int: The rank responsible for logging metrics:
            - Rank 0 for non-pipeline-parallel configs
            - Rank 0 for pipeline-parallel 'ZBVZeroBubble' schedule
            - The first rank of the last pipeline stage for other pipeline-parallel schedules
    """
    # Early return for non-pipeline-parallel configurations
    if not parallel_dims.pp_enabled:
        return 0

    # V Block Schedules return loss on rank 0
    if pp_schedule == "ZBVZeroBubble":
        return 0

    # Calculate first rank of the last pipeline stage
    world_size = parallel_dims.world_size
    pp_size = parallel_dims.pp
    return (world_size // pp_size) * (pp_size - 1)


class MetricsProcessor(Configurable):
    """Observability pipeline for training metrics.

    Manages step context, metric schedules, derived metrics (throughput, memory),
    and flushing aggregated metrics to TB/WandB + console.

    Caller must call init_observability() before constructing MetricsProcessor.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int = 10
        """How often to flush metrics to TB/WandB and print console."""

        log_tensor_metrics_freq: int = 10
        """How often to reduce tensor metrics (expensive all-reduce).
        Must be a multiple of log_freq."""

        enable_tensorboard: bool = False
        """Whether to log metrics to TensorBoard."""

        disable_color_printing: bool = False
        """Whether to disable color printing in logs."""

        save_tb_folder: str = "tb"
        """Folder to dump TensorBoard states."""

        save_for_all_ranks: bool = False
        """Raises NotImplementedError if True. All ranks write to JSONL;
        rank 0 aggregates and writes to TB/WandB."""

        enable_wandb: bool = False
        """Whether to log metrics to Weights & Biases."""

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

    # ---- Type annotations for public attributes set by trainer ----
    config: Config
    parallel_dims: ParallelDims
    device_memory_monitor: DeviceMemoryMonitor
    color: utils.NoColor | utils.Color
    ntokens_since_last_log: int
    num_flops_per_token: int
    validation_ntokens: int
    validation_time: float

    # Kept for backward compat (set by trainer, never read by MP)
    optimizers: OptimizersContainer | None
    model_parts: list[torch.nn.Module] | None
    lr_schedulers: LRSchedulersContainer | None

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        dump_folder: str = "./outputs",
        pp_schedule: str = "1F1B",
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        # Assumes init_observability() was already called by the caller.
        self.config = config
        self.parallel_dims = parallel_dims

        # Logging backends (TB/WandB).
        self._logger = self._build_metric_logger(
            config=config,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
            pp_schedule=pp_schedule,
            ft_enable=ft_enable,
            ft_replica_id=ft_replica_id,
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
        self.color = utils.NoColor() if config.disable_color_printing else utils.Color()
        self._gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )
        self._time_last_log = time.perf_counter()
        self.ntokens_since_last_log = 0
        self.num_flops_per_token = -1

        # Validation state (separate from training to avoid counter corruption).
        self.validation_ntokens = 0
        self.validation_time = time.perf_counter()

        # Kept for backward compat (set by trainer, never read by MP).
        self.optimizers = None
        self.model_parts = None
        self.lr_schedulers = None

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
        non_dp = self.parallel_dims.non_data_parallel_size
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
        record_metric("trainer/memory/max_reserved_pct_max", MaxMetric(value=mem.max_reserved_pct))
        record_metric("trainer/memory/max_active_gib_max", MaxMetric(value=mem.max_active_gib))
        record_metric("trainer/memory/max_active_pct_max", MaxMetric(value=mem.max_active_pct))
        record_metric("trainer/memory/num_alloc_retries_sum", SumMetric(value=mem.num_alloc_retries))
        record_metric("trainer/memory/num_ooms_sum", SumMetric(value=mem.num_ooms))

    # ----------------------------------------------------------------
    # Flush (called by train() on log steps or after validation)
    # ----------------------------------------------------------------

    def log(self, step: int) -> None:
        """Pure flush: barrier -> drain -> aggregate -> write -> console -> reset."""
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
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
    # Validation logging
    # ----------------------------------------------------------------

    def log_validation(
        self, loss: float, step: int, extra_metrics: dict[str, Any] | None = None
    ) -> None:
        """Record validation metrics to JSONL. No console, no flush — log() handles those."""
        time_delta = time.perf_counter() - self.validation_time
        device_mem_stats = self.device_memory_monitor.get_peak_stats()
        tps = (
            self.validation_ntokens
            / (time_delta * self.parallel_dims.non_data_parallel_size)
            if time_delta > 0
            else 0
        )

        record_metric("validator/loss_mean", MeanMetric(sum=float(loss)))
        record_metric("validator/throughput/tps_mean", MeanMetric(sum=tps))
        record_metric(
            "validator/memory/max_reserved_gib_max",
            MaxMetric(value=device_mem_stats.max_reserved_gib),
        )
        record_metric(
            "validator/memory/max_active_gib_max",
            MaxMetric(value=device_mem_stats.max_active_gib),
        )
        if extra_metrics:
            for k, v in extra_metrics.items():
                record_metric(f"validator/{k}", MeanMetric(sum=float(v)))

        # Reset validation accumulators (NOT training accumulators).
        self.validation_ntokens = 0
        self.validation_time = time.perf_counter()

    # ----------------------------------------------------------------
    # Console output
    # ----------------------------------------------------------------

    def _log_to_console(self, step: int, aggregated: dict[str, float]) -> None:
        c = self.color
        loss = aggregated.get("trainer/loss_mean", "--")
        grad_norm = aggregated.get("trainer/grad_norm_max", "--")
        mem_gib = aggregated.get("trainer/memory/max_reserved_gib_max", "--")
        mem_pct = aggregated.get("trainer/memory/max_reserved_pct_max", "--")
        tps = aggregated.get("trainer/throughput/tps_mean", "--")
        tflops = aggregated.get("trainer/throughput/tflops_mean", "--")
        mfu = aggregated.get("trainer/throughput/mfu_pct_mean", "--")

        def fmt(val, spec):
            return f"{val:{spec}}" if isinstance(val, (int, float)) else f"{val:>8}"

        logger.info(
            f"{c.red}step: {step:2}  "
            f"{c.green}loss: {fmt(loss, '8.5f')}  "
            f"{c.orange}grad_norm: {fmt(grad_norm, '7.4f')}  "
            f"{c.turquoise}memory: {fmt(mem_gib, '5.2f')}GiB"
            f"({fmt(mem_pct, '.2f')}%)  "
            f"{c.blue}tps: {fmt(tps, ',')}  "
            f"{c.cyan}tflops: {fmt(tflops, ',.2f')}  "
            f"{c.magenta}mfu: {fmt(mfu, '.2f')}%{c.reset}"
        )

        # Validation line (only present on validation steps).
        val_loss = aggregated.get("validator/loss_mean")
        if val_loss is not None:
            val_tps = aggregated.get("validator/throughput/tps_mean", "--")
            val_mem = aggregated.get("validator/memory/max_reserved_gib_max", "--")
            logger.info(
                f"{c.yellow}validate step: {step:2}  "
                f"{c.green}loss: {fmt(val_loss, '7.4f')}  "
                f"{c.turquoise}memory: {fmt(val_mem, '5.2f')}GiB  "
                f"{c.blue}tps: {fmt(val_tps, ',')}{c.reset}"
            )

    # ----------------------------------------------------------------
    # Logger setup
    # ----------------------------------------------------------------

    def _build_metric_logger(
        self,
        *,
        config: Config,
        parallel_dims: ParallelDims,
        dump_folder: str,
        pp_schedule: str,
        ft_enable: bool = False,
        ft_replica_id: int = 0,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ) -> BaseLogger:
        """Build TB/WandB loggers based on config and rank."""
        has_logging_enabled = config.enable_tensorboard or config.enable_wandb

        should_log = has_logging_enabled
        if should_log:
            metrics_rank = _get_metrics_rank(
                parallel_dims=parallel_dims, pp_schedule=pp_schedule
            )
            should_log = torch.distributed.get_rank() == metrics_rank

        if not should_log:
            return BaseLogger()

        base_log_dir = os.path.join(
            dump_folder,
            config.save_tb_folder,
            datetime.now().strftime("%Y%m%d-%H%M"),
        )

        if ft_enable:
            base_log_dir = os.path.join(base_log_dir, f"replica_{ft_replica_id}")

        logger_container = LoggerContainer()

        if config.enable_wandb:
            try:
                wandb_logger = WandBLogger(
                    base_log_dir, config_dict=config_dict, tag=tag
                )
                logger_container.add_logger(wandb_logger)
            except Exception as e:
                logger.error(f"Failed to create WandB logger: {e}")

        if config.enable_tensorboard:
            tensorboard_logger = TensorBoardLogger(base_log_dir, tag)
            logger_container.add_logger(tensorboard_logger)

        return logger_container

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def close(self) -> None:
        self._watcher.close()
        self._logger.close()
