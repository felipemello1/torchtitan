# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import os
import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims

# Observability imports are deferred to avoid circular dependency:
# components/metrics → observability/aggregation → components/metrics
# These are imported at first use in __init__ and cached as module-level vars.
EveryNSteps = None
MaxMetric = None
MeanMetric = None
NoOpMetric = None
SumMetric = None
record_metric = None
set_step = None
record_event = None


def _ensure_observability_imports():
    """Lazy-load observability modules to break circular import."""
    global EveryNSteps, MaxMetric, MeanMetric, NoOpMetric, SumMetric
    global record_metric, set_step, record_event
    if EveryNSteps is not None:
        return
    from torchtitan.observability.logging_boundary import (
        EveryNSteps as _EveryNSteps,
    )
    from torchtitan.observability.metrics import (
        MaxMetric as _MaxMetric,
        MeanMetric as _MeanMetric,
        NoOpMetric as _NoOpMetric,
        record_metric as _record_metric,
        SumMetric as _SumMetric,
    )
    from torchtitan.observability.step_state import set_step as _set_step
    from torchtitan.observability.structured_logging import (
        record_event as _record_event,
    )

    EveryNSteps = _EveryNSteps
    MaxMetric = _MaxMetric
    MeanMetric = _MeanMetric
    NoOpMetric = _NoOpMetric
    SumMetric = _SumMetric
    record_metric = _record_metric
    set_step = _set_step
    record_event = _record_event
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
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
    """Metrics recording, derived metrics, and logging subprocess.

    Records training/validation metrics to experiment JSONL via record_metric.
    A background subprocess reads JSONL, aggregates across ranks, and writes
    to WandB/TB/console. The training process only pays ~0.1ms per log step
    (queue.put cost).

    Method order mirrors experiments/observability/metrics_processor.py (the
    toy version) for easy side-by-side comparison during review.
    """

    _TRAIN_PREFIX = "trainer"
    _VAL_PREFIX = "validator"

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int = 10
        """How often to log to backends (e.g. wandb). Also gates expensive
        metrics computation (.item(), collectives)."""

        enable_tensorboard: bool = False
        """Whether to log metrics to TensorBoard"""

        save_tb_folder: str = "tb"
        """Folder to dump TensorBoard states"""

        enable_wandb: bool = False
        """Whether to log metrics to Weights & Biases"""

        console_log_metric_keys: list[str] = field(
            default_factory=lambda: [
                "training/loss_mean",
                "training/grad_norm_max",
                "trainer_memory/reserved_gib_max",
                "trainer_throughput/tps_mean",
                "trainer_throughput/tflops_mean",
                "trainer_throughput/mfu_pct_mean",
            ]
        )
        """Training metric keys to print to console each log step."""

        console_log_validation_keys: list[str] = field(
            default_factory=lambda: [
                "validation/loss_mean",
                "validator_memory/reserved_gib_max",
                "validator_throughput/tps_mean",
            ]
        )
        """Validation metric keys to print to console."""

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
        _ensure_observability_imports()
        self.config = config
        self.parallel_dims = parallel_dims
        self._force_log = False

        # Schedule: log on step 1 + every log_freq steps
        self._log_schedule = EveryNSteps(
            every_n=config.log_freq, additional_steps={1}
        )

        # Device memory monitor
        self.device_memory_monitor = build_device_memory_monitor()

        # TFLOPS/MFU: set by trainer after construction (-1 = skip)
        self.num_flops_per_token: int = -1
        self._gpu_peak_flops = utils.get_peak_flops(
            self.device_memory_monitor.device_name
        )

        self.reset_training_counters()
        self.reset_val_counters()

        # Determine which rank runs the logging subprocess
        self._metrics_rank = _get_metrics_rank(
            parallel_dims=parallel_dims, pp_schedule=pp_schedule
        )
        ensure_pp_loss_visible(
            parallel_dims=parallel_dims,
            pp_schedule=pp_schedule,
            color=Color(),
        )

        # Spawn the logging subprocess on the metrics rank.
        # logging_worker reads experiment JSONL from all ranks, aggregates
        # metrics, and flushes to WandB/TB/console. Runs in a separate
        # process so it never blocks training.
        needs_subprocess = (
            config.enable_wandb
            or config.enable_tensorboard
            or config.console_log_metric_keys
        )
        self._log_queue: multiprocessing.Queue | None = None
        self._log_process: multiprocessing.Process | None = None
        if torch.distributed.get_rank() == self._metrics_rank and needs_subprocess:
            # Lazy import to break circular dependency:
            # components/metrics → observability/aggregation → components/metrics
            from torchtitan.observability.aggregation import logging_worker

            self._log_queue = multiprocessing.Queue()
            self._log_process = multiprocessing.Process(
                target=logging_worker,
                args=(self._log_queue, dump_folder),
                kwargs={
                    "enable_wandb": config.enable_wandb,
                    "enable_tensorboard": config.enable_tensorboard,
                    "save_tb_folder": config.save_tb_folder,
                    "config_dict": config_dict,
                    "tag": tag,
                    "console_log_metric_keys": config.console_log_metric_keys,
                    "console_log_validation_keys": config.console_log_validation_keys,
                },
                daemon=True,
            )
            self._log_process.start()

    # ----
    # Step management
    # ----

    def set_step(self, step: int, force_log: bool = False) -> None:
        """Set current step. Call at the top of each training iteration.

        force_log: ensures loss is computed on validation steps even if
        the step isn't a regular log step.
        """
        self._step = step
        self._force_log = force_log
        set_step(step)
        record_event({"train.step": step})

    # ----
    # Schedule queries
    # ----

    def should_log(self, step: int) -> bool:
        """Returns True on log steps or when force_log was set."""
        return self._log_schedule(step) or self._force_log

    # ----
    # Counter resets
    # ----

    def reset_training_counters(self) -> None:
        """Reset throughput/memory counters for a new training measurement."""
        self._time_at_reset = time.perf_counter()
        self.ntokens_since_reset = 0
        self.device_memory_monitor.reset_peak_stats()

    def reset_val_counters(self) -> None:
        """Reset throughput/memory counters for a new validation measurement."""
        self._val_time_at_reset = time.perf_counter()
        self.val_ntokens_since_reset = 0
        self.device_memory_monitor.reset_peak_stats()

    # ----
    # Derived metrics (called every step, outside should_log gate)
    # ----

    def record_throughput(self, is_validation: bool = False) -> None:
        """Compute and record throughput from tokens since last reset."""
        if is_validation:
            time_delta = time.perf_counter() - self._val_time_at_reset
            ntokens = self.val_ntokens_since_reset
            prefix = self._VAL_PREFIX
        else:
            time_delta = time.perf_counter() - self._time_at_reset
            ntokens = self.ntokens_since_reset
            prefix = self._TRAIN_PREFIX

        tps = ntokens / time_delta if time_delta > 0 else 0
        record_metric(f"{prefix}_throughput/tps_mean", MeanMetric(sum=tps))

        if self.num_flops_per_token > 0:
            tflops = self.num_flops_per_token * tps / 1e12
            record_metric(f"{prefix}_throughput/tflops_mean", MeanMetric(sum=tflops))
            if self._gpu_peak_flops > 0:
                mfu = 100 * self.num_flops_per_token * tps / self._gpu_peak_flops
                record_metric(f"{prefix}_throughput/mfu_pct_mean", MeanMetric(sum=mfu))

    def record_memory(self, is_validation: bool = False) -> None:
        """Record GPU memory peak stats since last reset."""
        prefix = self._VAL_PREFIX if is_validation else self._TRAIN_PREFIX
        mem = self.device_memory_monitor.get_peak_stats()
        record_metric(
            f"{prefix}_memory/reserved_gib_max",
            MaxMetric(value=mem.max_reserved_gib),
        )
        record_metric(
            f"{prefix}_memory/active_gib_max",
            MaxMetric(value=mem.max_active_gib),
        )
        record_metric(
            f"{prefix}_memory/alloc_retries_sum",
            SumMetric(value=mem.num_alloc_retries),
        )
        record_metric(
            f"{prefix}_memory/ooms_sum",
            SumMetric(value=mem.num_ooms),
        )

    # ----
    # Flush
    # ----

    def log(self, step: int, is_validation: bool = False) -> None:
        """Signal the logging subprocess to aggregate and write.

        All ranks participate in the barrier so the subprocess can read
        all JSONL files. Non-blocking after barrier (~0.1ms).
        """
        torch.distributed.barrier()
        if self._log_queue is not None:
            self._log_queue.put((step, is_validation))

    def close(self) -> None:
        """Shut down the logging subprocess."""
        if self._log_queue is not None:
            self._log_queue.put(None)
        if self._log_process is not None:
            self._log_process.join(timeout=10)
            if self._log_process.is_alive():
                self._log_process.terminate()
