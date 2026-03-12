# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Toy MetricsProcessor for the observability experiment."""

import multiprocessing
import time
from dataclasses import dataclass, field

import torch

from torchtitan.config import Configurable
from torchtitan.observability import record_event
from torchtitan.observability.aggregation import logging_worker
from torchtitan.observability.logging_boundary import EveryNSteps
from torchtitan.observability.metrics import (
    MaxMetric,
    MeanMetric,
    record_metric,
)
from torchtitan.observability.step_state import set_step


class MetricsProcessor(Configurable):
    """Step context, derived metrics, and logging subprocess for the toy trainer.

    Mirrors the method order of components/metrics.py MetricsProcessor
    so the toy and production versions are easy to compare.
    """

    _TRAIN_PREFIX = "toy_trainer"
    _VAL_PREFIX = "toy_validator"

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        log_freq: int = 10
        """How often to log to backends (e.g. wandb). Also gates expensive
        metrics computation (.item(), collectives)."""
        enable_wandb: bool = True
        enable_tensorboard: bool = False
        disable_color_printing: bool = False
        console_log_metric_keys: list[str] = field(default_factory=list)
        console_log_validation_keys: list[str] = field(default_factory=list)

    def __init__(self, config: Config, *, dump_folder: str, rank: int):
        self.config = config
        self._step: int = 0
        self._rank = rank
        self._force_log = False

        # Schedule: log on step 1 + every log_freq steps
        self._log_schedule = EveryNSteps(every_n=config.log_freq, additional_steps={1})

        # Training measurement window
        self._time_last_log = time.perf_counter()
        self.ntokens_since_last_log: int = 0

        # Validation measurement window (reset at the start of each validate())
        self._val_time_last_log: float = 0.0
        self.val_ntokens_since_last_log: int = 0

        # Spawn the logging subprocess if any output is enabled.
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
        if rank == 0 and needs_subprocess:
            self._log_queue = multiprocessing.Queue()
            self._log_process = multiprocessing.Process(
                target=logging_worker,
                args=(self._log_queue, dump_folder),
                kwargs={
                    "enable_wandb": config.enable_wandb,
                    "enable_tensorboard": config.enable_tensorboard,
                    "console_log_metric_keys": config.console_log_metric_keys,
                    "console_log_validation_keys": config.console_log_validation_keys,
                    "disable_color_printing": config.disable_color_printing,
                },
                daemon=True,
            )
            self._log_process.start()

    # ----------------------------------------------------------------
    # Step management
    # ----------------------------------------------------------------

    def set_step(self, step: int, force_log: bool = False) -> None:
        """Set current step. Call before train_step().

        force_log: when True, should_log() returns True regardless of
        schedule. Used to ensure loss is computed on validation steps.
        """
        self._step = step
        self._force_log = force_log
        set_step(step)
        record_event({"train.step": step})

    # ----------------------------------------------------------------
    # Schedule queries
    # ----------------------------------------------------------------

    def should_log(self, step: int) -> bool:
        """Returns True on log steps or when force_log was set."""
        return self._log_schedule(step) or self._force_log

    # ----------------------------------------------------------------
    # Counter resets (measurement window boundaries)
    # ----------------------------------------------------------------

    def reset_counters(self, is_val: bool = False) -> None:
        """Reset throughput/memory counters at measurement window boundaries.

        Training: called by log() after flush — starts the next window.
        Validation: called by validate() at the top — starts a clean window.
        """
        if is_val:
            self.val_ntokens_since_last_log = 0
            self._val_time_last_log = time.perf_counter()
        else:
            self.ntokens_since_last_log = 0
            self._time_last_log = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    # ----------------------------------------------------------------
    # Derived metrics (called every step, outside should_log gate)
    # ----------------------------------------------------------------

    def record_throughput(self, is_val: bool = False) -> None:
        """Compute and record throughput from accumulated tokens."""
        if is_val:
            time_delta = time.perf_counter() - self._val_time_last_log
            ntokens = self.val_ntokens_since_last_log
            prefix = self._VAL_PREFIX
        else:
            time_delta = time.perf_counter() - self._time_last_log
            ntokens = self.ntokens_since_last_log
            prefix = self._TRAIN_PREFIX

        tps = ntokens / time_delta if time_delta > 0 else 0
        record_metric(f"{prefix}/tps_mean", MeanMetric(sum=tps))

    def record_memory(self, is_val: bool = False) -> None:
        """Record GPU memory peak stats since last reset."""
        prefix = self._VAL_PREFIX if is_val else self._TRAIN_PREFIX
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self._rank % torch.cuda.device_count()}")
            reserved_gib = torch.cuda.max_memory_reserved(device) / (1024**3)
            record_metric(
                f"{prefix}/memory_reserved_gib_max",
                MaxMetric(value=reserved_gib),
            )

    # ----------------------------------------------------------------
    # Flush (called by train() when should_log returns True)
    # ----------------------------------------------------------------

    def log(self, step: int, is_validation: bool = False) -> None:
        """Signal the logging subprocess to aggregate and write.

        All ranks participate in the barrier so the subprocess can read
        all JSONL files. Non-blocking after barrier (~0.1ms).
        """
        torch.distributed.barrier()
        if self._log_queue is not None:
            self._log_queue.put((step, is_validation))
        self.reset_counters(is_val=False)

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def close(self) -> None:
        """Shut down the logging subprocess."""
        if self._log_queue is not None:
            self._log_queue.put(None)
        if self._log_process is not None:
            self._log_process.join(timeout=10)
            if self._log_process.is_alive():
                self._log_process.terminate()
