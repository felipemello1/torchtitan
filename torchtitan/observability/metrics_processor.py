# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MetricsProcessor — observability pipeline for training metrics.

Manages step context, metric schedules, and logging backends (TB/WandB).
Caller must call init_observability() before constructing MetricsProcessor.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from torchtitan.components.metrics import (
    BaseLogger,
    LoggerContainer,
    TensorBoardLogger,
    WandBLogger,
)
from torchtitan.observability.common import clear_step_tags, set_step
from torchtitan.observability.logging_boundary import EveryNSteps
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import Color, NoColor


class MetricsProcessor:
    """Processes and logs training metrics.

    Provides:
    - Step context management (set_step)
    - Schedule queries (should_log, should_log_tensors)
    - Logging: write scalars to TB/WandB + console (log)
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

        # Logging backends (TB/WandB) — reuses Titan's existing logger classes.
        self._logger = self._build_metric_logger(
            config=config,
            dump_folder=dump_folder,
            rank=rank,
            config_dict=config_dict,
            tag=tag,
        )

        # Schedules
        self._tensor_metrics_schedule = EveryNSteps(
            every_n=config.log_tensor_metrics_freq, additional_steps={1}
        )
        self._log_schedule = EveryNSteps(every_n=config.log_freq, additional_steps={1})

        # Console color
        self.color = NoColor() if config.disable_color_printing else Color()

    def set_step(self, step: int) -> None:
        """Set current step. Call before train_step()."""
        self._step = step
        set_step(step)
        clear_step_tags()

    def should_log(self, step: int) -> bool:
        """Returns True on log steps. Caller uses this to gate log()."""
        return self._log_schedule(step)

    def should_log_tensors(self, step: int) -> bool:
        """Returns True on tensor reduction steps. Caller uses this to gate replicate_to_host."""
        return self._tensor_metrics_schedule(step)

    def log(self, step: int, scalars: dict[str, float]) -> None:
        """Write pre-reduced scalars to TB/WandB + console.

        Scalars are already final values from replicate_to_host.
        """
        if self._rank == 0:
            self._logger.log(scalars, step)
            self._log_to_console(step, scalars)

    def _log_to_console(self, step: int, scalars: dict[str, float]) -> None:
        c = self.color
        loss = scalars.get("trainer/loss_mean", "--")

        def fmt(val, spec):
            return f"{val:{spec}}" if isinstance(val, (int, float)) else f"{val:>8}"

        logger.info(
            f"{c.red}step: {step:2}  "
            f"{c.green}loss: {fmt(loss, '8.5f')}{c.reset}"
        )

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

    def close(self) -> None:
        self._logger.close()
