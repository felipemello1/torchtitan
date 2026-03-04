# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summary writer backends for experiment metrics visualization.

TensorBoardSummaryWriter, WandbSummaryWriter, LoggingSummaryWriter,
InMemorySummaryWriter — all composed via CompositeSummaryWriter.

Writer lifecycle follows TorchTitan's pattern: open() in setup, close()
explicitly. No try/finally, no atexit. __enter__/__exit__ exist as
convenience sugar from the reference for users who want context manager style.
"""


import logging
import os
from types import TracebackType
from typing import Any

import torch
from torchtitan.observability import tree
from torchtitan.observability.tensor_metrics import TMetricValue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def is_single_process_or_rank_zero() -> bool:
    """Returns True if not in a distributed setting or if this is rank 0."""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def _to_scalar(value: Any, path: str = "") -> int | float:
    """Converts a value to a Python scalar for logging.

    Args:
        value: The value to convert. Should be a TMetricValue containing scalars,
            or a Python scalar (int/float/bool).
        path: Optional key path for error messages.

    Returns:
        A Python int or float suitable for logging.

    Raises:
        ValueError: If the value contains a DTensor or Tensor.
        TypeError: If the value type is not supported.
    """
    prefix = f"Summary '{path}' " if path else "Value "

    if isinstance(value, TMetricValue):
        value = value.value()

    if isinstance(value, torch.Tensor):
        raise ValueError(
            f"{prefix}contains a {type(value).__name__}. Tensors must be converted "
            f"to Python scalars before logging. Call replicate_to_host() on your "
            f"summaries before passing to the summary writer."
        )

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value

    raise TypeError(
        f"{prefix}has unsupported type {type(value).__name__}. "
        f"Expected TMetricValue with scalar values, or Python scalar (int/float)."
    )


# ---------------------------------------------------------------------------
# SummaryWriter ABC


# ---------------------------------------------------------------------------


class SummaryWriter:
    """Base summary writer with lifecycle management.

    Usage:
        writer = TensorBoardSummaryWriter(log_dir="output/tb")
        writer.open()   # or use as context manager: with writer:
        for step in range(max_step):
            writer(step=step, values=scalars)
        writer.close()
    """

    def __init__(self) -> None:
        self._within_context = False

    def open(self) -> None:
        """Initialize the writer. Called once before the training loop."""
        if self._within_context:
            raise RuntimeError("Writer already open.")
        self._within_context = True

    def close(self) -> None:
        """Close the writer. Called once after the training loop."""
        self._within_context = False

    def __enter__(self) -> "SummaryWriter":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None

    def should_write(self, step: int) -> bool:
        """Override to gate logging frequency. Default: always write."""
        return True

    def __call__(self, step: int, values: dict[str, Any]) -> None:
        """Logs data if writer is open and should_write returns True.

        Args:
            step: The training step.
            values: The values to log. Must be Python scalars or TMetricValues
                containing scalars. Call replicate_to_host() on summaries
                before passing to ensure DTensors are converted to scalars.

        Raises:
            RuntimeError: If writer is not open.
            ValueError: If values contain unconverted Tensors.
        """
        if not self._within_context:
            raise RuntimeError("Writer not open. Call open() or use as context manager.")
        if not self.should_write(step):
            return

        # Convert all values to scalars before logging
        # Treat TMetricValue and Tensor as leaves, not containers to traverse
        def is_leaf(v):
            return isinstance(v, (TMetricValue, torch.Tensor))

        self._log(
            step,
            tree.map_with_path(
                lambda kp, v: _to_scalar(v, tree.path_str(kp)), values, is_leaf=is_leaf
            ),
        )

    def _log(self, step: int, values: dict[str, Any]) -> None:
        """Subclass hook: log the scalar values at the given step."""
        raise NotImplementedError(type(self))


# ---------------------------------------------------------------------------
# TensorBoardSummaryWriter

#  Preserved _replace_gfile, max_queue, per-key error handling.)
# ---------------------------------------------------------------------------


class TensorBoardSummaryWriter(SummaryWriter):
    """A SummaryWriter that logs to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard event files.
        max_queue: Maximum in-memory queue size for TensorBoard events.
    """

    def __init__(self, log_dir: str, max_queue: int = 10000) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.max_queue = max_queue
        self._tb_writer = None

    def open(self) -> None:
        super().open()
        if is_single_process_or_rank_zero():
            from torch.utils.tensorboard.writer import SummaryWriter as _TBSummaryWriter

            os.makedirs(self.log_dir, exist_ok=True)
            self._tb_writer = _TBSummaryWriter(
                log_dir=self.log_dir, max_queue=self.max_queue
            )
            _replace_gfile(self._tb_writer)
            logger.info(f"Initialized TensorBoardSummaryWriter at {self.log_dir}")

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
            logger.info("Flushed and closed TensorBoardSummaryWriter")
            self._tb_writer = None
        super().close()

    def _log(self, step: int, values: dict[str, int | float]) -> None:
        if self._tb_writer is not None:
            for k, v in values.items():
                try:
                    self._tb_writer.add_scalar(tag=k, scalar_value=v, global_step=step)
                except Exception as e:
                    logger.warning(
                        f"Error logging to TensorBoard at {step} for key {k}: {repr(e)}",
                        exc_info=e,
                    )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def _replace_gfile(tb_writer) -> None:
    """Replace TensorBoard's GFile with regular Python file for better performance.

    This optimization replaces the file implementation used by TensorBoard
    with a simple Python file: this replaces GFile, which closes and opens the file
    after appending each event (~50 bytes), which is fairly pathological for FUSE/NFS.

    All three reference libraries have this identical function.
    """
    try:
        from tensorboard.compat import tf

        file_writer = tb_writer._get_file_writer()
        if file_writer:
            event_writer = file_writer.event_writer
            # 16 MB buffer size for writes (empirically derived as working well for NFS/FUSE)
            events_file = open(
                event_writer._file_name, "wb", buffering=16 * 1024 * 1024
            )
            record_writer = event_writer._async_writer._writer

            if isinstance(record_writer._writer, tf.io.gfile.GFile):
                event_writer._general_file_writer = events_file
                record_writer._writer = events_file
                logger.info("Replaced TensorBoard GFile with buffered Python file")
            else:
                events_file.close()
    except Exception as e:
        logger.warning(f"Could not replace TensorBoard GFile: {repr(e)}")


# ---------------------------------------------------------------------------
# WandbSummaryWriter

# ---------------------------------------------------------------------------


class WandbSummaryWriter(SummaryWriter):
    """A SummaryWriter that logs to Weights & Biases.

    Args:
        project: W&B project name.
        entity: W&B entity (team or user).
        config_dict: Training config to log as run config.
        mode: W&B mode ("online", "offline", "disabled").
    """

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        config_dict: dict | None = None,
        mode: str = "offline",
    ) -> None:
        super().__init__()
        self.project = project
        self.entity = entity
        self.config_dict = config_dict
        self.mode = mode
        self._run = None

    def open(self) -> None:
        super().open()
        if is_single_process_or_rank_zero():
            import wandb

            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.config_dict,
                mode=self.mode,
            )
            wandb.define_metric("*", step_metric="global_step", step_sync=True)
            logger.info(f"Initialized WandbSummaryWriter (project={self.project})")

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
        super().close()

    def _log(self, step: int, values: dict[str, int | float]) -> None:
        if self._run is not None:
            values = {**values, "global_step": step}  # copy to avoid mutating shared dict
            self._run.log(values, step=step)


# ---------------------------------------------------------------------------
# CompositeSummaryWriter

# ---------------------------------------------------------------------------


class CompositeSummaryWriter(SummaryWriter):
    """A SummaryWriter that fans out to multiple writers.

    Per-writer error isolation: if one writer fails, others continue.

    Args:
        writers: Dict of name → SummaryWriter instances.
    """

    def __init__(self, writers: dict[str, SummaryWriter] | None = None) -> None:
        super().__init__()
        self._writers: dict[str, SummaryWriter | None] = dict(writers or {})

    def open(self) -> None:
        super().open()
        for name, writer in list(self._writers.items()):
            if writer is not None:
                try:
                    writer.open()
                    logger.info(f"Initialized {name} writer in CompositeSummaryWriter")
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize {name} writer: {repr(e)}", exc_info=e
                    )
                    self._writers[name] = None

    def close(self) -> None:
        for name, writer in list(self._writers.items()):
            if writer is not None:
                try:
                    writer.close()
                except Exception as e:
                    logger.warning(
                        f"Error during {name} writer close: {repr(e)}", exc_info=e
                    )
                finally:
                    self._writers[name] = None
        super().close()

    def _log(self, step: int, values: dict[str, Any]) -> None:
        """Log to all writers. Bypasses child should_write (composite gates globally)."""
        for name, writer in self._writers.items():
            if writer is not None:
                try:
                    writer._log(step, values)
                except Exception as e:
                    logger.warning(
                        f"Error logging to {name} writer for step {step}: {repr(e)}",
                        exc_info=e,
                    )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class LoggingSummaryWriter(SummaryWriter):
    """A SummaryWriter that logs to Python logger.info. Useful for console output."""

    def _log(self, step: int, values: dict[str, Any]) -> None:
        logger.info("Summaries at step %d: %s", step, values)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class InMemorySummaryWriter(SummaryWriter):
    """A SummaryWriter that stores values in memory. Useful for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.summaries: dict[int, dict[str, Any]] = {}

    def _log(self, step: int, values: dict[str, Any]) -> None:
        self.summaries[step] = values
