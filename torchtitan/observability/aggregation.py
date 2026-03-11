# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Aggregation: reduce experiment metrics from JSONL + logging subprocess."""

import glob
import json
import logging
import multiprocessing
import os
import time
from collections import defaultdict
from typing import Any

from torchtitan.observability.metrics import REDUCE_REGISTRY

logger = logging.getLogger(__name__)

_QUEUE_TIMEOUT_S = 600  # 10 minutes — if no signal, assume training crashed


def aggregate(entries: list[dict]) -> dict[str, float]:
    """Reduce a list of metric entries to a single dict. Pure function.

    Groups entries by key and delegates to REDUCE_REGISTRY.

    Example::

        entries = [
            {"key": "loss", "reduce": "MeanMetric", "sum": 6.0, "weight": 3.0},
            {"key": "loss", "reduce": "MeanMetric", "sum": 4.0, "weight": 2.0},
        ]
        aggregate(entries)  # {"loss": 2.0}
    """
    if not entries:
        return {}
    result: dict[str, float] = {}
    by_key: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_key[entry["key"]].append(entry)
    for key, key_entries in by_key.items():
        cls = REDUCE_REGISTRY[key_entries[0]["reduce"]]
        result[key] = cls.get_reduced_value_from_states(key_entries)
    return result


def _read_new_lines(
    log_dir: str,
    offsets: dict[str, int],
    buffer: dict[int, list[dict]],
) -> None:
    """Read all new JSONL lines into buffer, grouped by step.

    Opens files fresh each time (NFS-friendly — forces cache invalidation).
    Tracks offsets to avoid re-reading old lines.
    """
    for fp in sorted(glob.glob(os.path.join(log_dir, "*.jsonl"))):
        with open(fp) as f:
            f.seek(offsets.get(fp, 0))
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = entry.get("step")
                if step is not None:
                    buffer[step].append(entry)
            offsets[fp] = f.tell()


def _flush_step(
    step: int,
    buffer: dict[int, list[dict]],
    is_validation: bool,
    logger_backend: Any = None,
) -> tuple[dict[str, float], int]:
    """Pop entries for step, aggregate, write to backends + console.

    Purges entries for steps older than the requested step.

    Returns:
        (aggregated_dict, num_entries)
    """
    entries = buffer.pop(step, [])
    for s in [s for s in buffer if s < step]:
        del buffer[s]

    aggregated = aggregate(entries)

    if aggregated:
        if logger_backend is not None:
            logger_backend.log(aggregated, step)
        _log_to_console(step, aggregated, is_validation)

    return aggregated, len(entries)


# ANSI colors for console output
_RED = "\033[91m"
_GREEN = "\033[92m"
_ORANGE = "\033[93m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"


def _log_to_console(
    step: int, aggregated: dict[str, float], is_validation: bool
) -> None:
    """Print training + optional validation console line."""
    loss = aggregated.get("loss/trainer_loss_mean")
    if loss is None:
        loss = aggregated.get("toy_trainer/loss_mean")
    if loss is None:
        return
    grad_norm = aggregated.get("trainer_gradient/norm_max", 0)
    if grad_norm == 0:
        grad_norm = aggregated.get("toy_trainer/grad_norm_max", 0)
    mem_gib = aggregated.get("trainer_memory/reserved_gib_max", 0)
    tps = aggregated.get("trainer_throughput/tps_mean", 0)
    mfu = aggregated.get("trainer_throughput/mfu_pct_mean", 0)

    logger.info(
        f"{_RED}step: {step:2}  "
        f"{_GREEN}loss: {loss:8.5f}  "
        f"{_ORANGE}grad_norm: {grad_norm:7.4f}  "
        f"{_CYAN}memory: {mem_gib:5.2f}GiB  "
        f"{_BLUE}tps: {round(tps):,}  "
        f"{_MAGENTA}mfu: {mfu:.2f}%{_RESET}"
    )


def _build_metric_logger(
    dump_folder: str,
    *,
    enable_wandb: bool = False,
    enable_tensorboard: bool = False,
    save_tb_folder: str = "tb",
    config_dict: dict[str, Any] | None = None,
    tag: str | None = None,
) -> Any:
    """Build WandB/TB logger for the logging subprocess.

    Returns a simple object with log(metrics, step) and close() methods.
    """

    class _LoggerContainer:
        def __init__(self):
            self._tb_writer = None
            self._wandb_enabled = False

            if enable_tensorboard:
                from datetime import datetime

                from torch.utils.tensorboard import SummaryWriter

                tb_dir = os.path.join(
                    dump_folder, save_tb_folder, datetime.now().strftime("%Y%m%d-%H%M")
                )
                self._tb_writer = SummaryWriter(tb_dir, max_queue=1000)
                logger.info("[logging process] TensorBoard: %s", tb_dir)

            if enable_wandb:
                import wandb

                wandb.init(
                    project=os.getenv("WANDB_PROJECT", "torchtitan"),
                    name=os.getenv("WANDB_RUN_NAME", None),
                    dir=dump_folder,
                    config=config_dict,
                )
                self._wandb_enabled = True
                logger.info("[logging process] WandB initialized")

        def log(self, metrics: dict[str, float], step: int) -> None:
            if self._tb_writer is not None:
                for k, v in metrics.items():
                    self._tb_writer.add_scalar(
                        k if tag is None else f"{tag}/{k}", v, step
                    )
            if self._wandb_enabled:
                import wandb

                wandb.log(metrics, step=step)

        def close(self) -> None:
            if self._tb_writer is not None:
                self._tb_writer.close()
            if self._wandb_enabled:
                import wandb

                wandb.finish()

    return _LoggerContainer()


def logging_worker(
    queue: multiprocessing.Queue,
    dump_folder: str,
    *,
    enable_wandb: bool = False,
    enable_tensorboard: bool = False,
    save_tb_folder: str = "tb",
    config_dict: dict[str, Any] | None = None,
    tag: str | None = None,
    queue_timeout_s: float = _QUEUE_TIMEOUT_S,
) -> None:
    """Logging subprocess entry point.

    Reads experiment JSONL, aggregates, writes to WandB/TB/console.
    Receives ``(step, is_validation)`` tuples via queue.
    Shuts down on ``None`` sentinel or queue timeout (crash recovery).

    Args:
        queue: Receives (step, is_validation) or None to shut down.
        dump_folder: Root output directory containing experiment_logs/.
        enable_wandb: Whether to log to WandB.
        enable_tensorboard: Whether to log to TensorBoard.
        save_tb_folder: Subfolder for TensorBoard files.
        config_dict: Full config for wandb.init(config=...).
        tag: Prefix for TB/WandB scalar keys.
        queue_timeout_s: Seconds to wait before assuming training crashed.
            Default 600 (10 min). Set to 1 for unit tests.
    """
    from torchtitan.observability import init_observability

    init_observability(source="logging", output_dir=dump_folder, rank=0)

    log_dir = os.path.join(dump_folder, "experiment_logs")
    buffer: dict[int, list[dict]] = defaultdict(list)

    # Skip historical data from previous runs / checkpoint resume.
    offsets: dict[str, int] = {}
    for fp in glob.glob(os.path.join(log_dir, "*.jsonl")):
        offsets[fp] = os.path.getsize(fp)

    logger_backend = _build_metric_logger(
        dump_folder,
        enable_wandb=enable_wandb,
        enable_tensorboard=enable_tensorboard,
        save_tb_folder=save_tb_folder,
        config_dict=config_dict,
        tag=tag,
    )
    logger.info("[logging process] started, reading from %s", log_dir)

    while True:
        try:
            msg = queue.get(timeout=queue_timeout_s)
        except Exception:
            logger.warning(
                "[logging process] no signal in %ds, assuming training crashed",
                queue_timeout_s,
            )
            break

        if msg is None:
            break

        step, is_validation = msg
        time.sleep(0.02)  # let filesystem propagate writes

        t_read_start = time.perf_counter()
        _read_new_lines(log_dir, offsets, buffer)
        t_read_end = time.perf_counter()

        aggregated, num_entries = _flush_step(
            step, buffer, is_validation, logger_backend
        )

        logger.debug(
            "[obs] step %d: read=%.1fms entries=%d",
            step,
            (t_read_end - t_read_start) * 1000,
            num_entries,
        )

    logger.info("[logging process] shutting down")
    logger_backend.close()
