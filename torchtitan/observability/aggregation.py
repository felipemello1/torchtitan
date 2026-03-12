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
from datetime import datetime
from typing import Any

from torchtitan.components.metrics import (
    BaseLogger,
    LoggerContainer,
    TensorBoardLogger,
    WandBLogger,
)
from torchtitan.observability.metrics import REDUCE_REGISTRY
from torchtitan.observability.structured_logging import init_observability
from torchtitan.tools.utils import Color

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
    logger_backend: BaseLogger,
    console_keys: list[str],
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
        logger_backend.log(aggregated, step)
        if console_keys:
            _log_to_console(step, aggregated, console_keys)

    return aggregated, len(entries)


# Colors cycle for console columns (assigned by position in the key list).
_COLORS = ["green", "yellow", "cyan", "blue", "magenta", "red"]


def _fmt_value(value: float) -> str:
    """Format a number showing first 2 non-zero digits after '.', up to 5 decimals.

    Examples::

        _fmt_value(3.67060)   → '3.6706'
        _fmt_value(0.00123)   → '0.0012'
        _fmt_value(1234.5)    → '1234.5'
        _fmt_value(0.5603)    → '0.5603'
    """
    if value == 0:
        return "0"
    # Find how many decimals needed to show 2 non-zero digits
    abs_frac = abs(value) - int(abs(value))
    if abs_frac == 0 or abs(value) >= 100:
        return f"{value:.1f}"
    decimals = 0
    non_zero_seen = 0
    temp = abs_frac
    while decimals < 5 and non_zero_seen < 2:
        decimals += 1
        temp *= 10
        digit = int(temp) % 10
        if digit != 0 or non_zero_seen > 0:
            non_zero_seen += 1
    decimals = max(decimals, 2)
    return f"{value:.{decimals}f}"


def _log_to_console(
    step: int, aggregated: dict[str, float], console_keys: list[str]
) -> None:
    """Print one console line with the configured metric keys.

    Colors are assigned by position in the key list and cycle through
    _COLORS. Missing metrics show '--'.
    """
    color = Color()
    parts = [f"{color.red}step: {step:2}"]
    for i, key in enumerate(console_keys):
        c = getattr(color, _COLORS[i % len(_COLORS)])
        label = key.rsplit("/", 1)[-1]  # e.g., "toy_trainer/loss_mean" → "loss_mean"
        val = aggregated.get(key)
        if val is None:
            parts.append(f"{c}{label}: --")
        else:
            parts.append(f"{c}{label}: {_fmt_value(val)}")
    parts.append(color.reset)
    logger.info("  ".join(parts))


def _build_metric_logger(
    dump_folder: str,
    *,
    enable_wandb: bool = False,
    enable_tensorboard: bool = False,
    save_tb_folder: str = "tb",
    config_dict: dict[str, Any] | None = None,
    tag: str | None = None,
) -> BaseLogger:
    """Build WandB/TB logger for the logging subprocess."""
    container = LoggerContainer()
    if enable_tensorboard:
        tb_dir = os.path.join(
            dump_folder, save_tb_folder, datetime.now().strftime("%Y%m%d-%H%M")
        )
        container.add_logger(TensorBoardLogger(log_dir=tb_dir, tag=tag))
    if enable_wandb:
        container.add_logger(
            WandBLogger(log_dir=dump_folder, config_dict=config_dict, tag=tag)
        )
    return container


def logging_worker(
    queue: multiprocessing.Queue,
    dump_folder: str,
    *,
    enable_wandb: bool = False,
    enable_tensorboard: bool = False,
    save_tb_folder: str = "tb",
    config_dict: dict[str, Any] | None = None,
    tag: str | None = None,
    console_keys: list[str] | None = None,
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
        console_keys: Metric keys to print to console. Empty list disables.
        queue_timeout_s: Seconds to wait before assuming training crashed.
            Default 600 (10 min). Set to 1 for unit tests.
    """
    init_observability(source="logging_worker", output_dir=dump_folder, rank=0)

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
            step, buffer, is_validation, logger_backend, console_keys or []
        )

        logger.debug(
            "[obs] step %d: read=%.1fms entries=%d",
            step,
            (t_read_end - t_read_start) * 1000,
            num_entries,
        )

    logger.info("[logging process] shutting down")
    logger_backend.close()
