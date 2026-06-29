# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RL render policy for the generic gantt renderer (lands as experiments/rl/gantt.py).

The generic `gantt_generator` knows nothing about RL. This module supplies the RL view:
pin the pipeline loops + the vLLM engine as their own rows, keep one rollout worker, and
collapse the per-request / per-sibling `Task-N` fan-out to a few rows.
"""

import os
import time
from glob import glob

from torchtitan.experiments.rl.trace_names import (
    BATCHER_TASK_NAME,
    DATA_INPUT_TASK_NAME,
    ROLLOUT_WORKER_TASK_NAME_PREFIX,
    TRAINER_TASK_NAME,
    VLLM_ENGINE_TASK_NAME,
    WEIGHT_SYNC_MANAGER_TASK_NAME,
)
from torchtitan.observability.structured_logger.gantt_generator import (
    CollapsedTasks,
    generate_gantt_trace,
)
from torchtitan.tools.logging import logger

# Pipeline loops (controller) + the weight-sync tasks + the vLLM engine loop (generator), each its
# own row, in this order. trainer is the clock; the engine is the generator's main loop.
RL_PINNED_TASKS: tuple[str, ...] = (
    TRAINER_TASK_NAME,
    BATCHER_TASK_NAME,
    DATA_INPUT_TASK_NAME,
    WEIGHT_SYNC_MANAGER_TASK_NAME,
    VLLM_ENGINE_TASK_NAME,
)

# Process order: controller (orchestrator) -> trainer (the clock) -> generators (producers).
RL_SOURCE_ORDER: tuple[str, ...] = ("rl_controller", "rl_trainer", "rl_generator")

RL_COLLAPSE: tuple[CollapsedTasks, ...] = (
    # 32 persistent rollout_worker_N (one per active buffer slot) -> one representative row.
    CollapsedTasks(
        match=rf"^{ROLLOUT_WORKER_TASK_NAME_PREFIX}_\d+$",
        max_rows=1,
        label="rollout worker",
    ),
    # The auto-named Task-N family is one-RPC-per-task; label it by what the RPC IS on each source
    # (first match wins per source). ~num_workers*group_size concurrent -> 8 rows (raw=True for all).
    CollapsedTasks(
        match=r"^Task-\d+$",
        max_rows=8,
        label="trainer endpoint",
        source_match="rl_trainer",
    ),
    CollapsedTasks(
        match=r"^Task-\d+$",
        max_rows=8,
        label="generation request",
        source_match="rl_generator",
    ),
    CollapsedTasks(match=r"^Task-\d+$", max_rows=8, label="task"),
)


def generate_rl_gantt(
    log_dir: str, output_path: str, **selection_and_window_kwargs
) -> dict:
    r"""Render an RL gantt: the generic renderer with the RL pin/collapse policy applied.

    Extra kwargs (``ranks``, ``sources``, ``file_name_regex``, ``last_seconds``, ``start_step`` /
    ``end_step``, ``hide_spans``, ``raw``) pass straight through to ``generate_gantt_trace``.

    Defaults to rank 0 per source because non-rank-0 ranks usually repeat the same span families and
    make Perfetto harder to read. Use ``raw=True`` or ``ranks=None`` when debugging cross-rank
    duration skew or stragglers.

    Example:

        # last 50 steps of one run picked out of a mixed dir
        generate_rl_gantt("outputs/rl/structured_logs/", "outputs/rl/gantt.json",
                          file_name_regex="20260625-1941", start_step=51, end_step=100)
    """
    return generate_gantt_trace(
        log_dir,
        output_path,
        pinned_tasks=RL_PINNED_TASKS,
        collapse=RL_COLLAPSE,
        source_order=RL_SOURCE_ORDER,
        **selection_and_window_kwargs,
    )


def best_effort_generate_rl_gantt_on_shutdown(
    dump_folder: str,
    *,
    expected_rank0_files: int | None,
    timeout_s: float = 30.0,
    stable_s: float = 2.0,
    output_name: str = "gantt.json",
) -> None:
    """Render the RL gantt to ``dump_folder/output_name`` once the run's logs have landed.

    Best-effort: a debug artifact must never crash or hang the run, so this catches everything and
    caps its wait. The wait exists because on MAST the per-rank JSONL is written to warm storage
    (OILFS), which lags — an actor's final bytes become visible some time after it closes. On local
    FS the snapshot is already stable, so it returns at once.

    Args:
        dump_folder: the run's output root; logs are read from ``dump_folder/structured_logs``.
        expected_rank0_files: wait until at least this many ``*.global_rank_0.*.jsonl`` files exist
            (``2 + num_generators``: controller + trainer + one per generator). None = skip the floor.
        timeout_s: total budget for the whole wait; on timeout, render what is visible + warn.
        stable_s: the ``(file_count, total_size, latest_mtime)`` snapshot must hold this long.
        output_name: gantt filename under ``dump_folder``.

    Example:

        # 3 generators -> wait for 5 rank-0 files, then render the full run
        best_effort_generate_rl_gantt_on_shutdown("outputs/rl/run42", expected_rank0_files=5)
    """
    log_dir = os.path.join(dump_folder, "structured_logs")
    try:
        status = _wait_until_structured_logs_settle(
            log_dir,
            expected_rank0_files=expected_rank0_files,
            timeout_s=timeout_s,
            stable_s=stable_s,
        )
        if status == "no_logs":
            logger.warning(f"gantt-on-shutdown: {log_dir} has no logs; skipping")
            return
        # status is "settled" or "partial" (the partial timeout already warned). Render either way.
        output_path = os.path.join(dump_folder, output_name)
        generate_rl_gantt(log_dir, output_path)
        if os.path.exists(output_path):
            logger.info(f"gantt-on-shutdown: wrote {output_path}")
        else:
            logger.warning(
                f"gantt-on-shutdown: no records in {log_dir}; nothing written"
            )
    except Exception as e:
        logger.warning(f"gantt-on-shutdown: skipped ({type(e).__name__}: {e})")


def _structured_logs_snapshot(
    log_dir: str,
) -> tuple[tuple[int, int, float] | None, int]:
    """Return ``((file_count, total_size, latest_mtime), rank0_file_count)`` for ``log_dir/*.jsonl``,
    or ``(None, 0)`` if the directory is absent."""
    if not os.path.isdir(log_dir):
        return None, 0
    file_count = total_size = rank0 = 0
    latest_mtime = 0.0
    for path in glob(os.path.join(log_dir, "*.jsonl")):
        stat = os.stat(path)
        file_count += 1
        total_size += stat.st_size
        latest_mtime = max(latest_mtime, stat.st_mtime)
        if ".global_rank_0." in os.path.basename(path):
            rank0 += 1
    return (file_count, total_size, latest_mtime), rank0


def _wait_until_structured_logs_settle(
    log_dir: str, *, expected_rank0_files: int | None, timeout_s: float, stable_s: float
) -> str:
    """Block until ``log_dir`` has >= ``expected_rank0_files`` rank-0 JSONL files AND its snapshot is
    unchanged for ``stable_s``, or until ``timeout_s`` elapses. All waiting shares one budget.

    Returns:
        ``"settled"`` -- the file floor was met and the snapshot held for ``stable_s``.
        ``"partial"`` -- timed out with >=1 JSONL visible (caller renders a partial gantt; warned here).
        ``"no_logs"`` -- timed out with no JSONL visible (caller skips rendering).
    """
    deadline = time.monotonic() + timeout_s
    last_snapshot: tuple[int, int, float] | None = None
    stable_since: float | None = None
    while True:
        snapshot, rank0 = _structured_logs_snapshot(log_dir)
        enough = expected_rank0_files is None or rank0 >= expected_rank0_files
        if snapshot is not None and snapshot == last_snapshot:
            stable_since = (
                stable_since if stable_since is not None else time.monotonic()
            )
            if enough and time.monotonic() - stable_since >= stable_s:
                return "settled"
        else:
            last_snapshot, stable_since = snapshot, None
        if time.monotonic() >= deadline:
            file_count = snapshot[0] if snapshot is not None else 0
            if file_count == 0:
                return "no_logs"
            floor = (
                f"/{expected_rank0_files}" if expected_rank0_files is not None else ""
            )
            logger.warning(
                f"gantt-on-shutdown: timeout after {timeout_s:.0f}s "
                f"(rank0_files={rank0}{floor}); rendering partial gantt"
            )
            return "partial"
        time.sleep(min(0.5, stable_s / 2))
