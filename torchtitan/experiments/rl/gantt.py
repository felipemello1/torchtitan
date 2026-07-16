# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RL render policy for the generic gantt renderer.

The generic `gantt_generator` knows nothing about RL. This module supplies the RL view:
pin the pipeline loops + the vLLM engine as their own rows, keep one rollout worker, and
collapse the per-request / per-endpoint `Task-N` fan-out to a few rows.
"""

import os

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
        label="rollout_worker_loop",
    ),
    # The auto-named Task-N family, labeled by what the tasks ARE on each source (first match
    # wins per source). On the controller they drive per-sample rollouts; on the trainer and
    # generator they are Monarch endpoint dispatches. raw=True shows all of them.
    CollapsedTasks(
        match=r"^Task-\d+$",
        max_rows=8,
        label="rollout task",
        source_match="rl_controller",
    ),
    CollapsedTasks(
        match=r"^Task-\d+$",
        max_rows=8,
        label="trainer endpoint",
        source_match="rl_trainer",
    ),
    CollapsedTasks(
        match=r"^Task-\d+$",
        max_rows=8,
        label="generator endpoint",
        source_match="rl_generator",
    ),
    CollapsedTasks(match=r"^Task-\d+$", max_rows=8, label="task"),
)


def generate_rl_gantt(
    log_dir: str, output_path: str, **selection_and_window_kwargs
) -> dict:
    r"""Render an RL gantt: the generic renderer with the RL pin/collapse policy applied.

    Extra kwargs (``ranks``, ``sources``, ``file_name_regex``, ``start_time_us``, ``last_steps``,
    ``start_step`` / ``end_step``, ``raw``) pass straight through to ``generate_gantt_trace``.

    Defaults to rank 0 per source because non-rank-0 ranks usually repeat the same span families and
    make Perfetto harder to read. Use ``raw=True`` or ``ranks=None`` when debugging cross-rank
    duration skew or stragglers.

    Example:

        # last 10 steps of one run picked out of a mixed dir
        generate_rl_gantt("outputs/rl/structured_logs/", "outputs/rl/gantt.json",
                          file_name_regex="20260625-1941", last_steps=10)
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
    start_time_us: int,
    last_steps: int | None = 10,
    output_name: str = "gantt.json",
) -> None:
    """Render the RL gantt to ``dump_folder/output_name`` after a run ends.

    Best-effort: a debug artifact must never crash the run, so every failure is
    caught and logged. ``start_time_us`` isolates the current run -- a reused dump
    folder may hold older runs' JSONL, which must not leak into the render.

    Args:
        dump_folder: the run's output root; logs are read from ``dump_folder/structured_logs``.
        start_time_us: the run's start (epoch us), captured before logger init.
        last_steps: render only the final N steps (None = full run; unusable at ~1000 steps).
        output_name: gantt filename under ``dump_folder``.

    Example:

        best_effort_generate_rl_gantt_on_shutdown(
            "outputs/rl/run42", start_time_us=run_started_us, last_steps=10)
    """
    try:
        generate_rl_gantt(
            os.path.join(dump_folder, "structured_logs"),
            os.path.join(dump_folder, output_name),
            start_time_us=start_time_us,
            last_steps=last_steps,
        )
    except Exception:
        logger.warning("Failed to generate shutdown gantt", exc_info=True)
