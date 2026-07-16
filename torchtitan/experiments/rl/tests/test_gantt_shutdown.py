# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""best_effort_generate_rl_gantt_on_shutdown: renders when logs are present, isolates the
current run via start_time_us, and is best-effort (never raises) when logs are missing or
rendering fails. No GPU."""

import json
import os
import time

from torchtitan.experiments.rl import gantt as rl_gantt
from torchtitan.experiments.rl.gantt import best_effort_generate_rl_gantt_on_shutdown


def _write_run(
    dump_folder: str, *, num_generators: int, base_time_us: int, step: int = 1
) -> None:
    """Write a tiny per-rank structured_logs/ for one run (controller + trainer + N generators, rank 0)."""
    logs = os.path.join(dump_folder, "structured_logs")
    os.makedirs(logs, exist_ok=True)

    def span(task, name, source):
        base = {
            "global_rank": 0,
            "source": source,
            "step": step,
            "task_name": task,
            "log_type": "event",
        }
        return [
            {**base, "log_type_name": f"{name}_start", "time_us": base_time_us},
            {
                **base,
                "log_type_name": f"{name}_end",
                "time_us": base_time_us + 1000,
                "value": 1.0,
            },
        ]

    def write(fn, recs):
        with open(os.path.join(logs, fn), "w") as f:
            for record in recs:
                f.write(json.dumps(record) + "\n")

    write(
        f"rl_controller.global_rank_0.t{base_time_us}.jsonl",
        span("trainer_loop", "train_step", "rl_controller"),
    )
    write(
        f"rl_trainer.global_rank_0.t{base_time_us}.jsonl",
        span("Task-1", "forward_backward", "rl_trainer"),
    )
    for gen in range(num_generators):
        write(
            f"rl_generator.global_rank_0.t{base_time_us}g{gen}.jsonl",
            span("vllm_engine", "vllm_engine_step_burst", "rl_generator"),
        )


def _now_us() -> int:
    return time.time_ns() // 1_000


def test_renders_gantt_when_logs_present(tmp_path):
    dump = str(tmp_path)
    now_us = _now_us()
    _write_run(dump, num_generators=3, base_time_us=now_us)
    best_effort_generate_rl_gantt_on_shutdown(dump, start_time_us=now_us - 1)
    assert os.path.exists(os.path.join(dump, "gantt.json"))


def test_skips_render_when_structured_logs_missing(tmp_path):
    # No structured_logs/ at all: nothing rendered, nothing raised.
    best_effort_generate_rl_gantt_on_shutdown(str(tmp_path), start_time_us=_now_us())
    assert not os.path.exists(os.path.join(tmp_path, "gantt.json"))


def test_start_time_us_excludes_older_run_records(tmp_path):
    # A reused dump folder holds an old run (steps up to 1000, in the past) and the
    # current 1-step run; only the current run's records may render.
    dump = str(tmp_path)
    now_us = _now_us()
    old_us = now_us - 3_600_000_000  # one hour earlier
    _write_run(dump, num_generators=1, base_time_us=old_us, step=1000)
    _write_run(dump, num_generators=1, base_time_us=now_us, step=1)

    best_effort_generate_rl_gantt_on_shutdown(dump, start_time_us=now_us - 1)

    trace = json.load(open(os.path.join(dump, "gantt.json")))
    steps = {
        e["args"]["step"]
        for e in trace["traceEvents"]
        if e.get("ph") == "X" and "step" in e.get("args", {})
    }
    assert steps == {1}


def test_no_raise_when_render_fails(tmp_path, monkeypatch):
    dump = str(tmp_path)
    now_us = _now_us()
    _write_run(dump, num_generators=1, base_time_us=now_us)

    def boom(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(rl_gantt, "generate_rl_gantt", boom)
    # best-effort: the rendering failure is swallowed, not propagated.
    best_effort_generate_rl_gantt_on_shutdown(dump, start_time_us=now_us)
