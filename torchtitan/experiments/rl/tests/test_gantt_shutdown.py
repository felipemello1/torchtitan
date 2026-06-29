# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""best_effort_generate_rl_gantt_on_shutdown: renders when logs are present, and is best-effort
(never raises) when logs are missing or rendering fails. No GPU."""

import json
import os

from torchtitan.experiments.rl import gantt as rl_gantt
from torchtitan.experiments.rl.gantt import best_effort_generate_rl_gantt_on_shutdown


def _write_run(dump_folder: str, *, num_generators: int) -> None:
    """Write a tiny per-rank structured_logs/ for one run (controller + trainer + N generators, rank 0)."""
    logs = os.path.join(dump_folder, "structured_logs")
    os.makedirs(logs, exist_ok=True)

    def span(task, name, source):
        base = {
            "global_rank": 0,
            "rank": 0,
            "source": source,
            "step": 1,
            "task_name": task,
        }
        return [
            {**base, "log_type_name": f"{name}_start", "time_us": 0},
            {**base, "log_type_name": f"{name}_end", "time_us": 1000, "value": 1.0},
        ]

    def write(fn, recs):
        with open(os.path.join(logs, fn), "w") as f:
            for record in recs:
                f.write(json.dumps(record) + "\n")

    write(
        "rl_controller.global_rank_0.t.jsonl",
        span("trainer", "train_step", "rl_controller"),
    )
    write(
        "rl_trainer.global_rank_0.t.jsonl",
        span("Task-1", "forward_backward", "rl_trainer"),
    )
    for gen in range(num_generators):
        write(
            f"rl_generator.global_rank_0.g{gen}.jsonl",
            span("vllm_engine", "vllm_engine_step", "rl_generator"),
        )


def test_renders_gantt_when_logs_present(tmp_path):
    dump = str(tmp_path)
    _write_run(dump, num_generators=3)
    best_effort_generate_rl_gantt_on_shutdown(
        dump, expected_rank0_files=2 + 3, timeout_s=2.0, stable_s=0.1
    )
    assert os.path.exists(os.path.join(dump, "gantt.json"))


def test_skips_render_when_structured_logs_missing(tmp_path, monkeypatch):
    # No structured_logs/ -> "no_logs": must NOT call generate_rl_gantt (no false "wrote" log) or write.
    calls = []
    monkeypatch.setattr(
        rl_gantt, "generate_rl_gantt", lambda *a, **k: calls.append((a, k))
    )
    best_effort_generate_rl_gantt_on_shutdown(
        str(tmp_path), expected_rank0_files=5, timeout_s=0.3, stable_s=0.1
    )
    assert calls == []
    assert not os.path.exists(os.path.join(tmp_path, "gantt.json"))


def test_partial_render_on_timeout_with_some_logs(tmp_path):
    dump = str(tmp_path)
    _write_run(
        dump, num_generators=1
    )  # 3 rank-0 files present (controller + trainer + 1 generator)
    # Demand more rank-0 files than exist -> floor never met -> timeout -> "partial" -> still renders.
    best_effort_generate_rl_gantt_on_shutdown(
        dump, expected_rank0_files=99, timeout_s=0.5, stable_s=0.1
    )
    assert os.path.exists(os.path.join(dump, "gantt.json"))


def test_no_raise_when_render_fails(tmp_path, monkeypatch):
    dump = str(tmp_path)
    _write_run(dump, num_generators=1)

    def boom(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(rl_gantt, "generate_rl_gantt", boom)
    # best-effort: the rendering failure is swallowed, not propagated.
    best_effort_generate_rl_gantt_on_shutdown(
        dump, expected_rank0_files=3, timeout_s=2.0, stable_s=0.1
    )
