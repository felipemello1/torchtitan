# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Proposed additions to tests/unit_tests/observability/test_structured_logging.py.

Generic renderer (policy supplied by the test, as the RL preset does) + windowing.
Synthetic fixtures only. Run against this RC via the bundled ``run_tests.py``.
"""

import json
from collections import defaultdict

from torchtitan.observability.structured_logger.gantt_generator import (
    _pack_intervals,
    CollapsedTasks,
    generate_gantt_trace,
)

# RL-shaped policy used by several tests (mirrors experiments/rl/gantt.py).
RL_PINNED = ("trainer", "batcher", "data_input", "vllm_engine")
RL_COLLAPSE = (
    CollapsedTasks(match=r"^rollout_worker_\d+$", max_rows=1, label="rollout worker"),
    CollapsedTasks(match=r"^Task-\d+$", max_rows=8, label="task"),
)


def _span(task, name, t0, t1, *, rank=0, source="rl_controller", step=1):
    dur = (t1 - t0) / 1000
    base = {
        "log_type": "event",
        "global_rank": rank,
        "rank": rank,
        "source": source,
        "step": step,
    }
    if task is not None:
        base["task_name"] = task
    return [
        {**base, "log_type_name": f"{name}_start", "time_us": t0},
        {**base, "log_type_name": f"{name}_end", "time_us": t1, "value": dur},
    ]


def _write(tmp_path, fn, recs):
    with open(tmp_path / fn, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _rows(trace):
    pid_name, tids = {}, defaultdict(set)
    for e in trace["traceEvents"]:
        if e["ph"] == "M" and e["name"] == "process_name":
            pid_name[e["pid"]] = e["args"]["name"]
        elif e["ph"] == "X":
            tids[e["pid"]].add(e["tid"])
    return {pid_name[p]: len(t) for p, t in tids.items()}


def _labels(trace):
    pid_name = {
        e["pid"]: e["args"]["name"]
        for e in trace["traceEvents"]
        if e["ph"] == "M" and e["name"] == "process_name"
    }
    out = defaultdict(dict)
    for e in trace["traceEvents"]:
        if e["ph"] == "M" and e["name"] == "thread_name":
            out[pid_name[e["pid"]]][e["tid"]] = e["args"]["name"]
    return out


def _controller(tmp_path):
    recs = []
    recs += _span("trainer", "train_step", 0, 1000)
    recs += _span("batcher", "batcher_pack", 0, 1000)
    recs += _span("data_input", "get_training_sample", 0, 1000)
    for i in range(4):
        recs += _span(f"rollout_worker_{i}", "rollout_group", 0, 1000)
    for n in range(1, 21):
        recs += _span(f"Task-{n}", "generate", 100, 200)  # 20 concurrent
    recs += _span("trainer", "PolicyTrainer.Config.build", 10, 20)
    recs.append(
        {
            "log_type": "instant",
            "log_type_name": "metric_value",
            "event_name": "loss",
            "value": 2.5,
            "time_us": 150,
            "global_rank": 0,
            "rank": 0,
            "source": "rl_controller",
            "task_name": "Task-19",
            "step": 1,
        }
    )
    _write(tmp_path, "rl_controller.global_rank_0.20260101-000000-AAAAAA.jsonl", recs)


# --- generic mechanism ---


def test_generic_default_is_faithful(tmp_path):
    recs = _span("trainer", "train_step", 0, 1000) + _span(
        "trainer", "X.Config.build", 10, 20
    )
    for n in range(3):
        recs += _span(f"Task-{n}", "generate", 100, 200)
    _write(tmp_path, "rl_controller.global_rank_0.t.jsonl", recs)
    t = generate_gantt_trace(str(tmp_path), str(tmp_path / "g.json"))  # no policy
    assert _rows(t)["rl_controller"] == 4  # trainer + 3 Task-N, each own row
    assert "X.Config.build" in {
        e["name"] for e in t["traceEvents"] if e["ph"] == "X"
    }  # visible


def test_pinned_order_and_labels(tmp_path):
    _controller(tmp_path)
    t = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "g.json"),
        pinned_tasks=RL_PINNED,
        collapse=RL_COLLAPSE,
    )
    lbl = next(iter(_labels(t).values()))
    assert lbl[0] == "trainer" and lbl[1] == "batcher" and lbl[2] == "data_input"
    assert lbl[3] == "rollout worker"  # collapse max_rows=1 -> unnumbered label


def test_collapse_interval_caps_rows_no_pool(tmp_path):
    _controller(tmp_path)
    collapse = (CollapsedTasks(match=r"^Task-\d+$", max_rows=3, label="task"),)
    t = generate_gantt_trace(str(tmp_path), str(tmp_path / "g.json"), collapse=collapse)
    rows = sorted(
        v for v in next(iter(_labels(t).values())).values() if v.startswith("task")
    )
    assert rows == ["task 0", "task 1", "task 2"]  # 20 concurrent -> 3 rows, numbered


def test_pack_intervals_reuses_lowest_free_row(tmp_path):
    # Task-3 starts after Task-1 ends; it must reuse row 0 (lowest free), not the
    # earliest-ending row 1 -- otherwise a max_rows=1 cap wrongly drops it.
    assert _pack_intervals(
        {"Task-1": (0, 100), "Task-2": (0, 90), "Task-3": (101, 110)}
    ) == {
        "Task-1": 0,
        "Task-2": 1,
        "Task-3": 0,
    }
    recs = (
        _span("Task-1", "generate", 0, 100)
        + _span("Task-2", "generate", 0, 90)
        + _span("Task-3", "generate", 101, 110)
    )
    _write(tmp_path, "rl_controller.global_rank_0.t.jsonl", recs)
    collapse = (CollapsedTasks(match=r"^Task-\d+$", max_rows=1, label="task"),)
    t = generate_gantt_trace(str(tmp_path), str(tmp_path / "g.json"), collapse=collapse)
    assert "generate" in {
        e["name"] for e in t["traceEvents"] if e["ph"] == "X"
    }  # Task-1 & Task-3 kept on row 0


def test_collapse_max_rows_one_is_unnumbered(tmp_path):
    _controller(tmp_path)
    collapse = (
        CollapsedTasks(
            match=r"^rollout_worker_\d+$", max_rows=1, label="rollout worker"
        ),
    )
    t = generate_gantt_trace(str(tmp_path), str(tmp_path / "g.json"), collapse=collapse)
    labels = set(next(iter(_labels(t).values())).values())
    assert "rollout worker" in labels  # exactly the label, no trailing number
    assert "rollout worker 0" not in labels


def test_thread_name_for_every_row(tmp_path):
    _controller(tmp_path)
    t = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "g.json"),
        pinned_tasks=RL_PINNED,
        collapse=RL_COLLAPSE,
    )
    src = "rl_controller"
    used = {e["tid"] for e in t["traceEvents"] if e["ph"] == "X"}
    assert used == set(_labels(t)[src])


def test_hide_spans_default_empty_and_opt_in(tmp_path):
    _controller(tmp_path)
    t = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "g.json"),
        pinned_tasks=RL_PINNED,
        collapse=RL_COLLAPSE,
    )
    assert "PolicyTrainer.Config.build" in {
        e["name"] for e in t["traceEvents"] if e["ph"] == "X"
    }
    t = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "g2.json"),
        pinned_tasks=RL_PINNED,
        collapse=RL_COLLAPSE,
        hide_spans=(r"\.Config\.build$",),
    )
    assert "PolicyTrainer.Config.build" not in {
        e["name"] for e in t["traceEvents"] if e["ph"] == "X"
    }
    assert "trainer" in next(iter(_labels(t).values())).values()  # row still present


def test_raw_all_ranks_no_labels(tmp_path):
    _controller(tmp_path)
    _write(
        tmp_path,
        "rl_controller.global_rank_1.b.jsonl",
        _span("trainer", "train_step", 0, 1000, rank=1),
    )
    t = generate_gantt_trace(str(tmp_path), str(tmp_path / "g.json"), raw=True)
    assert len(_rows(t)) == 2  # all ranks
    assert _labels(t) == {}


def test_rank_filter_default(tmp_path):
    _controller(tmp_path)
    _write(
        tmp_path,
        "rl_controller.global_rank_1.b.jsonl",
        _span("trainer", "train_step", 0, 1000, rank=1),
    )
    t = generate_gantt_trace(
        str(tmp_path), str(tmp_path / "g.json"), pinned_tasks=RL_PINNED
    )
    assert list(_rows(t)) == ["rl_controller"]  # only the rank-0 source survives


def test_file_name_regex(tmp_path):
    _controller(tmp_path)
    _write(
        tmp_path,
        "rl_controller.global_rank_0.20259999-OTHER.jsonl",
        _span("trainer", "t", 0, 1000),
    )
    t = generate_gantt_trace(
        str(tmp_path), str(tmp_path / "g.json"), file_name_regex="20260101"
    )
    assert list(_rows(t)) == ["rl_controller"]


def test_step_window(tmp_path):
    recs = []
    for stp in range(1, 11):
        recs += _span("trainer", "train_step", stp * 1000, stp * 1000 + 500, step=stp)
    _write(tmp_path, "rl_controller.global_rank_0.t.jsonl", recs)
    t = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "g.json"),
        pinned_tasks=("trainer",),
        start_step=8,
        end_step=10,
    )
    assert sum(1 for e in t["traceEvents"] if e["ph"] == "X") == 3


def test_last_seconds_window_keeps_overlapping(tmp_path):
    # span "long" runs the whole time; "late" only near the end. Keep the last ~0.6ms.
    recs = _span("trainer", "long", 0, 10000) + _span("trainer", "late", 9400, 9900)
    _write(tmp_path, "rl_controller.global_rank_0.t.jsonl", recs)
    t = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "g.json"),
        pinned_tasks=("trainer",),
        last_seconds=0.0006,
    )  # window ~[9400, 10000]
    names = {e["name"] for e in t["traceEvents"] if e["ph"] == "X"}
    assert names == {"long", "late"}  # 'long' overlaps the window, 'late' is inside


def test_spmd_main(tmp_path):
    for rank in (0, 1):
        _write(
            tmp_path,
            f"training.global_rank_{rank}.t.jsonl",
            _span(None, "fwd_bwd", 0, 1000, rank=rank, source="training"),
        )
    t = generate_gantt_trace(str(tmp_path), str(tmp_path / "g.json"))
    assert _rows(t) == {"training": 1}
    assert set(next(iter(_labels(t).values())).values()) == {"main"}


def test_instant_on_dropped_task_falls_back(tmp_path):
    _controller(tmp_path)
    t = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "g.json"),
        pinned_tasks=RL_PINNED,
        collapse=RL_COLLAPSE,
    )
    loss = [
        e for e in t["traceEvents"] if e["ph"] == "i" and e["name"].startswith("loss")
    ]
    assert loss and all(e["tid"] == 0 for e in loss)


def test_determinism(tmp_path):
    _controller(tmp_path)
    a = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "a.json"),
        pinned_tasks=RL_PINNED,
        collapse=RL_COLLAPSE,
    )
    b = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "b.json"),
        pinned_tasks=RL_PINNED,
        collapse=RL_COLLAPSE,
    )
    assert json.dumps(a) == json.dumps(b)


def test_engine_pin_separates_from_task_rows(tmp_path):
    recs = _span(
        "vllm_engine", "vllm_engine_step_burst", 0, 10000, source="rl_generator"
    )
    for k in range(5):
        recs += _span(
            "vllm_engine",
            "vllm_engine_step",
            100 + k * 100,
            150 + k * 100,
            source="rl_generator",
        )
    for n in range(10):
        recs += _span(f"Task-{100+n}", "generate", 200, 9000, source="rl_generator")
    _write(tmp_path, "rl_generator.global_rank_0.t.jsonl", recs)
    t = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "g.json"),
        pinned_tasks=RL_PINNED,
        collapse=RL_COLLAPSE,
    )
    lbl = next(iter(_labels(t).values()))
    assert lbl[0] == "vllm_engine"  # engine pinned as its own first row
    assert sum(1 for v in lbl.values() if v.startswith("task")) == 8


def test_process_names_simplified_default_full_in_raw(tmp_path):
    # one controller + two generator rank-0 files (different runs) -> rl_controller, rl_generator_0/1
    _write(
        tmp_path,
        "rl_controller.global_rank_0.AAA.jsonl",
        _span("trainer", "train_step", 0, 1000),
    )
    _write(
        tmp_path,
        "rl_generator.global_rank_0.AAA.jsonl",
        _span("vllm_engine", "vllm_engine_step", 0, 1000, source="rl_generator"),
    )
    _write(
        tmp_path,
        "rl_generator.global_rank_0.BBB.jsonl",
        _span("vllm_engine", "vllm_engine_step", 0, 1000, source="rl_generator"),
    )
    t = generate_gantt_trace(
        str(tmp_path), str(tmp_path / "g.json"), pinned_tasks=("trainer", "vllm_engine")
    )
    assert set(_rows(t)) == {"rl_controller", "rl_generator_0", "rl_generator_1"}
    raw = generate_gantt_trace(str(tmp_path), str(tmp_path / "raw.json"), raw=True)
    assert set(_rows(raw)) == {
        "rl_controller.global_rank_0.AAA",
        "rl_generator.global_rank_0.AAA",
        "rl_generator.global_rank_0.BBB",
    }


def test_thread_sort_index_present_and_matches_tid(tmp_path):
    _controller(tmp_path)
    t = generate_gantt_trace(
        str(tmp_path),
        str(tmp_path / "g.json"),
        pinned_tasks=RL_PINNED,
        collapse=RL_COLLAPSE,
    )
    sort_idx = [
        e
        for e in t["traceEvents"]
        if e["ph"] == "M" and e["name"] == "thread_sort_index"
    ]
    assert sort_idx and all(e["args"]["sort_index"] == e["tid"] for e in sort_idx)
    rendered = {(e["pid"], e["tid"]) for e in t["traceEvents"] if e["ph"] == "X"}
    assert rendered <= {(e["pid"], e["tid"]) for e in sort_idx}
