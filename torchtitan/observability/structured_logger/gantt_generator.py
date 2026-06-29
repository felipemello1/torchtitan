# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""Render per-rank structured-log JSONL as a Perfetto flame chart.

One JSONL file per rank -> one Perfetto "process"; each asyncio task -> one or more "rows".
The renderer is GENERIC (no RL knowledge): YOU say which tasks to pin and which task
families to collapse. RL ships a ready-made policy in `experiments/rl/gantt.py`.

Quickstart:

    from torchtitan.observability.structured_logger.gantt_generator import (
        generate_gantt_trace, CollapsedTasks)

    generate_gantt_trace("outputs/rl/structured_logs/", "gantt.json",
        pinned_tasks=("trainer", "batcher", "data_input"),               # each gets its own row
        collapse=(CollapsedTasks(r"^Task-\d+$", max_rows=8, label="task"),))  # 256 tasks -> 8 rows
    # RL users skip the policy: `from torchtitan.experiments.rl.gantt import generate_rl_gantt`

What PRODUCES the JSONL (the emit side, for reference) -- `task_name` is the asyncio task name:

    with sl.log_trace_span("rollout_group"):     # running inside asyncio task "rollout_worker_3"
        ...
    # appends two JSONL lines (one physical line each):
    #   {"log_type_name": "rollout_group_start", "task_name": "rollout_worker_3", "time_us": 1782..., "step": 5}
    #   {"log_type_name": "rollout_group_end",   "task_name": "rollout_worker_3", "value": 12.4,  "step": 5}
    #                                                                             value = elapsed ms

What you GET (Perfetto; one process per rank file, nested spans stack as a flame chart):

    rl_controller ┐ trainer         ▓ wait ▓▓ fwd_bwd ▓▓ optim ▓        <- pinned
                  │ batcher          ▓ take ▓▓ pack ▓                    <- pinned
                  │ data_input       ▓ get_sample ▓                      <- pinned
                  │ rollout worker   ▓ rollout_group ▓▓ score ▓          <- collapsed: 32 workers -> 1 row
                  └ task 0..7        ▓ generate ▓ ▓ generate ▓ ...       <- collapsed: 256 tasks -> 8 rows

How it works (per source):

    JSONL records
      -> select records:  file_name_regex, ranks, sources, step window
      -> pair *_start / *_end   (LIFO stack per (source, task_name); nested spans pair correctly)
      -> last_seconds window:   keep spans that overlap it
      -> hide_spans:            drop matching span names from the view (opt-in; default keeps all)
      -> assign rows:           pinned_tasks  -> one row each, in your order
                                collapse      -> pack a task family into <= max_rows rows (extras dropped from the VIEW)
                                task_name=None -> one "main" row  (SPMD / non-async)
                                anything else  -> one row per task
      -> emit Chrome trace:     process_name + thread_name(label) + X spans + i instants

`raw=True` renders every task on its own row, all ranks, no labels (the exhaustive view).
The JSONL on disk is the complete, all-ranks event store and is never changed here.

TODO: file rotation / mtime-skipping for very long runs (windowing still reads the selected files first).
"""

import argparse
import heapq
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from typing import Any

from torchtitan.observability.structured_logger.structured_logging import LogType
from torchtitan.tools.logging import logger


@dataclass(frozen=True)
class CollapsedTasks:
    r"""Collapse a family of same-kind asyncio tasks (matched by name) into a few rows.

    A run spawns hundreds of short tasks (e.g. one ``Task-N`` per generate request); drawing
    each as its own Perfetto row is unreadable. This keeps ``max_rows`` representative rows and
    drops the rest FROM THE VIEW ONLY (the JSONL still has every task). Tasks that never overlap
    in time share a row; only genuinely-concurrent tasks need separate rows.

    Args:
        match: Regex matched (``re.search``) against the asyncio ``task_name``, e.g. ``r"^Task-\d+$"``.
        max_rows: Keep at most this many rows for the family (``None`` keeps all).
        label: Row label. With ``max_rows=1`` the row is just ``label``; otherwise rows are
            numbered ``"label 0"``, ``"label 1"``, ... .
        source_match: If set, this rule only applies to sources whose name matches (``re.search``), so
            the same ``Task-N`` family can be labeled per source. ``None`` = every source.

    Example:

        # a run has Task-10 .. Task-265 (256 concurrent generate tasks)
        CollapsedTasks(match=r"^Task-\d+$", max_rows=8, label="task")
        # -> rows "task 0" .. "task 7"   (8 of 256 kept; the rest dropped from the view)

        # 32 persistent rollout_worker_0 .. rollout_worker_31 (all alive the whole run)
        CollapsedTasks(match=r"^rollout_worker_\d+$", max_rows=1, label="rollout worker")
        # -> row "rollout worker"   (worker 0, which starts first)

        # label the same Task-N family differently on the generator vs elsewhere
        CollapsedTasks(match=r"^Task-\d+$", max_rows=8, label="generation request", source_match="rl_generator")
    """

    match: str
    max_rows: int | None
    label: str
    source_match: str | None = None


def generate_gantt_trace(
    log_dir: str,
    output_path: str,
    *,
    raw: bool = False,
    ranks: tuple[int, ...] | None = (0,),
    sources: tuple[str, ...] | None = None,
    file_name_regex: str | None = None,
    last_seconds: float | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    pinned_tasks: tuple[str, ...] = (),
    collapse: tuple[CollapsedTasks, ...] = (),
    hide_spans: tuple[str, ...] = (),
    source_order: tuple[str, ...] = (),
) -> dict:
    r"""Render per-rank JSONL into a Perfetto Chrome-Trace JSON (see the module docstring).

    Args:
        log_dir: Directory of per-rank ``*.jsonl`` trace files.
        output_path: Where to write the Chrome-Trace JSON. Open it in https://ui.perfetto.dev.
        raw: Render every task on its own row, all ranks, no labels (exhaustive). Selectors
            below (``ranks`` excepted) still apply. Example: ``raw=True``.
        ranks: Keep records with these ``global_rank``s; ``None`` = all. Ignored when ``raw=True``
            (raw shows all ranks). Example: ``ranks=(0,)`` (just rank 0).
        sources: Keep only these source names; ``None`` = all. Example: ``sources=("rl_controller",)``.
        file_name_regex: Only load files whose basename matches -- pick one run from a dir holding
            several. Example: ``file_name_regex="20260625-1941"``.
        last_seconds: Keep only the last N seconds of the run (spans overlapping the window).
            Example: ``last_seconds=120`` (last 2 minutes of a long run).
        start_step, end_step: Inclusive training-step window. Example: ``start_step=90, end_step=100``.
        pinned_tasks: Exact task names that each get their own row, in this order.
            Example: ``pinned_tasks=("trainer", "batcher", "data_input")``.
        collapse: Rules to fold high-cardinality task families into a few rows (see
            :class:`CollapsedTasks`). Example: ``collapse=(CollapsedTasks(r"^Task-\d+$", 8, "task"),)``.
        hide_spans: Span-name regexes to drop from the view (opt-in; default hides nothing).
            Example: ``hide_spans=(r"\.Config\.build$",)`` to mute env/model build spam.
        source_order: Process order by source-name prefix (rows from unlisted prefixes sort last, then
            by name). Default ``()`` = lexicographic. Example:
            ``source_order=("rl_controller", "rl_trainer", "rl_generator")``.

    Returns:
        The Chrome-Trace dict (``{"traceEvents": [...]}``), also written to ``output_path``.
    """
    records = load_all_records(log_dir, file_name_regex=file_name_regex)
    if not records:
        logger.info(f"No records found in {log_dir}")
        return {"traceEvents": []}

    # Per-record selection. raw shows all ranks; the other selectors still apply.
    records = _filter_records(
        records,
        ranks=None if raw else ranks,
        sources=sources,
        start_step=start_step,
        end_step=end_step,
    )
    if not records:
        logger.info(f"No records left in {log_dir} after selection")
        return {"traceEvents": []}

    sources_seen = _order_sources({r["_source_file"] for r in records}, source_order)
    source_to_pid = {s: i for i, s in enumerate(sources_seen)}

    paired, instants = _collect_paired_and_instants(records, source_to_pid)

    # last_seconds window after pairing so overlapping spans survive (no clipping).
    paired, instants = _apply_last_seconds(
        paired, instants, records, last_seconds=last_seconds
    )

    if raw:
        tid_by_source_and_task = _assign_tids_raw(paired)
        row_labels: dict[tuple[str, int], str] = {}
    else:
        paired = _drop_hidden_spans(paired, hide_spans)
        paired, tid_by_source_and_task, row_labels = _assign_tids_policy(
            paired, pinned_tasks=pinned_tasks, collapse=collapse
        )

    # raw keeps the full per-rank basenames; the default view shortens them to role (+ index).
    process_labels = (
        {s: s for s in sources_seen} if raw else _simplify_source_names(sources_seen)
    )

    events = _emit_chrome_events(
        paired=paired,
        instants=instants,
        source_to_pid=source_to_pid,
        process_labels=process_labels,
        tid_by_source_and_task=tid_by_source_and_task,
        row_labels=row_labels,
    )

    trace = {"traceEvents": events}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2)

    logger.info(f"Chrome Trace: {output_path}  (raw={raw})")
    logger.info(f"  {len(events)} events from {len(sources_seen)} sources")
    logger.info("  View in: chrome://tracing or https://ui.perfetto.dev")
    return trace


def load_all_records(log_dir: str, *, file_name_regex: str | None = None) -> list[dict]:
    r"""Load JSONL records from a ``structured_logs/`` directory.

    Args:
        log_dir: Directory of ``*.jsonl`` files (one per rank).
        file_name_regex: If set, only load files whose basename matches (``re.search``), so a
            directory holding several runs can be narrowed to one.

    Returns:
        All records across the selected files, each annotated with ``"_source_file"`` (the
        filename minus ``.jsonl``) for per-process grouping.
    """
    pattern = re.compile(file_name_regex) if file_name_regex else None
    records = []
    for path in sorted(glob(os.path.join(log_dir, "*.jsonl"))):
        basename = os.path.basename(path)
        if pattern is not None and not pattern.search(basename):
            continue
        source_name = basename.rsplit(".", 1)[0] if "." in basename else basename
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_source_file"] = source_name
                    records.append(r)
    return records


def _filter_records(
    records: list[dict],
    *,
    ranks: tuple[int, ...] | None,
    sources: tuple[str, ...] | None,
    start_step: int | None,
    end_step: int | None,
) -> list[dict]:
    """Per-record selection by rank, source, and step window (``None`` = no filter)."""
    rank_set = set(ranks) if ranks is not None else None
    source_set = set(sources) if sources is not None else None
    out = []
    for r in records:
        if (
            rank_set is not None
            and r.get("global_rank", r.get("rank", 0)) not in rank_set
        ):
            continue
        if source_set is not None and r.get("source") not in source_set:
            continue
        step = r.get("step")
        if start_step is not None and (step is None or step < start_step):
            continue
        if end_step is not None and (step is None or step > end_step):
            continue
        out.append(r)
    return out


def _apply_last_seconds(
    paired: list[dict],
    instants: list[dict],
    records: list[dict],
    *,
    last_seconds: float | None,
) -> tuple[list[dict], list[dict]]:
    """Keep only the last ``last_seconds`` of the run: spans overlapping the window, instants inside it.

    The window is ``[max_time_us - last_seconds*1e6, max_time_us]`` over the selected records.
    Spans are kept (not clipped) if they overlap, so context at the window edge survives.
    """
    if last_seconds is None:
        return paired, instants
    hi = max((r.get("time_us") or 0) for r in records)
    lo = hi - last_seconds * 1_000_000
    paired = [p for p in paired if p["start_ts"] <= hi and p["end_ts"] >= lo]
    instants = [i for i in instants if lo <= i["time_us"] <= hi]
    return paired, instants


def _drop_hidden_spans(paired: list[dict], hide_spans: tuple[str, ...]) -> list[dict]:
    """Drop spans whose ``display_name`` matches any hide regex (default: hide nothing)."""
    if not hide_spans:
        return paired
    patterns = [re.compile(rx) for rx in hide_spans]
    return [
        p for p in paired if not any(rx.search(p["display_name"]) for rx in patterns)
    ]


def _pack_intervals(ranges: dict[str | None, tuple[int, int]]) -> dict[str | None, int]:
    """Interval scheduling: non-overlapping tasks reuse a row, preferring low row ids.

    Row count equals peak concurrency. Among free rows we reuse the LOWEST id (not the
    earliest-ending one), so a downstream cap (keep first K rows) retains as much
    non-overlapping work as possible instead of stranding it on a high, dropped row.

    Args:
        ranges: ``task_name -> (min_start_ts, max_end_ts)``.

    Returns:
        ``task_name -> row`` over ``0..K-1`` where K is the peak concurrency.

    Example:

        _pack_intervals({"a": (0, 100), "b": (0, 90), "c": (101, 110)})
        # -> {"a": 0, "b": 1, "c": 0}   # c reuses row 0 (free after 100), not row 1
    """
    busy: list[tuple[int, int]] = []  # heap of (end_ts, row) currently occupied
    available: list[int] = []  # min-heap of freed row ids (reuse the lowest)
    next_row = 0
    row_of: dict[str | None, int] = {}
    for task_name, (start, end) in sorted(ranges.items(), key=lambda kv: kv[1][0]):
        # Free every row whose task has ended at/before this task starts.
        while busy and busy[0][0] <= start:
            _, freed = heapq.heappop(busy)
            heapq.heappush(available, freed)
        if available:
            row = heapq.heappop(available)
        else:
            row = next_row
            next_row += 1
        heapq.heappush(busy, (end, row))
        row_of[task_name] = row
    return row_of


def _task_ranges(spans: list[dict]) -> dict[str | None, tuple[int, int]]:
    """Compute each task's ``[min(start_ts), max(end_ts)]`` over its spans."""
    ranges: dict[str | None, tuple[int, int]] = {}
    for s in spans:
        tn = s.get("task_name")
        start, end = s["start_ts"], s["end_ts"]
        cur = ranges.get(tn)
        ranges[tn] = (
            (start, end) if cur is None else (min(cur[0], start), max(cur[1], end))
        )
    return ranges


def _assign_tids_raw(paired: list[dict]) -> dict[tuple[str, str | None], int]:
    """Exhaustive layout: pack every ``(source, task)`` onto its own row (no labels)."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for p in paired:
        by_source[p["source"]].append(p)
    tid_by_source_and_task: dict[tuple[str, str | None], int] = {}
    for source, spans in by_source.items():
        row_of = _pack_intervals(_task_ranges(spans))
        for task_name, row in row_of.items():
            tid_by_source_and_task[(source, task_name)] = row
        for s in spans:
            s["tid"] = row_of[s.get("task_name")]
    return tid_by_source_and_task


def _assign_tids_policy(
    paired: list[dict],
    *,
    pinned_tasks: tuple[str, ...],
    collapse: tuple[CollapsedTasks, ...],
) -> tuple[list[dict], dict[tuple[str, str | None], int], dict[tuple[str, int], str]]:
    """Caller-policy row assignment, per source: pinned -> collapse groups -> main -> other.

    Returns ``(kept_paired, tid_by_source_and_task, row_labels)``; ``row_labels`` maps
    ``(source, tid) -> label`` for ``thread_name``. Overflow tasks are dropped from the view.
    A task is claimed by the FIRST collapse group whose ``match`` hits it.
    """
    compiled = [
        (g, re.compile(g.match), re.compile(g.source_match) if g.source_match else None)
        for g in collapse
    ]

    by_source: dict[str, list[dict]] = defaultdict(list)
    for p in paired:
        by_source[p["source"]].append(p)

    kept_paired: list[dict] = []
    tid_by_source_and_task: dict[tuple[str, str | None], int] = {}
    row_labels: dict[tuple[str, int], str] = {}

    for source, spans in by_source.items():
        spans_by_task: dict[str | None, list[dict]] = defaultdict(list)
        for s in spans:
            spans_by_task[s.get("task_name")].append(s)

        next_tid = 0
        claimed: set[str | None] = set()

        def _row(label: str) -> int:
            nonlocal next_tid
            tid = next_tid
            next_tid += 1
            row_labels[(source, tid)] = label
            return tid

        def _place(task_name: str | None, tid: int) -> None:
            tid_by_source_and_task[(source, task_name)] = tid
            kept_paired.extend({**s, "tid": tid} for s in spans_by_task[task_name])

        # 1) pinned, in caller order
        for task_name in pinned_tasks:
            if task_name in spans_by_task:
                _place(task_name, _row(task_name))
                claimed.add(task_name)

        # 2) collapse groups, in caller order (first match wins). A rule with source_match only
        #    applies to matching sources, so the same Task-N family can be labeled per source.
        for group, rx, source_rx in compiled:
            if source_rx is not None and not source_rx.search(source):
                continue
            members = [
                t
                for t in spans_by_task
                if t is not None and t not in claimed and rx.search(t)
            ]
            if not members:
                continue
            claimed.update(members)
            row_of = _pack_intervals(
                {t: _task_ranges(spans_by_task[t])[t] for t in members}
            )
            kept_rows = sorted(
                {
                    r
                    for r in row_of.values()
                    if group.max_rows is None or r < group.max_rows
                }
            )
            row_to_tid = {
                r: _row(group.label if group.max_rows == 1 else f"{group.label} {r}")
                for r in kept_rows
            }
            for task_name, row in row_of.items():
                tid = row_to_tid.get(row)
                if tid is not None:  # else overflow -> dropped from the view
                    _place(task_name, tid)

        # 3) main (SPMD / non-async)
        if None in spans_by_task:
            _place(None, _row("main"))

        # 4) other named tasks, one row each
        for task_name in sorted(
            (t for t in spans_by_task if t is not None and t not in claimed), key=str
        ):
            _place(task_name, _row(task_name))

    return kept_paired, tid_by_source_and_task, row_labels


def _collect_paired_and_instants(
    records: list[dict], source_to_pid: dict[str, int]
) -> tuple[list[dict], list[dict]]:
    """Pair ``_start`` / ``_end`` via a per-(source, task_name) LIFO stack.

    Nested spans in one task pair correctly; SPMD code has ``task_name=None`` throughout and
    pairs on a single stack per source. Instants cover ``log_trace_instant``,
    ``log_trace_scalar`` metric values, and ``_error`` records.
    """
    paired: list[dict[str, Any]] = []
    instants: list[dict[str, Any]] = []
    pending: dict[tuple[str, str | None], list[dict]] = defaultdict(list)

    for r in records:
        event_type = r.get("log_type_name", "")
        log_type = r.get("log_type", "")
        time_us = r.get("time_us") or (r.get("time_ms") or 0) * 1000
        pid = source_to_pid[r["_source_file"]]
        rank = r.get("rank", 0)
        step = r.get("step")
        task_name = r.get("task_name")
        source = r["_source_file"]
        caller = r.get("caller")
        key = (source, task_name)

        if log_type == str(LogType.INSTANT):
            if event_type == "metric_value":
                event_name = r.get("event_name", "metric")
                value = r.get("value") or 0
                display_name = f"{event_name}={value:.4f}"
            else:
                display_name = event_type
            instants.append(
                {
                    "name": display_name,
                    "time_us": time_us,
                    "pid": pid,
                    "source": source,
                    "task_name": task_name,
                    "step": step,
                    "caller": caller,
                }
            )
        elif event_type.endswith("_start"):
            type_name = event_type.removesuffix("_start")
            display_name = r.get("event_name") or type_name
            pending[key].append(
                {
                    "ts": time_us,
                    "step": step,
                    "display_name": display_name,
                    "pid": pid,
                    "rank": rank,
                    "task_name": task_name,
                    "source": source,
                    "caller": caller,
                }
            )
        elif event_type.endswith("_end"):
            type_name = event_type.removesuffix("_end")
            duration_ms = r.get("value", 0)
            duration_us = (duration_ms or 0) * 1000
            stack = pending.get(key)
            start = stack.pop() if stack else None
            if start is not None:
                start_ts = start["ts"]
                end_ts = start_ts + duration_us
            else:
                end_ts = time_us
                start_ts = end_ts - duration_us
            paired.append(
                {
                    "pid": pid,
                    "rank": (start or {}).get("rank", rank),
                    "task_name": (start or {}).get("task_name", task_name),
                    "source": (start or {}).get("source", source),
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "display_name": (start or {}).get("display_name", type_name),
                    "step": step,
                    "duration_ms": duration_ms,
                    "caller": (start or {}).get("caller", caller),
                }
            )
        elif event_type.endswith("_error"):
            type_name = event_type.removesuffix("_error")
            instants.append(
                {
                    "name": f"ERROR: {type_name}",
                    "time_us": time_us,
                    "pid": pid,
                    "source": source,
                    "task_name": task_name,
                    "step": step,
                    "caller": caller,
                }
            )
        else:
            instants.append(
                {
                    "name": event_type,
                    "time_us": time_us,
                    "pid": pid,
                    "source": source,
                    "task_name": task_name,
                    "step": step,
                    "caller": caller,
                }
            )

    return paired, instants


def _order_sources(sources: set[str], source_order: tuple[str, ...]) -> list[str]:
    """Order sources (= process rows) by ``source_order`` prefix, then by name. Sources whose name
    matches no prefix sort last. ``source_order=()`` = plain lexicographic.

    Example:

        _order_sources({"rl_generator.g1", "rl_controller.c", "rl_trainer.t"},
                       ("rl_controller", "rl_trainer", "rl_generator"))
        # -> ["rl_controller.c", "rl_trainer.t", "rl_generator.g1"]
    """

    def key(source: str) -> tuple[int, str]:
        for rank, prefix in enumerate(source_order):
            if source.startswith(prefix):
                return (rank, source)
        return (len(source_order), source)

    return sorted(sources, key=key)


def _simplify_source_names(sources: list[str]) -> dict[str, str]:
    """Map each per-rank source basename to a short process label (role, plus an index when a role
    has more than one source). ``raw`` callers skip this and keep the full basename.

    Example:

        _simplify_source_names([
            "rl_controller.global_rank_0.20260629-082321-4MCHI5",
            "rl_generator.global_rank_0.20260629-082314-AAA",
            "rl_generator.global_rank_0.20260629-082314-BBB",
        ])
        # -> {"...controller...": "rl_controller",
        #     "...generator...AAA": "rl_generator_0", "...generator...BBB": "rl_generator_1"}
    """
    role_of = {s: s.split(".global_rank")[0] for s in sources}
    role_counts: dict[str, int] = defaultdict(int)
    for role in role_of.values():
        role_counts[role] += 1
    seen: dict[str, int] = defaultdict(int)
    labels: dict[str, str] = {}
    for source in sorted(sources):
        role = role_of[source]
        if role_counts[role] > 1:
            labels[source] = f"{role}_{seen[role]}"
            seen[role] += 1
        else:
            labels[source] = role
    return labels


def _resolve_instant_tid(
    *,
    source: str,
    task_name: str | None,
    tid_by_source_and_task: dict[tuple[str, str | None], int],
) -> int:
    """Instant goes on its task's row if kept, else the source's first row (row 0)."""
    return tid_by_source_and_task.get((source, task_name), 0)


def _chrome_tid_for_row(pid: int, row_index: int) -> int:
    """Map a semantic row index to the Chrome-trace ``tid`` Perfetto should use.

    Perfetto's Chrome importer floats the thread whose ``tid == pid`` to the top as the process
    "main thread", OVER our ``thread_sort_index``. So give row 0 (the pinned/most-important row)
    ``tid == pid`` to claim that slot, and shift the rest off ``pid`` so none collide.

    Example:

        [_chrome_tid_for_row(1, r) for r in (0, 1, 2)]  # pid 1 -> [1, 0, 2]  (row 0 floats; others avoid 1)
        [_chrome_tid_for_row(4, r) for r in (0, 1, 2)]  # pid 4 -> [4, 0, 1]
    """
    if row_index == 0:
        return pid
    tid = row_index - 1
    return tid + 1 if tid >= pid else tid


def _emit_chrome_events(
    *,
    paired: list[dict],
    instants: list[dict],
    source_to_pid: dict[str, int],
    process_labels: dict[str, str],
    tid_by_source_and_task: dict[tuple[str, str | None], int],
    row_labels: dict[tuple[str, int], str],
) -> list[dict]:
    """Build the Chrome Trace events: ``process_name`` per source, ``thread_name`` per labeled
    row, ``thread_sort_index`` so Perfetto honors our row order, ``X`` per span, ``i`` per instant."""
    events: list[dict[str, Any]] = []

    for source, pid in source_to_pid.items():
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": process_labels[source]},
            }
        )

    # thread_name + thread_sort_index per row. The emitted tid claims the main-thread slot for row 0;
    # sort_index keeps our semantic row order (row_index) for the rest.
    for (source, row_index), label in row_labels.items():
        pid = source_to_pid[source]
        chrome_tid = _chrome_tid_for_row(pid, row_index)
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": chrome_tid,
                "args": {"name": label},
            }
        )
        events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": chrome_tid,
                "args": {"sort_index": row_index},
            }
        )

    for p in paired:
        duration_ms = p["duration_ms"] or 0
        args: dict = {
            **({"step": p["step"]} if p["step"] is not None else {}),
            "duration_ms": f"{duration_ms:.2f}" if duration_ms else "0.00",
        }
        if p.get("caller"):
            args["caller"] = p["caller"]
        events.append(
            {
                "name": p["display_name"],
                "ph": "X",
                "ts": p["start_ts"],
                "dur": p["end_ts"] - p["start_ts"],
                "pid": p["pid"],
                "tid": _chrome_tid_for_row(p["pid"], p.get("tid", 0)),
                "args": args,
            }
        )

    for i in instants:
        row_index = _resolve_instant_tid(
            source=i["source"],
            task_name=i["task_name"],
            tid_by_source_and_task=tid_by_source_and_task,
        )
        args = {**({"step": i["step"]} if i["step"] is not None else {})}
        if i.get("caller"):
            args["caller"] = i["caller"]
        events.append(
            {
                "name": i["name"],
                "ph": "i",
                "ts": i["time_us"],
                "pid": i["pid"],
                "tid": _chrome_tid_for_row(i["pid"], row_index),
                "s": "t",
                "args": args,
            }
        )

    return events


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render per-rank structured-log JSONL as a Perfetto trace. "
        "For collapse rules use the Python API or experiments/rl/gantt.generate_rl_gantt."
    )
    p.add_argument("--log-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--raw",
        action="store_true",
        help="Exhaustive layout (all ranks, every task its own row).",
    )
    p.add_argument(
        "--ranks",
        type=str,
        default="0",
        help="Comma-separated global_ranks, or 'all' (default: 0).",
    )
    p.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated source names (default: all).",
    )
    p.add_argument(
        "--file-name-regex",
        type=str,
        default=None,
        help="Only load files whose basename matches.",
    )
    p.add_argument(
        "--last-seconds",
        type=float,
        default=None,
        help="Keep only the last N seconds of the run.",
    )
    p.add_argument("--start-step", type=int, default=None)
    p.add_argument("--end-step", type=int, default=None)
    p.add_argument(
        "--pinned",
        type=str,
        default=None,
        help="Comma-separated task names to pin as their own rows.",
    )
    p.add_argument(
        "--hide-span",
        action="append",
        default=None,
        help="Span-name regex to hide (repeatable).",
    )
    args = p.parse_args()

    ranks = (
        None if args.ranks == "all" else tuple(int(x) for x in args.ranks.split(","))
    )
    sources = tuple(args.sources.split(",")) if args.sources else None
    pinned = tuple(args.pinned.split(",")) if args.pinned else ()
    hide = tuple(args.hide_span) if args.hide_span else ()

    generate_gantt_trace(
        args.log_dir,
        args.output,
        raw=args.raw,
        ranks=ranks,
        sources=sources,
        file_name_regex=args.file_name_regex,
        last_seconds=args.last_seconds,
        start_step=args.start_step,
        end_step=args.end_step,
        pinned_tasks=pinned,
        hide_spans=hide,
    )


if __name__ == "__main__":
    main()
