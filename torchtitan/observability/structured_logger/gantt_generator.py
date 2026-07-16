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

How it works (two streaming passes over the selected files; memory stays bounded to the
window, not the run):

    pass 1: resolve last_steps / start_step / end_step / start_time_us
              to a wall-clock window [start_us, end_us]
    pass 2: stream records -> pair *_start / *_end (LIFO stack per execution key)
              -> keep spans overlapping the window, clipped to its edges
              -> assign rows:  pinned_tasks   -> one row each, in your order
                               collapse       -> pack a task family into <= max_rows rows
                               task_name=None -> one row per native thread (SPMD / non-async)
                               anything else  -> interval-packed shared rows
              -> emit Chrome trace: process/thread metadata + X spans + i instants
                 (compact JSON, atomic replace)

`raw=True` renders every task on its own row, all ranks, no labels (the exhaustive view).
The JSONL on disk is the complete, all-ranks event store and is never changed here.
"""

import argparse
import heapq
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterator
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
        CollapsedTasks(match=r"^Task-\d+$", max_rows=8, label="generator endpoint", source_match="rl_generator")
    """

    match: str
    max_rows: int | None
    label: str
    source_match: str | None = None


# Row label for instants whose task was dropped by a collapse cap. They must stay visible
# (errors!) but must not masquerade as events of a semantic row like "trainer_loop".
OVERFLOW_ROW_LABEL = "overflow"


def generate_gantt_trace(
    log_dir: str,
    output_path: str,
    *,
    raw: bool = False,
    ranks: tuple[int, ...] | None = (0,),
    sources: tuple[str, ...] | None = None,
    file_name_regex: str | None = None,
    start_time_us: int | None = None,
    last_steps: int | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    pinned_tasks: tuple[str, ...] = (),
    collapse: tuple[CollapsedTasks, ...] = (),
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
        start_time_us: Drop records before this epoch-microsecond timestamp; files whose mtime
            predates it are skipped without reading. Pass the run's start time to isolate the
            current run in a reused dump folder. Example: ``start_time_us=run_started_us``.
        last_steps: Keep only the final N observed training steps, resolved from the records
            themselves -- no need to know the run's final step number. Example: ``last_steps=10``.
        start_step, end_step: Inclusive training-step window. Example: ``start_step=90, end_step=100``.
        pinned_tasks: Exact task names that each get their own row, in this order.
            Example: ``pinned_tasks=("trainer", "batcher", "data_input")``.
        collapse: Rules to fold high-cardinality task families into a few rows (see
            :class:`CollapsedTasks`). Example: ``collapse=(CollapsedTasks(r"^Task-\d+$", 8, "task"),)``.
        source_order: Process order by source-name prefix (rows from unlisted prefixes sort last, then
            by name). Default ``()`` = lexicographic. Example:
            ``source_order=("rl_controller", "rl_trainer", "rl_generator")``.

    Returns:
        The Chrome-Trace dict (``{"traceEvents": [...]}``), also written to ``output_path``.
    """
    paths = _selected_jsonl_paths(
        log_dir, file_name_regex=file_name_regex, min_time_us=start_time_us
    )
    effective_ranks = None if raw else ranks

    # Pass 1: resolve the step selectors to a wall-clock window (None bound = unbounded).
    window_start_us, window_end_us = _resolve_time_window(
        paths,
        ranks=effective_ranks,
        sources=sources,
        start_time_us=start_time_us,
        last_steps=last_steps,
        start_step=start_step,
        end_step=end_step,
    )
    if window_start_us is not None:
        # A file whose last write predates the window start closed all its spans
        # before the window; skip it without reading.
        paths = _selected_jsonl_paths(
            log_dir, file_name_regex=file_name_regex, min_time_us=window_start_us
        )

    # Pass 2: stream, pair, and keep only window-overlapping spans/instants.
    paired, instants, sources_seen = _collect_paired_and_instants(
        _iter_selected_records(paths, ranks=effective_ranks, sources=sources),
        window_start_us=window_start_us,
        window_end_us=window_end_us,
    )
    if not paired and not instants:
        logger.info(f"No records found in {log_dir} (after selection)")
        return {"traceEvents": []}

    ordered_sources = _order_sources(sources_seen, source_order)
    source_to_pid = {s: i for i, s in enumerate(ordered_sources)}

    if raw:
        tid_by_key = _assign_tids_raw(paired)
        row_labels: dict[tuple[str, int], str] = {}
    else:
        paired, tid_by_key, row_labels = _assign_tids_policy(
            paired, pinned_tasks=pinned_tasks, collapse=collapse
        )

    # raw keeps the full per-rank basenames; the default view shortens them to role (+ index).
    process_labels = (
        {s: s for s in ordered_sources}
        if raw
        else _simplify_source_names(ordered_sources)
    )

    events = _emit_chrome_events(
        paired=paired,
        instants=instants,
        source_to_pid=source_to_pid,
        process_labels=process_labels,
        tid_by_key=tid_by_key,
        row_labels=row_labels,
    )

    trace = {"traceEvents": events}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Atomic + compact: a reader never sees a half-written file, and Perfetto
    # ignores whitespace (indent=2 was +50% file size).
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(trace, f, separators=(",", ":"))
    os.replace(tmp_path, output_path)

    logger.info(f"Chrome Trace: {output_path}  (raw={raw})")
    logger.info(f"  {len(events)} events from {len(ordered_sources)} sources")
    logger.info("  View in: chrome://tracing or https://ui.perfetto.dev")
    return trace


def _selected_jsonl_paths(
    log_dir: str, *, file_name_regex: str | None, min_time_us: int | None
) -> list[str]:
    """The ``*.jsonl`` files to read: basename-filtered, then mtime-skipped.

    A file whose mtime predates ``min_time_us`` received its last record before the
    threshold, so nothing in it can land at/after it. This keeps a reused dump folder
    O(current run) instead of O(every run ever dumped there). Mtime only ever SKIPS
    files; record selection is always by the records' own timestamps. A 2s slack
    absorbs filesystems that round mtime down to whole seconds.
    """
    pattern = re.compile(file_name_regex) if file_name_regex else None
    paths = []
    for path in sorted(glob(os.path.join(log_dir, "*.jsonl"))):
        if pattern is not None and not pattern.search(os.path.basename(path)):
            continue
        if (
            min_time_us is not None
            and (os.stat(path).st_mtime + 2.0) * 1_000_000 < min_time_us
        ):
            continue
        paths.append(path)
    return paths


def _iter_selected_records(
    paths: list[str],
    *,
    ranks: tuple[int, ...] | None,
    sources: tuple[str, ...] | None,
) -> Iterator[dict]:
    """Stream records from ``paths`` (rank/source-filtered), tagging each with
    ``_source_file`` (the basename minus ``.jsonl``) for per-process grouping."""
    rank_set = set(ranks) if ranks is not None else None
    source_set = set(sources) if sources is not None else None
    for path in paths:
        basename = os.path.basename(path)
        source_name = basename.rsplit(".", 1)[0] if "." in basename else basename
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if rank_set is not None and r.get("global_rank") not in rank_set:
                    continue
                if source_set is not None and r.get("source") not in source_set:
                    continue
                r["_source_file"] = source_name
                yield r


def _resolve_time_window(
    paths: list[str],
    *,
    ranks: tuple[int, ...] | None,
    sources: tuple[str, ...] | None,
    start_time_us: int | None,
    last_steps: int | None,
    start_step: int | None,
    end_step: int | None,
) -> tuple[int | None, int | None]:
    """Resolve the step selectors to a ``[start_us, end_us]`` wall-clock window (pass 1).

    Steps are stamps on records, but spans are drawn in time -- so the step selectors
    become the wall-clock extent of the selected steps, and pass 2 keeps whatever
    overlaps that extent. ``None`` bound = unbounded on that side. Memory is bounded:
    only per-step ``[min, max]`` timestamps of the still-relevant steps are retained.

    Example:

        # records: step 1 @ [t1a, t1b], ..., step 20 @ [t20a, t20b]
        _resolve_time_window(paths, last_steps=10, ...)            # -> (t11a, t20b)
        _resolve_time_window(paths, start_step=5, end_step=6, ...) # -> (t5a, t6b)
    """
    if last_steps is None and start_step is None and end_step is None:
        return start_time_us, None

    step_bounds: dict[int, list[int]] = {}
    max_step: int | None = None
    for r in _iter_selected_records(paths, ranks=ranks, sources=sources):
        step = r.get("step")
        if step is None:
            continue
        time_us = r["time_us"]
        if start_time_us is not None and time_us < start_time_us:
            continue
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue
        if last_steps is not None:
            max_step = step if max_step is None else max(max_step, step)
            cutoff = max_step - last_steps + 1
            if step < cutoff:
                continue
            for old_step in [s for s in step_bounds if s < cutoff]:
                del step_bounds[old_step]
        bounds = step_bounds.setdefault(step, [time_us, time_us])
        bounds[0] = min(bounds[0], time_us)
        bounds[1] = max(bounds[1], time_us)

    if not step_bounds:
        # No stepped record matched (e.g. logs from before the first set_step); fall
        # back to the explicit time bound only.
        return start_time_us, None

    window_start = min(b[0] for b in step_bounds.values())
    if start_time_us is not None:
        window_start = max(window_start, start_time_us)
    window_end = max(b[1] for b in step_bounds.values())
    return window_start, window_end


def _execution_key(record: dict) -> tuple[str, str, Any]:
    """The LIFO-stack identity a span belongs to.

    Asyncio spans pair per task; SPMD / non-async spans (``task_name=None``) pair per
    native thread so two threads' spans never cross-pair.

    Example:

        _execution_key({"_source_file": "a", "task_name": "Task-3", "tid": 7})
        # -> ("a", "task", "Task-3")
        _execution_key({"_source_file": "a", "tid": 41})
        # -> ("a", "thread", 41)
    """
    task_name = record.get("task_name")
    if task_name is not None:
        return (record["_source_file"], "task", task_name)
    return (record["_source_file"], "thread", record.get("tid"))


def _collect_paired_and_instants(
    records: Iterator[dict],
    *,
    window_start_us: int | None,
    window_end_us: int | None,
) -> tuple[list[dict], list[dict], set[str]]:
    """Pair ``_start`` / ``_end`` via a per-execution-key LIFO stack (pass 2, streaming).

    Spans overlapping the window are kept and CLIPPED to its edges, so one long outer
    span (a checkpoint save, a run-long wait) can't stretch the rendered view to the
    whole run; ``duration_ms`` keeps the true unclipped duration for the tooltip.
    Instants must lie inside the window. Instants cover ``log_trace_instant``,
    ``log_trace_scalar`` metric values, and ``_error`` records.
    """
    lo = window_start_us if window_start_us is not None else float("-inf")
    hi = window_end_us if window_end_us is not None else float("inf")

    paired: list[dict[str, Any]] = []
    instants: list[dict[str, Any]] = []
    pending: dict[tuple[str, str, Any], list[dict]] = defaultdict(list)
    sources_seen: set[str] = set()

    for r in records:
        event_type = r.get("log_type_name") or ""
        log_type = r.get("log_type", "")
        time_us = r["time_us"]
        step = r.get("step")
        task_name = r.get("task_name")
        source = r["_source_file"]
        caller = r.get("caller")
        key = _execution_key(r)
        sources_seen.add(source)

        if log_type == str(LogType.INSTANT):
            if event_type == "metric_value":
                event_name = r.get("event_name", "metric")
                value = r.get("value") or 0
                display_name = f"{event_name}={value:.4f}"
            else:
                display_name = event_type
            if lo <= time_us <= hi:
                instants.append(
                    {
                        "name": display_name,
                        "time_us": time_us,
                        "source": source,
                        "key": key,
                        "step": step,
                        "caller": caller,
                    }
                )
        elif event_type.endswith("_start"):
            type_name = event_type.removesuffix("_start")
            pending[key].append(
                {
                    "ts": time_us,
                    "step": step,
                    "display_name": r.get("event_name") or type_name,
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
            if start_ts > hi or end_ts < lo:
                continue
            paired.append(
                {
                    "source": (start or {}).get("source", source),
                    "key": key,
                    "start_ts": int(max(start_ts, lo)),
                    "end_ts": int(min(end_ts, hi)),
                    "display_name": (start or {}).get("display_name", type_name),
                    "step": step,
                    "duration_ms": duration_ms,
                    "caller": (start or {}).get("caller", caller),
                }
            )
        elif event_type.endswith("_error"):
            type_name = event_type.removesuffix("_error")
            if lo <= time_us <= hi:
                instants.append(
                    {
                        "name": f"ERROR: {type_name}",
                        "time_us": time_us,
                        "source": source,
                        "key": key,
                        "step": step,
                        "caller": caller,
                    }
                )
        else:
            if lo <= time_us <= hi:
                instants.append(
                    {
                        "name": event_type,
                        "time_us": time_us,
                        "source": source,
                        "key": key,
                        "step": step,
                        "caller": caller,
                    }
                )

    return paired, instants, sources_seen


def _pack_intervals(ranges: dict[Any, tuple[int, int]]) -> dict[Any, int]:
    """Interval scheduling: non-overlapping tasks reuse a row, preferring low row ids.

    Row count equals peak concurrency. Among free rows we reuse the LOWEST id (not the
    earliest-ending one), so a downstream cap (keep first K rows) retains as much
    non-overlapping work as possible instead of stranding it on a high, dropped row.

    Args:
        ranges: ``key -> (min_start_ts, max_end_ts)``.

    Returns:
        ``key -> row`` over ``0..K-1`` where K is the peak concurrency.

    Example:

        _pack_intervals({"a": (0, 100), "b": (0, 90), "c": (101, 110)})
        # -> {"a": 0, "b": 1, "c": 0}   # c reuses row 0 (free after 100), not row 1
    """
    busy: list[tuple[int, int]] = []  # heap of (end_ts, row) currently occupied
    available: list[int] = []  # min-heap of freed row ids (reuse the lowest)
    next_row = 0
    row_of: dict[Any, int] = {}
    for key, (start, end) in sorted(
        ranges.items(), key=lambda kv: (kv[1][0], str(kv[0]))
    ):
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
        row_of[key] = row
    return row_of


def _key_ranges(spans: list[dict]) -> dict[Any, tuple[int, int]]:
    """Each execution key's ``[min(start_ts), max(end_ts)]`` over its spans."""
    ranges: dict[Any, tuple[int, int]] = {}
    for s in spans:
        key = s["key"]
        start, end = s["start_ts"], s["end_ts"]
        cur = ranges.get(key)
        ranges[key] = (
            (start, end) if cur is None else (min(cur[0], start), max(cur[1], end))
        )
    return ranges


def _assign_tids_raw(paired: list[dict]) -> dict[tuple[str, str, Any], int]:
    """Exhaustive layout: pack every execution key onto its own row (no labels)."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for p in paired:
        by_source[p["source"]].append(p)
    tid_by_key: dict[tuple[str, str, Any], int] = {}
    for spans in by_source.values():
        row_of = _pack_intervals(_key_ranges(spans))
        for key, row in row_of.items():
            tid_by_key[key] = row
        for s in spans:
            s["tid"] = row_of[s["key"]]
    return tid_by_key


def _assign_tids_policy(
    paired: list[dict],
    *,
    pinned_tasks: tuple[str, ...],
    collapse: tuple[CollapsedTasks, ...],
) -> tuple[list[dict], dict[tuple[str, str, Any], int], dict[tuple[str, int], str]]:
    """Caller-policy row assignment, per source: pinned -> collapse groups -> threads -> other.

    Returns ``(kept_paired, tid_by_key, row_labels)``; ``row_labels`` maps
    ``(source, tid) -> label`` for ``thread_name``. Overflow tasks are dropped from the
    view and counted (their instants land on the overflow row; see ``_emit_chrome_events``).
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
    tid_by_key: dict[tuple[str, str, Any], int] = {}
    row_labels: dict[tuple[str, int], str] = {}
    dropped_span_count = 0
    dropped_task_count = 0

    for source, spans in by_source.items():
        spans_by_key: dict[tuple[str, str, Any], list[dict]] = defaultdict(list)
        for s in spans:
            spans_by_key[s["key"]].append(s)

        next_tid = 0
        claimed: set[tuple[str, str, Any]] = set()

        def _row(label: str) -> int:
            nonlocal next_tid
            tid = next_tid
            next_tid += 1
            row_labels[(source, tid)] = label
            return tid

        def _place(key: tuple[str, str, Any], tid: int) -> None:
            tid_by_key[key] = tid
            kept_paired.extend({**s, "tid": tid} for s in spans_by_key[key])

        # 1) pinned, in caller order
        for task_name in pinned_tasks:
            key = (source, "task", task_name)
            if key in spans_by_key:
                _place(key, _row(task_name))
                claimed.add(key)

        # 2) collapse groups, in caller order (first match wins). A rule with source_match only
        #    applies to matching sources, so the same Task-N family can be labeled per source.
        for group, rx, source_rx in compiled:
            if source_rx is not None and not source_rx.search(source):
                continue
            members = [
                k
                for k in spans_by_key
                if k[1] == "task" and k not in claimed and rx.search(k[2])
            ]
            if not members:
                continue
            claimed.update(members)
            row_of = _pack_intervals(
                {k: _key_ranges(spans_by_key[k])[k] for k in members}
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
            for key, row in row_of.items():
                tid = row_to_tid.get(row)
                if tid is not None:
                    _place(key, tid)
                else:  # overflow: dropped from the view; instants go to the overflow row
                    dropped_span_count += len(spans_by_key[key])
                    dropped_task_count += 1

        # 3) native threads (SPMD / non-async): one row per thread. A single-thread source
        #    keeps the plain "main" label.
        thread_keys = sorted(
            (k for k in spans_by_key if k[1] == "thread"), key=lambda k: str(k[2])
        )
        for key in thread_keys:
            label = "main" if len(thread_keys) == 1 else f"main {key[2]}"
            _place(key, _row(label))

        # 4) other named tasks: interval-packed so sequential one-shot tasks share a row
        #    (e.g. 5 sequential endpoint calls -> 1 row, not 5). A row holding exactly one
        #    task keeps that task's name as its label.
        other = [k for k in spans_by_key if k[1] == "task" and k not in claimed]
        if other:
            row_of = _pack_intervals(
                {k: _key_ranges(spans_by_key[k])[k] for k in other}
            )
            keys_per_row: dict[int, list] = defaultdict(list)
            for key, row in row_of.items():
                keys_per_row[row].append(key)
            row_to_tid = {}
            for row in sorted(keys_per_row):
                keys = keys_per_row[row]
                label = keys[0][2] if len(keys) == 1 else f"tasks {row}"
                row_to_tid[row] = _row(label)
            for key, row in row_of.items():
                _place(key, row_to_tid[row])

    if dropped_task_count:
        logger.info(
            f"Gantt view omitted {dropped_span_count} spans across {dropped_task_count} "
            f"tasks due to collapse row caps; use raw=True for an exhaustive render"
        )

    return kept_paired, tid_by_key, row_labels


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
    tid_by_key: dict[tuple[str, str, Any], int],
    row_labels: dict[tuple[str, int], str],
) -> list[dict]:
    """Build the Chrome Trace events: ``process_name`` + ``process_sort_index`` per source,
    ``thread_name`` + ``thread_sort_index`` per labeled row, ``X`` per span, ``i`` per instant.

    An instant whose task was dropped by a collapse cap lands on a per-source ``overflow``
    row (created on demand), NOT on row 0 -- an error from a dropped task must stay visible
    without masquerading as a main-loop event.
    """
    events: list[dict[str, Any]] = []

    # Resolve instant rows first: they may mint overflow rows, which must be labeled
    # before the thread-metadata pass below.
    next_row_by_source: dict[str, int] = defaultdict(int)
    for source, row_index in row_labels:
        next_row_by_source[source] = max(next_row_by_source[source], row_index + 1)
    overflow_row_by_source: dict[str, int] = {}

    def _overflow_row(source: str) -> int:
        if source not in overflow_row_by_source:
            row_index = next_row_by_source[source]
            next_row_by_source[source] = row_index + 1
            overflow_row_by_source[source] = row_index
            row_labels[(source, row_index)] = OVERFLOW_ROW_LABEL
        return overflow_row_by_source[source]

    instant_rows = []
    for i in instants:
        row_index = tid_by_key.get(i["key"])
        if row_index is None:
            # raw mode has no labels and keeps every key; only the policy view drops tasks.
            row_index = _overflow_row(i["source"]) if row_labels else 0
        instant_rows.append(row_index)

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
        # Perfetto orders processes by this metadata; pids are assigned in source_order.
        events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"sort_index": pid},
            }
        )

    # thread_name + thread_sort_index per row. The emitted tid claims the main-thread slot for
    # row 0; sort_index keeps our semantic row order (row_index) for the rest.
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
        pid = source_to_pid[p["source"]]
        events.append(
            {
                "name": p["display_name"],
                "ph": "X",
                "ts": p["start_ts"],
                "dur": p["end_ts"] - p["start_ts"],
                "pid": pid,
                "tid": _chrome_tid_for_row(pid, p.get("tid", 0)),
                "args": args,
            }
        )

    for i, row_index in zip(instants, instant_rows):
        args = {**({"step": i["step"]} if i["step"] is not None else {})}
        if i.get("caller"):
            args["caller"] = i["caller"]
        pid = source_to_pid[i["source"]]
        events.append(
            {
                "name": i["name"],
                "ph": "i",
                "ts": i["time_us"],
                "pid": pid,
                "tid": _chrome_tid_for_row(pid, row_index),
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
        "--start-time-us",
        type=int,
        default=None,
        help="Drop records before this epoch-microsecond timestamp.",
    )
    p.add_argument(
        "--last-steps",
        type=int,
        default=None,
        help="Keep only the final N observed training steps.",
    )
    p.add_argument("--start-step", type=int, default=None)
    p.add_argument("--end-step", type=int, default=None)
    p.add_argument(
        "--pinned",
        type=str,
        default=None,
        help="Comma-separated task names to pin as their own rows.",
    )
    args = p.parse_args()

    ranks = (
        None if args.ranks == "all" else tuple(int(x) for x in args.ranks.split(","))
    )
    sources = tuple(args.sources.split(",")) if args.sources else None
    pinned = tuple(args.pinned.split(",")) if args.pinned else ()

    generate_gantt_trace(
        args.log_dir,
        args.output,
        raw=args.raw,
        ranks=ranks,
        sources=sources,
        file_name_regex=args.file_name_regex,
        start_time_us=args.start_time_us,
        last_steps=args.last_steps,
        start_step=args.start_step,
        end_step=args.end_step,
        pinned_tasks=pinned,
    )


if __name__ == "__main__":
    main()
