# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Visualize structured logs as Chrome Trace (Gantt chart).

Usage:
    # Generate trace from toy_spmd.py output:
    python visualize.py /tmp/toy_spmd_output/system_logs /tmp/toy_spmd_output/trace.json

    # Generate trace from any system_logs directory:
    python visualize.py <system_logs_dir> <output_trace.json>

    # View: open trace.json in chrome://tracing or https://ui.perfetto.dev
"""

import json
import os
import sys
from glob import glob


def load_all_records(log_dir: str) -> list[dict]:
    """Load all JSONL records from a system_logs directory."""
    records = []
    for path in sorted(glob(os.path.join(log_dir, "*.jsonl"))):
        source = os.path.basename(path).replace(".jsonl", "")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_source_file"] = source
                    records.append(r)
    return records


def to_chrome_trace(log_dir: str, output_path: str) -> dict:
    """Convert structured logs to Chrome Trace JSON format.

    Each source file (e.g., trainer_rank_0_system) becomes a process.
    Each rank becomes a thread within that process. Start/end events
    become duration spans visible as a Gantt chart.

    Returns:
        The trace dict (also written to output_path).
    """
    records = load_all_records(log_dir)
    if not records:
        print(f"No records found in {log_dir}")
        return {"traceEvents": []}

    # Map source files to process IDs
    sources = sorted(set(r["_source_file"] for r in records))
    source_to_pid = {s: i for i, s in enumerate(sources)}

    events = []

    # Process name metadata
    for source, pid in source_to_pid.items():
        events.append({
            "name": "process_name", "ph": "M", "pid": pid, "tid": 0,
            "args": {"name": source},
        })

    for r in records:
        normal = r.get("normal", {})
        event_type = normal.get("log_type_name", "")
        time_ms = r.get("int", {}).get("time_ms", 0)
        time_us = time_ms * 1000  # Chrome Trace uses microseconds
        pid = source_to_pid[r["_source_file"]]
        rank = r.get("int", {}).get("rank", 0)
        step = r.get("int", {}).get("step")

        if event_type.endswith("_start"):
            name = event_type.replace("_start", "")
            events.append({
                "name": name, "ph": "B", "ts": time_us,
                "pid": pid, "tid": rank,
                "args": {"step": step},
            })
        elif event_type.endswith("_end"):
            name = event_type.replace("_end", "")
            duration_ms = r.get("double", {}).get("value", 0)
            events.append({
                "name": name, "ph": "E", "ts": time_us,
                "pid": pid, "tid": rank,
                "args": {"step": step, "duration_ms": f"{duration_ms:.2f}"},
            })
        elif event_type == "metric_value":
            event_name = normal.get("event_name", "metric")
            value = r.get("double", {}).get("value", 0)
            events.append({
                "name": f"{event_name}={value:.4f}", "ph": "i", "ts": time_us,
                "pid": pid, "tid": rank, "s": "t",
                "args": {"step": step},
            })

    trace = {"traceEvents": events}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trace, f)

    print(f"Chrome Trace: {output_path}")
    print(f"  {len(events)} events from {len(sources)} sources")
    print(f"  View in: chrome://tracing or https://ui.perfetto.dev")
    return trace


def summary(log_dir: str) -> None:
    """Print summary statistics from structured logs."""
    records = load_all_records(log_dir)

    sources: dict[str, int] = {}
    for r in records:
        src = r["_source_file"]
        sources[src] = sources.get(src, 0) + 1

    print(f"Total records: {len(records)}")
    print("Sources:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    # Phase timing
    print("\nPhase timing (from _end events):")
    phase_durations: dict[str, list[float]] = {}
    for r in records:
        event_type = r.get("normal", {}).get("log_type_name", "")
        if event_type.endswith("_end"):
            phase = event_type.replace("_end", "")
            duration = r.get("double", {}).get("value", 0)
            if duration > 0:
                phase_durations.setdefault(phase, []).append(duration)

    for phase, durations in sorted(phase_durations.items()):
        avg = sum(durations) / len(durations)
        mn, mx = min(durations), max(durations)
        print(f"  {phase}: avg={avg:.1f}ms, min={mn:.1f}ms, max={mx:.1f}ms, n={len(durations)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    to_chrome_trace(sys.argv[1], sys.argv[2])
