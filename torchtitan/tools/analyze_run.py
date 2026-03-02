# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Post-hoc analysis script for TorchTitan training outputs.

Reads an output directory and produces a structured performance report by:
1. Discovering artifacts (traces, memory snapshots, JSONL logs)
2. Running HTA (Holistic Trace Analysis) on profiler traces
3. Classifying CUDA memory allocations from snapshots
4. Running 7 detectors on HTA/memory results to identify bottlenecks
5. Printing a human-readable report or JSON output

Usage:
    python -m torchtitan.tools.analyze_run outputs/
    python -m torchtitan.tools.analyze_run outputs/ --trace-step iteration_6 --hta temporal_breakdown comm_comp_overlap --memory --detectors
    python -m torchtitan.tools.analyze_run outputs/ --json
"""

import argparse
import json
import os
import pickle
import sys
import time
from glob import glob
from typing import Any


# ─── Default HTA analyses ────────────────────────────────────────────────────

DEFAULT_HTA_ANALYSES = [
    "temporal_breakdown",
    "comm_comp_overlap",
    "gpu_kernel_breakdown",
    "idle_time_breakdown",
    "cuda_kernel_launch_stats",
    "potential_stragglers",
]


# ─── Step 1: Discover artifacts ──────────────────────────────────────────────


def _resolve_trace_dir(
    trace_dirs: list[str], trace_step: str | None
) -> str | None:
    """Pick the trace directory to analyze.

    If --trace-step is given (e.g., "iteration_6"), find the matching dir.
    Otherwise, use the LAST directory (highest iteration number = most
    representative of steady-state training, past compilation warm-up).
    Returns None if no traces found.
    """
    if not trace_dirs:
        return None
    if trace_step:
        matches = [d for d in trace_dirs if trace_step in os.path.basename(d)]
        if not matches:
            print(
                f"  WARNING: --trace-step={trace_step} not found in {trace_dirs}"
            )
            return None
        return matches[0]
    return trace_dirs[-1]  # last = highest iteration


def discover_artifacts(output_dir: str, trace_step: str | None) -> dict:
    """Find traces, snapshots, JSONL in the output directory."""
    trace_dirs = sorted(
        glob(os.path.join(output_dir, "profiling", "traces", "iteration_*"))
    )
    snapshot_files = sorted(
        glob(
            os.path.join(
                output_dir,
                "profiling",
                "memory_snapshot",
                "*",
                "rank*0*.pickle",
            )
        )
    )
    return {
        "trace_dir": _resolve_trace_dir(trace_dirs, trace_step),
        "snapshot": snapshot_files[-1] if snapshot_files else None,
        "experiment_logs": sorted(
            glob(os.path.join(output_dir, "experiment_logs", "*.jsonl"))
        ),
        "system_logs": sorted(
            glob(os.path.join(output_dir, "system_logs", "*.jsonl"))
        ),
    }


# ─── Step 2: Run HTA analyses ────────────────────────────────────────────────


def _serialize_hta_result(result: Any) -> Any:
    """Convert HTA return values to JSON-serializable dicts.

    HTA methods return diverse types:
    - DataFrame                          -> list of row dicts
    - tuple[DataFrame, DataFrame]        -> {"df_0": [...], "df_1": [...]}
    - dict[int, DataFrame]               -> {"rank_0": [...], "rank_1": [...]}
    - list[int]                          -> pass through (e.g., stragglers)
    - other                              -> str() fallback
    """
    try:
        import pandas as pd
    except ImportError:
        return str(result)

    if isinstance(result, pd.DataFrame):
        return result.to_dict(orient="records")
    if isinstance(result, tuple):
        return {
            f"df_{i}": (
                df.to_dict(orient="records")
                if isinstance(df, pd.DataFrame)
                else df
            )
            for i, df in enumerate(result)
        }
    if isinstance(result, dict):
        serialized = {}
        for k, v in result.items():
            key = f"rank_{k}" if isinstance(k, int) else str(k)
            serialized[key] = (
                v.to_dict(orient="records")
                if isinstance(v, pd.DataFrame)
                else v
            )
        return serialized
    if isinstance(result, list):
        return result
    return str(result)


def run_hta(trace_dir: str, analyses: list[str]) -> dict[str, Any]:
    """Run named HTA analyses. Each name maps to TraceAnalysis.get_{name}()."""
    try:
        from hta.trace_analysis import TraceAnalysis
    except ImportError:
        print(
            "  WARNING: HTA not installed. pip install HolisticTraceAnalysis"
        )
        return {}

    ta = TraceAnalysis(trace_dir=trace_dir)
    results = {}
    for name in analyses:
        method = getattr(ta, f"get_{name}", None)
        if method is None:
            print(f"  WARNING: TraceAnalysis has no method get_{name}")
            continue
        t0 = time.time()
        try:
            # Pass visualize=False to prevent HTA from opening Jupyter server
            import inspect

            sig = inspect.signature(method)
            kwargs = {"visualize": False} if "visualize" in sig.parameters else {}
            result = method(**kwargs)
            results[name] = _serialize_hta_result(result)
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  WARNING: get_{name} failed: {e}")
        elapsed = time.time() - t0
        print(f"  get_{name}: {elapsed:.1f}s")

    # Save results in parent dir (NOT in trace_dir — HTA reads all JSON files in
    # trace_dir as rank files, so hta_analysis.json would corrupt the analysis)
    out_path = os.path.join(os.path.dirname(trace_dir), "hta_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    return results


# ─── Step 3: Memory snapshot analysis ────────────────────────────────────────


def _classify(block: dict) -> str:
    """Classify by stack frames.

    Memory snapshot structure (from torch.cuda.memory._snapshot()):
      block["history"] = [
          {"addr": int, "frames": [{"name": str, "filename": str, "line": int}, ...]},
          ...
      ]
    Each history entry is an allocation event. frames[0] is the innermost frame.
    We concatenate all frame names from the most recent allocation event.

    NOTE: frame availability depends on torch.cuda.memory._record_memory_history()
    having been called with stacks="all" or stacks="python". If stacks were
    not recorded, all blocks classify as "unknown".
    """
    history = block.get("history", [])
    if not history:
        return "unknown"
    # Use the most recent allocation event
    latest = history[-1] if history else {}
    frame_names = " ".join(
        f.get("name", "") for f in latest.get("frames", [])
    )
    if not frame_names:
        return "unknown"
    if any(k in frame_names for k in ("Parameter", "nn.Module", "init_weights")):
        return "model_parameters"
    if any(k in frame_names for k in ("optimizer", "adam", "Adam", "step")):
        return "optimizer_states"
    if any(
        k in frame_names
        for k in ("checkpoint", "saved_tensor", "recompute")
    ):
        return "activation_checkpoint"
    if any(
        k in frame_names
        for k in ("nccl", "ProcessGroup", "all_gather", "reduce_scatter")
    ):
        return "nccl_buffers"
    if any(
        k in frame_names
        for k in ("forward", "backward", "matmul", "linear", "attention")
    ):
        return "activations"
    return "unknown"


def analyze_memory(snapshot_path: str) -> dict:
    """Classify allocations from a CUDA memory snapshot pickle."""
    with open(snapshot_path, "rb") as f:
        snapshot = pickle.load(f)

    categories: dict[str, int] = {}
    total_allocated = 0
    total_reserved = 0

    for segment in snapshot.get("segments", []):
        total_reserved += segment.get("total_size", 0)
        for block in segment.get("blocks", []):
            size = block.get("size", 0)
            if block.get("state") == "active_allocated":
                total_allocated += size
                cat = _classify(block)
                categories[cat] = categories.get(cat, 0) + size

    def to_gib(b: int) -> float:
        return b / (1024**3)

    result = {
        "total_allocated_gib": to_gib(total_allocated),
        "total_reserved_gib": to_gib(total_reserved),
        "fragmentation_pct": 100
        * (total_reserved - total_allocated)
        / max(total_reserved, 1),
        "categories": {
            k: to_gib(v)
            for k, v in sorted(categories.items(), key=lambda x: -x[1])
        },
    }
    # Save alongside snapshot
    out_path = os.path.join(
        os.path.dirname(snapshot_path), "memory_breakdown.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result


# ─── Step 4: Detectors ───────────────────────────────────────────────────────


def detect_low_overlap(hta: dict, threshold: float = 0.3) -> dict:
    """Comm/compute overlap < 30% -> communication is exposed."""
    rows = hta.get("comm_comp_overlap", [])
    if not rows or isinstance(rows, dict):
        return {"is_problem": False, "note": "no overlap data"}
    avg_overlap = sum(r.get("comp_comm_overlap_pctg", 0) for r in rows) / len(
        rows
    )
    is_problem = avg_overlap < threshold * 100
    return {
        "overlap_pctg": avg_overlap,
        "is_problem": is_problem,
        "severity": "warning" if is_problem else "info",
        "recommendation": (
            "Improve overlap: separate NCCL PGs, collective bucketing, prefetch tuning."
            if is_problem
            else "Overlap is adequate."
        ),
    }


def detect_high_idle(hta: dict, threshold: float = 30.0) -> dict:
    """GPU idle > 30%."""
    rows = hta.get("temporal_breakdown", [])
    if not rows or isinstance(rows, dict):
        return {"is_problem": False, "note": "no temporal data"}
    avg_idle = sum(r.get("idle_time_pctg", 0) for r in rows) / len(rows)
    idle_values = [r.get("idle_time_pctg", 0) for r in rows]
    max_idle = max(idle_values)
    min_idle = min(idle_values)
    is_problem = avg_idle > threshold
    return {
        "avg_idle_pctg": avg_idle,
        "max_idle_pctg": max_idle,
        "idle_range": max_idle - min_idle,
        "is_problem": is_problem,
        "severity": "warning" if is_problem else "info",
        "recommendation": (
            f"GPU idle {avg_idle:.0f}% (threshold {threshold:.0f}%). "
            f"Likely CPU-bound or communication-bound. "
            f"Check idle_time_breakdown for cause (host_wait vs kernel_kernel)."
            + (
                f" High rank variance ({max_idle - min_idle:.0f}pp) suggests straggler."
                if max_idle - min_idle > 10
                else ""
            )
        )
        if is_problem
        else "GPU utilization is adequate.",
    }


def detect_kernel_launch_overhead(
    hta: dict, threshold: float = 0.25
) -> dict:
    """Many kernels shorter than CPU launch time -> CUDAGraph candidate."""
    stats = hta.get("cuda_kernel_launch_stats", {})
    if isinstance(stats, dict) and "error" in stats:
        return {"is_problem": False, "note": stats["error"]}
    # Find rank 0 data — serialized as "rank_0" key
    rank0_rows = stats.get("rank_0", [])
    if not rank0_rows:
        # Fallback: try first available rank
        for key, val in stats.items():
            if isinstance(val, list) and val:
                rank0_rows = val
                break
    if not rank0_rows:
        return {"is_problem": False, "note": "no kernel launch data"}
    total = len(rank0_rows)
    short = sum(
        1
        for r in rank0_rows
        if r.get("gpu_dur", 0) < r.get("cpu_dur", 0)
    )
    short_ratio = short / total if total > 0 else 0
    mean_delay = (
        sum(r.get("launch_delay", 0) for r in rank0_rows) / total
        if total > 0
        else 0
    )
    is_problem = short_ratio > threshold
    return {
        "short_kernel_ratio": short_ratio,
        "short_kernel_count": short,
        "total_kernels": total,
        "mean_launch_delay_us": mean_delay,
        "is_problem": is_problem,
        "severity": "warning" if is_problem else "info",
        "recommendation": (
            f"{short_ratio * 100:.0f}% of kernels shorter than CPU launch "
            f"(mean delay {mean_delay:.0f}us). CUDAGraph would eliminate launch overhead."
        )
        if is_problem
        else "Kernel launch overhead is acceptable.",
    }


def detect_stragglers(hta: dict) -> dict:
    """Any rank flagged as potential straggler by HTA."""
    stragglers = hta.get("potential_stragglers", [])
    if isinstance(stragglers, dict) and "error" in stragglers:
        return {"is_problem": False, "note": stragglers["error"]}
    if not stragglers:
        return {"is_problem": False, "stragglers": []}
    return {
        "stragglers": stragglers,
        "is_problem": True,
        "severity": "warning",
        "recommendation": (
            f"Ranks {stragglers} flagged as potential stragglers. "
            f"Check temporal_breakdown per-rank idle for these ranks. "
            f"Common causes: data loading imbalance, hardware issue, "
            f"uneven MoE routing."
        ),
    }


def detect_queue_starvation(
    hta: dict, min_median_queue: float = 5.0
) -> dict:
    """GPU queue often empty -> CPU not feeding GPU fast enough."""
    rows = hta.get("queue_length_summary", [])
    if not rows or isinstance(rows, dict):
        return {"is_problem": False, "note": "no queue data"}
    medians = [
        r.get("median", r.get("mean", 0))
        for r in rows
        if isinstance(r, dict)
    ]
    if not medians:
        return {"is_problem": False, "note": "no queue metrics found"}
    avg_median = sum(medians) / len(medians)
    is_problem = avg_median < min_median_queue
    return {
        "avg_median_queue_length": avg_median,
        "is_problem": is_problem,
        "severity": "warning" if is_problem else "info",
        "recommendation": (
            f"Median queue length {avg_median:.1f} (< {min_median_queue}). "
            f"CPU may not be enqueuing kernels fast enough. "
            f"Check for Python overhead, data loading stalls, or excessive host-side sync."
        )
        if is_problem
        else "GPU queue is healthy.",
    }


def detect_moe_comm_overhead(hta: dict, threshold: float = 0.2) -> dict:
    """AllToAll > 20% of GPU time. MoE communication bottleneck."""
    kernel_data = hta.get("gpu_kernel_breakdown", {})
    if isinstance(kernel_data, dict) and "error" in kernel_data:
        return {"is_problem": False, "note": kernel_data["error"]}
    per_kernel = kernel_data.get("df_1", [])
    if not per_kernel:
        return {"is_problem": False, "note": "no kernel breakdown data"}
    total_us = sum(r.get("sum_us", 0) for r in per_kernel)
    if total_us == 0:
        return {"is_problem": False, "note": "zero total kernel time"}
    a2a_us = sum(
        r.get("sum_us", 0)
        for r in per_kernel
        if "alltoall" in r.get("kernel_name", "").lower()
        or "all_to_all" in r.get("kernel_name", "").lower()
    )
    fraction = a2a_us / total_us
    is_problem = fraction > threshold
    return {
        "a2a_fraction": fraction,
        "a2a_time_ms": a2a_us / 1000,
        "is_problem": is_problem,
        "severity": "warning" if is_problem else "info",
        "recommendation": (
            f"AllToAll is {fraction * 100:.0f}% of GPU time ({a2a_us / 1000:.1f}ms). "
            f"Consider DeepEP for hardware-optimized all-to-all "
            f"and node-limited routing to reduce cross-node traffic."
        )
        if is_problem
        else "AllToAll overhead is acceptable.",
    }


def detect_memory_fragmentation(
    mem: dict | None, threshold: float = 15.0
) -> dict:
    """Reserved >> allocated -> fragmentation. Threshold: >15% fragmented."""
    if not mem:
        return {"is_problem": False, "note": "no memory data"}
    frag = mem.get("fragmentation_pct", 0)
    is_problem = frag > threshold
    return {
        "fragmentation_pct": frag,
        "allocated_gib": mem.get("total_allocated_gib", 0),
        "reserved_gib": mem.get("total_reserved_gib", 0),
        "is_problem": is_problem,
        "severity": "warning" if is_problem else "info",
        "recommendation": (
            f"Fragmentation {frag:.1f}% (allocated={mem.get('total_allocated_gib', 0):.1f}GiB, "
            f"reserved={mem.get('total_reserved_gib', 0):.1f}GiB). "
            f"Consider torch.cuda.memory.set_per_process_memory_fraction() "
            f"or reducing peak allocation variance."
        )
        if is_problem
        else "Memory fragmentation is acceptable.",
    }


def _has_alltoall(hta_results: dict) -> bool:
    """Check if gpu_kernel_breakdown contains alltoall kernels."""
    kernel_data = hta_results.get("gpu_kernel_breakdown", {})
    if not isinstance(kernel_data, dict):
        return False
    per_kernel = kernel_data.get("df_1", [])
    if not isinstance(per_kernel, list):
        return False
    return any(
        "alltoall" in r.get("kernel_name", "").lower()
        or "all_to_all" in r.get("kernel_name", "").lower()
        for r in per_kernel
        if isinstance(r, dict)
    )


def run_detectors(
    hta_results: dict, memory_results: dict | None = None
) -> dict[str, dict]:
    """Run all detectors. Each returns {metric values, is_problem, severity, recommendation}."""
    detectors = [
        detect_low_overlap,
        detect_high_idle,
        detect_kernel_launch_overhead,
        detect_stragglers,
        detect_queue_starvation,
        detect_memory_fragmentation,
    ]
    if _has_alltoall(hta_results):
        detectors.append(detect_moe_comm_overhead)

    results = {}
    for det in detectors:
        try:
            data = (
                memory_results
                if det == detect_memory_fragmentation
                else hta_results
            )
            results[det.__name__] = det(data)
        except Exception as e:
            results[det.__name__] = {"error": str(e), "is_problem": False}
    return results


# ─── Step 5: Print report ────────────────────────────────────────────────────


def _fmt_size(path: str) -> str:
    """Return human-readable size of a directory or file."""
    total = 0
    if os.path.isdir(path):
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
    elif os.path.isfile(path):
        total = os.path.getsize(path)
    if total < 1024:
        return f"{total}B"
    elif total < 1024**2:
        return f"{total / 1024:.1f}KB"
    elif total < 1024**3:
        return f"{total / 1024 ** 2:.1f}MB"
    else:
        return f"{total / 1024 ** 3:.2f}GB"


def print_report(
    artifacts: dict,
    hta_results: dict,
    memory_results: dict | None,
    detector_results: dict,
) -> None:
    """Print human-readable report to stdout."""
    print()
    print("=" * 60)
    print("  TorchTitan Run Analysis")
    print("=" * 60)

    # Artifacts summary
    trace_dir = artifacts.get("trace_dir")
    if trace_dir:
        print(f"\nTrace: {trace_dir} ({_fmt_size(trace_dir)})")
    else:
        print("\nTrace: (none found)")

    snapshot = artifacts.get("snapshot")
    if snapshot:
        print(f"Snapshot: {snapshot}")
    else:
        print("Snapshot: (none found)")

    exp_logs = artifacts.get("experiment_logs", [])
    sys_logs = artifacts.get("system_logs", [])
    print(
        f"Logs: {len(exp_logs)} experiment, {len(sys_logs)} system"
    )

    # HTA results
    if hta_results:
        print("\n--- HTA Analyses ---")
        for name, data in hta_results.items():
            if isinstance(data, dict) and "error" in data:
                print(f"  {name}: ERROR - {data['error']}")
                continue

            if name == "temporal_breakdown" and isinstance(data, list):
                # Summarize: average across ranks
                avg_compute = (
                    sum(r.get("compute_time_pctg", 0) for r in data) / len(data)
                )
                avg_comm = (
                    sum(r.get("non_compute_time_pctg", 0) for r in data)
                    / len(data)
                )
                avg_idle = (
                    sum(r.get("idle_time_pctg", 0) for r in data) / len(data)
                )
                print(
                    f"  {name}: compute={avg_compute:.1f}% | comm={avg_comm:.1f}% | idle={avg_idle:.1f}%"
                )
            elif name == "comm_comp_overlap" and isinstance(data, list):
                avg = sum(
                    r.get("comp_comm_overlap_pctg", 0) for r in data
                ) / len(data)
                print(
                    f"  {name}: {avg:.1f}% average across {len(data)} ranks"
                )
            elif name == "gpu_kernel_breakdown" and isinstance(data, dict):
                per_kernel = data.get("df_1", [])
                if per_kernel:
                    top = sorted(
                        per_kernel,
                        key=lambda r: r.get("sum_us", 0),
                        reverse=True,
                    )[:3]
                    total = sum(r.get("sum_us", 0) for r in per_kernel)
                    parts = []
                    for r in top:
                        pct = (
                            r.get("sum_us", 0) / total * 100
                            if total > 0
                            else 0
                        )
                        kname = r.get("kernel_name", "?")
                        # Truncate long kernel names
                        if len(kname) > 40:
                            kname = kname[:37] + "..."
                        parts.append(f"{kname} ({pct:.1f}%)")
                    print(f"  {name}:")
                    for part in parts:
                        print(f"    Top: {part}")
                else:
                    print(f"  {name}: (no per-kernel data)")
            elif name == "potential_stragglers":
                if isinstance(data, list) and data:
                    print(f"  {name}: ranks {data}")
                else:
                    print(f"  {name}: none detected")
            elif name == "idle_time_breakdown" and isinstance(data, dict):
                df0 = data.get("df_0", [])
                if df0 and isinstance(df0, list):
                    # Summarize idle categories
                    cats = {}
                    for row in df0:
                        for k, v in row.items():
                            if k != "rank" and isinstance(v, (int, float)):
                                cats[k] = cats.get(k, 0) + v
                    n = len(df0)
                    parts = [
                        f"{k}={v / n:.1f}%"
                        for k, v in sorted(
                            cats.items(), key=lambda x: -x[1]
                        )[:3]
                    ]
                    print(f"  {name}: {', '.join(parts)}")
                else:
                    print(f"  {name}: (data available)")
            else:
                # Generic summary
                if isinstance(data, list):
                    print(f"  {name}: {len(data)} rows")
                elif isinstance(data, dict):
                    print(f"  {name}: {len(data)} entries")
                else:
                    print(f"  {name}: {data}")

    # Memory results
    if memory_results:
        print("\n--- Memory Analysis ---")
        alloc = memory_results.get("total_allocated_gib", 0)
        reserved = memory_results.get("total_reserved_gib", 0)
        frag = memory_results.get("fragmentation_pct", 0)
        print(
            f"  Allocated: {alloc:.1f} GiB | Reserved: {reserved:.1f} GiB | Fragmentation: {frag:.1f}%"
        )
        cats = memory_results.get("categories", {})
        if cats:
            for cat, gib in cats.items():
                pct = gib / alloc * 100 if alloc > 0 else 0
                print(f"  {cat:25s} {gib:6.1f} GiB ({pct:.1f}%)")

    # Detector results
    if detector_results:
        print("\n--- Detectors ---")
        for name, result in detector_results.items():
            if isinstance(result, dict) and "error" in result:
                print(f"  ? {name}: ERROR - {result['error']}")
                continue
            is_problem = result.get("is_problem", False)
            marker = "!!" if is_problem else "ok"
            # Build a brief summary
            summary_parts = []
            for k, v in result.items():
                if k in (
                    "is_problem",
                    "severity",
                    "recommendation",
                    "note",
                ):
                    continue
                if isinstance(v, float):
                    summary_parts.append(f"{k}={v:.1f}")
                elif isinstance(v, (int, list)):
                    summary_parts.append(f"{k}={v}")
            summary = ", ".join(summary_parts[:3])
            if result.get("note"):
                summary = result["note"]
            print(f"  [{marker}] {name}: {summary}")
            if is_problem and result.get("recommendation"):
                print(f"       -> {result['recommendation']}")

    print()


# ─── CLI main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a TorchTitan training output directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m torchtitan.tools.analyze_run outputs/\n"
            "  python -m torchtitan.tools.analyze_run outputs/ --trace-step iteration_6\n"
            "  python -m torchtitan.tools.analyze_run outputs/ --hta temporal_breakdown comm_comp_overlap\n"
            "  python -m torchtitan.tools.analyze_run outputs/ --memory --detectors\n"
            "  python -m torchtitan.tools.analyze_run outputs/ --json\n"
        ),
    )
    parser.add_argument(
        "output_dir",
        help="Path to TorchTitan output directory (e.g., outputs/)",
    )
    parser.add_argument(
        "--trace-step",
        default=None,
        help="Trace iteration to analyze (e.g., iteration_6). Default: last available.",
    )
    parser.add_argument(
        "--hta",
        nargs="*",
        default=None,
        help=(
            "HTA analyses to run (space-separated). "
            "Each maps to TraceAnalysis.get_{name}(). "
            f"Default: {' '.join(DEFAULT_HTA_ANALYSES)}"
        ),
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        default=False,
        help="Enable memory snapshot analysis.",
    )
    parser.add_argument(
        "--detectors",
        action="store_true",
        default=False,
        help="Enable detector pass on HTA/memory results.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output full results as JSON to stdout (suppresses human-readable report).",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"ERROR: {output_dir} is not a directory")
        sys.exit(1)

    # If no flags specified, enable everything by default
    run_all = args.hta is None and not args.memory and not args.detectors
    hta_analyses = (
        args.hta
        if args.hta is not None
        else (DEFAULT_HTA_ANALYSES if run_all else [])
    )
    do_memory = args.memory or run_all
    do_detectors = args.detectors or run_all

    # Step 1: Discover
    print(f"Discovering artifacts in {output_dir}...")
    artifacts = discover_artifacts(output_dir, args.trace_step)
    trace_dir = artifacts["trace_dir"]
    snapshot_path = artifacts["snapshot"]

    if trace_dir:
        print(f"  Trace dir: {trace_dir} ({_fmt_size(trace_dir)})")
    else:
        print("  No trace directories found.")
    if snapshot_path:
        print(f"  Snapshot: {snapshot_path}")
    else:
        print("  No memory snapshots found.")
    print(
        f"  Logs: {len(artifacts['experiment_logs'])} experiment, {len(artifacts['system_logs'])} system"
    )

    # Step 2: HTA
    hta_results: dict[str, Any] = {}
    if hta_analyses and trace_dir:
        print(f"\nRunning HTA analyses: {', '.join(hta_analyses)}...")
        hta_results = run_hta(trace_dir, hta_analyses)
    elif hta_analyses and not trace_dir:
        print("\nSkipping HTA: no trace directory found.")

    # Step 3: Memory
    memory_results: dict | None = None
    if do_memory and snapshot_path:
        print(f"\nAnalyzing memory snapshot: {snapshot_path}...")
        memory_results = analyze_memory(snapshot_path)
    elif do_memory and not snapshot_path:
        print("\nSkipping memory analysis: no snapshot found.")

    # Step 4: Detectors
    detector_results: dict[str, dict] = {}
    if do_detectors and (hta_results or memory_results):
        print("\nRunning detectors...")
        detector_results = run_detectors(hta_results, memory_results)
        # Save detector report
        report_path = os.path.join(output_dir, "detector_report.json")
        with open(report_path, "w") as f:
            json.dump(detector_results, f, indent=2, default=str)
        print(f"  Saved: {report_path}")
    elif do_detectors:
        print("\nSkipping detectors: no HTA or memory results to analyze.")

    # Step 5: Output
    if args.json:
        combined = {
            "artifacts": {
                k: v
                for k, v in artifacts.items()
                if k != "trace_dir" or v is not None
            },
            "hta": hta_results,
            "memory": memory_results,
            "detectors": detector_results,
        }
        print(json.dumps(combined, indent=2, default=str))
    else:
        print_report(artifacts, hta_results, memory_results, detector_results)

    print("Done.")


if __name__ == "__main__":
    main()
