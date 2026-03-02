# analyze_run.py Guide

## Overview

`analyze_run.py` reads a TorchTitan output directory and produces a structured
performance report. It combines HTA trace analysis, memory snapshot analysis,
and automated bottleneck detection.

## Usage

```bash
# Full analysis (all defaults):
python -m torchtitan.tools.analyze_run <output_dir>/

# Specific HTA analyses only:
python -m torchtitan.tools.analyze_run <output_dir>/ \
    --hta temporal_breakdown comm_comp_overlap

# Memory analysis only:
python -m torchtitan.tools.analyze_run <output_dir>/ --memory

# Detectors only:
python -m torchtitan.tools.analyze_run <output_dir>/ --detectors

# JSON output for programmatic use:
python -m torchtitan.tools.analyze_run <output_dir>/ --json
```

## How It Works

### Step 1: Discover Artifacts
Scans the output directory for:
- `profiling/traces/iteration_*/` — Chrome trace directories
- `profiling/memory_snapshot/*/rank*.pickle` — memory snapshots
- `experiment_logs/*.jsonl` — experiment metrics
- `system_logs/*.jsonl` — system phase timing

### Step 2: HTA Analysis
For each `--hta` method name, calls `TraceAnalysis.get_{name}()`.
Results are saved to `hta_analysis.json` in the trace directory.

### Step 3: Memory Analysis
Loads the memory snapshot pickle and classifies allocations:
- **model_parameters**: nn.Module, Parameter, init_weights
- **optimizer_states**: Adam, optimizer, step
- **activations**: forward, backward, matmul, attention
- **activation_checkpoint**: checkpoint, saved_tensor, recompute
- **nccl_buffers**: nccl, ProcessGroup, all_gather
- **unknown**: unclassifiable

### Step 4: Detectors
7 automated checks on HTA + memory results. Each returns:
```json
{
    "is_problem": true/false,
    "severity": "info" | "warning",
    "recommendation": "specific action to take",
    ...metric values...
}
```

### Step 5: Report
Prints a formatted summary. With `--json`, outputs all results as JSON to stdout.

## Reading the Output

### Good Signs
- All detectors show `is_problem: false`
- Comm/compute overlap > 50%
- GPU idle < 15%
- No stragglers
- Fragmentation < 10%

### Red Flags
- `detect_high_idle` fires → GPU is underutilized
- `detect_low_overlap` fires → communication is exposed
- `detect_stragglers` fires → one rank is slowing everyone down
- `detect_moe_comm_overhead` fires → AllToAll is a bottleneck

## Visualizing Traces

For visual inspection, open the Chrome traces directly in Perfetto:
1. Go to https://ui.perfetto.dev
2. Open the trace file from `profiling/traces/iteration_N/rank000000_trace.json.gz`
3. Look for `profile_annotation` labels (forward_backward, optimizer, Attention, FFN, MoE)

For system-level phase timing:
```bash
python torchtitan/experiments/observability/visualize.py \
    <output_dir>/system_logs/ trace.json
```
Open `trace.json` in Perfetto to see a Gantt chart of training phases across ranks.
