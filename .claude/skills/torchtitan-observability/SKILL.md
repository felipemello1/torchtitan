# TorchTitan Observability Analysis

Analyze training runs using structured JSONL, profiler traces, and memory snapshots.

## When to Use

- User asks to analyze a training run's performance
- User has profiler traces or memory snapshots to examine
- User wants to identify bottlenecks (idle GPU, stragglers, communication overhead)
- User asks about MFU, throughput, or memory usage

## Quick Start

```bash
# Analyze a run's output directory:
python -m torchtitan.tools.analyze_run <output_dir>/

# With specific options:
python -m torchtitan.tools.analyze_run <output_dir>/ \
    --trace-step iteration_7 \
    --hta temporal_breakdown comm_comp_overlap gpu_kernel_breakdown \
    --memory --detectors

# JSON output (for programmatic consumption):
python -m torchtitan.tools.analyze_run <output_dir>/ --json
```

## Full Workflow

### Step 1: Run Training with Profiling

```bash
torchrun --nproc_per_node=8 torchtitan/train.py \
    --model.name llama3 --model.flavor 8B \
    --training.steps=10 \
    --profiling.enable_profiling --profiling.profile_freq=10 \
    --profiling.enable_memory_snapshot \
    --job.dump_folder outputs/
```

This profiles the last steps (after compilation warm-up).

### Step 2: Run Analysis

```bash
python -m torchtitan.tools.analyze_run outputs/
```

This produces:
- `hta_analysis.json` — HTA results (temporal breakdown, overlap, kernel stats)
- `memory_breakdown.json` — memory classification by category
- `detector_report.json` — automated bottleneck detection

### Step 3: Interpret Results

The 7 detectors check for common issues:

| Detector | Fires when | Action |
|----------|-----------|--------|
| `detect_low_overlap` | Comm/compute overlap < 30% | Improve overlap: separate PGs, bucketing, prefetch |
| `detect_high_idle` | GPU idle > 30% | Check idle_time_breakdown for cause |
| `detect_kernel_launch_overhead` | >25% kernels shorter than CPU launch | Consider CUDAGraph |
| `detect_stragglers` | HTA flags rank(s) as slow | Check data loading balance, hardware |
| `detect_queue_starvation` | Median queue length < 5 | CPU may not feed GPU fast enough |
| `detect_moe_comm_overhead` | AllToAll > 20% of GPU time | Consider DeepEP, node-limited routing |
| `detect_memory_fragmentation` | Reserved >> allocated by >15% | Reduce peak variance |

### Step 4: Visualize System Logs

```bash
# Generate Chrome Trace from structured JSONL:
python torchtitan/experiments/observability/visualize.py \
    outputs/system_logs/ outputs/trace.json

# View in browser: chrome://tracing or https://ui.perfetto.dev
```

## Output Directory Structure

```
outputs/
├── system_logs/                    ← Per-rank JSONL (phase timing, events)
│   └── trainer_rank_0_system.jsonl
├── experiment_logs/                ← Per-rank JSONL (metrics)
│   └── trainer_rank_0_experiment.jsonl
├── profiling/                      ← Profiler artifacts
│   ├── traces/iteration_N/         ← Chrome traces per rank
│   └── memory_snapshot/step_N/     ← Memory snapshot pickles
├── tb/                             ← TensorBoard (if enabled)
├── trace.json                      ← Chrome Trace (from visualize.py)
├── hta_analysis.json               ← HTA results
├── memory_breakdown.json           ← Memory classification
└── detector_report.json            ← Bottleneck detection
```

## Key Metrics

| Metric | Source | What it means |
|--------|--------|---------------|
| `loss/global_avg` | experiment JSONL | Training loss (cross-entropy) |
| `throughput/tps` | experiment JSONL | Tokens per second per device |
| `throughput/mfu_pct` | experiment JSONL | Model FLOPS utilization |
| `time/end_to_end_s` | experiment JSONL | Wall-clock time per step |
| `memory/max_reserved_gib` | experiment JSONL | Peak GPU memory reserved |
| `temporal_breakdown` | HTA | Compute vs comm vs idle % |
| `comm_comp_overlap` | HTA | How much comm is hidden behind compute |

## Reference Files

See `references/` directory for:
- `bottleneck-signatures.md` — threshold tables and decision tree
- `hta-api-guide.md` — all HTA methods, return types, column names
- `gpu-specs.md` — peak TFLOPS and HBM bandwidth per GPU family
- `analyze-run-guide.md` — detailed guide for using analyze_run.py
