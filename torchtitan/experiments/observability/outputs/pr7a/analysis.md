# PR7a Output Analysis

**Run:** `torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py`
**Date:** 2026-03-01
**Config:** 4 GPUs, 2DP x 2TP, 20 steps, TinyModel (64d, 128h, 3 layers)

## System JSONL

- **360 records** across 4 ranks (90 per rank)
- Each step produces: fwd_bwd_start, fwd_bwd_end, optim_start, optim_end = 4 events
- Plus metric_value events on logging steps (1, 5, 10, 15, 20)
- 20 steps x 4 events = 80 span events per rank. Plus ~10 metric events = ~90 total.

## Phase Timing

| Phase | Avg (ms) | Min (ms) | Max (ms) | Count |
|-------|----------|----------|----------|-------|
| fwd_bwd | 212.7 | 14.4 | 4005.2 | 80 |
| optim | 8.5 | 2.5 | 121.2 | 80 |

**Step 1** is an outlier: ~4s for fwd_bwd (torch.compile compilation), ~121ms for
optimizer. Steps 2+ are steady at ~15ms fwd_bwd and ~2.5ms optimizer.

## Chrome Trace

- **364 events** (360 records + 4 process metadata)
- Open `trace.json` in https://ui.perfetto.dev to see Gantt chart
- Each rank is a separate thread within its process
- fwd_bwd and optim spans are clearly visible as duration bars

## Scalars (Tensor Metrics)

Written on logging steps (1, 5, 10, 15, 20). Example from step 1:
- loss: ~3.42 (cross-entropy, decreasing over training)
- grad_norm: ~4000 (high on step 1 due to random init, drops to ~0.4 by step 20)
- w2_partial: ~0.1 (Partial DTensor metric from TransformerBlock)
- layer_0/w1_norm: ~6.5 (weight norm, stable)

## Experiment JSONL

Contains CPU metrics (learning_rate, step_time_ms) on every step, plus bridged
tensor metrics (loss, grad_norm, w1_norm, w2_partial with all_reduced=True) on
logging steps.

## Observations

1. Compilation overhead dominates step 1 (4s vs 15ms steady-state)
2. Optimizer step is ~6x faster than forward/backward (expected for this tiny model)
3. All 4 ranks produce identical event counts (symmetric workload)
4. The trace visualization makes it easy to spot the compilation spike on step 1
