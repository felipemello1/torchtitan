# HTA (Holistic Trace Analysis) API Guide

Install: `pip install HolisticTraceAnalysis`

## Basic Usage

```python
from hta.trace_analysis import TraceAnalysis
ta = TraceAnalysis(trace_dir="outputs/profiling/traces/iteration_7/")
```

## Methods

| Method | Return Type | Key Columns | What It Tells You |
|--------|------------|-------------|-------------------|
| `get_temporal_breakdown()` | DataFrame | rank, idle_pctg, compute_pctg, non_compute_pctg | Where time goes per rank |
| `get_comm_comp_overlap()` | DataFrame | rank, comp_comm_overlap_pctg | Is communication hidden behind compute? |
| `get_gpu_kernel_breakdown()` | tuple[DataFrame, DataFrame] | df_0: type percentages; df_1: per-kernel sum_us, count, mean_us | Which kernels dominate? |
| `get_idle_time_breakdown()` | tuple[DataFrame, DataFrame?] | idle category, duration | WHY is GPU idle? |
| `get_cuda_kernel_launch_stats()` | dict[int, DataFrame] | gpu_dur, cpu_dur, launch_delay | Is CPU launch overhead limiting? |
| `get_potential_stragglers()` | list[int] | — | Which ranks are slow? |
| `get_gpu_kernels_with_user_annotations()` | DataFrame? | annotation, kernel_name, dur | Maps kernels to profile_annotation labels |
| `get_queue_length_summary()` | DataFrame? | rank, stream, min, max, median | Is GPU pipeline full? |
| `get_memory_bw_summary()` | DataFrame | bandwidth utilization | Memory bandwidth usage |
| `get_frequent_cuda_kernel_sequences()` | DataFrame | sequences, count | Kernel fusion opportunities |

## Idle Time Categories

From `get_idle_time_breakdown()`:
- **host_wait**: GPU waiting for CPU to enqueue work (data loading, Python overhead)
- **kernel_kernel**: Gap between consecutive GPU kernels (small kernel overhead)
- **other**: Synchronization barriers, NCCL waits

## Straggler Detection

`get_potential_stragglers()` returns rank IDs where:
- Idle time significantly exceeds the median across ranks
- Often caused by: data loading imbalance, hardware degradation, uneven MoE routing
