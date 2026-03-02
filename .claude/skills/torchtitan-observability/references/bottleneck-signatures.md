# Bottleneck Signatures and Thresholds

## Decision Tree

```
Is GPU idle > 30%?
├── YES → Check idle_time_breakdown
│   ├── host_wait dominant → CPU-bound (data loading, Python overhead)
│   ├── kernel_kernel dominant → Small kernels, fragmented execution
│   └── other dominant → Synchronization barriers
├── NO → Check comm/compute overlap
│   ├── Overlap < 30% → Communication exposed
│   │   ├── AllToAll dominant → MoE routing overhead
│   │   └── AllGather/ReduceScatter → FSDP/TP comm
│   └── Overlap > 30% → Good overlap
│       └── Check kernel launch overhead
│           ├── >25% short kernels → CUDAGraph candidate
│           └── OK → Check MFU vs peak
```

## Threshold Table

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| GPU idle % | < 15% | 15-30% | > 30% |
| Comm/compute overlap | > 50% | 30-50% | < 30% |
| Short kernel ratio | < 10% | 10-25% | > 25% |
| Memory fragmentation | < 10% | 10-15% | > 15% |
| AllToAll % of GPU time | < 10% | 10-20% | > 20% |
| Queue length (median) | > 10 | 5-10 | < 5 |
| Straggler idle spread | < 5pp | 5-10pp | > 10pp |

## Common Fixes

| Bottleneck | Fix | Expected Impact |
|-----------|-----|----------------|
| High idle (data loading) | Increase num_workers, prefetch | 10-30% step time reduction |
| High idle (Python) | torch.compile, CUDAGraph | 15-40% |
| Exposed communication | Separate NCCL PGs, bucket_cap_mb tuning | 5-20% |
| MoE AllToAll | DeepEP, node-limited routing, expert capacity tuning | 10-25% |
| Short kernels | CUDAGraph, operator fusion | 10-30% |
| Memory fragmentation | set_per_process_memory_fraction, reduce activation peaks | Avoid OOM |
| Straggler | Fix data loading balance, check NVLink topology | 5-15% |
