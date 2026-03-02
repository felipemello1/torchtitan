# GPU Specifications for MFU Calculation

## Peak TFLOPS

| GPU | BF16 TFLOPS | FP16 TFLOPS | TF32 TFLOPS | FP32 TFLOPS | HBM BW (TB/s) |
|-----|-------------|-------------|-------------|-------------|---------------|
| A100 SXM 80GB | 312 | 312 | 156 | 19.5 | 2.0 |
| A100 PCIe 80GB | 312 | 312 | 156 | 19.5 | 2.0 |
| H100 SXM 80GB | 990 | 990 | 495 | 67 | 3.35 |
| H100 PCIe 80GB | 756 | 756 | 378 | 51 | 2.0 |
| H200 SXM 141GB | 990 | 990 | 495 | 67 | 4.8 |
| B200 SXM 192GB | 2250 | 2250 | 1125 | 150 | 8.0 |
| GB200 NVL72 | 2250 | 2250 | 1125 | 150 | 8.0 |

## MFU Formula

```
MFU = (num_flops_per_token * tokens_per_second_per_device) / peak_flops
```

Where `num_flops_per_token` depends on the model:
- Dense transformer: `6 * num_params` (forward pass) or `6 * num_params * seq_len` per sequence
- MoE: `6 * (dense_params + active_expert_params * top_k)`

## Roofline Ridge Points

The ridge point is where compute becomes the bottleneck instead of memory bandwidth:

```
ridge_point = peak_flops / memory_bandwidth  (in ops/byte)
```

| GPU | Ridge Point (BF16) |
|-----|-------------------|
| A100 SXM | 156 ops/byte |
| H100 SXM | 295 ops/byte |
| H200 SXM | 206 ops/byte |
| B200 SXM | 281 ops/byte |

Operations below the ridge point are memory-bound. Above = compute-bound.
For LLM training, most operations are compute-bound (large matmuls).
