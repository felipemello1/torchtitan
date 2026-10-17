# Qwen3.5 regional compile traces

Profiles five Qwen3.5-27B regions at the 40,960-token training shape:

1. OffsetRMSNorm (Q-norm shape)
2. GDN RMSNormGated
3. SwiGLU
4. Partial RoPE plus Q/K reconstruction
5. Residual-add plus FFN OffsetRMSNorm

```bash
python benchmarks/qwen3_5_compile_regions/trace_regions.py \
  --output-dir region_traces
```

The script writes synchronized forward/backward timings to `results.json` and
one compressed Chrome/Perfetto trace per region and implementation. Eager and
regional `torch.compile` paths run by default.

GB300 results (`40,960` tokens, BF16, 3 warmups, 10 measurements):

| Region | Eager | Compile | Kernel reference |
|---|---:|---:|---:|
| OffsetRMSNorm | 7.186 ms | 0.507 ms | stock Triton fails at this shape |
| GDN RMSNormGated | 11.281 ms | 0.811 ms | FLA 0.851 ms |
| SwiGLU | 8.263 ms | 2.586 ms | Triton 2.622 ms; packed 1.743 ms |
| Partial RoPE + reconstruction | 4.854 ms | 0.742 ms | Helion + reconstruction 2.966 ms |
| Residual-add + FFN RMSNorm | 6.271 ms | 0.817 ms | FLA 0.952 ms |

Installed kernel references can also be traced:

```bash
python benchmarks/qwen3_5_compile_regions/trace_regions.py \
  --implementations eager,compile,triton,triton_packed,fla,helion \
  --output-dir region_traces
```

Kernel coverage is explicit: Triton covers OffsetRMSNorm and SwiGLU (hosted and
packed-backward variants), FLA covers GDN RMSNormGated and residual RMSNorm,
and Helion covers RoPE rotation followed by eager reconstruction. Missing
optional packages are skipped. The stock OffsetRMSNorm Triton backward is
expected to fail at the Qwen3.5-27B production shape.

The measured traces and raw JSON are in `results/gb300_40960`.
