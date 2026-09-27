# fp32 weight gradients, end to end (demo)

Owner: @felipemello1. Status: demo of a proposal; everything here is meant to move into PyTorch.

`training.mixed_precision_grad="float32"` (core, PR below this one in the stack) makes `Linear` and `GroupedLinear` write fp32 weight gradients instead of rounding them to bf16. Two PyTorch changes complete it. This folder runs both on a stock PyTorch nightly, so the whole stack can be tried and measured today:

```text
main:          GEMM (fp32 accumulate) -> write dW bf16 -> FSDP copy-in (bf16 -> fp32 buffer) -> reduce-scatter
this stack:    GEMM (fp32 accumulate) -> write dW fp32 ------------------------------------> reduce-scatter
```

1. **`grouped_mm.py`: `torch._grouped_mm(..., out_dtype=torch.float32)` for bf16 inputs.**
   - It's needed for MoE experts; PyTorch rejects it today, as its CUTLASS kernel hardcodes a bf16 output.
   - `grouped_mm_fp32.cu` is that kernel with the output dtype as a template parameter, JIT-built as an extension.
   - `install()` routes exactly those calls to it.
2. **`fsdp_reduce_scatter.py`: FSDP2 reduce-scatter without the copy-in.**
   - When every gradient of an FSDP unit already has the reduce dtype, it reduce-scatters each gradient straight into its output slice (one coalesced collective) instead of copying all of them into one fp32 buffer first.
   - `install()` swaps in a copy of FSDP2's `foreach_reduce` with this change. It refuses to run on any PyTorch whose `foreach_reduce` differs from the one it was copied from.
   - With bf16 gradients (the default) nothing changes: the copy-in is still needed to convert.

`pytorch_patches/` has the upstream-shaped versions of both, against PyTorch main `e0fe13c`. Once they land, delete this folder: core's `_grouped_mm` probe then enables experts by itself.

## Run

Requirements:
- `torch==2.15.0.dev20260926+cu130`; the FSDP copy is pinned to it.
- nvcc matching torch's CUDA: `pip install ninja nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-crt==13.0.88`.
- CUTLASS at PyTorch's pin in `CUTLASS_DIR` (default `/tmp/$USER/cutlass`):

  ```bash
  git init $CUTLASS_DIR && git -C $CUTLASS_DIR fetch --depth 1 https://github.com/NVIDIA/cutlass e05f953a5b3d38adc240df2ff928e0421c2abba3 && git -C $CUTLASS_DIR checkout FETCH_HEAD
  ```

```bash
# The first run builds the extension (~1.5 min, cached); build once before launching many ranks
python -c "from torchtitan.experiments.fp32_weight_grads import grouped_mm; grouped_mm.load_extension()"
torchrun --nproc_per_node=8 -m torchtitan.experiments.fp32_weight_grads.train \
    --module qwen3 --config qwen3_30b_a3b --training.mixed_precision_grad float32
```

Tests (CPU/gloo for FSDP; the kernel test needs an SM90/SM100 GPU and CUTLASS):

```bash
pytest torchtitan/experiments/fp32_weight_grads/tests/
```

## Results (1x GB300, FSDP 1 rank, eager, seed 42, deterministic, 2 interleaved repeats)

Qwen3-30B-A3B layer shapes, 4 layers, 100 steps:

```text
                         tokens/s vs main   peak mem    dW error vs exact fp64, Linear / experts
main                     56,912             59.93 GiB   1.66e-03 / 1.66e-03
core only (experts bf16) (not faster)                   1.14e-05 / 1.66e-03
this stack               +10.1%             60.61 GiB   1.14e-05 / 3.75e-06
```

Llama3 1B, 300 steps: +2.7% tokens/s, weight-gradient error 1.66e-03 -> 9.27e-06.

About the numbers:
- **Where the speed comes from:** removing FSDP's copy-in.
- **Part of it is single-GPU only:** on one rank FSDP copies each gradient twice and stalls on the second copy. In a profiled step, 7.5 ms of the 16.7 ms saved exists on any number of GPUs, and 5.5 ms is single-GPU only. 8-GPU results are pending.
- **Memory:** the fp32 unsharded gradients of the block in backward cost ~1.8 GiB for this block size, the same at any depth.

The correctness checks in `tests/` show the in-place reduce-scatter's sharded gradients bitwise equal to FSDP's copy-in path in these cases:
- 1/2/4 ranks;
- HSDP;
- `Shard(1)` parameters;
- uneven dim-0;
- no-sync accumulation;
- a divide factor.

## Caveats

- **Pinned PyTorch:** the FSDP change is a copy of PyTorch internals, pinned to one nightly (`install()` checks it).
- **Vendor kernel:** the CUDA kernel is here only for the demo. Per the experiments guidelines, kernels belong upstream, and this one's upstream home is PyTorch's `_grouped_mm`.
- **Compile:** `mixed_precision_grad="float32"` doesn't support `compile.components` with `"model"` yet (core raises).
