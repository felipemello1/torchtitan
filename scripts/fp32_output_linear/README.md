# FP32OutputLinear: regenerate every number

These scripts regenerate the numbers in `FP32OutputLinear`'s docstrings, comments and TODOs, and in its PR description, on any GPU. Each script prints one text file; this README maps every number to the line that regenerates it.

## Run

```bash
# Real data: final-norm hidden states and lm_head weights, from Hugging Face, on the c4 sample in tests/assets.
PYTHONPATH=. python scripts/fp32_output_linear/prepare_data.py --cache-dir ~/.cache/fp32_output_linear [--with-27b]
# Every script, one text file each in <out_dir> (stderr in <out_dir>/logs). Needs 2 GPUs for the FSDP check.
CUDA_VISIBLE_DEVICES=0,1 PYTHON=python scripts/fp32_output_linear/run_all.sh <out_dir> [--with-27b]
```

- `run_all.sh` runs `prepare_data.py` itself and keeps cache files that exist. `CACHE_DIR` overrides the cache location.
- `--with-27b` downloads Qwen3.5-27B (54 GB) and regenerates the PR's "Controlled results" tables. Without it, those tables aren't regenerated.
- Needs a torch nightly from 2026-10-02 or later; older builds round an fp32 grad_weight to bf16 under FSDP. `transformers` is needed for `prepare_data.py` only.
- Every script runs alone too: each docstring has an `Example:` command.
- BF16x9 rows run on sm100+ (B200, B300, GB300) only; elsewhere they print "skipped". On those GPUs, the old router backward and `upcast + BF16x9` use BF16x9, as the trainer did before this PR.

| script | what it measures |
|---|---|
| `docstring_table.py` | the backward docstring's table: errors on real data, backward time eager / eager split / compiled |
| `rounding.py` | round vs truncate each piece: errors and cost; the 0.1 example; where 3 pieces stop being exact |
| `split_k.py` | the split-K TODO, implemented locally: grad_input error and backward time |
| `split_compile.py` | compiled vs eager split: kernels, time, static vs symbolic dims, one head + loss chunk |
| `grad_weight_layouts.py` | LM-head grad_weight: stack vs split, with and without `addmm(out=)` |
| `backward_memory.py` | size of each backward temporary, measured peaks, cost of a bf16 `.grad` |
| `gemm_throughput.py` | bf16 vs fp32 (IEEE, TF32, BF16x9) GEMM TFLOPS |
| `router_gate_comparison.py` | the deleted `RouterGateLinear` vs `FP32OutputLinear` at Qwen3.5-35B-A3B's router shape |
| `lm_head_variants.py` | the PR's "Controlled results": inference and training per LM-head variant |
| `fsdp_fp32_grad.py` | does FSDP2 keep the fp32 grad_weight (2 GPUs) |

Baselines that no longer exist in torchtitan are local copies, labeled in each script: `CastLinear` (`upcast + ... GEMM`), `RouterGateLinear`'s backward, a bf16 (Megatron-style) backward, the truncated split, and split-K.

## Where each number comes from

"line" = a string to grep for in `<out_dir>/<script>.txt`.

### `torchtitan/models/common/linear.py`

| number | where | script: line |
|---|---|---|
| 3 pieces cost 1.4-1.6x the 2-piece backward | `FP32OutputLinear.Config.higher_precision_bwd` | `docstring_table`: `3 pieces / 2 pieces backward time` |
| bf16 GEMMs ~15x faster than fp32 matmuls on H100 | `backward`, Option 1 | `gemm_throughput`: `bf16 (fp32 out) / fp32 IEEE` |
| bf16 keeps 8 of fp32's 24 significant bits | `backward`, Option 1 and 3 | definition (7 stored mantissa bits + 1 implicit) |
| 0.1 = 0.100097656 - 0.000097752 + 0.000000097; 2 pieces off by 1e-7 | `backward`; `_split_into_bf16_pieces_eager` Example | `rounding`: `the shipped split of 0.1` |
| 3 pieces exact for \|x\| >= 2^-110; 2 pieces keep about 16 bits | `_split_into_bf16_pieces_eager` Args | `rounding`: `3 pieces, \|x\| >= 2^-110` and `2 pieces, \|x\| >= 2^-110` (max relative error 2^-17 = 7.6e-6) |
| a third piece helps a router's grad_input a little, an LM head's not at all | `backward`, "2 or 3 pieces" | `docstring_table`: `grad_input error, 2 pieces / 3 pieces` |
| third piece ~1.5x slower | `backward`, "2 or 3 pieces" | `docstring_table`: `3 pieces / 2 pieces backward time` |
| table: relative errors (grad_input, grad_weight) | `backward` table | `docstring_table`: `relative error vs fp64 (real data)`, columns `grad_input`, `grad_weight` |
| table: backward times; bf16 backward 7.2 ms (head) and 0.29 ms (router) | `backward` table | `docstring_table`: `median of 3 runs`, `(bf16 Linear ... ms)` |
| stacking makes the LM head's grad_weight 1.4x faster | `backward`, Stacking | `grad_weight_layouts`: `qwen3_8b 2048 2 first split` (`x stack`) |
| copies x: 32 MiB; copying W instead: 2.3 GiB (LM head) | `backward`, Stacking | `backward_memory`: `cat([x] * P)`, `cat([W] * P)` under `Qwen3-8B head chunk` |
| copies W: 1 MiB; copying x instead: 512 MiB (router) | `backward`, Stacking | `backward_memory`: `cat([W] * P)`, `cat([x] * P)` under `router 64k tok` |
| split-K: grad_input error 2.9e-4 -> 1.3e-5, correctly rounded 95.2% -> 99.6%, +3-5% backward | `_wide_backward` TODO | `split_k`: `one GEMM (ships)` and `L=8192` (errors); `L=8192 ... x shipped` (time) |
| fused split kernel: 2.4e-6 | `_wide_backward` TODO | not regenerable, see below |
| addmm(out=weight.grad) saves 2.9 ms and a 2.3 GiB temporary per Qwen3-8B chunk | `_wide_backward` TODO | `grad_weight_layouts`: `qwen3_8b 2048 2 later stack` vs `stack_addmm` (time and `extra`) |
| copying x beats one GEMM per piece + add: 1.4x at 2048 tokens, 1.03-1.14x at 8k-32k | `_wide_backward` comment | `grad_weight_layouts`: `first split` rows at 2048, 8192, 16384, 32768 tokens, 2 pieces (`x stack`) |
| compiled split: 1 kernel instead of 6 (10 with 3 pieces), 6.80 -> 1.66 ms | split section comment | `split_compile`: `split + cat alone`, `LM head [2048, 151936]` rows |
| -15-17% per Qwen3-8B head + loss chunk | split section comment | `split_compile`: `LM head + loss chunk`, `T=2048` rows (CE eager and compiled) |
| a symbolic vocab dim makes the split 15-40% slower | `_split_into_bf16_pieces` comment | `split_compile`: `static or symbolic dims`, LM head `symbolic T and O` |
| LM-head grad_weight error 1.4e-5 (truncating) -> 7.6e-6 (rounding) | `_split_into_bf16_pieces_eager` | `rounding`: LM head `2 pieces truncate` / `2 pieces half-away`, column `grad_weight` |

### `torchtitan/config/transform/lm_head_fp32.py` and `torchtitan/components/loss.py`

| number | where | script: line |
|---|---|---|
| 3 pieces cost 1.4-1.6x the 2-piece backward | `LMHeadFP32OutputConverter.Config.higher_precision_bwd` | `docstring_table`: `3 pieces / 2 pieces backward time` |
| router grad_input error 7x lower with 3 pieces | same | `docstring_table`: `router: grad_input error, 2 pieces / 3 pieces` |
| compiling lm_head with the loss frees the fp32 dlogits, 1.2 GiB per Qwen3-8B chunk | `ChunkedLossWrapper` TODO in `loss.py` | `backward_memory`: `grad_output [T, O] fp32 (the dlogits)` |

### PR description

| number | section | script: line |
|---|---|---|
| grad_weight 1.4e-5 -> 7.6e-6; router grad_input 4.9e-6 -> 2.5e-6 (2 pieces) | Update after review 3 | `rounding`: `2 pieces truncate` / `2 pieces half-away` |
| rounding is free compiled, +4-9% eager backward | Update after review 3 | `rounding`: `split compiled` / `split eager` rows, `half-away` % |
| fwd/bwd 53 ms (BF16x9 backward) -> 20 ms, 16 ms compiled | TLDR | `lm_head_variants_qwen3_5_27b`: `fp32 bwd (router)` and `hi+lo bwd (this PR)` training rows (BF16x9: sm100+) |
| vocab 248320, hidden 5120 | Controlled results | `lm_head_variants_qwen3_5_27b`: `hidden (2048, 5120), weight (248320, 5120)` |
| inference table (1 seq, 16 seq, logprob \|err\|) | Controlled results | `lm_head_variants_qwen3_5_27b`: `inference, LM head alone` |
| bf16 rounding of exact gradients: 1.66e-03 (dh), 1.64e-03 (dW) | Controlled results | `lm_head_variants_qwen3_5_27b`: `bf16 rounding of the exact gradients alone` |
| training table (ms, logprob \|err\|, dh err, dW err; eager and compile) | Controlled results | `lm_head_variants_qwen3_5_27b`: `training, one 2048-token chunk` |
| 2 vs 3 pieces: ~16 bits, ~7x, ~1.5x, and the table | 2 vs 3 pieces | as the docstring rows above |
| BF16x9 router backward: dh 1.66e-03 vs 1.70e-03, 2.7x slower eager (53 vs 20 ms), 3.2x compiled (52 vs 16 ms) | Q&A | `lm_head_variants_qwen3_5_27b`: router and this-PR rows, eager and `(compile)` |
| Qwen3.5-35B-A3B router: 1.66e-03 for both, 99.5% bitwise equal, 44.8 -> 14.4 ms | Q&A | `router_gate_comparison`: `65536 tokens` block |
| FSDP keeps fp32 grad_weight: 2.7e-07 vs the 1.66e-03 bf16 floor | Code changes | `fsdp_fp32_grad`: `eager LM head (wide)` |
| a bf16 .grad costs +1.6 ms per Qwen3-8B head chunk | Code changes | `backward_memory`: `bf16 .grad +... ms` under `Qwen3-8B head chunk` |

### Extra numbers, not in the PR yet

| number | script: line |
|---|---|
| round vs truncate vs ties-to-even: errors (both layouts, 2 and 3 pieces) and eager / compiled cost | `rounding` |
| 2 vs 3 pieces, all columns | `docstring_table` |
| stack vs split, with and without `addmm(out=)`, 2k-64k tokens, Qwen3-8B and Qwen3.5-27B heads | `grad_weight_layouts` |
| compiled vs eager split: kernels and GPU time, head and router | `split_compile`: `split + cat alone` |
| split time with static or symbolic T and O | `split_compile`: `static or symbolic dims` |
| one head + loss chunk, compiled vs eager split, CE eager and compiled | `split_compile`: `LM head + loss chunk`; `split_compile_TORCH_COMPILE_DISABLE` runs the shipped code with `TORCH_COMPILE_DISABLE=1` |
| memory of each temporary, and measured backward peaks | `backward_memory` |
| bf16 vs fp32 GEMM throughput | `gemm_throughput` |
| % of grad_input correctly rounded | `docstring_table` (`rounded ok`), `rounding`, `split_k` |

## What a script can't regenerate

- **The Validation section (Qwen3-30B-A3B, 2 nodes x 4 GB300):** these are training runs. Their config isn't in torchtitan. The recipe: random init, c4, seq 4096, PP2 (1F1B, stages of 25/23 layers, lm_head on stage 1) x DP4, EP4, HybridEP + CUDA graphs, no AC, forced balanced routing, MBS4 x 16 microbatches (1,048,576 tokens/step), 100 and 300 steps, seeds 42 and 777. Four arms differ only in the lm_head (bf16 `Linear`, `CastLinear`, `FP32OutputLinear` via `LMHeadFP32OutputConverter`) and in whether the checkout ships `RouterGateLinear` or `FP32OutputLinear` for the gate. tps is the median over steps 21-300.
- **The fused split kernel's 2.4e-6 (`_wide_backward` TODO):** an sm90-only prototype kernel on a separate branch, not part of this PR.
- **The exact error values in the PR:** they were measured on hidden states of local notes, which aren't public. `prepare_data.py` uses the public c4 sample instead, with bitwise the same lm_head weights. On the old hidden states, these scripts print the PR's errors digit for digit (H100); on c4, errors differ by up to 1.6x, e.g. LM-head grad_input 2.9e-4 (old text) vs 2.1e-4 (c4). `prepare_data.py` is bitwise deterministic across runs on one machine; another GPU may change the last bits of the hidden states.
- **GB300 timings on other GPUs:** the scripts regenerate the PR's GB300 tables only on GB300/B300. On H100 they print H100 times, and skip the BF16x9 rows.
