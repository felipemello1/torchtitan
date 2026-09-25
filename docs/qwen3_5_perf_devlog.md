# Qwen3.5 hill-climb devlog (training + inference)

Goal (from Felipe): profile OSS Qwen3.5 in TorchTitan, in training and in the vLLM generator, and hill-climb until told to stop. Custom kernels are allowed; they must be robust at low and high context. Newest entries first. Each entry: what changed, the measurement, keep or drop.

Setup: 1x GB300 (SM103), TorchTitan `base/upstream-07b97cb` plus the draft PRs, vLLM nightly `0.1.dev1+g9d88ceb02`, torch `2.15.0.dev20260908+cu130`, attn-gym 0.0.13 plus local patches (overlays in `scratch/gen-compile-bench/site_*`). Harness: `investigations/generator_compile_bench/`.

## Baselines

Generator, decode ms/token (2k bs1 / 2k bs16 / 32k bs1 / 32k bs16), unseeded sampling (TorchTitan's RL default), all generator PRs stacked (`gen-parity-dev6`):
```text
4B   Titan        3.56  3.91  3.80  6.11      native V2 3.28 3.96 3.51 6.19    native V1 3.38 4.04 3.69 6.31
27B  Titan       11.51 12.55 11.89 16.14      native V2 11.15 12.93 11.52 16.47  native V1 11.29 13.06 11.64 16.57
```
Titan must run vLLM's V1 runner, which costs native 0.1–0.19 ms/step. Against native on the same V1 runner, Titan is at 95 / 103 / 97 / 103% (4B) and 98 / 104 / 98 / 103% (27B). A V2-runner port is being investigated separately.

Trainer, one Qwen3.5 GDN layer (fused projections, #57), fwd+bwd, 16k tokens in 4 documents, torch.compile, bf16 (`gdn_train_profile.py`):
```text
4B   6.54 ms GPU:  chunk GDN 2.39 | projection GEMMs 2.16 | copies/adds 0.82 | causal conv 0.51 | g/beta bwd + cat 0.30 | reductions 0.24 | l2norm 0.11
27B 12.07 ms GPU:  projection GEMMs 5.81 | chunk GDN 3.54 | copies/adds 1.24 | causal conv 0.62 | g/beta bwd + cat 0.47 | reductions 0.26 | l2norm 0.11
```
Most of the "copies/adds" and "reductions" come from attn_gym's Blackwell GDN backward (`attn_gym/linear/gdn/impl/chunk.py`, `_finish_chunk_gdn_bwd`). It reuses KDA's vector-gate kernels, so for grouped heads (Qwen3.5: 2 value heads per key head at 4B, 3 at 27B) it:
- `repeat_interleave`s q and k (2 copies, 0.23 ms at 4B)
- materializes an FP32 `[T, H, K]` vector gate (0.13 ms)
- adds the WY and intra FP32 dq/dk pieces (0.23 ms)
- sums each key head's group (0.20 ms)
- casts to bf16 (0.06 ms)

The Blackwell path is still faster than attn_gym's Triton GDN backward: 6.60 vs 6.81 ms (4B), 12.06 vs 12.47 ms (27B) (`gdn_bwd_path_bench.py`).

## Entries

### 2026-09-25 10:30 — generator: cheaper GDN metadata build; final medians (keep)
- Finding: a host-attributed profile (Python stacks) at 4B 32k bs16 puts Titan's `TorchTitanGDNAttentionMetadataBuilder.build` at 2.2 ms/step profiled, against 0 for vLLM's GDN builder on these ops. It launched ~9 small GPU ops per decode step, most only needed by the packed path.
- Change (#58): on the single-token path, skip the query_start_loc copy/fill and `split_decodes_and_prefills`, and write `has_initial_state` with one `torch.gt(seq_lens, 1, out=...)`.
- Result, pinned, 2 processes: 4B 3.05-3.15 / 3.69-3.71 / 3.31-3.39 / 5.91-5.94 -> 2.96-2.97 / 3.64 / 3.21 / 5.85-5.89 ms.
- Final medians against native V2: 4B 110 / 108 / 109 / 105%; 27B 99.7 / 105 / 100 / 104% (5 Titan processes, 2 native).
- Variance: at 27B bs1, 1 of 5 CuMem-pool processes was slow (11.43 vs ~11.16 ms). Four processes without the CuMem pool were all 11.11–11.16.

### 2026-09-25 09:10 — generator bs=1: remove extra kernels, not slow ones (keep)
- Finding: on vLLM's V2 runner, Titan's decode step has the same GPU busy time as native (27B 2k bs1: 10.79 vs 10.73 ms). But it launches 1183 kernels per step against 1018, and the gaps between CUDA-graph nodes add ~0.26 ms per step (`gpu_gaps.py`: in-step idle 0.40 vs 0.14 ms). The extra kernels:
  - 128 memcpys: inductor's `x + mm -> addmm` copies the residual `x` first
  - 48 contiguous copies of q|k|v in front of the conv
- Fix 1: attn_gym `causal_conv1d_decode` accepts row-strided input (attention-gym#1). Titan passes the q|k|v slice in place and copies only for prefill (#59).
- Fix 2: unfuse residual `addmm` only in the single-token graph (#56). vLLM `compile_sizes=[1]` plus a post-grad pass with `is_applicable_for_range(end <= 1)`, plus `realize_reads_threshold=1`.
  - Without the realize option, each RMSNorm re-sums a chain of 4 unrealized residual adds and reads them twice, going from 2.9 to 4.8 us, which undoes the win (27B 2k bs1 11.58 ms).
  - Applying the unfuse to all sizes regresses bs16 (27B 2k bs16 12.50 -> 12.73), hence the single-token gate.
- Result (unseeded, V2 runner, text check passes):
  ```text
                                   2k bs1  2k bs16  32k bs1  32k bs16
  4B  V2 runner                     3.42    3.79     3.71     6.21
  4B  + strided decode + unfuse     3.07    3.75     3.47     6.85
  4B  native V2                     3.28    3.96     3.51     6.19
  27B V2 runner                    11.43   12.50    11.88    16.59
  27B + strided decode + unfuse    11.31   12.40    11.80    16.70
  27B native V2                    11.15   12.93    11.52    16.47
  ```
- Caveat: the 32k numbers vary by up to ~10% between processes of the same config. A repeat of the V2-runner baseline gave 4B 32k 3.83 / 6.81 against 3.71 / 6.21 the first time; FlashInfer re-autotunes in every process. Cross-process repeats are running.
- Rejected: inductor `triton.multi_kernel` breaks vLLM's compile cache (`NameError: multi_kernel_0`), and `realize_acc_reads_threshold=2` doesn't help.

### 2026-09-25 08:00 — generator: vLLM V2 model runner (keep)
- TorchTitan forced vLLM's V1 runner, which costs native 0.1–0.19 ms/step, and V1's seeded sampling launches per-request generator kernels. The V2 runner port (`rl-gen-v2-runner`) needs:
  - `TorchTitanGDNAttentionMetadataBuilder` reports `UNIFORM_SINGLE_TOKEN_DECODE` on V2, so FULL graphs are decode-only
  - `_forward` treats missing GDN metadata as a warmup run
  - V1 is kept for batch-invariant mode and multi-GPU engines
- Text matches V1 at 4B and 27B. 27B: 11.51/12.55/11.89/16.14 (V1) -> 11.43/12.50/11.88/16.59 (V2).

### 2026-09-25 07:30 — inductor: lower `cat` as ConcatKernel, not pointwise (keep)
- Finding: in the fused-projection backward, inductor lowers the `[z|a|b]` gradient `cat` as one masked pointwise kernel over `T x 4160`. Every element branches over the three column ranges and does integer div/mod, reaching about a third of bandwidth: 0.30 ms per layer at 4B, 0.47 ms at 27B (`triton_poi_fused__to_copy_cat_exp_mul_neg_sigmoid_backward_silu_backward_softplus...`).
- Change: `torch.compile(block, options={"max_pointwise_cat_inputs": 0, "max_complex_pointwise_cat_inputs": 0})`. Each gradient producer then writes straight into its slice of the concatenated buffer. The cat is exact, so numerics are unchanged.
- One GDN layer fwd+bwd (with the previous entry): 4B 6.33 -> 6.00 ms; 27B 11.71 -> 11.36 ms.
- One full layer cycle (3 GDN blocks + 1 gated-attention block, including MLPs), compiled per block like the trainer, 16k tokens (`stack_train_bench.py`):
  ```text
                                         4B        27B
  attn-gym 0.13, default inductor      42.75 ms     -
  + head-sum epilogue                  42.03 ms   108.02 ms
  + ConcatKernel cat                   40.29 ms   106.07 ms   (-5.8% vs baseline at 4B)
  separate projections, same fixes     41.24 ms     -
  ```
  With both fixes, the fused layout (#57) is now 2.3% faster than separate projections in training, where before it was neutral.
- Keep. It needs a way to pass inductor options to the per-block compile (`torchtitan/distributed/compile.py` calls `transformer_block.compile(backend=..., fullgraph=True)` with no options).

### 2026-09-25 06:40 — attn_gym: fold add + group sum + cast into one kernel (keep)
- Change: `sum_expanded_head_gradients` (Triton, `attn_gym/linear/gdn/bwd/triton/chunk_gdn_bwd_head_sum.py`). One pass reads both FP32 pieces and writes the grouped bf16 gradient, instead of an add, a group sum, and a cast. Four rows per program and 8 warps (~6 TB/s on GB300; one row per program was 4.4 TB/s). int64 offsets. Branch `gdn-bwd-fused-group-epilogue` in `libraries/attention-gym`.
- Correctness:
  - kernel test: 19 cases (T in {1, 77, 4096}; groups 1/2/3; bf16/fp16) match the reference, with at most one FP32 rounding of difference
  - attention-gym GDN backward tests (Blackwell route, grouped heads): 32 passed
- One GDN layer fwd+bwd, 16k tokens: 4B 6.54 -> 6.33 ms (-3.3%); 27B 12.09 -> 11.71 ms (-3.1%).
- Keep.

### Rejected
- Unfusing the residual `addmm` in the generator (post-grad pass) removes 128 memcpy/step at 27B. But the add then lands in a slower add+RMSNorm kernel (2.9 -> 4.8 us at bs1), and bs16 regresses: 27B 12.69 -> 12.91 ms at 2k/bs16. In training it is neutral: 28.41 vs 28.59 ms per 27B block.
- `pattern_matcher=False` for the generator: bs1 slightly better, bs16 worse (loses other fusions). Dropped.
- vLLM's hand-written custom ops (`custom_ops=all`) on native are 1–3% slower than inductor.
- Single `[q|k|v|z|a|b]` GEMM in the model: +5% (27B) / +9% (4B) per GDN layer in training. Kept generator-only (#59).

### Open
- V2 runner prototype (branch `rl-gen-v2-runner`): bs1 faster (27B 11.51 -> 11.43, 4B 3.56 -> 3.42) but 32k/bs16 slower (27B 16.14 -> 16.59), with equal GPU busy time. Host/scheduling cause not yet found. Not opened as a PR.
- GQA-native Blackwell GDN backward: the `repeat_interleave` of q/k and the FP32 vector-gate materialization still cost 1.09 ms per 4-layer cycle at 4B (2.7%). Removing them needs CuTe kernel changes in attn_gym (`chunk_kda_bwd_wy_dqkg`, intra, recompute_aqk).
