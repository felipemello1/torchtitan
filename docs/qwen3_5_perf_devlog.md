# Qwen3.5 hill-climb devlog (training + inference)

Goal (from Felipe): profile OSS Qwen3.5 in TorchTitan, in training and in the vLLM generator, and hill-climb until told to stop. Custom kernels are allowed; they must be robust at low and high context. Newest entries first. Each entry: what changed, the measurement, keep or drop.

Setup: 1x GB300 (SM103), TorchTitan `base/upstream-07b97cb` plus the draft PRs, vLLM nightly `0.1.dev1+g9d88ceb02`, torch `2.15.0.dev20260908+cu130`, attn-gym 0.0.13 plus local patches (overlays in `scratch/gen-compile-bench/site_*`). Harness: `investigations/generator_compile_bench/`.

## Baselines

Generator, decode ms/token (2k bs1 / 2k bs16 / 32k bs1 / 32k bs16), unseeded sampling (TorchTitan's RL default), all generator PRs stacked (`gen-parity-dev6`):
```text
4B   Titan        3.56  3.91  3.80  6.11      native V2 3.28 3.96 3.51 6.19    native V1 3.38 4.04 3.69 6.31
27B  Titan       11.51 12.55 11.89 16.14      native V2 11.15 12.93 11.52 16.47  native V1 11.29 13.06 11.64 16.57
```
Titan must run vLLM's V1 runner, which costs native 0.1–0.19 ms/step. Against native on the same V1 runner, Titan is at 95 / 103 / 97 / 103% (4B) and 98 / 104 / 98 / 103% (27B). The V2-runner port is #61 (entry 08:00).

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

### 2026-09-25 16:45 — generator: TP-compatible GDN fusion (#64 supersedes #57/#59)
- **Problem:** #57's fused `[q|k|v]` / `[z|a|b]` parameters can't be colwise-sharded, so the model rejected TP>1.
- **Fix, #64:** keep the per-projection model, so trainer and generator share one definition. The generator lays each rank's local shards out contiguously (parameters become row views) and runs one GEMM and one conv.
- **Result, 4B:** 2.654 / 3.317 / 2.926 / 5.318 ms/token vs 2.683 / 3.281 / 2.947 / 5.081 for #57 + #59, and 2.650 / 3.929 / 2.966 / 6.279 unfused. Text identical.
- **Validation:** CPU tests shard like TP=2. A real TP=2 run needs a multi-GPU node.

### 2026-09-25 16:10 — generator: TRT-LLM cubins, host overhead, structured-log flushes (keep)
Full write-up: `investigations/generator/GENERATOR_ANALYSIS_AND_PLAN_20260925.md`.
- **The 32k:2048 "gap" was prefill-only padding.** #61 now caps V2's auto capture sizes. 27B prefill (bs1, ms): Titan 51.9 / 108.6 / 345.7 / 142.5 / 250.5 vs native 58.0 / 117.3 / 356.0 / 152.3 / 254.4.
- **Long decode (27B, 4096 tokens):** within 0.6% of native at bs1, 4-5% faster at bs16.
- **flashinfer-cubin.** vLLM skipped TRT-LLM attention because NVIDIA's artifactory is blocked, then synced `seq_lens.cpu()` every decode step.
  - With the 0.6.18.post1 cubin wheel from FlashInfer's GitHub release, 27B Titan decode goes 11.21 / 12.31 / 11.55 / 15.83 -> 10.84 / 11.89 / 11.18 / 14.27 ms/token, and prefill 32k:2048 goes 249 -> 159 ms.
  - Native with cubins: 10.78 / 12.50 / 11.13 / 14.80.
  - FairTitan #21 (with FA4 b32 and flashinfer-jit-cache).
- **Generator host overhead (#62).** detokenize=False, flat_logprobs, and one span per burst: 4B bs64 +10%, bs256 +26%.
- **Structured logger (#63).** Flush at most once per second, cache relpath: 565 -> 30 us per span on NFS. 4B bs64 with logs on NFS: 3.9k -> 9.4k tok/s.
- **#61 now warms up FlashInfer GDN prefill** during profiling.
- **Rejected:**
  - Engine core in its own process: +1.7% at 4B bs256.
  - FlashInfer mm_bf16 low-M GEMMs: about 1% at 27B bs1.
  - Inductor PDL for Titan: about 0.6%.

### 2026-09-25 12:45 — generator prefill: FlashInfer GDN kernel on V2, no prefill-sized graphs (keep, #61)
- Finding: decode was at parity, but bs1 prefill (time to one token) was 35–64% slower than native at 27B. Measured with the `bench.py --prefill-sweep cached:new` mode, which is new. From a profile of a 2048-token prefill (`--profile-prefill`):
  - on V2, packed GDN runs eagerly, and Titan's path launches ~12 kernels per layer plus Python. Attention Gym's jit-cache `make_runtime_key` / `_canonicalize` costs ~0.3 ms per call under the profiler. The GPU idles ~50 ms per prefill.
  - Attention Gym's chunk kernels take 13.7 ms on the GPU, against 4.1 ms for FlashInfer's single Blackwell GDN kernel (which native uses).
  - the auto capture sizes (up to `max_num_batched_tokens`) pad prefill batches to the next power of two, although V2 never replays a graph for prefill. No `cudaGraphLaunch` shows up in prefill traces, for Titan or native.
- Change:
  - on V2 on Blackwell, prefill uses vLLM's `fused_post_conv_prep` + FlashInfer's `chunk_gated_delta_rule` (the native kernels), fed only the real requests and tokens
  - `graph_prefill=False` caps V2's auto capture sizes at `max_num_seqs`
- Why the real-count slicing matters: FlashInfer pays a work tile for each empty interval. At T=512, 73 capacity intervals take 817 us against 46 us for 2. The first prototype passed full capacity and was slower than Attention Gym (2048 tokens: 216 ms against 187 ms).
- Result, 27B, ms (cached prefix : new tokens):
  ```text
                          0:512  0:2048  0:8192  32k:512  32k:2048
  before                   91.6   187.0   416.4   238.4    351.8
  + FlashInfer prefill     49.3   117.0   348.2   133.1    276.2
  + sizes <= 144           49.0   107.5   341.2   140.7    245.0
  native                   57.0   117.5   356.0   145.7    251.2
  ```
  At 4B: 55.6 / 110.3 / 136.9 / 149.5 / 153.0 -> 37.3 / 75.4 / 108.9 / 109.1 / 107.6, against native 43.1 / 85.9 / 113.7 / 116.7 / 122.0.
- Decode is unchanged across processes.
- 27B prompt logprobs (512 / 2048 / 6000 tokens of real text):
  - mean |Δ| between Attention Gym and FlashInfer prefill: 0.011
  - FlashInfer vs native: 0.015, against 0.016 for Attention Gym vs native
- Not done: Attention Gym's host overhead itself (`make_runtime_key` on every call). It also makes the trainer's forward host-bound at 16k tokens.

### 2026-09-25 11:10 — trainer: FA4 >= 4.0.0b32 for hd256 packed attention (keep)
- Finding: the FairTitan runtime pins FA4 at git `0f3fb00` (2026-09-11). There, the dedicated hd256 kernels (Qwen3.5 attention) size their grid by packed total tokens x number of documents, so packed batches launch mostly empty tiles. Upstream fixed this in flash-attention #2807 (2026-09-14), released in `flash-attn-4` 4.0.0b32.
- Attention fwd+bwd, hd256, 16 q / 4 kv heads (`/tmp/felipemello/hc/cudnn_varlen.py`):
  ```text
  docs x len     FA4 0f3fb00   FA4 b32   cuDNN (aten._cudnn_attention_forward, ragged)
  4 x 4096        2.88 ms      2.27      2.19
  16 x 4096      14.69         8.57      8.62
  64 x 1024      25.94         3.88      4.32
  1 x 65536     104.3        106.9     101.0
  ```
- With b32, FA4 and cuDNN are within ~10% everywhere, so no backend change is needed. cuDNN's forward is up to 3.6x faster on skewed lengths (e.g. 63 x 200 + 1 x 3784: 0.17 vs 0.62 ms), but its backward is slower. So fwd+bwd is about equal, and cuDNN is ~10% better under FullAC (which runs the forward twice).
- One 4B layer cycle, 16k tokens: attention 2.90 -> 2.24 ms (4 docs), 2.74 -> 1.04 ms (16 docs); cycle -1.6% / -3.5%.
- Full trainer step, 4B, 16k tokens of packed c4_test, FullAC, compile, 1 GPU (`hc_recipes.qwen35_4b_text_16k`, `trainstep.sh`): 28.7k -> 29.6k tokens/s (+3.0%). Losses match bitwise for 20 steps with `--debug.seed 42`.
- Caveat: with `max_num_documents` set (fixed-shape metadata for training CUDA graphs), `max_seqlen` is the context length, so the grid stays inflated even on b32 (26.1k vs 27.9k tokens/s in the same test). RL recipes leave it unset.
- Keep: README minimum bumped to b32 in #60. The FairTitan install script (`launcher/rl/install_titan_rl.sh`) still pins `0f3fb00`. b32 runs with the pinned apache-tvm-ffi 0.1.11, although its metadata asks for >= 0.1.12.

### 2026-09-25 11:10 — trainer: full-step profile, 4B, 16k (findings)
Profiled step with FA4 b32: 576 ms wall, 90% GPU busy.
- GEMM 257 ms (at cuBLAS's practical peak, 1.8-2.0 PFLOP/s; inductor max-autotune finds nothing better)
- chunk GDN 96 ms
- eager elementwise 46 ms
- inductor 44 ms
- optimizer 30 ms
- attention 20 ms
- memcpy 15 ms
- conv 14 ms

Where the rest goes:
- Idle GPU (44 ms): all of it is in the forward, because the forward is host-bound at 16k tokens. Per layer, FSDP `pre_forward` takes ~1.5 ms of host time (per-parameter casts and copies, even at world size 1) and the attn_gym op wrappers ~1.5 ms, against ~3 ms of GPU work per layer. This disappears at the RL shape (27B, 64k tokens), where GPU work per layer is ~16x larger.
- Chunked-loss lm_head gradient (13.5 ms, 2.3%): with gradient sync off for chunks 0..6, FSDP accumulates each chunk's bf16 lm_head gradient into an fp32 buffer with `aten::add_` (fp32 += bf16). The mixed dtypes make it a non-vectorized kernel at ~3.3 TB/s, split into two launches because the tensor exceeds 32-bit indexing: 7 chunks x 2 x 0.965 ms. Possible fix: accumulate inside the weight-gradient GEMM (beta=1, fp32 C). At 27B/64k this is ~0.4% of the step, so not pursued yet.
- FSDP at world size 1 (~38 ms): bf16 casts (729 launches, 14 ms), `chunk_cat` into fp32 (11 ms) and DtoD memcpy (13 ms). These are needed at 8 GPUs.

### 2026-09-25 10:30 — generator: cheaper GDN metadata build; final medians (keep)
- Finding: a host-attributed profile (Python stacks) at 4B 32k bs16 puts Titan's `TorchTitanGDNAttentionMetadataBuilder.build` at 2.2 ms/step profiled, against 0 for vLLM's GDN builder on these ops. It launched ~9 small GPU ops per decode step, most only needed by the packed path.
- Change (#58): on the single-token path, skip the query_start_loc copy/fill and `split_decodes_and_prefills`, and write `has_initial_state` with one `torch.gt(seq_lens, 1, out=...)`.
- Result, pinned, 2 processes: 4B 3.05-3.15 / 3.69-3.71 / 3.31-3.39 / 5.91-5.94 -> 2.96-2.97 / 3.64 / 3.21 / 5.85-5.89 ms.
- Final medians against native V2: 4B 110 / 108 / 109 / 105%; 27B 99.7 / 105 / 100 / 104% (5 Titan processes, 2 native).
- Variance: at 27B bs1, 1 of 5 CuMem-pool processes was slow (11.43 vs ~11.16 ms). Five processes without the CuMem pool were all 11.11–11.16. The CuMem pool is needed for RDMA weight transfer, so this is a caveat, not a fix.

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
- Unfusing the residual `addmm` for every batch size: bs16 regresses (27B 12.69 -> 12.91 ms at 2k/bs16), and in training it is neutral (28.41 vs 28.59 ms per 27B block). The kept version is gated to the single-token graph and adds `realize_reads_threshold=1` (entry 09:10).
- `pattern_matcher=False` for the generator: bs1 slightly better, bs16 worse (loses other fusions). Dropped.
- vLLM's hand-written custom ops (`custom_ops=all`) on native are 1–3% slower than inductor.
- Single `[q|k|v|z|a|b]` GEMM in the model: +5% (27B) / +9% (4B) per GDN layer in training. Kept generator-only (#59).

### Open
- CuMem-pool variance at bs1: about 1 in 5 processes decodes ~2.5% slower at 27B bs1 (11.43 vs 11.16 ms). A possible follow-up is a separate MemPool without expandable segments for the weights.
- GQA-native Blackwell GDN backward: the `repeat_interleave` of q/k and the FP32 vector-gate materialization still cost 1.09 ms per 4-layer cycle at 4B (2.7%). Removing them needs CuTe kernel changes in attn_gym (`chunk_kda_bwd_wy_dqkg`, intra, recompute_aqk).
