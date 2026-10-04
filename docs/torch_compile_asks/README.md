# Make torch.compile go BRRRRR

[Felipe Mello](mailto:felipemello@meta.com)

## TL;DR

- GB300 (SM103), torch 2.15.0.dev20260926+cu130, triton 3.8.0. TorchTitan with `local_compile` regions: small named functions compiled with `fullgraph=True` and the default `recompile_limit=8`, no eager fallback, the rest of the model eager.
- Models: DeepSeek-V4 flash, Kimi K3, Qwen3.5-MoE, DeepSeek-V3 671B (including DistMoE).
- For DeepSeek-V4 and Kimi K3, upstream's default regions left 40-56% of layer time on the table versus compiling whole layers. Closing the gap took rewrites Inductor could have found itself, plus guard and recompile workarounds.
- This doc lists 32 things torch.compile could not figure out on its own, as 27 requests. Each has numbers at real shapes, the workaround we used where there is one, and (for 25 of the 32) a minimal pure-torch repro in `repros/`.
- Biggest costs, in ms per 16k-token fwd+bwd microbatch: ~290 ms (Kimi K3) without a traceable deterministic scatter-add; ~240-300 ms (Kimi K3) from symbolic strides on views plus every call site running the newest, symbolic graph; up to ~230 ms (Kimi K3) from small reductions that are not unrolled.
- Root-cause themes:
  - symbolic shapes cost even when bounded: symbolic strides and offsets, int64 indexing, masked loads, no unrolling, the newest graph serving every call;
  - fixed thresholds and lowering choices with no cost model: the unroll threshold of 8, cat lowering, producer inlining, `topk`;
  - opaque custom ops and complex numbers block fusion;
  - under `fullgraph=True`, guards and graph budgets turn performance heuristics into hard failures;
  - tracing and checkpointing corner cases that crash or are silently wrong.
- First ask: performance heuristics must never install guards (request 1), plus the six other crash and silent-wrong bugs (requests 2-7). Top speed ask: a traceable fixed-order segmented sum (request 8).

Repros: `repros/` (pure torch; `./repros/run_all.sh cpu` runs the CPU ones, `./repros/run_all.sh gpu` the rest). Raw outputs: `repros/logs/`.

## Terms

- region: a function decorated with TorchTitan's `@local_compile`, bound to `torch.compile(fn, fullgraph=True, **options)` while the rest of the model runs eager.
- block compile: `torch.compile` of a whole transformer layer.
- SAC (SelectiveAC): selective activation checkpointing, i.e. `torch.utils.checkpoint` with a policy that saves some ops and recomputes the rest in backward.
- HOP: a higher-order op, a Dynamo operator that traces a function body as a subgraph (e.g. `torch.utils.checkpoint` becomes `tag_activation_checkpoint`).
- make_fx: torch's FX tracer. GraphTrainer, TorchTitan's experimental trainer, traces the whole training step with non-strict `make_fx`.
- Models: DSv4 = DeepSeek-V4 flash, DSv3 = DeepSeek-V3 671B, Kimi K3, Qwen3.5 (Qwen3.5-MoE, Qwen3.5-35B-A3B).
- Kimi residual (attention residual): Kimi K3's block-level residual, a softmax-weighted sum over a stack of N <= 8 earlier block outputs `[T, N, 7168]` plus the current partial sum `[T, 7168]` (the "partial"). 2 calls per layer, 186 per forward.
- mHC: DeepSeek-V4's hyper-connections. HcPre mixes 4 residual streams into one per token, with a 20-step Sinkhorn normalization of a `[T, 4, 4]` matrix. HcPost expands the output back to 4 streams.
- MLA: multi-head latent attention, the attention of DeepSeek-V3 and Kimi K3 (q, k and v come from low-rank projections).
- GDN: Gated DeltaNet, Qwen3.5's linear-attention layer (attn-gym kernels).
- FA4: FlashAttention 4 (CuTe DSL).
- DistMoE: the fused distributed routed-expert backend TorchTitan uses for DeepSeek-V3 671B. Its kernels are torch.library custom ops.
- EP / TP / PP: expert / tensor / pipeline parallelism. "Rank-local shapes" are one EP rank's shapes.
- WGRAD: the weight-gradient GEMMs of the backward. In-place WGRAD (TorchTitan's `inplace_wgrad_accum`, on by default) accumulates them directly into the existing `.grad`.
- BF16x9 (`torch.backends.cuda.matmul.fp32_precision = "bfx9"`): fp32 matmuls emulated with bf16 tensor-core GEMMs. TorchTitan enables it on SM100+.
- Shapes: T = tokens per microbatch (per rank), D = hidden size, F = FFN or expert hidden size, H = heads, K = routed experts per token (top-k), N = Kimi residual stack width. "16k" = 16384 tokens; fwd+bwd = forward + backward.
- packed training: several documents concatenated into a fixed-length microbatch, so T is fixed per step (in RL it varies).

## What compile leaves on the table

As compiled vs our rewrite or option, one GB300. Request numbers refer to the sections below. In this doc's code blocks, `#N` is the fork PR `https://github.com/felipemello1/torchtitan/pull/N`, and `attn-gym #10` is https://github.com/felipemello1/attention-gym/pull/10.

```text
req  issue                                          as compiled                        with rewrite / option            type
  1  mix-order heuristic installs shape guards      3 graphs                           2 graphs (non_strict_mode)       design disagreement
  2  FX graph cache ignores a custom op's fake      stale stride assert                fresh cache dir                  bug
  3  make_fx captures attribute tensor as const     rebinding ignored                  in-place update (#117)           bug (TorchTitan) / doc
  4  SAC + wrapped regions after a recompile        wrong saved outputs on             flag off (regions recomputed)    bug (crash)
                                                    recompute: RuntimeError / assert
  5  no_grad pass recompiles every region           +1 graph per class                 -                                feature request
     one norm region, mixed call sites, no_grad     9 graphs (limit 8)                 6 marks + option (#128)          feature request
     size-0/1 dims always specialize                +1 graph                           mark_unbacked                    doc / design
  6  new outer attr stored in a HOP: wrong error    Observed exception                 create state outside (#136)      error-message bug
  7  autograd.Fn reads a view._base/grad_dtype      BackendCompilerFailed              inplace_wgrad_accum=False        bug (crash)
  8  opaque custom op (deterministic scatter_add)   4248 us                            1102 us per-token Function       feature request
  9  packed views: symbolic stride/offset;          601 us                             361 us static F                  feature request
       newest symbolic graph serves every call
 10  small reduction not unrolled (N=8, symbolic)   1574 / 1245 us                     314 us (N=7) / 353 us thresh 9   feature request
     small non-innermost reduction off bandwidth    1354 us                            290 us thresh 17 (ideal 281)     default/feature
 11  tiny-K bmm -> extern cuBLAS + fp32 copy        3475 us                            2155 us sum / 2357 us pass       default/doc
 12  producer inlined + recomputed per element      1225 us                            278 us realize first             feature request
 13  masked loads with symbolic T (18 loads)        663 us                             314 us static T                  Triton triage
 14  pointwise-cat lowering vs ConcatKernel (a)     2155 us                            781 us ConcatKernel              feature request
       ... and the other way for (b)                1026 us                            1502 us ConcatKernel
     masked cat inside a persistent reduction       4 row re-loads                     -                                feature request
 15  symbolic size args typed int64 by design       1918 us                            1816 us assume_32bit_indexing    feature request
 16  Sinkhorn [T,4,4] realizes every step           289-316 us (159 kernels)           86 us per-entry (32 kernels)     feature request
 17  complex mul: no Inductor codegen               444 us (7 kernels)                 199 us real arithmetic           feature request
 18  region options lost when inlined               option ignored                     hand rewrite (#109, #110)        feature request (obs)
     functional region: no alias / in-place grad    419 us v-grad copy                 Triton override (DSv3)           feature request (obs)
 19  topk falls back to ATen                        97 / 378 us (4k/16k)               triton.decompose_sort_ops        default
 20  FMA contraction: compiled != eager by 1 ulp    81 / 4.2M elements                 0 (emulate_precision_casts)      doc
 21  maybe_mark_dynamic forbidden in graph          AssertionError                     guard with is_compiling()        doc
 22  functools.cache under fake tensors             eager breaks                       no cache                         doc
 23  make_fx specializes Python ints                silent wrong size                  fixed-capacity metadata          doc
 24  Dynamo/make_fx fakes forbid data_ptr           FA4 trace fails                    custom-op fake (attn-gym #10)    doc (obs)
 25  regional_inductor merges regions, tags comms   regions merge                      inductor_region key              bug / doc (obs)
 26  row gather does not exploit row reuse          649 us sequential                  - (floor ~281 us)                observation
 27  regional_inductor partitioning superlinear     382 s DSv4 4 layers                -                                perf bug (compile time)
```

## Requests, by impact

- Unit for the performance requests: ms saved per 16k fwd+bwd training microbatch of the model the request hits (rank-local shapes as measured), counting only the calls we measured. Tags in another unit say so (per call, per 16k forward, per 4k microbatch). Per-call numbers are in each section. Requests that fix the same op are marked as alternatives and do not add.
- Order: requests 1-7 are crashes and hard failures; 8-19 are performance, by the stated figure (where a request names two models, by the measured one); 20-26 are docs and observations; 27 is compile time.

For scale, estimated 16k fwd+bwd microbatch times on one GB300 (2-layer harness wall, no activation checkpointing, times the layer count; excludes embedding, head, optimizer and communication). Each line names the tree it scales:

- Kimi K3, upstream main's default regions: ~15 s (2 layers 331 ms, 324 ms with https://github.com/felipemello1/torchtitan/pull/113, request 9; EP8 rank-local shapes with 112 local experts, no EP communication; 93 layers). So ~290 ms (request 8) is ~2% of a microbatch.
- DeepSeek-V4 flash, our region stack (fork PRs 105-108, with and without 110): ~2.2-2.5 s (2 layers 103.6-114.5 ms; 43 layers; https://github.com/felipemello1/torchtitan/pull/137). On upstream main's default regions its 2 layers take 233.3 ms (https://github.com/felipemello1/torchtitan/pull/105), ~5.0 s per microbatch.
- DeepSeek-V3 671B, fused MLA override plus our regions (EP8 rank shapes): ~3.3 s at 16k and ~0.83 s at 4k (2 MoE layers 109.1 / 27.2 ms; 61 layers, its 3 dense layers counted as MoE layers; https://github.com/felipemello1/torchtitan/pull/110).

1. **[hard failure under fullgraph]** Never install guards in performance heuristics (mix-order reduction)
2. **[bug, silently stale]** Key the FX graph cache on a custom op's fake output metadata
3. **[bug, silently wrong]** make_fx: warn on, or lift, an attribute tensor captured as a constant
4. **[crash with the flag on]** SelectiveAC with `wrap_inductor_compiled_regions`: recompute is served from a different graph after a recompile
5. **[hard failure: `FailOnRecompileLimitHit`]** Graph budgets under fullgraph
6. **[hard failure under fullgraph, misleading error]** A new attribute stored on an outer-scope object inside a HOP fails with a generic "Observed exception"
7. **[crash]** An `autograd.Function` forward that reads `._base` and `grad_dtype` of a weight view crashes aot_eager and inductor
8. **[~290 ms, Kimi K3]** A traceable fixed-order segmented sum (deterministic scatter-add)
9. **[~240-300 ms, Kimi K3, fixed T]** Keep views' stride and offset static, and do not route every call site to the newest, symbolic graph
10. **[up to ~230 ms, Kimi K3]** Unroll small reductions
11. **[1.1-1.3 ms per call, Kimi K3; alternative to request 10]** Decompose memory-bound batched GEMVs by default
12. **[~175 ms per 16k no_grad forward, Kimi K3]** Inline a producer only when the consumer's broadcast factor is small
13. **[~65 ms per 16k forward, Kimi K3, RL only]** Triton: masked loads halve a many-load kernel on SM103
14. **[~59 ms, DeepSeek-V4 flash]** A cost model or autotuning for cat/stack lowering
15. **[up to ~26 ms, Kimi K3; part of request 10]** int32 indexing for bounded symbolic sizes
16. **[~20 ms, DeepSeek-V4 flash]** Fuse tiny-block reductions across steps (Sinkhorn)
17. **[~11 ms, DeepSeek-V4 flash]** Complex-multiply codegen
18. **[~10 ms Kimi K3; ~24 ms DeepSeek-V3, estimate]** Region semantics: options and aliasing
19. **[~3 ms per 4k microbatch, DeepSeek-V3 671B]** Lower small-k `topk` by default
20. **[doc]** FP contraction: compiled is not bitwise vs eager for fused pointwise math
21. **[doc]** `maybe_mark_dynamic` is forbidden in graphs
22. **[doc]** `functools.cache` under fake-tensor tracing
23. **[doc]** make_fx specializes Python ints silently
24. **[doc]** The fake-tensor `data_ptr` policy differs between `FakeTensorMode` and Dynamo / make_fx
25. **[bug / doc]** `regional_inductor` merges unkeyed regions and tags collectives
26. **[observation]** Row gathers do not exploit row reuse
27. **[compile time: 382 s per process start, DeepSeek-V4 4 layers]** `regional_inductor` partitioning scales superlinearly

---

### 1. Never install guards in performance heuristics (mix-order reduction)

- **Request:** a mode in which performance heuristics never add guards (`statically_known_true` / `guard_or_false`, taking the non-fused schedule when unknown), or a documented interaction with fullgraph and the recompile limit.
- **Estimated impact:** hard failures of fullgraph regions. Under `fullgraph=True` each guard is a graph against `recompile_limit=8`, ending in a hard `FailOnRecompileLimitHit`. The Kimi residual reached exactly 8 in the worst call order, and Qwen3.5's shared norm region 8 of 8 even with the option (request 5).
- **Type:** design disagreement: a performance heuristic becomes a hard failure under fullgraph.
- **Context:**
  1. The scheduler checks `nrow * ncol >= 5M` and `nrow >= 4096` with `evaluate_expr(..., size_oblivious=True, fallback_value=False)` (`torch/_inductor/scheduler.py:470-494`). That installs a guard on a symbolic size, so a later short batch recompiles.
  2. The guard is deliberate. The source says (`scheduler.py:476-478`): "Call evaluate_expr rather than statically_known_geq since nrow can have dynamic shape in real models. Don't use hint directly since hint can be non-representative."
  3. Our disagreement: in a `fullgraph=True` region every extra graph counts against `recompile_limit=8`, with a hard `FailOnRecompileLimitHit` and no eager fallback.
  4. The repro runs the full Kimi residual, fwd+bwd, `fullgraph=True`, at T = 16384, 8192, 2048:

```text
default:                                    3 graphs; guard failure "4096 <= stack.size()[0] <= 268435456
                                            # (_inductor/scheduler.py:492 in can_fuse)"
triton.mix_order_reduction_non_strict_mode: 2 graphs
```

- **Workaround** (https://github.com/felipemello1/torchtitan/pull/109): region-scoped `options={"triton.mix_order_reduction_non_strict_mode": True}`. Time-neutral (16k fwd+bwd 4801.6 vs 4800.9 us; decode T=1 8.8 vs 8.3 us); worst case 8 -> 6 graphs.
- **Repro:** `python repros/01_mix_order_guards.py` (GPU). Expected: the block above.

### 2. Key the FX graph cache on a custom op's fake output metadata

- **Request:** include the fake's output metadata (or the op library's version) in the FX graph cache key.
- **Estimated impact:** silently stale code. Fixing a fake keeps running code compiled from the broken one.
- **Type:** bug.
- **Context:**
  1. The repro has one custom op whose fake v1 claims the input's transposed strides (like a fake built on `empty_like(input)`), while the real kernel returns contiguous output. Fake v2 fixes it.

```text
fake v1, fresh cache A  -> AssertionError: expected size 32==32, stride 64==1 at dim=0; expected size 64==64, stride 1==32 at dim=1
fake v2, same cache A   -> the same AssertionError (code compiled from fake v1 is reused)
fake v2, fresh cache B  -> ok
CPU: TORCHINDUCTOR_AUTOGRAD_CACHE=0 (FX graph cache on) -> stale; TORCHINDUCTOR_FX_GRAPH_CACHE=0 -> v2 ok
```

  2. Inductor's FX graph cache key covers the graph but not the custom op's fake output metadata (the strides behind the generated `assert_size_stride`). So fixing a fake keeps running code generated from the old one until `TORCHINDUCTOR_CACHE_DIR` changes. The CPU variants isolate the FX graph cache.
  3. Found when fixing attn-gym's GDN fakes (https://github.com/felipemello1/attention-gym/pull/9).
- **Workaround:** a fresh `TORCHINDUCTOR_CACHE_DIR` per variant when testing fakes.
- **Repro:** `python repros/02_cache_ignores_fake.py` (GPU; the cache behavior also reproduces on CPU). Expected: the block above.

### 3. make_fx: warn on, or lift, an attribute tensor captured as a constant

- **Request:** warn when a traced graph captures a non-parameter, non-buffer tensor as a constant, or lift it to an input.
- **Estimated impact:** silently wrong results. This was a live TorchTitan GraphTrainer bug.
- **Type:** bug in TorchTitan's GraphTrainer path; doc / feature request for make_fx (tracing, not Inductor).
- **Context:**
  1. The repro's loss divides by a tensor reached through a class attribute:

```text
graph constants: ['_tensor_constant0']
traced, denominator 2:          2.0
after in-place fill_(4):        1.0   eager 1.0   (in-place update is seen)
after rebinding to tensor(8):   1.0   eager 0.5   (rebinding is silently ignored)
```

  2. TorchTitan rebinds the denominator every step: `AuxLoss.set_step_denominator` assigns `cls._step_denominator = denominator` (`torchtitan/models/common/aux_loss.py:165`), called with a fresh tensor each step (`torchtitan/training_engine.py:470-482`). That is the silently ignored case: a GraphTrainer graph traced at step 1 keeps dividing by step 1's token count (in the DeepSeek-V3 graph: `_tensor_constant0 -> reciprocal`).
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/117): keep one persistent denominator tensor and update it in place each step.
- **Repro:** `python repros/03_make_fx_tensor_constant.py` (CPU). Expected: the block above.

### 4. SelectiveAC with `wrap_inductor_compiled_regions`: recompute is served from a different graph after a recompile

- **Request:** replay in recompute the callable recorded in forward, or key saved outputs by call order within the checkpointed region rather than per graph.
- **Estimated impact:** a crash, or wrong saved outputs on recompute. With the flag on, every TorchTitan MoE model we ran crashed at step 1. Keeping the flag off is not free either: compiled regions are then recomputed under SAC instead of saved.
- **Type:** bug (crash).
- **Context:**
  1. The repro calls one `torch.compile` function at two shapes inside `checkpoint(..., context_fn=create_selective_checkpoint_contexts(policy))`, with a policy that saves `inductor_compiled_code` outputs (`MUST_SAVE`) and `torch._inductor.config.wrap_inductor_compiled_regions = True`:

```text
cold (the second call recompiles inside the checkpointed forward):
    RuntimeError: inductor_compiled_code invocation index 1 encountered during backward but not found in storage
warm (both graphs compiled before the checkpoint):
    OK, grad matches no-checkpoint: True
```

  2. SAC keeps one FIFO of saved compiled-region outputs per compiled callable (`torch/utils/checkpoint.py:1554-1568`, `_sac_storage_key`), i.e. per Dynamo cache entry.
  3. In forward, call 1 runs graph G1 (static) and call 2 compiles G2 (automatic dynamic). In recompute, Dynamo serves call 1 from the newest matching entry (request 9), G2, which pops call 2's saved outputs. Call 2 then finds G2's queue empty.
  4. General condition: any cache entry added between a checkpointed forward and its recompute. Examples: the automatic-dynamic recompile on step 1, a new token count, a later layer's recompile.
  5. In TorchTitan's MoE the routed experts (`[T*K, F]`) and the shared expert (`[T, F]`) call one activation region per block. So the wrong `[T, F]` tensor reaches the next grouped GEMM before the RuntimeError can fire: `GroupMMCommon.cuh:89` `offset <= tensor_ShapeA[0]`.
  6. With the flag on, every variant we ran crashed at step 1 before writing a result row: DeepSeek-V3, DeepSeek-V4, Kimi K3 and Qwen3.5-35B-A3B, each with its regions.
     - The crash is in SAC's backward recompute (`checkpoint.py unpack_hook -> recompute_fn`).
     - The error: `GroupMMCommon.cuh:89 prepare_grouped_gemm_data: Assertion offset <= tensor_ShapeA[0]`.
     - Trees: DeepSeek-V4 on https://github.com/felipemello1/torchtitan/pull/105 to https://github.com/felipemello1/torchtitan/pull/108 plus https://github.com/felipemello1/torchtitan/pull/110; Kimi K3 on https://github.com/felipemello1/torchtitan/pull/109 plus https://github.com/felipemello1/torchtitan/pull/110; Qwen3.5-35B-A3B on https://github.com/felipemello1/torchtitan/pull/110.
- **Workaround:** none practical. TorchTitan keeps the flag off, so compiled regions are invisible to SAC and are recomputed under its policy instead of being saved. The SAC rows of every region PR above include the regions' recompute.
- **Repro:** `python repros/04_sac_wrapped_region_recompile.py cold` and `... warm` (GPU; add `--cpu` to run on CPU). Expected: the block above; raw output in `repros/logs/04_sac_wrapped_region_recompile.txt`.

### 5. Graph budgets under fullgraph

- **Request:**
  - let a no_grad call reuse the grad-mode graph's forward, or count signatures per grad mode;
  - per-call-site graph budgets, or a recompile limit that does not count shape-class x grad-mode products against one code object;
  - make one escape hatch practical for regions known to see sizes 1..N, or document the per-signature budget arithmetic.
- **Estimated impact:** hard failures. A no_grad pass doubles every region's graphs. One region shared by call sites of different rank or layout reaches 8 of 8 with the region option alone, and 8 with eager-side marks alone; only both together reach 6.
- **Type:** feature request (grad mode, shared call sites); doc / design (size-0/1 specialization).
- **Context, (a) grad mode is a global-state guard, so validation recompiles every region:**
  1. Repro:

```text
after training at 3 token counts:  graphs = 2      (static T, then dynamic T)
after a no_grad validation pass:   graphs = 3      guard failure "GLOBAL_STATE changed: grad_mode"
5 ranks x 2 grad modes:            FailOnRecompileLimitHit at signature 9 (recompile_limit=8)
```

  2. Every region pays one more graph per shape class once validation runs under `torch.no_grad()`.
  3. On the defaults of https://github.com/felipemello1/torchtitan/pull/97, DeepSeek-V4 with `complex_rope` forced on (key None vs tensor, inverse, 1 vs 64 heads, dynamic T) completes training with 8 RoPE graphs and fails at the first no_grad pass. On the original recipe of https://github.com/felipemello1/torchtitan/pull/106 it already failed at the third training shape.
- **Context, (b) one norm region shared by call sites of different rank and layout runs out of graphs:**
  1. The repro runs Qwen3.5's `OffsetRMSNorm.forward` as one fullgraph region. It is used for the 2D layer norms `[T, 2048]` and the 3D q/k norms `[T, H, 256]` (q is a strided `chunk` view, k contiguous), with one attention and one GDN block per step: training at T = 16k, 8k, 2k, then no_grad at 16k, 2k. The limit is raised to count graphs.

```text
model level, Qwen3.5-35B-A3B layers, regions = offset_rmsnorm, training 16k/8k/2k then no_grad 16k/2k:
  upstream base                         graphs 0/0 .. 0/7, then FailOnRecompileLimitHit
  + region option non_strict_mode       8 graphs: at the limit, no headroom
  + eager-side marks (T dynamic,
    normalized dim static) + option     6 = 3 layouts x 2 grad modes                       (#128, 48280e292)
repro 05c, CPU aot_eager (Dynamo-level guards only; logs/05c_shared_norm_call_sites_cpu.txt):
  strided q 7 graphs | contiguous q 5 | eager-side marks 6 (non_strict changes nothing without Inductor)
repro 05b, GPU Inductor (adds the mix-order guards, e.g. `x.size()[1]*x.size()[0] >= 5242880`, where 5242880 = 5 * 2**20 is the
  mix-order size threshold, `scheduler.py:474`; the log attributes the guard to `autograd_cache.py:141` because it was replayed
  from the warm cache; logs/05b_shared_norm_call_sites.txt):
  strided q 9 graphs | non_strict 7 | contiguous q 7 | both 5 | eager-side marks 8 | marks + non_strict 6
  (the over-limit count needs Inductor's mix-order guards: CPU aot_eager stops at 7)
```

  2. Graphs come from rank (2D vs 3D), strides (q view vs k), dynamic T, the mix-order `nrow * ncol >= 5M` guards (`s1*s0 >= 5242880`, request 1) and grad mode (part (a)).
  3. In TorchTitan this is upstream's `offset_rmsnorm` region: the Qwen3.5 budget probe hit `FailOnRecompileLimitHit` there.
- **Context, (c) size-0 and size-1 dims always specialize:**
  1. Repro:

```text
default, widths 1,2,3,5,8,1:                            2 graphs (N=1 static, then one dynamic-N graph)
maybe_mark_dynamic on the size-1 dim, widths 1 then 4:  2 graphs (the size-1 dim still specializes)
mark_unbacked on the width, widths 1 then 4:            1 graph
```

  2. A region whose stack width varies 1..8 keeps a separate N = 1 graph even after N goes dynamic, and `maybe_mark_dynamic` does not change that. `torch._dynamo.mark_unbacked` does (1 graph), but it must be applied at every call site and disables other specializations on that dim. `torch.fx.experimental._config.backed_size_oblivious` is the global alternative.
  3. Size 0 specializes too: a routed MoE call with 0 or 1 rows (an EP rank receiving no tokens; PP shape inference on uninitialized inputs) costs one graph per grad mode. That is a more common trigger in MoE training than a stack width of 1.
  4. Measured graph accounting for the Kimi residual (all 93 layers' widths): training at two token counts uses 4 graphs ({N = 1, dynamic N} x {first T static, then dynamic}). A no_grad pass adds 2 (part (a)), and request 1's `T >= 4096` guard adds up to 2 more, reaching the limit of 8. With the mix-order option, 6.
- **Workarounds:**
  - (a) https://github.com/felipemello1/torchtitan/pull/106 / https://github.com/felipemello1/torchtitan/pull/107: enable `complex_rope` for DeepSeek-V4 only together with the glue regions that absorb most of its call signatures. Budgets were measured with a no_grad pass for every region (all <= 6 of 8).
  - (b) https://github.com/felipemello1/torchtitan/pull/128 (48280e292). Neither fix alone is enough on GPU: the region option alone gives 8 of 8 at model level, and the eager-side marks alone give 8 in the repro. So https://github.com/felipemello1/torchtitan/pull/128 does both:
    - the eager `forward` marks T dynamic and the normalized dim static (guarded by `is_compiling()`, request 21);
    - it calls the region with `triton.mix_order_reduction_non_strict_mode`;
    - that leaves one graph per layout and grad mode (6). A contiguous q (an eager copy) would save one more.
  - (c) https://github.com/felipemello1/torchtitan/pull/109: one call signature (the partial is always a tensor) plus request 1's option.
- **Repros:** `python repros/05a_grad_mode_guard.py` (GPU), `python repros/05b_shared_norm_call_sites.py` (GPU), `python repros/05c_shared_norm_call_sites_cpu.py` (CPU), `python repros/05d_size1_specialization.py` (GPU). Expected: the blocks above.

### 6. A new attribute stored on an outer-scope object inside a HOP fails with a generic "Observed exception"

- **Request:** raise the "HOP: Non-nullified side effect" error for a store to a new outer-scope attribute too, naming the attribute. Optionally, document that lazy-init caches must be created outside checkpointed compiled code.
- **Estimated impact:** a hard failure under `fullgraph=True` with a misleading message. With `fullgraph=False`, the whole enclosing function runs eagerly.
- **Type:** error-message bug.
- **Context:**
  1. The repro stores to objects defined outside the body of `torch.utils.checkpoint` (the `tag_activation_checkpoint` HOP), on CPU with `backend="eager"` and `fullgraph=True` unless noted:

```text
lazy init, fresh threading.local, top level                        OK
read-only getattr default, fresh threading.local, in HOP           OK
lazy init, fresh threading.local (closure), in HOP                 Unsupported: Observed exception | AttributeError("'_local' object has no attribute 'mesh_stack'")
store new attr, global threading.local, in HOP                     Unsupported: Observed exception | AttributeError("'_local' object has no attribute 'other'")
store new attr, global plain object, in HOP                        Unsupported: Observed exception | AttributeError("'Plain' object has no attribute 'other'")
store new attr, plain object (closure), in HOP                     Unsupported: Observed exception | AttributeError("'Plain' object has no attribute 'other'")
store new attr, non-constant value (tensor), in HOP                Unsupported: HOP: Unsafe side effect
store existing attr, global threading.local, in HOP                Unsupported: HOP: Non-nullified side effect
lazy init, threading.local subclass with __init__ default, in HOP  OK
lazy init, fresh threading.local, in HOP, fullgraph=False          OK (frames compiled: 0)
```

  2. The store raises, not the `getattr`. A body with no `getattr` fails the same way, and it is not specific to `threading.local`: plain global and closure objects fail identically.
  3. Mechanism: inside a HOP, Dynamo defers a constant attribute store on an outer-scope object and first snapshots the attribute's original value (`snapshot_attr_mutation`, `torch/_dynamo/side_effects.py:335-366`, called from `store_attr` at `:561`). For a new attribute that read raises AttributeError. Only `NotImplementedError` is caught, so it escapes as the generic graph break "Observed exception" (gb0088), with `AttributeError("'_local' object has no attribute 'mesh_stack'")`.
  4. The generic message appears only when the stored value is a constant (an empty list counts): `store_attr` reaches `snapshot_attr_mutation` only for constants (`side_effects.py:549-561`). A non-constant value, such as a tensor, gets "HOP: Unsafe side effect" (row 7).
  5. A store to an existing attribute gets the intended error, "HOP: Non-nullified side effect" (`side_effects.py:368-383`). That restriction is fundamental by design: only mutations undone before the subgraph exits are supported. So the general trigger is any store to an outer-scope attribute that outlives the HOP; the bug is that a new attribute gets a misleading message.
  6. With `fullgraph=False`, Dynamo does not just break at the checkpoint call: it falls back to eager for the whole enclosing function (0 frames compiled in the last row).
  7. Where we hit it: TorchTitan's `_spmd_mesh_stack()` (`torchtitan/distributed/spmd_types.py:142-147`, before https://github.com/felipemello1/torchtitan/pull/136) lazily stores a list on a module-level `threading.local`. The aux loss's `spmd_local_context` reaches it inside SelectiveAC-inside-compiled-region prototypes, on the first call in a fresh process or thread. TorchTitan's trainer is unaffected: `activate_spmd` creates the stack before the model runs (`torchtitan/distributed/parallelism_context.py:451`). Only harnesses and prototypes that skip it fail.
- **Likely fix:** in `snapshot_attr_mutation`, also catch `AttributeError` and record the original as missing, so `validate_deferred_attr_mutations` fires the intended "HOP: Non-nullified side effect" for the attribute.
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/136, c41884e97): a `threading.local` subclass whose `__init__` sets the default, so the attribute exists in every thread and nothing is stored inside the traced region.
- **Repro:** `python repros/06_hop_outer_attr_store.py` (CPU). Expected: the block above; raw output in `repros/logs/06_hop_outer_attr_store.txt`.

### 7. An `autograd.Function` forward that reads `._base` and `grad_dtype` of a weight view crashes aot_eager and inductor

- **Request:** Dynamo should guard on or specialize `._base` from the input's source, or graph-break on it, rather than emit it as a graph op on an input that has no `_base` under AOTAutograd. At minimum, a clear message instead of an internal `AttributeError` in the backend.
- **Estimated impact:** a crash. It breaks torch.compile around DistMoE with TorchTitan's default in-place WGRAD.
- **Type:** bug (crash; a hard failure even with `fullgraph=False`, which the repro uses).
- **Context:**
  1. The repro's `autograd.Function` forward takes a view of a parameter (`self.w.flatten(0, 1)`), resolves `w._base`, checks `is_leaf` / `requires_grad`, keeps a weakref, and optionally reads `grad_dtype`. It is compiled with each backend, plus a variant that passes the parameter itself (no view):

```text
with grad_dtype read               eager     ok
with grad_dtype read               aot_eager FAIL BackendCompilerFailed | AttributeError: 'NoneType' object has no attribute 'is_leaf'
with grad_dtype read               inductor  FAIL BackendCompilerFailed | AttributeError: 'NoneType' object has no attribute 'is_leaf'
without grad_dtype read            eager / aot_eager / inductor  ok
with grad_dtype read, no view      eager / aot_eager / inductor  ok
explain, with grad_dtype read:     3 graphs, 2 breaks ("torch.* op returned non-Tensor" at the grad_dtype read)
  graph 1: l_w_ = L_w_ ; getattr_1 = l_w_._base ; param = l_w_._base ; getattr_3 = param.is_leaf ; return ()
explain, without grad_dtype read:  1 graph, 0 breaks (autograd_function_apply HOP)
innermost torch frames of the aot_eager failure:
  _functorch/aot_autograd.py:596 create_aot_state -> run_functionalized_fw_and_collect_metadata
  _functorch/_aot_autograd/collect_metadata_analysis.py:225 -> graph_capture_wrappers.py:1537 PropagateUnbackedSymInts(mod).run
  fx/experimental/symbolic_shapes.py:9315 run_node -> fx/interpreter.py:377 call_function (getattr(param, 'is_leaf'))
```

  2. Mechanism, from the explain output and frames above:
     - With the `grad_dtype` read, Dynamo graph-breaks at that line (it returns a dtype), so the Function is not captured as one `autograd_function_apply` op. The forward's prefix becomes its own graph, whose input is the view and whose FX nodes are `l_w_._base` and `.is_leaf`.
     - With `backend="eager"` that graph runs on the real view, so `_base` is the parameter. Under AOTAutograd the input is traced as a fresh functionalized fake tensor with no view link, so `_base` is `None` and the next attribute read raises, in AOTAutograd's metadata pass (`run_functionalized_fw_and_collect_metadata`).
     - Without the read: 1 graph, no breaks. Without a view the forward takes the `w._base is None` branch, so no `_base` node is recorded and every backend compiles.
  3. Why the messages differ: the repro reads `param = w._base` and then `param.is_leaf`, so the first read on `None` is `is_leaf`. DistMoE's `_parameter_base` loops `while base._base is not None: base = base._base`, which ran one step at trace time, so the recorded graph next reads `._base` of the (now `None`) base: `'NoneType' object has no attribute '_base'`.
  4. Where we hit it: DistMoE's in-place WGRAD path (`dist_moe` at commit ab56bce). `_RegisteredBf16Autograd.forward` calls `_weak_parameter_ref` (`dist_moe/api.py:3614`), whose `_parameter_base` walks `tensor._base` (`dist_moe/_execution.py:409`). Dynamo also graph-breaks at `_execution.py:478` (`parameter.grad_dtype`). TorchTitan's DistMoE adapter passes a view (`w13.weight.flatten(1, 2)`), so torch.compile of a block around DistMoE crashes with TorchTitan's default `inplace_wgrad_accum=True`.
- **Workaround:** `inplace_wgrad_accum=False`. A block then compiles to one graph with 0 breaks, bitwise equal to eager for the output and all four grads. The core trainer does not hit it: it compiles regions around DistMoE and captures CUDA graphs, with no per-block torch.compile. GraphTrainer (make_fx) with in-place WGRAD was not measured (DistMoE documents make_fx support through opaque ops).
- **Repro:** `python repros/07_autograd_fn_param_base.py` (CPU). Expected: the block above; raw output in `repros/logs/07_autograd_fn_param_base.txt`.

### 8. A traceable fixed-order segmented sum (deterministic scatter-add)

- **Request:** a traceable deterministic scatter-add lowering when each output row has a fixed set of contributors, or a "fixed-order segmented sum" primitive Inductor can fuse.
- **Estimated impact:** ~290 ms per Kimi K3 16k fwd+bwd microbatch (3.1 ms x 92 MoE layers).
- **Type:** feature request.
- **Context:**
  1. The repro is the Kimi K3 MoE combine: 262144 expert-sorted rows x 3584 bf16, summed into 16384 tokens through a `torch.library.custom_op` deterministic `scatter_add`, fwd+bwd:

```text
eager as written            13540 us   42 kernels
compiled as written          4248 us   27 kernels   (indexing_backward_kernel 2722 us survives inside the custom op)
compiled per-token rewrite   1102 us    2 kernels   ideal 827 us
```

  2. The custom op exists to force deterministic accumulation. Once it is opaque, compile only fuses the scaling around it, and the op's sort-based deterministic `index_put` dominates.
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/110): every token owns exactly K rows, so combine is an `autograd.Function` whose forward is an unrolled per-token sum in fp32 and whose backward is a gather by token. Deterministic, no atomics, 2-3x more accurate vs fp64.
- **Repro:** `python repros/08_opaque_custom_op.py` (GPU). Expected: the block above.

### 9. Keep views' stride and offset static, and do not route every call site to the newest, symbolic graph

- **Request:**
  - (a) keep a view's stride and offset static when they are fixed multiples of a static or specialized size;
  - (b) a way to mark strides and offsets static alongside sizes;
  - (c) prefer a static graph that accepts the inputs over the newest symbolic one, or specialize a dim that takes a handful of large distinct values.
- **Estimated impact:** ~240-300 ms per Kimi K3 16k fwd+bwd microbatch with fixed T: ~2.6 ms (EP estimate) to 3.3 ms (measured, 2 layers without EP) per MoE layer x 92 MoE layers. Partly worked around in https://github.com/felipemello1/torchtitan/pull/113.
- **Type:** feature request.
- **Context:**
  1. The repro runs SwiGLU on `gate, up = unbind(-2)` views of a packed `[16384, 2, 6144]` bf16 projection, fwd+bwd. It is compiled at F = 6144 only (static) vs at F = 3072 first and then F = 6144 (automatic dynamic, as Kimi K3's routed and then shared experts):

```text
repro:  F static (compiled at 6144 only)    361 us   3 kernels (includes the eager UnbindBackward stack)
        F dynamic (3072 first, then 6144)   601 us   3 kernels
model, Kimi K3 situglu, shared expert T x 6144:
        static 462.9 us | dynamic F 845.6 us | mark_static on F 555.1 us | + torch._check on stride and offset 470.6 us
```

  2. With F symbolic, the views' row stride (2F) and `up`'s storage offset (F) get their own symbols, and the kernels index with `xindex % ks0` instead of a constant divisor.
  3. `mark_static` pins sizes only: on F it recovers most of the gap. `torch._check(gate.stride(0) == 2 * gate.size(-1))` plus a `_check` on the offset recovers the rest. Separately allocated contiguous gate/up with dynamic F show no slowdown (236 vs 235 us in a first version of this repro). `assume_aligned_inputs=True` does not help.
  4. Dynamo checks its most recently added cache entry first. So once one call site creates the symbolic graph, every call site that graph accepts runs it, including shapes that already had a static graph. The cost therefore hits fixed-T packed training, not only RL. Measured, 2 layers without EP: 331.07 -> 324.48 ms with the `mark_static` of https://github.com/felipemello1/torchtitan/pull/113 alone.
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/113): `torch._dynamo.mark_static` on the hidden dim inside the swiglu / situglu regions. The stride/offset `_check` is a follow-up. The packed `[rows, 2, F]` input of https://github.com/felipemello1/torchtitan/pull/99 gives static strides with `mark_static` alone.
- **Repro:** `python repros/09_symbolic_hidden_dim.py` (GPU). Expected: the repro rows above.

### 10. Unroll small reductions

- **Request:** unroll when a symbol's upper bound (value range) is below `unroll_reductions_threshold`. Raise or autotune the threshold (8, strict `<`) for reductions over a non-innermost dim, or improve the persistent-reduction schedule for that layout.
- **Estimated impact:**
  - Up to ~230 ms per Kimi K3 16k fwd+bwd microbatch for the residual as compiled on https://github.com/felipemello1/torchtitan/pull/109: up to 1.2 ms per call x 186 calls, the static-vs-dynamic gap. That is an upper bound: width-1 calls are static and not in the gap.
  - The hand-written `autograd.Function` of https://github.com/felipemello1/torchtitan/pull/131 closes most of it (16k fwd+bwd 3642 -> 2652 us vs 2403 static). That leaves ~0.25 ms per call (up to ~46 ms, same upper bound); request 15 is part of it.
  - Separately, ~90 ms for MoE dispatch/combine at K = 16 (~1 ms x 92 layers; worked around in https://github.com/felipemello1/torchtitan/pull/110).
- **Type:** feature request (symbolic sizes); default / feature request (non-innermost dim). One root cause: `unroll_reductions_threshold`.
- **Context, (a) small reductions are unrolled only below a static threshold:**
  1. The repro is the Kimi residual's weighted sum over its stack `[16384, N, 7168]` plus a partial, softmax weights recomputed per element, forward only:

```text
N=7 static                    314 us   pointwise (unrolled)        ideal 298 us
N=8 static                   1574 us   persistent reduction        ideal 331 us
N=8 static, threshold 9       353 us   pointwise
N=7 dynamic                  1245 us   looped reduction (triton_red)
N=7 dynamic, threshold 64    1245 us   looped reduction: the threshold never applies to symbolic sizes
```

  2. Inductor unrolls a reduction into a pointwise kernel only if its size is a static integer below `config.unroll_reductions_threshold` (default 8, strict `<`; `torch/_inductor/ir.py:1903-1908`). A symbolic size is never unrolled, whatever its value range, and static N = 8 (Kimi K3's widest stack) is not unrolled either.
  3. In production N varies 1-8 across layers. After the second width, automatic dynamic shapes make N symbolic, and every call takes the reduction kernel (Kimi residual fwd+bwd 2402 us static vs 3642 us dynamic, op level).
  4. Tried without effect on the dynamic graph: `split_reductions=False`, `triton.cooperative_reductions`, `triton.multi_kernel=1`. `coordinate_descent_tuning`: 1834 -> 1434 us.
- **Context, (b) a small reduction over a non-innermost dim runs far below bandwidth:**
  1. The repro is the Kimi K3 MoE dispatch-gather backward: summing each token's K = 16 rows of `[262144, 3584]` bf16 gradients, plus the same sum over a contiguous `[T, 16, 3584]`:

```text
gather, .sum over K (default threshold 8)   1354 us   persistent reduction
gather, .sum over K, threshold 16           1354 us   (strict <: 16 does nothing)
gather, .sum over K, threshold 17            290 us   pointwise
gather, unrolled Python loop                 290 us   pointwise                 ideal 281 us
contiguous [T,16,D], .sum over K             999 us   persistent reduction (no gather)
contiguous [T,16,D], threshold 17            278 us   pointwise
```

  2. K = 16 is above `unroll_reductions_threshold` (8, strict `<`), so `.sum(1)` is lowered as a persistent reduction over a dim with stride D. That kernel runs 3.6x below bandwidth even on contiguous input (999 vs 278 us); the gather adds ~35% on top.
  3. With the threshold at 17 (16 does nothing) the same `.sum(1)` becomes the pointwise kernel the hand-unrolled loop produces, at roofline.
- **Workarounds:**
  - (a) https://github.com/felipemello1/torchtitan/pull/131 (draft) for dynamic N: a custom `autograd.Function` with a hand-written backward (16k fwd+bwd 3642 -> 2652 us vs the static 2403 us of https://github.com/felipemello1/torchtitan/pull/109, medians of 3 processes). For static N = 8, a region-scoped `unroll_reductions_threshold=9` works (not shipped).
  - (b) https://github.com/felipemello1/torchtitan/pull/110: `_sum_token_rows` loops over `top_k` in Python; dispatch backward 2028 -> 989 us in the real op. A region-scoped `unroll_reductions_threshold=17` gives the same kernel bitwise (Kimi combine 1107 vs 1105 us, dispatch 997 vs 995 us).
- **Repros:** `python repros/10a_symbolic_small_dim.py`, `python repros/10b_small_k_in_tile.py` (GPU). Expected: the blocks above.

### 11. Decompose memory-bound batched GEMVs by default

- **Request:** enable the decomposition by default for memory-bound batched GEMVs (or gate it on a cost model), lower the default floor, and decide how it should interact with bfx9.
- **Estimated impact:** an alternative to request 10 for the Kimi residual (1.1-1.3 ms per call at 16k; does not add). Caveat: the pass skips fp32 bmm under `fp32_precision="bfx9"`, which TorchTitan sets on SM100+, so as is it does not help the trainer.
- **Type:** default / doc.
- **Context:**
  1. The repro is the Kimi K3 attention residual, `softmax(scores)[T, 1, 8] @ values.float()[T, 8, 7168]`, fwd+bwd:

```text
T=16384  bmm as written            3475 us   10 kernels   forward externs [extern_kernels.bmm]
                                             (gemv2N 1226 us + a separate fp32 upcast of values + gemv2T)
T=16384  bmm + decompose_mm_pass   2357 us    5 kernels   no externs
T=16384  explicit sum              2155 us    4 kernels   no externs                ideal 794 us
T=2048   bmm as written             459 us   10 kernels   extern bmm
T=2048   bmm + decompose_mm_pass    459 us   10 kernels   extern bmm (the pass does not fire below its 10240 floor)
T=2048   explicit sum               276 us    4 kernels                             ideal 99 us
```

  2. As written, Inductor sends the batched GEMV to cuBLAS, so the bf16 -> fp32 upcast of `[T, 8, 7168]` (1.9 GB at 16k) is materialized instead of fused into the reduction.
  3. Inductor's `decompose_mm_pass` already handles this shape (`torch/_inductor/fx_passes/decompose_mem_bound_mm.py:62-90`: `mat1.shape[0] >= 10240` and 2 of m, k, n < 32). But it is opt-in (`post_grad_fusion_options={"decompose_mm_pass": {}}`), and its first-dim floor excludes 2k-token prefill and decode.
  4. In the trainer the pass is not an option as is:
     - `should_decompose_bmm` returns False for an fp32 bmm when `torch.backends.cuda.matmul.fp32_precision == "bfx9"`, because BF16x9 emulation must stay an ATen extern (`fx_passes/decompose_mem_bound_mm.py:68-73`, `torch/_inductor/utils.py:3486-3495`).
     - TorchTitan enables bfx9 on SM100+ (`torchtitan/distributed/utils.py:255-266`, called at `:477`).
     - So the trainer's baseline is a BF16x9 cuBLAS bmm. The repro runs with the default fp32 precision.
  5. The 10240 floor is configurable (`post_grad_fusion_options={"decompose_mm_pass": {"min_first_dimension_decomposition": ...}}`, `decompose_mem_bound_mm.py:34-36`), so the ask is a default change.
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/109): the explicit weighted sum `(probs[..., None] * values.float()).sum(1)`, 9% faster than the pass and also effective at small T. A region-scoped `post_grad_fusion_options` would have been a simpler alternative at large T.
- **Repro:** `python repros/11_extern_gemv.py` (GPU). Expected: the block above.

### 12. Inline a producer only when the consumer's broadcast factor is small

- **Request:** account for the consumer's broadcast factor when deciding to inline a producer with expensive per-element ops (exp).
- **Estimated impact:** ~175 ms per Kimi K3 16k forward under no_grad (0.95 ms x 186 calls; evaluation, RL prefill).
- **Type:** feature request.
- **Context:**
  1. The repro is a softmax + zero pad of `[T, N+1]` scores feeding an 8-way unrolled weighted sum over `[16384, 8, 7168]` (N dynamic, clamped reads), no_grad forward:

```text
default                      1225 us   one pointwise kernel recomputes 8 exp/div + masked loads per output element
realize_reads_threshold=1     278 us   the [T, 8] probs are realized first            ideal 265 us
```

  2. When the padded probs are also needed for backward, they are realized and the kernel is fast. In a no_grad forward Inductor inlines the softmax into the consumer, multiplying its cost by D = 7168.
  3. `realize_reads_threshold=1` region-wide slows the backward (fwd+bwd 3106 us).
- **Workaround:** the `autograd.Function` of https://github.com/felipemello1/torchtitan/pull/131 saves the padded probs, so they are realized in training (fwd+bwd 2399 vs 3234 us, measured on its branch at the time). no_grad still takes the slow kernel.
- **Repro:** `python repros/12_inline_recompute.py` (GPU). Expected: the block above.

### 13. Triton: masked loads halve a many-load kernel on SM103

- **Request:** Triton triage with the standalone kernels. On the Inductor side, the mask is provably unnecessary only when XBLOCK divides 7168 (true for XBLOCK <= 1024, not for 2048, which autotuning can pick).
- **Estimated impact:** ~65 ms per Kimi K3 16k forward, RL only (dynamic T; 0.35 ms x 186).
- **Type:** Triton triage.
- **Context:**
  1. Inductor repro, then the two generated kernels run standalone:

```text
Inductor, fwd:  residual weighted sum   T static 314.1 us | T dynamic 662.6 us
                swiglu (control)        T static 133.9 us | T dynamic 133.8 us
standalone, us by XBLOCK/num_warps:
                unmasked  512/4 315, 512/8 563, 1024/4 322, 1024/8 317, 2048/4 308, 2048/8 328
                masked    512/4 597, 512/8 642, 1024/4 697, 1024/8 605, 2048/4 832, 2048/8 743
```

  2. The generated sources differ only by `xmask = xindex < xnumel` on 18 loads (8 broadcast loads of per-token softmax values). Both carry `tt.divisibility=16` on xnumel.
  3. The standalone sweep shows the slowdown at every XBLOCK / num_warps, so it is not autotuning. The mechanism (predication of broadcast loads vs register pressure) is not established; it looks like Triton codegen on SM103 rather than an Inductor decision.
- **Workaround:** none needed in packed training (T is fixed per microbatch). It matters for RL, where T varies.
- **Repros:** `python repros/13a_masked_loads_symbolic_T.py` (Inductor) and `python repros/13b_standalone_masked_kernels.py` (the two generated kernels, `13b_kernel_static.py` and `13b_kernel_masked.py`; GPU). Expected: the block above.

### 14. A cost model or autotuning for cat/stack lowering

- **Request:** a cost model for cat lowering that accounts for input layouts and downstream reductions, or autotuning between the two lowerings. Also lower an interleave/cat consumed by a reduction without re-loading the row per branch.
- **Estimated impact:** ~59 ms per DeepSeek-V4 flash 16k fwd+bwd microbatch (1.37 ms x 43 layers, inverse RoPE alone). Worked around per region.
- **Type:** feature request.
- **Context, (a) pointwise-cat lowering is right for some shapes and 2-3x wrong for others on GB300:**
  1. The repro splits, rotates the last 64 of 512 channels, and cats, on DeepSeek-V4 attention shapes `[16384, 64, 512]` bf16, fwd+bwd:

```text
(a) inverse rope on o        default cat 2155 us (2 kernels) | ConcatKernel  781 us (5 kernels)   ideal 605 us
(b) per-head RMS norm + rope  default cat 1026 us (2 kernels) | ConcatKernel 1502 us (7 kernels)   ideal 756 us
```

  2. The default lowering fuses the cat into one masked pointwise kernel with per-element index math. On GB300 that is 2.8x slower than per-input copies for (a). It is faster for (b), where it lets the cat fuse into the RMS norm's persistent reduction (ConcatKernel splits them).
  3. The repro matches the model: (a) 2177 vs 791 us, (b) 1044 vs 1542 us. (The first version of this repro fed a transposed view of o, as the model does; that only adds an eager 1.2 ms gradient layout copy outside the region to both variants.)
  4. In an earlier H100 study, the same ConcatKernel option made Qwen3.5's partial RoPE 24-26% slower in training and 3x slower at decode, so no global setting is right.
- **Context, (b) a masked cat/stack inside a persistent reduction re-loads the row per branch** (no standalone repro):
  1. DeepSeek-V4 `q_norm_rope`, 16k: the forward kernel takes 455 us for 2.15 GB (ideal 302 us).
  2. Inductor's output code for `triton_per_fused__to_copy_add_cat_mean_mul_pow_rsqrt_select_split_with_sizes_stack_sub_view_0` lowers the split + interleaved-pair stack + cat as masked pointwise code inside the RMS norm's persistent reduction. The q row is re-loaded 4 times under different masks, with int64 `%` / `//` index math. The ConcatKernel alternative cannot fuse into the reduction (part (a), row (b): 1542 vs 1044 us fwd+bwd).
  3. Region options tried: `coordinate_descent_tuning` 483 vs 485 us; `persistent_reductions=False` 1056 us; a cat-free rewrite 528 us.
  4. A related case: the real-arithmetic RoPE's `torch.stack((re, im), -1)` lowers as a masked cat (per element `x % 2`, then 8 masked loads). Casting each half before the stack, plus a rotation-by-minus-angle backward, took DeepSeek-V4 RoPE 217 -> 143 us (https://github.com/felipemello1/torchtitan/pull/115).
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/107): `options={"max_pointwise_cat_inputs": 0, "max_complex_pointwise_cat_inputs": 0}` on the inverse-RoPE region only.
- **Repro:** `python repros/14_cat_lowering.py` (GPU). Expected: part (a)'s block.

### 15. int32 indexing for bounded symbolic sizes

- **Request:** decide int32 vs int64 per index expression from value ranges, honoring `torch._check` bounds (today they are ignored), and/or select an int32 kernel at runtime when the sizes fit. Either avoids the hard failure a global `assume_32bit_indexing` hits (below). A per-expression decision alone recovers the cost only where T is bounded, because the costly expressions index the full `[T, N, D]` stack and T has no upper bound.
- **Estimated impact:**
  - Up to ~26 ms per Kimi K3 16k fwd+bwd microbatch on top of https://github.com/felipemello1/torchtitan/pull/131 (0.14 ms per call x 186 residual calls).
  - That is an upper bound: it assumes every call takes the dynamic-width graph, while width-1 calls are static and unaffected.
  - Part of request 10's remaining gap, not additive.
  - Today's only knob, `assume_32bit_indexing`, also slows the forward and fails to compile above ~37k tokens at width 8.
- **Type:** feature request.
- **Context:**
  1. The repro is the residual's weighted sum with N dynamic, fwd+bwd:

```text
default                      1918 us
torch._check(N <= 8)         1914 us   (no change)
assume_32bit_indexing=True   1816 us
```

  2. `_decide_tl_dtype` (`torch/_inductor/codegen/triton_utils.py:259-278`) types every symbolic size argument `ks*` as `tl.int64` for non-template kernels without block pointers. Its comment: "Even if the ks0 symbol itself is within tl.int32 range, it's risky to use tl.int32 dtype since we may have ks0*ks1 later for kernels like torch.mean when dynamic shape is enabled." A `torch._check` bound cannot change that by construction.
  3. In the full Kimi residual on the earlier branch, before the hand-written backward of https://github.com/felipemello1/torchtitan/pull/131, the effect was larger (fwd+bwd 3229 -> 2779 us, backward reduction kernel 1071 -> 727 us). On https://github.com/felipemello1/torchtitan/pull/131 it is 2652 -> 2514 us (see Workaround).
  4. `assume_32bit_indexing=True` switches the `ks*` arguments to int32 and checks every index expression against the int32 max in `expr_fits_within_32bit` (`torch/_inductor/utils.py:4689-4704`). Inputs that contradict it throw (`torch/_inductor/config.py:1825-1829`). It is input-checked, not unsafe.
- **Workaround:** none shipped. Region-scoped `options={"assume_32bit_indexing": True}` cuts the Kimi residual's fwd+bwd at 16k from 2652 to 2514 us (dynamic width, on https://github.com/felipemello1/torchtitan/pull/131). But the region then fails to compile once T*N*D > int32 max (T=65536, N=8: `expect_true failed for 7168*s0*s24 <= 2147483647`), and the forward gets slower (32k: 1515 -> 1578 us), so https://github.com/felipemello1/torchtitan/pull/131 does not use it.
- **Repro:** `python repros/15_int64_indexing_symbolic.py` (GPU). Expected: the block above.

### 16. Fuse tiny-block reductions across steps (Sinkhorn)

- **Request:** fuse reductions over a tiny trailing block (4 x 4 per token) across steps, e.g. by keeping the block in registers.
- **Estimated impact:** ~20 ms per DeepSeek-V4 flash 16k fwd+bwd microbatch (0.20-0.23 ms x 86 HcPre calls).
- **Type:** feature request.
- **Context:**
  1. The repro is the DeepSeek-V4 mHC Sinkhorn: 20 iterations of row then column normalization of `[16384, 4, 4]` fp32, fwd+bwd:

```text
[T,4,4] tensor form          289-316 us (across runs)   159 kernels   rel vs fp64 8.8e-8
16 per-entry [T] tensors      86 us    32 kernels   rel vs fp64 8.8e-8
```

  2. Each column normalization reads the previous step at transposed indices (`sum(-2)` after `sum(-1)`), so Inductor cannot fuse consecutive steps and realizes every intermediate. With each of the 16 entries as its own `[T]` tensor, every read is same-index and the steps fuse.
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/120, draft): the 16 per-entry `[T]` tensors in the compiled branch. In context 309 -> 105 us; in that PR, Sinkhorn alone fwd+bwd at 16k goes 303 -> 97 us kernel time (medians of 3 processes).
- **Repro:** `python repros/16_sinkhorn_transposed_reads.py` (GPU). Expected: the block above.

### 17. Complex-multiply codegen

- **Request:** lower complex `mul` (with `view_as_complex` / `view_as_real`) to real arithmetic.
- **Estimated impact:** ~11 ms per DeepSeek-V4 flash 16k fwd+bwd microbatch (0.25 ms x 43 attention RoPE calls; the Compressor and Indexer calls add more).
- **Type:** feature request.
- **Context:**
  1. The repro is DeepSeek-V4 RoPE on q `[16384, 64, 64]` bf16 with a complex64 cache `[16384, 1, 32]`, fwd+bwd:

```text
eager complex              445.5 us   7 kernels
compiled complex           443.7 us   7 kernels   (the aten elementwise complex-mul fallback, 214 us, stays)
compiled real arithmetic   199.0 us   2 kernels   (same math on view_as_real(cache))
```

  2. Inductor warns "Torchinductor does not support code generation for complex operators" and keeps the fallback, which also blocks fusing the split/cat glue around it. The warning prints only on a cold compile; `run_all.sh` reruns with `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` to show it.
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/106): `ComplexRoPE.apply_rotary_emb` takes a real-arithmetic branch under `torch.compiler.is_compiling()`; eager keeps the complex ops. With the glue in the same region (https://github.com/felipemello1/torchtitan/pull/107), DeepSeek-V4 q norm + RoPE went 1497 -> 1044 us.
- **Repro:** `python repros/17_complex_ops.py` (GPU). Expected: the block above.

### 18. Region semantics: options and aliasing

- **Request:**
  - carry an inlined compiled function's options as a region hint for the outer compile, or document that they are dropped;
  - let regions return views of their inputs without a gradient copy, and allow in-place updates of an incoming gradient that is not used elsewhere.
- **Estimated impact:** these block replacing hand rewrites and Triton overrides with compile. Kimi K3 MLA ~10 ms per 16k fwd+bwd microbatch (0.42 ms x 24 MLA layers); DeepSeek-V3 MLA ~24 ms per 16k fwd+bwd microbatch (~0.4 ms between the region's floor and the override x 61 layers, estimate).
- **Type:** feature request (observation, no standalone repro).
- **Context, (a) region-scoped options are lost when the region is inlined under an outer compile:**
  1. A region-scoped `unroll_reductions_threshold=17` makes `gather + .sum(1)` identical in speed and bits to the hand-unrolled loop of https://github.com/felipemello1/torchtitan/pull/110 (Kimi combine 1104.5 vs 1107.2 us, dispatch 995.1 vs 996.8 us).
  2. A nested `torch.compile(fn, options=...)` is inlined into an outer whole-model `torch.compile` (a user's, or vLLM's `support_torch_compile`, both of which also take TorchTitan's compiled branches). Only its `fullgraph` semantics are kept (`torch/_dynamo/variables/functions.py:3095-3110`); its Inductor options are not applied.
  3. That is why https://github.com/felipemello1/torchtitan/pull/109 and https://github.com/felipemello1/torchtitan/pull/110 ship hand rewrites instead of the option, and why several workarounds in this doc are code rather than config.
- **Context, (b) functional regions cannot alias outputs or update gradients in place:**
  1. Kimi K3 MLA k/v assembly (16k fwd+bwd): eager 2664.0 us; a region returning v as a view of kv 854.4-854.9 us over 3 processes. Of that, a 419 us eager copy of v's gradient remains (AOTAutograd's handling of an output that aliases an input).
  2. DeepSeek-V3 MLA: TorchTitan's opt-in fused Triton override (torch.library ops) runs fwd+bwd in 1024 us at 16k, bitwise vs eager, by keeping v a view and rotating the incoming q gradient in place. A compile region can do neither (AOTAutograd's backward writes a fresh grad_q): estimated region floor ~1.4 ms, measured 2.1 ms with the complex RoPE fallback.
- **Workarounds:** (a) hand rewrites in https://github.com/felipemello1/torchtitan/pull/109 and https://github.com/felipemello1/torchtitan/pull/110. (b) The hand-written override for DeepSeek-V3 (https://github.com/felipemello1/torchtitan/pull/124 adds recipes that turn it on); the Kimi region (https://github.com/felipemello1/torchtitan/pull/122) accepts the copy.
- **Repro:** none standalone.

### 19. Lower small-k `topk` by default

- **Request:** lower small-k `topk` by default, or by a size heuristic.
- **Estimated impact:** ~3 ms per DeepSeek-V3 671B 4k-token fwd+bwd microbatch (0.054 ms x 58 MoE layers; router glue 160 -> 106 us at 4k, the production shape). Per 16k not measured.
- **Type:** default.
- **Context:**
  1. DeepSeek-V3 router glue (scores, group top-2, top-4 groups, top-8 of 256 experts) at 4k tokens: eager 325.5 us (72 kernels), compiled 159.7 us (21), compiled with `triton.decompose_sort_ops` 106.4 us (17).
  2. The compiled glue is dominated by ATen `topk` (`sbtopk::gatherTopK`, top-8 of 256: 96.9 us at 4k, 378 us at 16k), which Inductor lowers only with the option (`torch/_inductor/lowering.py:8490-8493`). Ideal for the whole glue: 1.8 / 7.1 us at 4k / 16k.
  3. A sort-and-slice rewrite with default options was faster still (selection 56.2 vs 78.4 us at 4k with the option).
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/132): a DeepSeek-V3 router region whose compiled branch selects experts with argmax rounds instead of ATen `topk`.
- **Repro:** none standalone.

### 20. FP contraction: compiled is not bitwise vs eager for fused pointwise math

- **Request:** document that compiled vs eager is not bitwise for fused pointwise math. A per-region "no FP contraction" option would help numerics tests.
- **Estimated impact:** test failures and confusion, not speed: compiled differs from eager by 1 bf16 ulp on a few elements.
- **Type:** doc.
- **Context:**
  1. The repro runs RoPE on `[4096, 8, 128]` bf16 and compares forward outputs element by element:

```text
(a)  complex eager vs real-arithmetic compiled   44 / 4194304 elements differ, max abs diff 7.8e-3 (1 bf16 ulp)
(a') real eager vs real compiled                 92 / 4194304, max 1.6e-2 (1 ulp for |x| in [2, 4))
(b)  rotate-half (CosSinRoPE) eager vs compiled  81 / 4194304, max 7.8e-3
with torch._inductor.config.emulate_precision_casts=True (Inductor then passes enable_fp_fusion=False):
(a)  90 / 4194304 differ (different formula: ATen complex mul vs real arithmetic)
(a') 0 / 4194304   (b) 0 / 4194304   -> bitwise once FP contraction is off
TRITON_DEFAULT_FP_FUSION=0 has no effect: Inductor sets enable_fp_fusion explicitly (codegen/triton.py:7492)
```

  2. Eager computes `x*cos`, `rot*sin` and the add as separate ops, each rounded to fp32, so no contraction can happen across them (contraction inside a single eager kernel is still possible). Triton fuses the whole expression and contracts it into FMAs, so a few fp32 results round to a different bf16.
  3. An earlier H100 study saw bitwise forwards; on GB300 it is not bitwise by default.
  4. `emulate_precision_casts=True` also inserts low-precision round trips, so it is a test-only setting, not a fix.
  5. Consequence: TorchTitan's upstream test `TestRoPELocalCompile::test_forward_and_backward_match_eager` (CosSinRoPE, `rtol=atol=0`) fails on unmodified main on this GPU (1 of 65536 elements, 1 ulp).
- **Workaround:** tests compare within about 1 bf16 ulp (https://github.com/felipemello1/torchtitan/pull/106 for ComplexRoPE; https://github.com/felipemello1/torchtitan/pull/111 for the upstream CosSinRoPE test, which keeps bitwise checks for batch invariance).
- **Repro:** `python repros/20_fma_bitwise.py` (GPU); `run_all.sh gpu` also runs it with `EMULATE_PRECISION_CASTS=0/1` and `TRITON_DEFAULT_FP_FUSION=0`. Expected: the block above.

### 21. `maybe_mark_dynamic` is forbidden in graphs

- **Request:** a doc note, or make `maybe_mark_dynamic` a no-op in graphs, as `mark_static` is.
- **Estimated impact:** a hard failure once a helper that calls it is itself compiled (block compile, GraphTrainer).
- **Type:** doc.
- **Context:**
  1. Repro:

```text
unguarded:                   AssertionError: Attempt to trace forbidden callable <function maybe_mark_dynamic>
guarded by is_compiling():   compiled fine
```

  2. `maybe_mark_dynamic` / `mark_dynamic` are `@forbid_in_graph` by design (`torch/_dynamo/decorators.py:1347`), unlike `mark_static`, which works in-graph.
  3. `if not torch.compiler.is_compiling(): torch._dynamo.maybe_mark_dynamic(...)` traces fine.
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/109): dropped the call; one call signature plus request 1's option kept the residual within budget.
- **Repro:** `python repros/21_maybe_mark_dynamic_traced.py` (GPU). Expected: the block above.

### 22. `functools.cache` under fake-tensor tracing

- **Request:** a warning from non-strict tracers when a fake tensor escapes into module-level state, or a doc note.
- **Estimated impact:** eager breaks after a trace.
- **Type:** doc.
- **Context:**
  1. A cached tensor factory first called under `FakeTensorMode` (make_fx, GraphTrainer's non-strict tracer) caches a FakeTensor. The next eager call fails with `AssertionError: Please convert all Tensors to FakeTensors first ...`.
  2. Dynamo ignores the cache wrapper and warns, so only non-Dynamo tracers are affected.
- **Workaround** (https://github.com/felipemello1/torchtitan/pull/108): removed the cache.
- **Repro:** `python repros/22_cache_under_fake_tensors.py` (CPU).

### 23. make_fx specializes Python ints silently

- **Request:** guard or error on a changed int input in the traced callable, or document it prominently.
- **Estimated impact:** silently wrong sizes.
- **Type:** doc.
- **Context:**
  1. A graph traced with `n=3` returns `[0.0, 2.0, 4.0]` when called with `n=5`.
  2. DeepSeek-V4 sizes tensors from the Python int `VarlenMetadata.max_k`, and FA4 varlen takes `max_seqlen` as an int, so GraphTrainer graphs freeze them without an error.
- **Workaround:** fixed-capacity varlen metadata in any GraphTrainer recipe (no PR yet).
- **Repro:** `python repros/23_make_fx_int_specialization.py` (CPU).

### 24. The fake-tensor `data_ptr` policy differs between `FakeTensorMode` and Dynamo / make_fx

- **Request:** document the policy difference, or make it consistent.
- **Estimated impact:** library fakes that pass under plain `FakeTensorMode` fail under Dynamo and make_fx.
- **Type:** doc (observation).
- **Context:**
  1. FA4's `_flash_attn_fwd` JIT-compiles its kernel on first use from `from_dlpack(tensor)` (`flash_attn/cute/interface.py:1482-1501`, `cute_dsl_utils.py:163`).
  2. Dynamo and make_fx create fake tensors with `fake_tensor_allow_unsafe_data_ptr_access=False` (`torch/_dynamo/output_graph.py:817`, `torch/fx/experimental/proxy_tensor.py:3078`), so the DLPack export throws. Under a plain `FakeTensorMode` the same path works.
  3. Library authors who test fakes under plain `FakeTensorMode` miss this.
- **Workaround:** attn-gym registers its FA4 path as a custom op whose fake never touches data (https://github.com/felipemello1/attention-gym/pull/10).
- **Repro:** none standalone.

### 25. `regional_inductor` merges unkeyed regions and tags collectives

- **Request:** change, or document, how `regional_inductor` groups annotated nodes without an `inductor_region` key; and document that `fx.traceback.annotate` tags collectives created inside the context.
- **Estimated impact:** adjacent regions merge into one partition, and collectives lose overlap.
- **Type:** bug / doc (observation; `torch.fx.passes.regional_inductor`, used by GraphTrainer).
- **Context:**
  1. Without an `inductor_region` key, every annotated node joins one default region, so adjacent annotated regions merge into one partition (`torch/fx/passes/regional_inductor.py:143-160`).
  2. (doc) `fx.traceback.annotate` tags every node created inside the context. Under SimpleFSDP that includes the parameter all-gathers, and backward reduce-scatters inherit the tag through TorchTitan's `_copy_fwd_metadata_to_bw_nodes` (`torchtitan/experiments/graph_trainer/make_fx_tracer.py:167`), as torch's own `copy_fwd_metadata_to_bw_nodes` does for AOTAutograd (`torch/_functorch/_aot_autograd/utils.py:660`). So collectives are compiled into the partition and lose overlap.
- **Workaround:** TorchTitan's GraphTrainer bridge adds `"inductor_region": name` to its annotation (from the review of https://github.com/felipemello1/torchtitan/pull/118).
- **Repro:** none standalone.

### 26. Row gathers do not exploit row reuse

- **Request:** a gather schedule that reads each source row once and writes its K copies, or at least reaches write bandwidth.
- **Estimated impact:** the Kimi K3 dispatch gather at 16k takes 649 us against a ~281 us write-bound floor.
- **Type:** observation (one probe; not confirmed further).
- **Context:**
  1. Kimi K3 MoE dispatch gather at 16k: 1.88 GB of output from 117 MB of distinct rows.
  2. A plain copy of the same output takes 554 us. The gather takes 596-633 us in expert-sorted order and 649 us with each token's K rows adjacent (perfect reuse), for both ATen `index_select` and Inductor `x[idx]`. The write-bound floor is ~281 us.
- **Workaround:** none.
- **Repro:** none standalone.

### 27. `regional_inductor` partitioning scales superlinearly

- **Request:** build the dependency map once and share it across regions (or partition all region ids in one pass), and bound the merge loop's cycle checks.
- **Estimated impact:** DeepSeek-V4 regional compile takes 382 s at 4 layers on every process start (a warm cache does not help); extrapolated, ~1.5 h for an 11-layer pipeline stage.
- **Type:** performance bug (compile time).
- **Context:**
  1. The repro is a synthetic graph of L layers (untagged matmul + 8 tagged pointwise ops each), timing only `_RegionScooper.scoop_regions`, the partitioning that runs before any partition is compiled:

```text
DSv4 GraphTrainer regional compile:
  2 layers, 2690 tagged nodes      61.9 s   (11 foreign GPU processes)
  4 layers, 5402 tagged nodes     382.1 s   (4 foreign GPU processes: the larger run was the less contended one)
  ~N^2.6 is the slope between these two runs under different contention, not a fit
  2 layers across reps: regional 91.8, 65.4, 65.8 s; full Inductor on the same graph 39.4, 1.5, 1.9 s   (warm cache from rep 2)
repro 27, scoop_regions only:
  tagged nodes                     400     800    1600    3200
  one shared region id            0.24    1.44   10.95       -  s   (closest to production, see below)
  one region id per layer         0.17    1.16    7.79   63.99 s   (~N^2.9)
extrapolations (labelled, not measured):
  shared-region row to ~5400 nodes: ~350-400 s, close to the measured 382 s
  an 11-layer DSv4 pipeline stage at ~N^2.6: 382 s x (11/4)^2.6 = ~1.5 h of partitioning per process start
```

  2. A warm cache barely helps the regional pass, while full Inductor drops to 1.5-1.9 s, so the time is not Inductor lowering.
  3. GraphTrainer's bridge (https://github.com/felipemello1/torchtitan/pull/118, `torchtitan/distributed/local_compile.py:86-92` at af93854ad) sets `"inductor_region": name`, the region name, so every layer's instance of a region shares one id: DeepSeek-V4 has 9 ids at any depth. Production is therefore closest to the repro's shared-region row.
  4. Where the repro spends it (cProfile at 800 tagged nodes, `repros/logs/27_regional_partition_scaling.txt`):
     - one shared region (closest to production's 9 ids): `propose_partitions` 2.07 s, of which 45,448 `maybe_merge_partition` attempts with a cycle check each (`dfs_iter_find_cycle`, 1.22 s). A separate toy-model measurement of the same cycle check: 0.3 / 0.8 / 5.7 s at 400 / 800 / 1,600 nodes;
     - one region id per layer (many ids): `regional_inductor.py:177-183` builds one `CapabilityBasedPartitioner` per region over the whole graph, and each builds `_DependencyViewer`, a transitive-downstream set per node (`torch/fx/passes/infra/partitioner.py:50-58`): 0.78 s for 100 constructions, plus 0.62 s in 100 `propose_partitions`.
- **Workaround:** none; every process start pays it.
- **Repro:** `python repros/27_regional_partition_scaling.py` (CPU). Expected: the "repro 27" rows above.

---

## Observations without a minimal repro

- **Gradient-accumulation adds fold into GEMM epilogues only under whole-graph compile.** Qwen3.5-35B-A3B, 2 layers: 61 bf16 add launches with regions vs 36 under block compile, where cuBLAS `*_bx_*` (beta != 0) variants absorb them. Grouped-GEMM weight gradients have no accumulate-in-place path at all, so autograd's standalone bf16 AccumulateGrad adds remain with regions: ~1.3 ms per DeepSeek-V3 MoE layer at any token count (8.5 GB per add).
- **Whole-layer Inductor is slower than the region boundary in two places** (DeepSeek-V3 671B layers, 4096 tokens, GraphTrainer full Inductor vs the same regions; repro `repros/obs_whole_graph_vs_region.py`, GPU):
  - Harness note: the GraphTrainer-trace rows below ran with the layer input `x` detached, so the first layer's input gradient (its RMSNorm dx and the residual add into `x`) was not computed. Every compared variant skips the same work, so the comparisons hold; absolute times are slightly low. The repro's numbers keep `requires_grad=True` and are unaffected.
  - The routed SwiGLU backward: the backward of `unbind(-2)` is lowered as a masked pointwise cat fused into it (`triton_poi_fused_mul_silu_silu_backward_stack_unbind_view`), 0.298 ms per layer vs 0.113 ms with ConcatKernel lowering (knob `max_complex_pointwise_cat_inputs`; request 14's lowering choice). On GB300 ConcatKernel halves this kernel class: 0.755 -> 0.368 ms on the Llama3-8B FFN (https://github.com/felipemello1/torchtitan/pull/60).
  - The MoE dispatch backward per-token sum:
    - `triton_poi_fused__to_copy_add_div_gather_index_mul_select_sigmoid_unsqueeze_view` recomputes the router's sigmoid and divide through 4 dependent index loads, with 5 bounds asserts per expert per element.
    - 0.42 ms per layer vs 0.075 ms for the region of https://github.com/felipemello1/torchtitan/pull/110 (0.26 ms without the asserts).
    - The repro's `token_sum_chain` gives 0.206 ms in one graph, 0.115 ms without index asserts, 0.079 ms with the scores passed in (ideal 0.074).
    - Same producer-inlining pattern as request 12.
  - Full-Inductor knobs that recover both: `max_pointwise_cat_inputs=0` + `max_complex_pointwise_cat_inputs=0` (2 MoE layers 27.09 -> 26.03 ms), plus `assert_indirect_indexing=False` (25.52 ms), passed through GraphTrainer's `full_inductor_compilation_pass(inductor_configs=...)`.
- **Multi-output reductions are partitioned by output shape.** The DeepSeek-V4 HcPost backward (16k) is 3 kernels, 558 us (ideal ~265). Inductor splits `grad_comb [T,4,4]`, `grad_residual [T,4,D]` and `grad_x [T,D]` into separate kernels, so the gradient is read 3 times and the residual twice (a ~474 us schedule). The 285 us `grad_comb` kernel computes T*16 separate 4096-long dot products. One kernel producing all three outputs from one read is not generated.
- **Harness note: `torch.compile` of a SelectiveAC-wrapped DeepSeek-V4 block produced 0 graphs** (silent fallback with `fullgraph=False`). Our harness wrapped attn-gym's `gather_attn` in `torch.compiler.disable` inside the checkpointed region; Kimi K3's SAC-wrapped block compiles. Not isolated further.
- **SelectiveAC's dispatch mode costs ~20 us of CPU per op** (8.5 ms self CPU over 435 dispatches), making small units host-bound: Qwen3.5-MoE 2 layers, kernel time 35.2 -> 23.0 ms with our region but wall 38.1 -> 38.6 ms.
- **Libraries that switch kernels on `torch.compiler.is_compiling()` change what compile measures** (attn-gym behavior, not torch): attn-gym's `gather_attn` used its CuTe/FA4 path eagerly and Triton under compile (8.6x slower DeepSeek-V4 attention). Under GraphTrainer's non-strict tracer `is_compiling()` is False, so the CuTe path ran on fake tensors and crashed (fixed in https://github.com/felipemello1/attention-gym/pull/10).

## Background

**Problem.** TorchTitan compiles small named regions and runs the rest of the model eager. Each region is compiled with `fullgraph=True` and the default `recompile_limit=8`, with no eager fallback. For DeepSeek-V4 and Kimi K3, upstream's default regions left 40-56% of layer time on the table versus compiling whole layers; closing the gap took rewrites Inductor could have found itself, plus guard and recompile workarounds. This doc answers: which compiler behaviors forced hand rewrites or region-scoped options, how big each is at real shapes, and what we did instead.

**Grouping of the requests by kind:** codegen gaps (8, 9, 10, 11, 13, 15, 17), guards and recompiles (1, 5), lowering and scheduling choices (10, 12, 14, 16, 19, 26), numerics (20), tracing and caching (2, 3, 6, 7, 21, 22, 23, 24, 25), regions, checkpointing and functionalization (4, 18), compile time (27). Request 10's two parts share one root cause (`unroll_reductions_threshold`).

**How we measured:**

- Hardware and software: one GB300 (SM103), torch 2.15.0.dev20260926+cu130, triton 3.8.0.
- Kernel time = sum of profiler CUDA kernel durations per call. "Ideal" = minimum HBM bytes / 7.1 TB/s (measured).
- Repro numbers come from the runs in `repros/logs/` (one process per run). Numbers quoted from TorchTitan measurements say when they are medians of 3 processes.
- Per-microbatch estimates multiply a measured per-call saving by the number of calls in the model (stated in each request).
- TorchTitan `file:line` citations are at upstream `db050eb3f` (the tree the PRs above are based on) unless noted; torch citations are at the torch version above.

## Raw logs

- `repros/logs/batch_v2.txt`: requests 2, 5 (05d), 8, 10, 11, 13, 14, 17, 21.
- `repros/logs/batch_rest.txt` and `repros/logs/batch_round2.txt`: requests 1, 5 (05a), 20.
- `repros/logs/batch_cpu.txt`: requests 3, 22, 23 (CPU).
- `repros/logs/05b_shared_norm_call_sites.txt` (GPU) and `repros/logs/05c_shared_norm_call_sites_cpu.txt` (CPU): request 5 (b).
- `repros/logs/batch_v3.txt`: requests 9, 12, 15, 16 (request 9's rerun with packed views appended).
- `repros/logs/04_sac_wrapped_region_recompile.txt`: request 4 (CPU and GPU).
- `repros/logs/06_hop_outer_attr_store.txt`: request 6 (CPU).
- `repros/logs/07_autograd_fn_param_base.txt`: request 7 (CPU: backend results, explain output, innermost frames).
- `repros/logs/27_regional_partition_scaling.txt`: request 27 (CPU timings and cProfile).
- Requests without a repro (14 (b), 18, 19, 24-26 and the observations) cite measurements from TorchTitan model and PR runs.

## References

- torch (2.15.0.dev20260926):
  - Inductor: `torch/_inductor/scheduler.py:470-494`, `ir.py:1903-1908`, `fx_passes/decompose_mem_bound_mm.py:34-36,62-90`, `utils.py:3486-3495,4689-4704`, `config.py:1825-1829`, `codegen/triton_utils.py:259-278`, `codegen/triton.py:7492`, `lowering.py:8490-8493`.
  - Dynamo: `torch/_dynamo/side_effects.py:335-383,549-561`, `decorators.py:1347`, `variables/functions.py:3095-3110`, `output_graph.py:817`.
  - FX and AOTAutograd: `torch/fx/experimental/proxy_tensor.py:3078`, `torch/fx/passes/regional_inductor.py:143-160,177-183`, `torch/fx/passes/infra/partitioner.py:50-58`, `torch/_functorch/aot_autograd.py:596`, `torch/_functorch/_aot_autograd/utils.py:660`.
  - Checkpointing: `torch/utils/checkpoint.py:1554-1568`.
- TorchTitan (upstream `db050eb3f`): `torchtitan/models/common/aux_loss.py:165`, `torchtitan/training_engine.py:470-482`, `torchtitan/distributed/utils.py:255-266,477`, `torchtitan/distributed/spmd_types.py:142-147`, `torchtitan/distributed/parallelism_context.py:451`, `torchtitan/experiments/graph_trainer/make_fx_tracer.py:167`.
- Other code: FA4 `flash_attn/cute/interface.py:1482-1501`, `cute_dsl_utils.py:163`; DistMoE (`dist_moe` at ab56bce) `dist_moe/api.py:3614`, `dist_moe/_execution.py:409,478`.
- TorchTitan fork PRs (ours):
  - https://github.com/felipemello1/torchtitan/pull/60
  - https://github.com/felipemello1/torchtitan/pull/97
  - https://github.com/felipemello1/torchtitan/pull/99
  - https://github.com/felipemello1/torchtitan/pull/105
  - https://github.com/felipemello1/torchtitan/pull/106
  - https://github.com/felipemello1/torchtitan/pull/107
  - https://github.com/felipemello1/torchtitan/pull/108
  - https://github.com/felipemello1/torchtitan/pull/109
  - https://github.com/felipemello1/torchtitan/pull/110
  - https://github.com/felipemello1/torchtitan/pull/111
  - https://github.com/felipemello1/torchtitan/pull/113
  - https://github.com/felipemello1/torchtitan/pull/115
  - https://github.com/felipemello1/torchtitan/pull/117
  - https://github.com/felipemello1/torchtitan/pull/118
  - https://github.com/felipemello1/torchtitan/pull/120
  - https://github.com/felipemello1/torchtitan/pull/122
  - https://github.com/felipemello1/torchtitan/pull/124
  - https://github.com/felipemello1/torchtitan/pull/128
  - https://github.com/felipemello1/torchtitan/pull/131
  - https://github.com/felipemello1/torchtitan/pull/132
  - https://github.com/felipemello1/torchtitan/pull/136
  - https://github.com/felipemello1/torchtitan/pull/137
- attention-gym fork PRs (ours):
  - https://github.com/felipemello1/attention-gym/pull/9
  - https://github.com/felipemello1/attention-gym/pull/10
