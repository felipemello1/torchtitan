# Tensor logging

Tensor logging publishes distributed statistics for selected model, optimizer, router, and data tensors without retaining full tensors or exposing process groups to recipes.

```text
authoritative producer
  -> fixed sufficient statistics on device
  -> packed reductions over setup-bound owners
  -> one canonical writer derives scalars
  -> existing TensorBoard or W&B logger
```

## Recipe API

Tensor logging has three recipe fields:

```bash
--tensor-logging.enable \
--tensor-logging.families PARAMETER PRECLIP_GRADIENT DATASET_LOSS \
--tensor-logging.layer-ids 0 7
```

- `enable` turns the component on.
- `families` selects semantic built-in metrics. The default is `PARAMETER PRECLIP_GRADIENT`.
- `layer_ids` selects global decoder layers for layer-owned families. Job/data-only selections keep the default `(0,)`; a different value fails setup instead of being ignored.

Publication uses `metrics.log_freq`. There is no per-family cadence, mesh, process-group, reduction, or arbitrary metric-name knob.

At least one existing scalar sink must be enabled:

```bash
--metrics.enable-tensorboard
# or
--metrics.enable-wandb
```

## Built-in families

Dense boundaries:

```text
BOUNDARY_OUTPUT
BOUNDARY_OUTPUT_COTANGENT
```

Selected parameters and completed gradients:

```text
PARAMETER
PRECLIP_GRADIENT
```

Qwen3 MoE router and expert state:

```text
ROUTER_DISTRIBUTION
OFFERED_ASSIGNMENTS
PER_SEQUENCE_ROUTING
EXPERT_COMPUTE_ROWS
EXPERT_BIAS
```

`ROUTER_DISTRIBUTION` reports the entropy of the L1-normalized sigmoid/softmax of the count-weighted mean router logits plus shifted expert bias. It is measured in nats in `[0, ln(num_experts)]`; zero total activated mass has entropy zero, and this is not mean per-token entropy. Offered assignments describe the physical routed positions seen by the dispatcher, including sequence padding introduced by the supported EP/SP path.

Whole-model and AdamW state:

```text
WHOLE_GRADIENT
OPTIMIZER_DISTRIBUTION
MOMENTUM_GRADIENT_COSINE
```

Trainer data and objective state:

```text
DATASET_LOSS
DOCUMENT_SEGMENTS
BLOCK_CAUSAL_MOMENTS
```

Keys begin with `tensor_metrics/`. Layer-owned keys use the canonical global layer and parameter names; data keys use the stable dataset config ID.

## Two recording mechanisms

### Module and parameter observation

Boundary and parameter families are installed during setup on selected post-parallelization modules and parameters. Setup reads the direct projection sharding contract, DTensor placements, and `ParallelDims`; the recipe does not supply topology.

Hooks immediately reduce each observed tensor to bounded sufficient statistics such as element count, absolute sum, square sum, maximum, zero count, and nonfinite count. Full activations and gradients are not retained.

### Prebound authoritative-owner recording

Router assignments, expert rows, optimizer state, and data statistics do not exist at a useful module return. Their authoritative owner receives a prebound recorder:

```python
if self.expert_compute_rows_recorder is not None:
    self.expert_compute_rows_recorder(num_global_tokens_per_local_expert)
```

The call carries only the semantic payload. Setup already fixed its family, shape, global IDs, contributor ranks, reduction, and cadence. Do not pass a metric dictionary or topology through model returns.

## Distributed behavior

A semantic population is not necessarily every physical rank.

```text
TP-replicated value        -> one TP representative contributes
TP-sharded value           -> every shard contributes its logical elements
CP-split loss              -> every CP partial contributes
full pre-CP data batch     -> one CP representative contributes
DP batches                 -> every distinct batch owner contributes
EP local-expert vector     -> each owner contributes its global expert slice
```

Compatible rows are packed by dtype and owner cohort. One collective is not launched per scalar metric. Ratios and norms are derived only after exact numerators and denominators reach the writer.

## Cadence

Boundary, parameter, whole-gradient, expert-bias, and optimizer families are point samples from the selected logging step. Router-distribution, expert-count, and data families accumulate every training step and publish/reset at `metrics.log_freq`. `PER_SEQUENCE_ROUTING` instead publishes the final forward sample in that window.

`observation_count` uses the family's authoritative unit: module calls for boundary rows, producer calls after replica exclusion for interval rows, and one for point samples. `window_steps` states how many successful optimizer steps elapsed since the preceding publication. A checkpoint load resets nonpersistent partial-window state.

Publication steps perform the packed reductions, host transfer, scalar derivation, and sink write, so they are slower than adjacent non-publication steps. `metrics.log_freq` amortizes that work; choose it with the selected family count and sink cost in mind.

## Current support boundary

The component supports the core Trainer with ordinary Llama3 or Qwen3 model configs, the default communication mode, and TensorBoard or W&B on one canonical writer.

- Parameter and trainer-owned families support eager execution, SelectiveAC, and inductor model compile where their source runs outside recomputed model hooks.
- Boundary and internal MoE forward families currently require eager execution without activation checkpointing, model compile, or validation-enabled jobs.
- Data-only families support CP, DP replicate, EP coexistence, and the `spmd_types` backend. `full_dtensor` remains unsupported.
- Pipeline parallelism, CPU offload, Graph Trainer, CUDA graphs, quantized model configs, and all-rank scalar sinks fail setup.
- Document/data metrics currently require one `HuggingFaceTextDataLoader` dataset carrying packed-document positions.
- AdamW distribution/cosine families require the public FP32-state AdamW implementation.

Unsupported combinations fail during setup rather than silently changing their population or failing at publication.

## Adding a built-in site

1. Choose the authoritative tensor and decide whether it is a module/parameter boundary or needs a prebound recorder.
2. Define its exact semantic population, sufficient statistics, denominator, cadence, absence rule, and emitted keys.
3. Add one `TensorMetricFamily` value and bind it during `TensorLogging` setup. Do not add a user process-group or reduction knob.
4. For an internal site, pass only the tensor payload at the authoritative callsite.
5. Add an exact distributed test that would fail for a plausible wrong cohort, plus one normal Trainer run proving sink output.
6. Add execution-mode support only after the real producer passes that mode; otherwise fail setup and document the reopening condition.

Prefer extending an existing recorder when the new row has the same owner cohort, dtype, cadence, and lifecycle. Keep a separate literal recorder when those semantics differ; do not introduce a generic metric registry.
