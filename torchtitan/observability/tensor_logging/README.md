# Tensor logging

Tensor logging records distributions of activations, gradients, parameters, optimizer state, router state, and data statistics without retaining the original tensors. Model code declares stable names during construction and records values where they are produced; the trainer handles cadence, distributed reduction, and publication to TensorBoard or Weights & Biases.

```text
model construction     register owner + fixed names
selected train step    accumulate local sufficient statistics
backward               record incoming cotangents for paired names
end of step            reduce three packed WORLD buffers
host                    derive scalar metrics, filter, publish, clear
```

Registration gives every metric a fixed row in preallocated buffers. Adding an ordinary tensor metric increases the buffer size but does not add a collective.

## Quickstart

Enable tensor logging and a metrics sink on an existing recipe:

```bash
NGPU=8 MODULE=llama3 CONFIG=llama3_8b ./run_train.sh \
    --metrics.enable-tensor-logging \
    --metrics.enable-tensorboard
```

The relevant defaults are:

```python
enable_tensor_logging = False
tensor_logging_freq = 5
tensor_logging_metrics_filter_regex = (
    r"\.w:(?:kurtosis|zero_frac)$"
    r"|:(?:numel|nonfinite_count|abs_mean|square_mean|abs_max)$"
)
tensor_logging_adam_momentum_gradient_cosine = False
```

The effective cadence is `max(tensor_logging_freq, log_freq)`, so tensor scans never run more frequently than ordinary logging. For example, `tensor_logging_freq=5` and `log_freq=10` records tensor statistics at steps 10, 20, and so on. Tensor logging does not inherit ordinary metrics' special publication at step 1.

The filter matches `<full_name>:<statistic>` and controls publication only. It does not skip tensor scans or reduce GPU work. Use `'.*'` while debugging to publish every derived statistic.

See the [observability README](../README.md) for TensorBoard and Weights & Biases setup.

## Record tensors in model code

Every recording point has two parts:

1. Register fixed names in `__init__`, before the trainer initializes tensor logging.
2. Record tensors where they exist in `forward` or another execution path.

The first argument to both calls is the **owner**. Its path in the model supplies the metric-name prefix.

### Forward tensor and backward cotangent

Use `register_fwd_bwd` and `log_fwd_bwd_stats` to record a tensor now and the cotangent that reaches it during backward:

```python
class FeedForward(nn.Module):
    def __init__(self, ...):
        ...
        register_fwd_bwd(self, ["act_out"])

    def forward(self, hidden):
        act_out = self.activation(self.w1(hidden))
        log_fwd_bwd_stats(self, act_out=act_out)
        return self.w2(act_out)
```

This produces names ending in `act_out.x` and `act_out.dx`. The call attaches an autograd hook to `act_out`, returns `None`, and does not replace the tensor. Keep using the original `act_out` downstream.

Use `log_fwd_bwd_stats` only with differentiable tensors. Under `torch.no_grad()`, it is a no-op so validation does not enter the training window.

### Forward-only or internal tensor

Use `register` and `log_stats` when there is no meaningful backward value:

```python
class Router(nn.Module):
    def __init__(self, gate):
        ...
        register(self, ["router_logits"])

    def forward(self, hidden):
        router_logits = self.gate(hidden)
        log_stats(self, router_logits=router_logits)
        return router_logits
```

The same API works at module boundaries and for intermediate calculations inside a module. Nothing discovers module boundaries automatically, and callers do not pass a device mesh.

### Owner and name rules

The owner passed during recording must be the same object used during registration:

```python
register_fwd_bwd(self.lm_head, ["output"])
...
logits = self.lm_head(hidden)
log_fwd_bwd_stats(self.lm_head, output=logits)
```

Register every possible name before initialization. Recording an unknown name raises `KeyError`; registering the same full name twice raises `ValueError`. Different pipeline ranks may own different names because initialization gathers the complete name set across ranks.

## From tensors to metrics

Each observation contributes one row of mergeable sufficient statistics:

```text
counts = [numel, nonfinite_count, zero_count, observation_count]
sums   = [abs_sum, square_sum, fourth_moment_sum]
maxima = [abs_max]
```

At publication, the trainer performs three packed collectives over the default WORLD group:

```text
all count rows     --SUM--> one count buffer
all sum rows       --SUM--> one sum buffer
all maximum rows   --MAX--> one maximum buffer
```

For example, two ranks record one tensor each:

```text
rank 0: [1, -3]    numel=2  abs_sum=4  square_sum=10  zeros=0  abs_max=3
rank 1: [2,  0]    numel=2  abs_sum=2  square_sum= 4  zeros=1  abs_max=2

WORLD:              numel=4  abs_sum=6  square_sum=14  zeros=1  abs_max=3
derived:             abs_mean=1.5  square_mean=3.5  zero_frac=0.25
```

The full derived set includes counts, sums, `zero_frac`, `abs_mean`, `square_mean`, RMS, `abs_max`, and excess kurtosis. The configured regex selects which leaves reach the metrics sink.

Ordinary rows count physical observations. A replicated tensor contributes once per rank that holds it, so absolute counts include replica multiplicity. Ratios such as `abs_mean`, `zero_frac`, and kurtosis remain unchanged under uniform replication.

## Metrics that need topology

The common recording API deliberately does not guess TP, CP, DP, or EP semantics. Most tensor distributions can merge physical observations over WORLD, but metrics such as expert load require a semantic population to be reconstructed first.

Router statistics therefore follow this lifecycle:

```text
each layer buffers local router state
        -> stack all layers
        -> reduce over the groups that shard that population
        -> derive entropy, load, or imbalance
        -> log_stats(router, derived_name=derived_tensor)
        -> ordinary packed WORLD publication
```

Keeping mesh-specific mathematics next to the producer makes the required population explicit. It also batches each topology reduction across layers instead of issuing one collective per layer.

Dataset and document metrics use a separate exact-counter path. Contributor flags avoid duplicate TP/CP observations, and one packed `float64` SUM reconstructs weighted loss and document statistics for the publication window.

## Names and built-in coverage

Names combine the owner's model path, registered key, and derived statistic:

```text
layers.0.attention.xq.x.abs_mean
layers.0.attention.xq.dx.abs_max
layers.0.feed_forward.act_out.x.square_mean
layers.0.moe.router.expert_load.abs_max
layers.0.attention.wo.weight.w.zero_frac
layers.0.attention.wo.weight.dw.abs_mean
layers.0.attention.wo.weight.normed_dw.abs_max
layers.0.attention.wo.weight.numerator.abs_mean
gradients.all.abs_max
gradients.moe.abs_max
data/datasets.<dataset_id>.valid_token_count
data/documents.segment_length_mean
```

Built-in instrumentation covers decoder input and logits; transformer residual streams and branch outputs; Q/K/V and attention-head values; dense feed-forward and MoE intermediates; trainable parameters and gradients before and after clipping; Adam state; whole-model and MoE gradient aggregates; router health; and dataset/document counters. Exact activation names follow each model implementation.

`tensor_logging_adam_momentum_gradient_cosine=True` additionally records the per-parameter cosine between the Adam first moment and gradient. It is disabled by default because exact reconstruction of a sharded parameter can require per-parameter communication.

## Add a metric

Choose the smallest existing path that matches the quantity:

```text
ordinary tensor distribution
    register / register_fwd_bwd in setup
    log_stats / log_fwd_bwd_stats where the tensor exists

semantic value requiring a specific topology
    reconstruct the required population beside its producer
    derive the semantic tensor, then pass it to log_stats

exact counter or weighted mean
    accumulate exact sufficient statistics like DataStatistics

scalar already owned by the trainer
    keep it in the ordinary MetricsProcessor path
```

An ordinary tensor should not need a new wrapper, metric family, mesh registry, or dictionary threaded through model code.

## Execution modes

- Full and selective activation checkpointing preserve each original forward observation exactly once while still recording its backward cotangent.

- Regional full-graph `torch.compile` is supported. The recording decision lives in a device buffer, so selected and unselected steps use the same compiled graph.

- In-process Graph Trainer tracing and CUDA-graph replay are supported on their validated model paths. Separately produced or loaded precompiled artifacts reject tensor logging at setup because registered names and live buffers are not portable across artifacts.

- Ordinary and interleaved pipeline schedules are supported. Multiple local model parts map back to global model paths, so names remain `layers.<global_id>...` instead of depending on rank-local part indices.

- The strict `spmd_types` and default DTensor backends are supported on their validated paths. The combined Graph Trainer plus strict-`spmd_types` path is currently blocked by existing Graph Trainer mesh/backend setup before tensor logging initializes.

- CUDA uses a Triton accumulator and CPU uses an eager reference implementation. ROCm has not been validated.

## Current limitations

- GraphPP ZBV training publishes backward and router metrics, but its forward `.x` accumulator mutations are not yet preserved through the production GraphPP path.

- Publication runs synchronously with the training step. A broad filter can produce thousands of sink calls, so use a narrow filter for routine runs and widen it for focused debugging.

- The publication regex controls sink volume, not GPU collection work. Increase `tensor_logging_freq` to reduce collection frequency.

- Adam numerator and denominator metrics apply only to Adam and AdamW optimizers.
