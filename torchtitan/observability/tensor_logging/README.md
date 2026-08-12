# Tensor logging

Tensor logging records distributions of model tensors at the trainer's normal logging cadence. Model code has one recording API for both module boundaries and internal tensors:

```python
class FeedForward(nn.Module):
    def __init__(self, ...):
        ...
        register_fwd_bwd(self, ["act_out"])

    def forward(self, value):
        act_out = self.activation(self.w1(value))
        act_out = log_fwd_bwd_stats(self, act_out=act_out)
        return self.w2(act_out)
```

`register_fwd_bwd()` creates `act_out.x` and `act_out.dx`. `log_fwd_bwd_stats()` returns the observed tensor, records it immediately, and records its incoming cotangent during backward. Assign the return value so `torch.compile` preserves the backward observation. Use `register()` and `log_stats()` when only the current tensor is needed.

The trainer owns the lifecycle:

```text
model construction       register fixed keys
trainer initialization   init fixed device slots
selected train step      record into those slots
end of selected step     reduce packed slots, publish scalars, clear
```

All ordinary keys share three WORLD collectives per publication: integer counts use SUM, floating sums use SUM, and maxima use MAX. Adding an ordinary key grows the packed slabs but does not add another collective.

## Configuration

```python
MetricsProcessor.Config(
    log_freq=10,
    enable_tensor_logging=True,
    tensor_logging_freq=100,
    tensor_logging_metrics_filter_regex=(
        r"layers\..*\.(attn_stream|ffn_stream).*:(?:abs_mean|abs_max)$"
    ),
    tensor_logging_optimizer_cosine=False,
)
```

`tensor_logging_freq` follows Llama4x's separate tensor-stat cadence and must be a multiple of `log_freq`; unlike ordinary trainer logging, it does not special-case step 1. The regex is an allowlist over `<full_key>:<statistic>` and only filters publication. It does not change model computation or slot accumulation. Adam momentum/gradient cosine is separately opt-in because reconstructing a sharded parameter's scalar products can require communication per parameter.

The regex applies to ordinary tensor rows. Trainer-owned `data/*` rows are exact counters and weighted statistics and remain visible whenever tensor logging is selected.

## Built-in coverage

TorchTitan records decoder input/logits; transformer residual streams and branch outputs; Q/K/V, normalized Q/K, and attention-head outputs; dense-FFN and MoE intermediates; every trainable parameter's gradient, clipped gradient, post-step weight, and Adam state; whole-model and MoE gradient aggregates; router entropy/load/imbalance/bias; and weighted dataset/document statistics. Names include the concrete module path, for example `layers.0.attention.xq.dx.abs_mean`.

## Topology-specific metrics

The recording call does not accept a mesh. Generic tensor distributions describe the physical observations emitted by model ranks.

Metrics whose meaning requires topology reconstruction do that before ordinary publication. For example, router code stacks every layer's expert counts, reduces the stack over the required TP/DP/CP groups, derives expert load and imbalance, and then records those derived tensors with `log_stats()`. This keeps mesh knowledge in the producer that understands the metric rather than in the general recording API.

## Execution modes

The accumulation operation has fixed nonpersistent buffers, an opaque custom operation, and a custom-autograd identity, so forward and backward observations work with Graph Trainer and ordinary full-graph `torch.compile`. Full and selective activation checkpointing disable recording during recomputation; the original forward and backward cotangent are each observed once.

Pipeline parallel tensor logging currently fails at setup. It needs stable global stage names before local pipeline parts can publish unambiguous keys.
