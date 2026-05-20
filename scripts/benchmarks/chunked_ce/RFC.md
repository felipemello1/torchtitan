# Chunked Cross-Entropy Reducers for SFT and RL

## TL;DR

TorchTitan's chunked cross-entropy path saves memory by running the LM head and cross entropy over sequence chunks, but the existing public boundary is only a scalar SFT loss. RL losses such as GRPO need selected token logprobs, and sequence-level losses such as GSPO need a real `[batch, sequence]` selected-logprob tensor.

This RFC proposes keeping one `ChunkedCELoss` class and adding two boundaries:

```python
loss = chunked_ce.reduce_selected_token_logprobs(hidden_states, labels, reducer)
logprobs = chunked_ce.compute_selected_token_logprobs(hidden_states, labels)
```

`reduce_selected_token_logprobs` is for token-local losses such as SFT, GRPO, DAPO, CISPO, and SAPO. The reducer receives one `[batch, chunk]` selected-logprob tensor at a time, returns a scalar, and the implementation immediately backprops through the chunk. This preserves the memory pattern of chunked CE and avoids materializing `[batch, sequence]`.

`compute_selected_token_logprobs` is for sequence-wise losses such as GSPO. It returns a real `[batch, sequence]` autograd tensor and replays the LM head during backward so full `[batch, sequence, vocab]` logits are never retained.

What this branch has already demonstrated:

- CPU correctness passes for SFT, GRPO, and GSPO-style losses. The tests compare scalar loss, selected logprobs, hidden-state gradients, LM-head gradients, uneven chunk offsets, and external loss scaling.
- SFT distributed smoke matches same-SHA `upstream/main@52a292d29` scalar loss at step 1 under FSDP=4, FSDP=2+TP=2, TP=4, and compiled FSDP=2+TP=2 with `num_chunks=8`.
- The TP configurations intentionally change `grad_norm` versus current upstream because the branch fixes the hidden-gradient placement path. At `L=2048`, FSDP=2+TP=2 changes step-1 `grad_norm` from `1.3237` to `1.5195`; TP=4 changes it from `1.2737` to `1.5189`.
- The long-context compiled stress at `L=131072`, `BS=1`, FSDP=2+TP=2, `num_chunks=8` matches step-1 scalar loss and memory versus same-SHA upstream: both report `loss=8.12721` and `memory=4.88 GiB`; candidate step-1 `grad_norm=1.5613` versus upstream `1.3576`.
- The RL smoke runs Qwen3-0.6B with 2 generator GPUs + 2 trainer GPUs, trainer/generator TP=2, `num_chunks=8`. It emits TensorBoard metrics `train/grad_norm/mean=11.4375`, `perf/tokens_per_second=53.0643`, and `train/memory/max_active_gib=2.01168`.

Do not overread the smoke data. Exact SFT bitwise claims still need TensorBoard export or `scripts/loss_compare.py`; stdout is only five-significant-digit evidence. The RL smoke proves the distributed actor path and nonzero backward under TP=2, not FSDP-only RL.

## Walking Example

```python
B, L, V, H = 2, 16, 32, 8
num_chunks = 4
hidden_states = torch.randn(B, L, H, dtype=torch.bfloat16)
labels = torch.randint(0, V, (B, L))
lm_head = nn.Linear(H, V, bias=False, dtype=torch.bfloat16)
loss_fn = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=num_chunks))
loss_fn.set_lm_head(lm_head)
```

SFT remains scalar:

```python
loss = loss_fn(hidden_states, labels)
```

GRPO uses the reducer path:

```python
reducer = grpo_loss.make_token_reducer(packed_policy_inputs)
loss = loss_fn.reduce_selected_token_logprobs(hidden_states, labels, reducer.chunk_loss)
```

GSPO uses the selected-logprob tensor path:

```python
policy_logprobs = loss_fn.compute_selected_token_logprobs(hidden_states, labels)
loss = gspo_loss.sequence_loss(policy_logprobs, packed_policy_inputs)
```

## Current Limitation

The existing chunked CE loop is shaped like this:

```python
total_loss = 0
for chunk in chunks:
    h_chunk = hidden_states[:, chunk, :].detach().requires_grad_(True)
    logits = lm_head(h_chunk)
    chunk_loss = F.cross_entropy(logits.flatten(0, 1), labels[:, chunk].flatten(), reduction="sum")
    chunk_loss.backward()
    grad_accumulator.add(h_chunk.grad)
    total_loss = total_loss + chunk_loss.detach()
return bridge_to_decoder_backward(hidden_states, grad_accumulator.result(), total_loss)
```

That is memory efficient because only one `[batch, chunk, vocab]` logits tensor exists at a time. The problem is that RL losses need a boundary between logits and scalar loss:

```python
selected_logprobs = -F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
ratio = torch.exp(selected_logprobs - generator_logprobs)
loss = clipped_policy_loss(ratio, advantages, mask)
```

The current chunked CE API never exposes `selected_logprobs`, so RL code either falls back to dense logits or duplicates the chunk loop in the trainer.

## Proposed API

```python
def selected_token_logprobs(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
    losses = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="none", ignore_index=ignore_index).view_as(labels)
    return -losses
```

`selected_token_logprobs` is the primitive. It uses `F.cross_entropy(reduction="none")` instead of `log_softmax(...).gather(...)` so ignored labels are handled correctly and the implementation stays aligned with optimized PyTorch CE.

```python
class ChunkedCELoss(BaseLoss):
    def set_lm_head(self, lm_head: nn.Module) -> None: ...

    def reduce_selected_token_logprobs(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        reducer: Callable[[torch.Tensor, torch.Tensor, slice], torch.Tensor],
        *,
        ignore_index: int = -100,
    ) -> torch.Tensor: ...

    def compute_selected_token_logprobs(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        *,
        ignore_index: int = -100,
    ) -> torch.Tensor: ...
```

The reducer receives `policy_logprobs_chunk`, `labels_chunk`, and the absolute sequence `slice`. The slice lets the reducer index side inputs such as generator logprobs, advantages, masks, and per-token weights without forcing those tensors into the chunked CE class.

## GRPO Reducer Example

```python
class GRPOTokenReducer:
    def __init__(self, inputs: PackedPolicyLossInputs, clip_eps: float):
        self.inputs = inputs
        self.clip_eps = clip_eps
        self.loss_sum = torch.zeros((), device=inputs.loss_mask.device, dtype=torch.float32)
        self.ratio_sum = self.loss_sum.clone()
        self.clipped_sum = self.loss_sum.clone()

    def chunk_loss(self, policy_logprobs: torch.Tensor, labels: torch.Tensor, token_slice: slice) -> torch.Tensor:
        generator = self.inputs.generator_logprobs[:, token_slice]
        advantages = self.inputs.advantages[:, token_slice]
        weights = self.inputs.loss_weights[:, token_slice].to(policy_logprobs.dtype)
        ratio = torch.exp(policy_logprobs - generator.detach())
        clipped_ratio = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
        pg_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
        chunk_loss = (pg_loss * weights).sum()
        with torch.no_grad():
            self.loss_sum = self.loss_sum + chunk_loss.detach().float()
            self.ratio_sum = self.ratio_sum + (ratio.detach().float() * weights).sum()
            self.clipped_sum = self.clipped_sum + ((ratio != clipped_ratio).float() * weights).sum()
        return chunk_loss
```

This works because GRPO token loss is chunk-local. Every token's contribution depends on the current policy logprob, generator logprob, token advantage, and token weight for that token. No sequence-level reduction is needed before clipping.

## GSPO Example

GSPO clips at sequence level, so the reducer path is not enough:

```python
policy_logprobs = loss_fn.compute_selected_token_logprobs(hidden_states, labels)
response_logprobs = extract_response_logprobs(policy_logprobs, seq_lens, prompt_lens, response_lens)
loss = gspo_loss(response_logprobs, advantages, generator_token_logprobs)
```

This path materializes `[batch, sequence]`, not `[batch, sequence, vocab]`. The forward chunk loop computes selected logprobs chunk by chunk and concatenates them. Backward replays the LM head chunk by chunk and accumulates gradients for hidden states and LM-head parameters.

## Autograd Bridge

The scalar reducer path runs inner `chunk_loss.backward()` calls during forward, then returns a scalar whose backward must propagate the accumulated hidden-state and LM-head parameter gradients through the outer graph.

The bridge must multiply all captured gradients by upstream `grad_output`:

```python
class _ChunkedLossWithParamGrads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, accumulated_hidden_grad, loss, lm_head, fsdp_enabled, *params):
        ctx.save_for_backward(accumulated_hidden_grad, *captured_param_grads)
        return loss.detach()

    @staticmethod
    def backward(ctx, grad_output):
        accumulated_hidden_grad, *param_grads = ctx.saved_tensors
        hidden_grad = accumulated_hidden_grad * grad_output
        scaled_param_grads = tuple(None if grad is None else grad * grad_output for grad in param_grads)
        return hidden_grad, None, None, None, None, *scaled_param_grads
```

Ignoring `grad_output` silently breaks `(scale * loss).backward()` and gradient accumulation schemes that scale the loss outside the chunked CE call.

## FSDP and DTensor Notes

For FSDP, the LM head should remain unsharded across the per-sequence chunk loop and gradient sync should happen once at the end of the loop. A small context manager should temporarily set the FSDP reshard and gradient-sync flags and restore them in `finally`/`__exit__`.

For tensor parallel outputs, hidden states may need to be replicated along the TP axis before applying the LM head in the loss. The selected-logprob primitive must also handle DTensor logits and plain tensor labels. Upstream TorchTitan now has a DTensor-aware `cross_entropy_loss` path for SFT; this RFC only needs extra label preparation for the new selected-logprob primitive unless a test proves SFT CE still needs it.

## Alternatives Considered

Dense logits would compute `[batch, sequence, vocab]` and then run the normal RL loss. This is the simplest API, but it loses the memory benefit that motivated chunked CE.

Returning `[batch, sequence]` for every RL loss would also be simple. The problem is that token-local losses such as GRPO do not need a sequence tensor before reducing. Forcing them through `compute_selected_token_logprobs` adds backward replay and can add extra FSDP LM-head gathers that the scalar reducer avoids.

Using only the scalar reducer would keep the implementation small, but it fails for GSPO and other sequence-wise losses because they need the full selected-logprob tensor before the sequence reduction.

An online-LSE backend, as used by Liger and prime-rl style implementations, can be more memory efficient for very large vocabularies. This RFC defers it because it is a backend choice, not the API boundary. The reducer and tensor APIs should be able to host this backend later.

Vocab-parallel CE/logprob implementations, as used by Megatron-style, slime-style, and AReaL-style code, are useful when the vocabulary is sharded. This RFC defers that path because TorchTitan first needs an API that works cleanly with its FSDP, TP, CP, and DTensor stack.

A trainer-owned chunk loop, as used by the internal reference implementation, puts selected-logprob computation and policy-loss orchestration directly in the trainer. This RFC only adopts the useful loss ownership idea through the reducer callable. The LM-head, FSDP, TP, and DTensor mechanics stay inside `ChunkedCELoss` so every trainer loss does not have to duplicate model-parallel details.

## Related Library Patterns

Across libraries, the stable RL boundary is selected token logprobs plus side tensors such as generator logprobs, reference logprobs, advantages, masks, and normalization weights. The main differences are who owns chunking, whether selected logprobs are produced by dense CE, online-LSE, or vocab-parallel kernels, and whether the implementation has to preserve model-parallel details.

### Liger

Liger's `_ChunkedSelectiveLogProbFunction` returns per-token selected logprobs from hidden states and an LM-head weight. It is an online-LSE custom autograd function, so the largest temporary is `[seq_chunk, vocab_chunk]` instead of `[seq_chunk, vocab]`.

```python
class _ChunkedSelectiveLogProbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, bias, temperature, vocab_chunk_size):
        logprobs, log_z = _selective_logprob_forward(hidden, weight, targets, bias, temperature, vocab_chunk_size)
        ctx.save_for_backward(hidden, weight, targets, bias, log_z)
        ctx.temperature = temperature
        ctx.vocab_chunk_size = vocab_chunk_size
        return logprobs

    @staticmethod
    def backward(ctx, grad_logprobs):
        hidden, weight, targets, bias, log_z = ctx.saved_tensors
        return _selective_logprob_backward(hidden, weight, targets, bias, log_z, grad_logprobs, ctx.temperature, ctx.vocab_chunk_size)
```

The useful idea is the selected-logprob boundary. The part not copied here is Liger's distribution model: Liger has no FSDP/TP/CP/DTensor logic in this primitive, so TorchTitan still needs the FSDP reshard and DTensor placement logic around the LM head.

### prime-rl

prime-rl replaces the output linear layer with a module whose forward returns selected logprobs and entropy instead of logits. It also uses online-LSE and computes entropy from the same recurrence.

```python
class FusedOutputLinear(torch.nn.Linear):
    def forward(self, hidden_states, labels, temperature):
        b, s, h = hidden_states.shape
        hidden = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()
        inv_t = 1.0 / temperature.reshape(b * s).contiguous()
        logprobs, entropy = _SequenceChunkedLogProbEntropyFn.apply(hidden, self.weight, labels, inv_t, self.chunk_size)
        return PrimeLmOutput(logprobs=logprobs.reshape(b, s), entropy=entropy.reshape(b, s))
```

The key design is that the trainer sees an LM output object with exactly the tensors RL needs:

```python
@dataclass
class LossInputs:
    trainer_logprobs: torch.Tensor
    inference_logprobs: torch.Tensor
    advantages: torch.Tensor
    loss_mask: torch.Tensor

def default_loss_fn(inputs, config):
    log_ratio = inputs.trainer_logprobs - inputs.inference_logprobs
    ratio = torch.exp(log_ratio)
    return ((-inputs.advantages * ratio) * inputs.loss_mask).sum()
```

The useful idea is that RL losses consume selected logprobs and side tensors, not logits. The part deferred here is the online-LSE backend; this RFC keeps `F.cross_entropy(reduction="none")` as the first backend because it matches PyTorch CE semantics, handles `ignore_index`, and keeps the SFT and RL paths close to TorchTitan's existing chunked CE.

### Internal Reference

The internal reference RL stack is trainer-owned. The trainer chunks the work, calls a selected-logprob primitive inside each chunk, passes all side tensors through a loss input object, and backprops each chunk. Its pretraining scalar CE path is separate from the RL path.

```python
def selective_log_softmax(logits, target_ids):
    return F.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

class LossFn:
    def __call__(self, inputs: LossInput) -> LossOutput:
        ratio = torch.exp(inputs.trainer_log_probs - inputs.inference_log_probs)
        numerator = (inputs.advantages * ratio * inputs.loss_mask).sum()
        denominator = inputs.loss_mask.sum()
        return LossOutput(loss=Fraction(numerator, denominator), metrics=...)
```

That shape is good for a stack where the trainer already owns custom distributed selected-logprob primitives, including vocab-to-sequence all-to-all. The design in this RFC is different on purpose:

- The internal reference puts the chunk loop in the trainer; this RFC keeps the chunk loop in `ChunkedCELoss`.
- The internal reference lets each trainer path choose its selected-logprob primitive; this RFC uses `F.cross_entropy(reduction="none")` first so ignored-label handling and SFT CE semantics stay aligned.
- The internal reference loss consumes a `LossInput` object and returns numerator/denominator style outputs; this RFC uses a reducer callable for token-local losses and a `[batch, sequence]` selected-logprob tensor for sequence-level losses.
- The internal reference can expose custom distributed primitives directly to the trainer; this RFC hides FSDP resharding, TP placement, CP layout, LM-head ownership, and DTensor handling inside `ChunkedCELoss` so RL losses can remain ordinary policy-loss code.

### NeMo-RL

NeMo-RL dispatches on a logprob boundary and chooses dense, TP-aware, or fused hidden-state-to-logprob backends.

```python
def get_next_token_logprobs_from_logits(logits_or_hidden, model, ...):
    if dense_path:
        return F.log_softmax(logits, dim=-1).gather(-1, labels[..., None]).squeeze(-1)
    if tp_dtensor_path:
        return ChunkedDistributedLogprob.apply(logits, labels, ...)
    return ChunkedDistributedHiddenStatesToLogprobs.apply(hidden_states, model.lm_head.weight, labels, tp_group, chunk_size, ...)

class ClippedPGLossFn:
    input_type = LossInputType.LOGPROB
```

The useful idea is backend selection behind a selected-logprob API. TorchTitan can add online-LSE or vocab-parallel backends later without changing the reducer or sequence-wise loss APIs. This argues for naming the new API around selected token logprobs rather than around one implementation detail.

### torchtune

torchtune's `LinearCrossEntropyLoss` is the small scalar-SFT reference: the loss owns the LM head, chunks hidden states, and reduces each chunk to a scalar.

```python
class LinearCrossEntropyLoss(nn.Module):
    def set_model_output(self, model):
        model.skip_output_layer = True
        self.linear_projection = model.output

    def forward(self, outputs, targets):
        hidden, target = self.mask_inputs(outputs, targets)
        loss = 0.0
        for h_chunk, t_chunk in zip(hidden.tensor_split(self.num_output_chunks), target.tensor_split(self.num_output_chunks)):
            logits = self.linear_projection(h_chunk)
            loss = loss + F.cross_entropy(logits.float(), t_chunk, reduction="sum", ignore_index=self.ignore_index)
        return loss
```

This scalar pattern is useful for SFT, but returning a `[batch, sequence]` selected-logprob tensor from a plain Python chunk loop keeps the per-chunk graphs alive. The tensor path therefore needs a custom autograd replay boundary.

### Megatron-Style, Slime-Style, AReaL-Style, Verl, OpenRLHF, And trl

Megatron-style code is strongest when vocab-parallel CE is the native distributed primitive. That path avoids gathering a full vocabulary shard on every rank, but it assumes the rest of the stack is organized around vocab-parallel logits.

Slime-style code wraps vocab-parallel selected-logprob work in a sequence-chunked path. The useful idea is that selected logprobs can be chunked without materializing full logits. The part that does not directly transfer is the Megatron-style distribution substrate.

AReaL-style code exposes vocab-parallel selected logprobs through a custom autograd function. This is a possible later backend for TorchTitan if the DTensor semantics are pinned down.

Verl, OpenRLHF, and trl share the same high-level policy-loss boundary: policy logprobs, behavior or generator logprobs, optional reference logprobs, advantages, masks, and normalization. Some variants chunk SFT NLL or async-GRPO logprob computation, but the loss itself still wants selected token logprobs rather than full logits.

The conclusion for TorchTitan is narrow: implement the selected-logprob boundary once in `ChunkedCELoss`, keep token-local RL losses on the scalar reducer path, provide a tensor path for sequence-wise losses, and leave lower-memory online-LSE or vocab-parallel backends as later replacements behind the same API.

## Evidence So Far

The standalone prototype uses `B=2, L=17, V=32, H=8, num_chunks=5` and compares dense SFT, GRPO, and sequence-reducing selected-logprob losses against the chunked paths. It checks scalar loss, selected logprobs, hidden gradients, LM-head gradients, uneven chunk offsets, and external loss scaling. Confidence is high for CPU numerical parity.

The integrated SFT unit tests use `B=2, L=8, V=64, H=32, num_chunks=4` and compare dense CE to `ChunkedCELoss`, including external loss scaling and LM-head gradients. Confidence is high for CPU numerical parity.

The integrated RL unit tests use `B=1, L=8, V=17, H=11, num_chunks=4` and compare dense selected-logprob GRPO to the chunked reducer path, including hidden and LM-head gradients. They also compare dense selected-logprob GSPO to `compute_selected_token_logprobs`, including loss metrics, hidden gradients, and LM-head gradients. Confidence is high for CPU numerical parity.

The real TorchTitan SFT deterministic matrix uses the Llama debug model, `num_chunks=8`, `L=2048`, FSDP=4, FSDP=2+TP=2, TP=4, and compiled FSDP=2+TP=2. A fresh same-SHA comparison against current `upstream/main` shows scalar loss matches at step 1 in all cells. TP cells intentionally change grad_norm from upstream's known placement behavior: FSDP=2+TP=2 step 1 `1.3237 -> 1.5195`; TP=4 step 1 `1.2737 -> 1.5189`. FSDP=4 has small step-10 stdout drift after updates: upstream `loss=4.01313 grad_norm=1.9032`, candidate `loss=4.01671 grad_norm=1.9059`. Confidence is medium-high; stdout is sufficient for smoke validation but TensorBoard export is still required before claiming bitwise equality.

The real TorchTitan long-context stress uses the Llama debug model, `num_chunks=8`, `L=131072`, `BS=1`, FSDP=2+TP=2, and compiled loss. A fresh same-SHA comparison shows upstream step 1 `loss=8.12721 grad_norm=1.3576 mem=4.88GiB tok/s=10656`; candidate step 1 `loss=8.12721 grad_norm=1.5613 mem=4.88GiB tok/s=10723`. Step 3 candidate is `loss=7.04638 grad_norm=1.9557 mem=4.88GiB tok/s=32869`; upstream is `loss=7.04837 grad_norm=1.6096 mem=4.88GiB tok/s=28808`. Confidence is medium; single-run throughput is noisy, memory matches, scalar loss matches at step 1, and the larger long-context TP grad_norm delta should be checked with TensorBoard before making convergence claims.

The real RL trainer smoke uses Qwen3-0.6B GRPO, `num_chunks=8`, one prompt, group size 2, max generated tokens 20, 2 generator GPUs, and 2 trainer GPUs with trainer/generator TP=2. TensorBoard metrics from the run are `loss/mean=0.0`, `train/grad_norm/mean=11.4375`, `perf/tokens_per_second=53.0643`, `train/memory/max_active_gib=2.01168`, `reward/_mean=0.15000001`, `reward/group_std/mean=0.15000001`, and `reward/zero_std_frac=0.0`. Confidence is medium; it proves nonzero reward variance and nonzero backward through the RL actor path. Direct RDMA failed before the loss path on this host, so the run disables direct RDMA. vLLM logs a cleanup `GPUModelRunner.shutdown` AttributeError after metrics are emitted.

The validation artifact is `scripts/benchmarks/chunked_ce/validation_summary_2026-05-19.csv`. It records loss, grad_norm, tok/s, memory, source logs, and comparator for the fresh runs.

The GSPO memory delta is still only medium confidence. The sequence-wise selected-logprob tensor path costs more than the reducer path under FSDP because backward replay can trigger a second LM-head all-gather, and production-vocab memory remains unmeasured for this implementation.

## Common Pitfalls

### Ignored Labels

Wrong:

```python
logprobs = F.log_softmax(logits, dim=-1).gather(-1, labels[..., None]).squeeze(-1)
```

If `labels` contains `-100`, gather reads an invalid label index. Use `F.cross_entropy(reduction="none", ignore_index=-100)` and negate the result.

### Uneven Chunks

Wrong:

```python
start = chunk_idx * chunk_len
```

When `sequence_len` is not divisible by `num_chunks`, each chunk can have a different length. Track a cumulative `start` offset instead.

### External Loss Scaling

Wrong:

```python
def backward(ctx, grad_output):
    return ctx.accumulated_hidden_grad
```

The bridge must scale hidden and parameter gradients by `grad_output`.

### Plain Python Chunking Returning `[B, L]`

Returning `torch.cat(logprob_chunks, dim=1)` from a normal Python loop keeps every chunk's `[batch, chunk, vocab]` graph alive. Use a custom autograd function that replays the LM head during backward.

### FSDP Reshard State

Do not permanently pin `set_reshard_after_forward(False)` or `set_requires_gradient_sync(False)`. Scope those toggles to the chunk loop and restore them.

### TP Placement For Hidden Gradients

Wrong:

```python
return DTensor.from_local(local_hidden_grad, mesh, placements=hidden_states.placements)
```

If the local gradient has been computed after a TP all-gather, wrapping it back with stale shard/partial placements can silently change the effective gradient. Redistribute or construct the gradient with placements that describe the tensor actually being returned, then validate grad_norm under TP and FSDP+TP.

### Two Sources Of Chunk Count

Wrong:

```python
self.chunked_loss_num_chunks = config.chunked_loss_num_chunks
self.chunked_ce_loss = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=config.chunked_loss_num_chunks))
if self.chunked_loss_num_chunks > 1:
    ...
```

Use the constructed `ChunkedCELoss` as the runtime source of truth:

```python
self.chunked_ce_loss = None
if config.chunked_loss_num_chunks > 1:
    self.chunked_ce_loss = ChunkedCELoss(ChunkedCELoss.Config(num_chunks=config.chunked_loss_num_chunks), compile_config=compile_config)

if self.chunked_ce_loss is not None:
    ...
```

### `loss_parallel()` Scope

PyTorch `loss_parallel()` currently supports a 1D mesh. FSDP+TP still uses chunked CE, but loss-parallel wrapping may be skipped on a 2D mesh. This does not switch the implementation to dense CE.

### Compile Is A Separate Validation Cell

Do not assume `torch.compile` proves eager correctness or hides placement issues. In the fresh same-SHA run, compiled FSDP=2+TP=2 preserves the same scalar-loss match and the same TP grad_norm delta as eager: upstream step 1 `grad_norm=1.3237`, candidate step 1 `grad_norm=1.5195`.

### Stdout Precision

TorchTitan stdout prints limited precision. Use TensorBoard/event exports or `scripts/loss_compare.py` before claiming bitwise parity.

## Entropy and Metrics

Entropy is not part of the CE or GRPO objective here. If RL metrics need entropy, compute it separately under `torch.no_grad()` from chunk logits or add a later online-LSE backend that exposes analytical entropy cheaply. Do not force entropy through `PackedPolicyLossInputs` as an input to the policy loss.

Metrics should normalize using the same token weights and global valid-token counts as the loss. Reducers can accumulate detached numerator and denominator statistics per chunk and expose them after the scalar loss is built.

## Open Technical Questions

1. Re-run the real TorchTitan SFT matrix with TensorBoard exports before claiming bitwise parity in a PR.
2. Add an FSDP-only RL trainer smoke; the current nonzero-reward RL smoke covers trainer/generator TP=2.
3. Production-vocab GSPO memory remains unmeasured for this implementation.
4. If compiled TP without loss parallel still hits a mixed Tensor/DTensor CE crash, extend the label-preparation helper into SFT `cross_entropy_loss` with a regression test.
5. Explain or rule out the small FSDP=4 step-10 drift with full-precision TensorBoard logs before claiming exact SFT parity.
6. For long-context TP, compare full-precision loss and grad_norm at 64k/128k before treating the larger grad_norm delta as fully characterized.

## Out Of Scope For The First PR

Full Forge loss-package port, DAPO/CISPO/SAPO production implementations, analytical entropy, z-loss, adaptive `max_tokens_per_chunk`, vocab-parallel selected-logprob, online-LSE backend, internal-reference-style vocab-to-sequence all-to-all, and broad 4D mesh production validation are follow-up work.
