# RL Tuning Guide

This runbook pairs with the `[RL config resolved]` block printed at startup.
Use that block as the launch-time source of truth, then use the metrics below
to decide which knob to change.

## Capacity

RL batch capacity is owned by `BatchConfig` through `batcher.batch`.

```text
global_batch_rows = (
    batcher.batch.global_batch_size
    if batcher.batch.global_batch_size > 0
    else batcher.batch.local_batch_size * trainer_dp_degree
)
prompt_groups_per_batch = ceil(global_batch_rows / group_size)
rollout_concurrency_groups = prompt_groups_per_batch * (max_offpolicy_steps + 1)
replay_buffer_samples = max(
    global_batch_rows,
    rollout_concurrency_groups * group_size,
)
max_admitted_generation_prompts = max(
    rollout_concurrency_groups * group_size,
    num_validation_prompts,
)
```

Example:

```py
BatchConfig(local_batch_size=8, global_batch_size=64, seq_len=2048)
group_size = 8
max_offpolicy_steps = 1

# Startup resolves:
# prompt_groups_per_batch = 8
# rollout_concurrency_groups = 16
# replay_buffer_samples = 128
# max_admitted_generation_prompts = 128
```

## Migration

| Old field | Replacement |
| --- | --- |
| `num_prompts_per_step` | Derived from `batcher.batch.global_batch_size`, `batcher.batch.local_batch_size`, and `group_size` |
| `rollout_group_size` | `group_size` |
| `num_validation_samples=64` | `num_validation_prompts=64` |
| `async_rollout_groups` | `async_pipeline.rollout_concurrency_groups` |
| `replay_buffer_groups` | `async_pipeline.replay_buffer_samples` |
| `max_admitted_generation_prompts` | `async_pipeline.max_admitted_generation_prompts` |

## Metrics

| Question | Metrics |
| --- | --- |
| Did replay keep the trainer fed? | `timing/replay_wait`, `trainer/idle_ratio` |
| Is replay too deep or stale? | `replay/buffer/depth_samples_post_pull`, `replay/buffer/stale_drop_rate` |
| Is controller admission backing up? | `generation_scheduler/queued_prompts/mean`, `generation_scheduler/admitted_prompts/mean`, `generation_scheduler/queue_wait_seconds/mean` |
| Is vLLM saturated? | `generator/live/kv_cache_usage_pct/max`, `generator/live/num_preempted_reqs/max`, `generator/live/{idx}/queue_depth/max` |
| Did numerical health degrade? | `health/loss/policy_logprob_nonfinite_frac`, `health/loss/ref_logprob_nonfinite_frac`, `health/train/skipped_nonfinite_loss`, `health/train/skipped_nonfinite_grad_norm` |
| Is policy drift in band? | `logprob_drift/diff_mean`, `logprob_drift/diff_max_abs`, `logprob_drift/diff_fraction`, `loss/ratio/mean` |
| Is reward signal alive? | `reward/_mean`, `reward/group_std/mean`, `reward/zero_std_frac/mean` |
| Are rollouts being dropped? | `rollout/dropped_empty_groups`, `rollout/dropped_zero_advantage_groups`, `rollout/error_rate/mean`, `rollout/truncation_rate/mean` |

## Adjustments

| Observation | First action |
| --- | --- |
| `trainer/idle_ratio` is high and vLLM saturation is low | Add generator capacity with `num_generator_instances` or generator TP |
| `replay/buffer/stale_drop_rate` is high | Lower `async_pipeline.rollout_concurrency_groups` or raise `max_offpolicy_steps` |
| `replay/buffer/depth_samples_post_pull` stays near capacity | Lower `async_pipeline.replay_buffer_samples` or add trainer capacity |
| `generation_scheduler/queued_prompts/mean` keeps rising | Increase generator capacity or lower rollout concurrency |
| `generator/live/kv_cache_usage_pct/max` is above 90 | Raise `generator.gpu_memory_limit` or reduce admitted prompt pressure |
| `reward/zero_std_frac/mean` is 1.0 | Use a harder task, adjust sampling, or disable zero-advantage drops for debugging |

## Known Follow-Ups

- Cache `_build_sampling_params` by equality key to avoid rebuilding identical
  vLLM sampling params.
- Admit replay groups atomically so partial groups cannot enter replay under
  tight capacity.
- Pull generator weights concurrently after the trainer push when the transport
  path can handle parallel readers reliably.
- Support multiple generator instances across multi-node host meshes.
