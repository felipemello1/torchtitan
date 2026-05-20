# Chunked CE Reducer Review Pack

This folder is the review surface for the chunked cross-entropy reducer work. It contains the RFC, one standalone prototype, the validation harnesses, and the current small CSV summary.

## Files

`RFC.md` describes the reducer API, selected-logprob tensor API, walking examples, related library patterns, known limitations, and current validation evidence.

`prototype.py` is a standalone CPU proof for scalar SFT, token-local RL reducers, and sequence-level selected-logprob replay.

`launch_validation.sh` is the consolidated validation harness. It defaults to CPU tests; GPU cells require explicit `CELLS` and `CUDA_VISIBLE_DEVICES`.

`parity_real_titan.sh` compares the candidate worktree against an upstream-main worktree for SFT smoke runs.

`loss_compare_chunked_ce.sh` wraps TorchTitan's existing `scripts/loss_compare.py` with the chunked CE defaults used here. Run it only from a clean worktree because `loss_compare.py` checks out commits.

`memory_sweep.py` is a standalone peak-memory shape sweep. It is useful for local shape pressure, but it is not a current-TorchTitan distributed comparator.

`validation_summary_2026-05-19.csv` records the latest small validation summary with loss, grad_norm, tok/s, memory, source log, and comparison baseline.

## Current Results

All SFT cells below use `num_chunks=8` and compare the candidate branch against same-SHA `upstream/main@52a292d29`.

CPU correctness passes for the standalone prototype, core `ChunkedCELoss` tests, and RL loss tests. These tests compare scalar loss, selected logprobs, hidden-state gradients, LM-head gradients, uneven chunk offsets, and external loss scaling.

At `L=2048`, `local_batch_size=8`, FSDP=4 matches step-1 scalar loss: upstream `loss=8.14122`, candidate `loss=8.14122`; upstream `grad_norm=1.5182`, candidate `grad_norm=1.5148`; both report `memory=0.68 GiB`.

At `L=2048`, `local_batch_size=8`, FSDP=2+TP=2 matches step-1 scalar loss: upstream `loss=8.11665`, candidate `loss=8.11665`; upstream `grad_norm=1.3237`, candidate `grad_norm=1.5195`; both report `memory=0.48 GiB`. The grad_norm delta is expected from the hidden-gradient placement fix in this branch.

At `L=2048`, `local_batch_size=8`, TP=4 matches step-1 scalar loss: upstream `loss=8.13643`, candidate `loss=8.13643`; upstream `grad_norm=1.2737`, candidate `grad_norm=1.5189`; both report `memory=0.34 GiB`. The grad_norm delta is expected from the same TP placement fix.

At `L=2048`, `local_batch_size=8`, compiled FSDP=2+TP=2 preserves the same step-1 scalar-loss match and TP grad_norm delta: upstream `loss=8.11665`, candidate `loss=8.11665`; upstream `grad_norm=1.3237`, candidate `grad_norm=1.5195`.

At `L=131072`, `local_batch_size=1`, compiled FSDP=2+TP=2 matches step-1 scalar loss and memory: upstream `loss=8.12721`, candidate `loss=8.12721`; upstream `memory=4.88 GiB`, candidate `memory=4.88 GiB`; upstream `grad_norm=1.3576`, candidate `grad_norm=1.5613`.

The RL smoke runs Qwen3-0.6B with 2 generator GPUs + 2 trainer GPUs, trainer/generator TP=2, one prompt, group size 2, max generated tokens 20, and `num_chunks=8`. TensorBoard metrics from the run include `train/grad_norm/mean=11.4375`, `perf/tokens_per_second=53.0643`, `train/memory/max_active_gib=2.01168`, `reward/_mean=0.15000001`, and `reward/group_std/mean=0.15000001`.

Do not overread stdout. TorchTitan console logs print limited precision; use TensorBoard exports or `scripts/loss_compare.py` before claiming bitwise parity in a PR.

## Commands

Use the `titan` environment for pretraining/core tests and `titan_rl` for RL tests that import the RL package with vLLM available.

Standalone CPU prototype:

```bash
/home/felipemello/.conda/envs/titan/bin/python scripts/benchmarks/chunked_ce/prototype.py
```

CPU validation:

```bash
BASE=/home/felipemello/torchtitan/.worktrees/34-chunked-ce-v2-handoff \
OUT=/tmp/chunked_ce_v2_validation_cpu \
bash scripts/benchmarks/chunked_ce/launch_validation.sh
```

SFT distributed validation on GPUs `0,1,2,3`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
BASE=/home/felipemello/torchtitan/.worktrees/34-chunked-ce-v2-handoff \
OUT=/tmp/chunked_ce_v2_validation_gpu \
CELLS="A B C D" \
bash scripts/benchmarks/chunked_ce/launch_validation.sh
```

RL smoke on GPUs `0,1,2,3`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
BASE=/home/felipemello/torchtitan/.worktrees/34-chunked-ce-v2-handoff \
OUT=/tmp/chunked_ce_v2_validation_rl \
CELLS=E \
bash scripts/benchmarks/chunked_ce/launch_validation.sh
```

Quick SFT parity against an upstream-main worktree:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CANDIDATE_WORKTREE=/home/felipemello/torchtitan/.worktrees/34-chunked-ce-v2-handoff \
MAIN_WORKTREE=/home/felipemello/torchtitan/.worktrees/upstream-main-52a292d-codex \
OUT=/tmp/chunked_ce_v2_real_train \
bash scripts/benchmarks/chunked_ce/parity_real_titan.sh
```

TensorBoard loss comparison through TorchTitan's comparator:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
BASELINE_COMMIT=52a292d29 \
TEST_COMMIT=HEAD \
OUT=/tmp/chunked_ce_v2_loss_compare \
bash scripts/benchmarks/chunked_ce/loss_compare_chunked_ce.sh
```

Cell E defaults to the cached Qwen3-0.6B HF snapshot at `/home/felipemello/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`, `rl_grpo_qwen3_0_6b_batch_invariant`, `num_steps=1`, `num_prompts_per_step=1`, `generator.sampling.n=2`, `max_tokens=20`, and `trainer.chunked_loss_num_chunks=8`. It also defaults `RL_DIRECT_RDMA=0` and `TORCHSTORE_RDMA_ENABLED=0` because direct RDMA failed before the loss path on this machine. Override with `RL_HF_ASSETS_PATH`, `RL_CONFIG`, `RL_STEPS`, `RL_PROMPTS`, `RL_GROUP_SIZE`, `RL_MAX_TOKENS`, `RL_DIRECT_RDMA`, `TORCHSTORE_RDMA_ENABLED`, or `RL_EXTRA`.

The final handoff gate should use fresh pretraining and RL runs from the consolidated worktree. For each run, record loss, grad_norm, tok/s, memory, source log, and comparison baseline; do not infer these from CPU tests.
