# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy RL Example with Monarch Actors + Full Observability (PR6)

Architecture (pure Monarch, no Forge service layer):
- TrainerActor: 4 procs (TP=2 × DP=2), torch.compiled model with DTensor
- RewardActor: 1 CPU proc (scores completions)
- Controller: RL loop with DefaultAggregator + RolloutLogger

Demonstrates ALL observability features:
- PR1: init_observability, set_step, record_span, record_event
- PR2: InvocationContext, record_tensor_metric, replicate_to_host
- PR3: EveryNSteps, CompositeSummaryWriter
- PR4: record_metric, log_reduced_metrics, DefaultAggregator
- PR5: profile_annotation
- PR6: RolloutLogger

Outputs to /tmp/toy_rl_output/ — integration tests read from there.

Run (requires 6 GPUs):
    python torchtitan/experiments/observability/toy_rl.py
"""

import asyncio
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from monarch.actor import Actor, current_rank, endpoint, this_host

# Observability imports (PR1 through PR6)
from torchtitan.observability import (
    add_step_tag,
    clear_step_tags,
    CompositeSummaryWriter,
    DefaultAggregator,
    EventType,
    EveryNSteps,
    init_observability,
    InMemorySummaryWriter,
    InvocationContext,
    log_reduced_metrics,
    MaxMetric,
    MaxTMetric,
    MeanMetric,
    MeanTMetric,
    profile_annotation,
    record_event,
    record_metric,
    record_span,
    record_tensor_metric,
    replicate_to_host,
    set_step,
)
from torchtitan.observability.rollout_logger import RolloutLogger, filter_top_bottom

# ---- Config ----
NUM_STEPS = 6
GROUP_SIZE = 4
D_MODEL = 64
HIDDEN = 128
VOCAB_SIZE = 16
LOG_EVERY = 2
OUTPUT_DIR = "/tmp/toy_rl_output"


# ---- Model ----
class TinyMLP(nn.Module):
    def __init__(self, d=D_MODEL, h=HIDDEN, v=VOCAB_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h, d)
        self.head = nn.Linear(d, v)

    def forward(self, x):
        return self.head(self.fc2(self.relu(self.fc1(x))))


# ---- Helpers ----
def make_completions(group_size, d):
    return [torch.arange(d, dtype=torch.float32) * (i + 1) / d for i in range(group_size)]


def compute_advantages(rewards):
    rewards_t = torch.tensor(rewards)
    std = rewards_t.std()
    if std > 1e-6:
        return (rewards_t - rewards_t.mean()) / (std + 1e-8)
    return torch.ones_like(rewards_t)


def make_batch(completions, advantages):
    return {
        "inputs": torch.stack(completions),
        "targets": torch.zeros(len(completions), dtype=torch.long),
        "advantages": advantages,
    }


# ---- Actors ----
class TrainerActor(Actor):
    @endpoint
    async def setup(self):
        rank = current_rank().rank
        self._rank = rank
        self.device = f"cuda:{rank}"
        torch.cuda.set_device(self.device)

        if not torch.distributed.is_initialized():
            import socket
            os.environ.setdefault("MASTER_ADDR", socket.gethostname())
            os.environ.setdefault("MASTER_PORT", "29500")
            torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=4)

        # PR1: Init observability (per-rank system + experiment JSONL)
        init_observability(rank=rank, source="trainer", output_dir=OUTPUT_DIR)

        self.mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
        model = TinyMLP().to(self.device)
        parallelize_module(model, self.mesh["tp"], {
            "fc1": ColwiseParallel(), "fc2": RowwiseParallel(), "head": ColwiseParallel(),
        })
        self.model = torch.compile(model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.weight_version = 0

    @endpoint
    async def train_step(self, batch: dict, step: int = 0) -> dict:
        # Per-endpoint init (MONARCH_COMPAT)
        init_observability(rank=self._rank, source="trainer", output_dir=OUTPUT_DIR)
        set_step(step)

        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        # PR2: InvocationContext for tensor metrics
        with InvocationContext() as ctx:
            # PR1 + PR5: record_span + profile_annotation
            with record_span("Forward/Backward", EventType.FWD_BWD):
                with profile_annotation("forward"):
                    output = self.model(inputs)
                loss = F.cross_entropy(output, targets)
                # PR2: record_tensor_metric (GPU tensor, no .item())
                record_tensor_metric("loss", MeanTMetric(mean=loss, weight=torch.tensor(1.0, device=self.device)))
                self.optimizer.zero_grad()
                loss.backward()

            with record_span("Optimizer", EventType.OPTIM):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.weight_version += 1

            # PR2: grad_norm as tensor metric (to_local if DTensor)
            gn = grad_norm.to_local() if hasattr(grad_norm, "to_local") else grad_norm
            record_tensor_metric("grad_norm", MaxTMetric(gn.detach()))

        # PR4: CPU metrics
        record_metric("weight_version", MeanMetric(value=float(self.weight_version)))

        # PR2: replicate_to_host on logging steps
        scalars = {}
        if step % LOG_EVERY == 0 or step == 1:
            scalars = replicate_to_host(ctx.summaries())
            # PR4: bridge tensor metrics → experiment JSONL
            log_reduced_metrics(scalars)

        return {
            "weight_version": self.weight_version,
            "loss": loss.detach().item(),
            "grad_norm": grad_norm.item(),
            "scalars": scalars,
        }

    @endpoint
    async def teardown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class RewardActor(Actor):
    @endpoint
    async def setup(self):
        self.target = torch.ones(D_MODEL)
        self._rank = current_rank().rank
        # PR1: Init observability for reward actor
        init_observability(rank=0, source="reward", output_dir=OUTPUT_DIR)

    @endpoint
    async def score(self, completions: list[torch.Tensor], step: int = 0) -> list[float]:
        init_observability(rank=0, source="reward", output_dir=OUTPUT_DIR)
        set_step(step)
        with record_span("Score", EventType.EVAL):
            rewards = [-((c - self.target) ** 2).mean().item() for c in completions]
        # PR4: CPU metrics from reward actor
        record_metric("reward/mean", MeanMetric(value=sum(rewards) / len(rewards)))
        record_metric("reward/max", MaxMetric(value=max(rewards)))
        return rewards


# ---- Main ----
async def main():
    t0 = time.time()
    print(f"Toy RL: {NUM_STEPS} steps, TP=2 × DP=2")

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Init observability for controller
    init_observability(rank=0, source="controller", output_dir=OUTPUT_DIR)

    # Spawn actors
    host = this_host()
    trainer_mesh = host.spawn_procs(per_host={"gpus": 4}, name="trainer")
    trainer = trainer_mesh.spawn("trainer", TrainerActor)
    await trainer.setup.call()

    reward_mesh = host.spawn_procs(per_host={"procs": 1}, name="reward")
    reward_actor = reward_mesh.spawn("reward", RewardActor)
    await reward_actor.setup.call()

    print("Actors spawned. Starting RL loop.\n")

    # PR3: Writer + schedule
    mem_writer = InMemorySummaryWriter()
    writer = CompositeSummaryWriter(writers={"memory": mem_writer})
    writer.open()
    log_schedule = EveryNSteps(every_n=LOG_EVERY, additional_steps={1})

    # PR4: Aggregator + PR6: RolloutLogger
    aggregator = DefaultAggregator(experiment_log_dir=os.path.join(OUTPUT_DIR, "experiment_logs"))
    rollout_logger = RolloutLogger(output_dir=os.path.join(OUTPUT_DIR, "rollouts"))
    executor = ThreadPoolExecutor(max_workers=2)

    # RL loop
    for step in range(1, NUM_STEPS + 1):
        set_step(step)
        clear_step_tags()

        # Generate completions (deterministic)
        completions = make_completions(GROUP_SIZE, D_MODEL)

        # Score
        reward_results = await reward_actor.score.call(completions, step=step)
        rewards = next(iter(reward_results.values()))

        # Compute advantages and build batch
        advantages = compute_advantages(rewards)
        batch = make_batch(completions, advantages)

        # Train
        with record_span("Training", EventType.FWD_BWD):
            all_rank_results = await trainer.train_step.call(batch, step=step)

        per_rank = list(all_rank_results.values())
        r0 = per_rank[0]

        # PR6: Log rollouts
        rollout_records = [
            {"prompt_id": i, "reward": rewards[i], "policy_version": r0["weight_version"]}
            for i in range(len(rewards))
        ]
        executor.submit(rollout_logger.log, rollout_records, step, lambda r: filter_top_bottom(r, k=2))

        # PR3 + PR4: Logging gate with non-blocking aggregation
        if log_schedule(step):
            add_step_tag("logging")
            record_event({"train.loss": r0["loss"], "train.grad_norm": r0["grad_norm"]})

            def _aggregate_and_write(agg, w, s):
                aggregated = agg.collect_and_aggregate(step=s)
                if aggregated:
                    w(step=s, values=aggregated)

            executor.submit(_aggregate_and_write, aggregator, writer, step)

        print(
            f"  Step {step:>2}/{NUM_STEPS}: "
            f"loss={r0['loss']:.4f}, "
            f"grad_norm={r0['grad_norm']:.4f}, "
            f"rewards={[f'{r:.3f}' for r in rewards]}"
        )

    # Shutdown
    executor.shutdown(wait=True)
    rollout_logger.close()
    writer.close()
    await trainer.teardown.call()

    # Save outputs for integration test
    with open(os.path.join(OUTPUT_DIR, "writer_summaries.json"), "w") as f:
        json.dump({str(k): v for k, v in mem_writer.summaries.items()}, f)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
