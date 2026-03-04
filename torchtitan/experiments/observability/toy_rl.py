# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy RL Example with Monarch Actors

Reuses ToyTrainer from toy_spmd.py inside a Monarch actor.
Controller owns aggregation, rollout logging, and backend writes.
Actors write to per-rank JSONL; controller reads via DefaultAggregator.

Run (requires 4 GPUs):
    python -m torchtitan.experiments.observability.toy_rl
"""

import asyncio
import os
import shutil
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from monarch.actor import Actor, current_rank, endpoint, this_host
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.experiments.observability.toy_spmd import (
    BATCH_SIZE,
    D_MODEL,
    LoggingConfig,
    SEQ_LEN,
    ToyTrainer,
    VOCAB_SIZE,
)
from torchtitan.observability import (
    clear_step_tags,
    CompositeSummaryWriter,
    DefaultAggregator,
    EventType,
    init_observability,
    InMemorySummaryWriter,
    MaxMetric,
    MeanMetric,
    record_event,
    record_metric,
    record_span,
    set_step,
)
from torchtitan.observability.rollout_logger import filter_top_bottom, RolloutLogger

NUM_STEPS = 6
TRAINER_LOG_FREQ = 2
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "toy_rl")


class TrainerActor(Actor):
    @endpoint
    async def setup(self):
        rank = current_rank().rank
        self._rank = rank
        self.device = f"cuda:{rank}"
        torch.cuda.set_device(self.device)
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", socket.gethostname())
            os.environ.setdefault("MASTER_PORT", "29500")
            world_size = int(os.environ.get("WORLD_SIZE", 4))
            torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
        self.trainer = ToyTrainer(
            self.device, mesh["dp"], mesh["tp"], OUTPUT_DIR,
            logging_cfg=LoggingConfig(
                log_freq=TRAINER_LOG_FREQ,
                enable_console=False,
                enable_tensorboard=False,
                enable_wandb=False,
            ),
        )

    @endpoint
    async def set_step(self, step: int):
        """Receive step via broadcast. FIFO ordering guarantees this runs before train_step."""
        self.trainer.set_step(step)

    @endpoint
    async def train_step(self, tokens, labels, loss_mask) -> float:
        """Returns loss as a scalar. Metrics are logged to JSONL inside the trainer."""
        # Controller sends CPU tensors; move to this actor's GPU
        tokens = tokens.to(self.device)
        labels = labels.to(self.device)
        loss_mask = loss_mask.to(self.device)
        loss, _grad_norm = self.trainer.train_step(tokens, labels, loss_mask)
        return loss.item()

    @endpoint
    async def teardown(self):
        self.trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class RewardActor(Actor):
    @endpoint
    async def setup(self):
        self.target = torch.ones(D_MODEL)
        rank = current_rank().rank
        init_observability(rank=rank, source="reward", output_dir=OUTPUT_DIR)

    @endpoint
    async def score(self, completions: list[torch.Tensor], step: int = 0) -> list[float]:
        """Score completions against a target. Returns list of rewards."""
        set_step(step)
        with record_span("RewardScoring", EventType.RL_SCORING):
            rewards = [-((c - self.target) ** 2).mean().item() for c in completions]
        record_metric("reward/mean", MeanMetric(sum=sum(rewards), weight=len(rewards)))
        record_metric("reward/max", MaxMetric(value=max(rewards)))
        return rewards


async def main():
    t0 = time.time()
    print(f"Toy RL: {NUM_STEPS} steps")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Initialize controller observability ---
    init_observability(rank=0, source="controller", output_dir=OUTPUT_DIR)

    # --- Spawn actors ---
    host = this_host()
    trainer_mesh = host.spawn_procs(per_host={"gpus": 4}, name="trainer")
    trainer = trainer_mesh.spawn("trainer", TrainerActor)
    await trainer.setup.call()

    reward_mesh = host.spawn_procs(per_host={"procs": 1}, name="reward")
    reward_actor = reward_mesh.spawn("reward", RewardActor)
    await reward_actor.setup.call()
    print("Actors spawned.\n")

    # --- Initialize backends and aggregator ---
    writer = CompositeSummaryWriter(writers={"memory": InMemorySummaryWriter()})
    writer.open()
    aggregator = DefaultAggregator(experiment_log_dir=os.path.join(OUTPUT_DIR, "experiment_logs"))
    rollout_logger = RolloutLogger(output_dir=os.path.join(OUTPUT_DIR, "rollouts"))
    executor = ThreadPoolExecutor(max_workers=2)

    # --- Fixed data for overfitting (generated once, reused every step) ---
    torch.manual_seed(42)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    loss_mask = torch.ones(BATCH_SIZE, SEQ_LEN)  # all tokens valid in RL toy

    # Dummy completions for reward scoring (fixed vectors for overfitting)
    completions = [torch.arange(D_MODEL, dtype=torch.float32) * (i + 1) / D_MODEL for i in range(4)]

    def _aggregate_and_write(aggregator, writer, step):
        aggregated = aggregator.collect_and_aggregate(step=step)
        if aggregated:
            writer(step=step, values=aggregated)

    # --- RL training loop ---
    for step in range(1, NUM_STEPS + 1):
        set_step(step)  # Controller's own ContextVar
        clear_step_tags()

        # Broadcast step to all actors (fire-and-forget, FIFO ordering).
        # Actors receive this before train_step executes.
        await trainer.set_step.call(step)

        # Score completions
        with record_span("RewardScoring", EventType.RL_SCORING):
            reward_results = await reward_actor.score.call(completions, step=step)
        # Monarch returns dict[rank, result] — single reward actor has one rank
        rewards = next(iter(reward_results.values()))

        # Train (no step parameter — set via broadcast above)
        with record_span("Training", EventType.FWD_BWD):
            loss_results = await trainer.train_step.call(tokens, labels, loss_mask)
        loss = next(iter(loss_results.values()))

        # Rollout logging (non-blocking)
        records = [{"prompt_id": i, "reward": rewards[i], "policy_version": step} for i in range(len(rewards))]
        executor.submit(rollout_logger.log, records, step, lambda r: filter_top_bottom(r, k=1))

        # System metrics (always, every step)
        record_event({"train.loss": loss, "reward_mean": sum(rewards) / len(rewards)})

        # Aggregate and write to backends (non-blocking)
        executor.submit(_aggregate_and_write, aggregator, writer, step)

        print(f"  Step {step}/{NUM_STEPS}: loss={loss:.4f}, rewards={[f'{r:.3f}' for r in rewards]}")

    executor.shutdown(wait=True)
    rollout_logger.close()
    writer.close()
    await trainer.teardown.call()
    print(f"\nDone in {time.time() - t0:.1f}s. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
