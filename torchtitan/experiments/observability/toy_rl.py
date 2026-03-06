# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy RL Example with Monarch Actors

Reuses ToyTrainer from toy_spmd.py inside a Monarch actor.
Controller owns aggregation, rollout logging, and console output.
Actors write to per-rank JSONL; controller reads via FileWatcher + aggregate.

Run (requires 4 GPUs):
    python -m torchtitan.experiments.observability.toy_rl
"""

import asyncio
import logging
import os
import shutil
import socket
import time

import torch
from monarch.actor import Actor, current_rank, endpoint, this_host
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.experiments.observability.toy_spmd import (
    BATCH_SIZE,
    D_MODEL,
    RepeatDataset,
    SEQ_LEN,
    ToyTrainer,
    VOCAB_SIZE,
)
from torchtitan.observability import (
    aggregate,
    clear_step_tags,
    EventType,
    FileWatcher,
    init_observability,
    MaxMetric,
    MeanMetric,
    MetricsProcessor,
    record_event,
    record_metric,
    record_span,
    set_step,
)
from torchtitan.observability.rollout_logger import filter_top_bottom, RolloutLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

NUM_STEPS = 6
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
        dp_rank = mesh["dp"].get_local_rank()

        # Dummy dataloader (data comes from controller via train_step args).
        torch.manual_seed(42 + dp_rank)
        tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=self.device)
        labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=self.device)
        loss_mask = torch.ones(BATCH_SIZE, SEQ_LEN, device=self.device)
        dataloader = RepeatDataset(tokens, labels, loss_mask)

        self.trainer = ToyTrainer(
            self.device, mesh["dp"], mesh["tp"], OUTPUT_DIR, dataloader,
        )

    @endpoint
    async def set_step(self, step: int):
        """Receive step via broadcast. Just save — don't call trainer."""
        self.step = step

    @endpoint
    async def train_step(self, tokens, labels, loss_mask) -> float:
        """Train one step. Sets step + ContextVar, then trains."""
        self.trainer.step = self.step
        self.trainer.metrics_processor.set_step(self.step)
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
    logger.info(f"Toy RL: {NUM_STEPS} steps")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Controller observability.
    init_observability(rank=0, source="controller", output_dir=OUTPUT_DIR)

    # Spawn actors.
    host = this_host()
    trainer_mesh = host.spawn_procs(per_host={"gpus": 4}, name="trainer")
    trainer = trainer_mesh.spawn("trainer", TrainerActor)
    await trainer.setup.call()

    reward_mesh = host.spawn_procs(per_host={"procs": 1}, name="reward")
    reward_actor = reward_mesh.spawn("reward", RewardActor)
    await reward_actor.setup.call()
    logger.info("Actors spawned.")

    # Controller aggregation (reads JSONL from all actors).
    watcher = FileWatcher(log_dir=os.path.join(OUTPUT_DIR, "experiment_logs"))
    rollout_logger = RolloutLogger(output_dir=os.path.join(OUTPUT_DIR, "rollouts"))

    # Fixed data for overfitting.
    torch.manual_seed(42)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    loss_mask = torch.ones(BATCH_SIZE, SEQ_LEN)

    # Dummy completions for reward scoring.
    completions = [torch.arange(D_MODEL, dtype=torch.float32) * (i + 1) / D_MODEL for i in range(4)]

    def _log_to_console(step, aggregated):
        loss = aggregated.get("trainer/loss_mean", "--")
        reward_mean = aggregated.get("reward/mean", "--")

        def fmt(val, spec):
            return f"{val:{spec}}" if isinstance(val, (int, float)) else f"{val:>8}"

        logger.info(
            f"step: {step}  loss: {fmt(loss, '8.5f')}  "
            f"reward_mean: {fmt(reward_mean, '8.5f')}"
        )

    # RL training loop.
    for step in range(1, NUM_STEPS + 1):
        set_step(step)
        clear_step_tags()

        await trainer.set_step.call(step)

        # Score completions.
        with record_span("RewardScoring", EventType.RL_SCORING):
            reward_results = await reward_actor.score.call(completions, step=step)
        rewards = next(iter(reward_results.values()))

        # Train.
        with record_span("Training", EventType.FWD_BWD):
            loss_results = await trainer.train_step.call(tokens, labels, loss_mask)
        loss = next(iter(loss_results.values()))

        # Rollout logging.
        records = [{"prompt_id": i, "reward": rewards[i], "policy_version": step} for i in range(len(rewards))]
        rollout_logger.log(records, step, lambda r: filter_top_bottom(r, k=1))

        # System metrics.
        record_event({"train.loss": loss, "reward_mean": sum(rewards) / len(rewards)})

        # Aggregate and log.
        aggregated = aggregate(watcher.drain(step))
        if aggregated:
            _log_to_console(step, aggregated)

    rollout_logger.close()
    watcher.close()
    await trainer.teardown.call()
    logger.info(f"Done in {time.time() - t0:.1f}s. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
