# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy RL Example with Monarch Actors (PR6)

Reuses ToyTrainer from toy_spmd.py inside a Monarch actor.
Controller owns aggregation + rollout logging.

Run (requires 4 GPUs):
    python torchtitan/experiments/observability/toy_rl.py
"""

import asyncio
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from monarch.actor import Actor, current_rank, endpoint, this_host

from torchtitan.experiments.observability.toy_spmd import (
    LoggingConfig, ToyTrainer, VOCAB_SIZE, D_MODEL,
)
from torchtitan.observability import (
    add_step_tag, clear_step_tags, CompositeSummaryWriter, DefaultAggregator,
    EventType, init_observability, InMemorySummaryWriter,
    MaxMetric, MeanMetric, record_event, record_metric, record_span, set_step,
)
from torchtitan.observability.rollout_logger import RolloutLogger, filter_top_bottom

NUM_STEPS = 6
LOG_EVERY = 2
OUTPUT_DIR = "/tmp/toy_rl_output"


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
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
        # No backends in actor — controller handles aggregation + backends.
        # log_freq must match LOG_EVERY so trainer reduces tensor metrics
        # on the same steps the controller aggregates.
        self.trainer = ToyTrainer(
            self.device, mesh["dp"], mesh["tp"], OUTPUT_DIR,
            logging_cfg=LoggingConfig(log_freq=LOG_EVERY, enable_console=False),
        )

    @endpoint
    async def train_step(self, tokens, labels, step) -> dict:
        loss, grad_norm, dt_ms, ctx = self.trainer.train_step(tokens, labels, step)
        return {"loss": loss.item(), "grad_norm": grad_norm.item(), "dt_ms": dt_ms}

    @endpoint
    async def teardown(self):
        self.trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class RewardActor(Actor):
    @endpoint
    async def setup(self):
        self.target = torch.ones(D_MODEL)
        init_observability(rank=0, source="reward", output_dir=OUTPUT_DIR)

    @endpoint
    async def score(self, completions: list[torch.Tensor], step: int = 0) -> list[float]:
        set_step(step)
        with record_span("Score", EventType.EVAL):
            rewards = [-((c - self.target) ** 2).mean().item() for c in completions]
        record_metric("reward/mean", MeanMetric(value=sum(rewards) / len(rewards)))
        record_metric("reward/max", MaxMetric(value=max(rewards)))
        return rewards


async def main():
    t0 = time.time()
    print(f"Toy RL: {NUM_STEPS} steps")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_observability(rank=0, source="controller", output_dir=OUTPUT_DIR)

    host = this_host()
    trainer_mesh = host.spawn_procs(per_host={"gpus": 4}, name="trainer")
    trainer = trainer_mesh.spawn("trainer", TrainerActor)
    await trainer.setup.call()

    reward_mesh = host.spawn_procs(per_host={"procs": 1}, name="reward")
    reward_actor = reward_mesh.spawn("reward", RewardActor)
    await reward_actor.setup.call()
    print("Actors spawned.\n")

    # Controller owns aggregation + backends + rollout logging.
    # Controller aggregates every step — the trainer gates its own
    # replicate_to_host via log_freq, so tensor metrics only appear on
    # scheduled steps. CPU metrics appear every step.
    writer = CompositeSummaryWriter(writers={"memory": InMemorySummaryWriter()})
    writer.open()
    aggregator = DefaultAggregator(experiment_log_dir=os.path.join(OUTPUT_DIR, "experiment_logs"))
    rollout_logger = RolloutLogger(output_dir=os.path.join(OUTPUT_DIR, "rollouts"))
    executor = ThreadPoolExecutor(max_workers=2)

    for step in range(1, NUM_STEPS + 1):
        set_step(step)
        clear_step_tags()

        # Score
        completions = [torch.arange(D_MODEL, dtype=torch.float32) * (i+1) / D_MODEL for i in range(4)]
        reward_results = await reward_actor.score.call(completions, step=step)
        rewards = next(iter(reward_results.values()))

        # Train
        tokens = torch.randint(0, VOCAB_SIZE, (8, 16))
        labels = torch.randint(0, VOCAB_SIZE, (8, 16))
        with record_span("Training", EventType.FWD_BWD):
            results = await trainer.train_step.call(tokens, labels, step)
        r0 = next(iter(results.values()))

        # Rollout logging (non-blocking)
        records = [{"prompt_id": i, "reward": rewards[i], "policy_version": step} for i in range(len(rewards))]
        executor.submit(rollout_logger.log, records, step, lambda r: filter_top_bottom(r, k=2))

        # Aggregation every step (non-blocking, controller-side)
        add_step_tag("logging")
        record_event({"train.loss": r0["loss"], "train.grad_norm": r0["grad_norm"]})

        def _agg(a, w, s):
            agged = a.collect_and_aggregate(step=s)
            if agged:
                w(step=s, values=agged)

        executor.submit(_agg, aggregator, writer, step)

        print(f"  Step {step}/{NUM_STEPS}: loss={r0['loss']:.4f}, rewards={[f'{r:.3f}' for r in rewards]}")

    executor.shutdown(wait=True)
    rollout_logger.close()
    writer.close()
    await trainer.teardown.call()
    print(f"\nDone in {time.time()-t0:.1f}s. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
