# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This is not a production training recipe. It is just dummy incomplete code
to demonstrate observability APIs.

Toy RL with Monarch actors: GeneratorActor produces completions,
RewardActor scores them, TrainerActor trains on them. Controller
orchestrates the loop.

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
    SEQ_LEN,
    setup_data,
    ToyTrainer,
    VOCAB_SIZE,
)
from torchtitan.observability import (
    clear_step_tags,
    EventType,
    init_observability,
    record_event,
    record_span,
    set_step,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

NUM_STEPS = 6
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "toy_rl")


class GeneratorActor(Actor):
    """Dummy generator that returns fixed tokens as if they were generated."""

    @endpoint
    async def setup(self):
        dataset = setup_data()
        self.tokens = dataset.tokens
        self.labels = dataset.labels
        self.loss_mask = dataset.loss_mask

    @endpoint
    async def set_step(self, step: int):
        """Receive step from controller."""
        set_step(step)

    @endpoint
    async def generate(
        self, prompts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate completions from prompts. Returns (tokens, labels, loss_mask)."""
        return self.tokens, self.labels, self.loss_mask


class TrainerActor(Actor):
    @endpoint
    async def setup(self):
        rank = current_rank().rank
        self.device = f"cuda:{rank}"
        torch.cuda.set_device(self.device)
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", socket.gethostname())
            os.environ.setdefault("MASTER_PORT", "29500")
            world_size = int(os.environ.get("WORLD_SIZE", 4))
            torch.distributed.init_process_group(
                backend="nccl", rank=rank, world_size=world_size
            )
        init_observability(source="trainer", output_dir=OUTPUT_DIR, rank=rank)
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
        self.trainer = ToyTrainer(self.device, mesh["dp"], mesh["tp"], OUTPUT_DIR)

    @endpoint
    async def set_step(self, step: int):
        """Receive step from controller. Sets step on trainer + ContextVar."""
        self.trainer.step = step
        self.trainer.metrics_processor.set_step(step)

    @endpoint
    async def train_step(self, tokens, labels, loss_mask) -> float:
        """Train one step on generated completions. Returns loss value."""
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
    """Dummy reward actor. Scores are not used by the trainer — this actor
    exists to demonstrate multi-actor observability patterns."""

    @endpoint
    async def setup(self):
        rank = current_rank().rank
        init_observability(source="reward", output_dir=OUTPUT_DIR, rank=rank)

    @endpoint
    async def set_step(self, step: int):
        """Receive step from controller."""
        set_step(step)

    @endpoint
    async def score(self, completions: list[torch.Tensor]) -> list[float]:
        """Score completions. Returns dummy constant rewards."""
        with record_span("rl_time/scoring_s", EventType.RL_SCORING):
            rewards = [1.0] * len(completions)
        return rewards


async def main():
    t0 = time.time()
    logger.info(f"Toy RL: {NUM_STEPS} steps")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    init_observability(source="controller", output_dir=OUTPUT_DIR, rank=0)

    # ---- Setup ----
    host = this_host()

    generator_mesh = host.spawn_procs(per_host={"procs": 1}, name="generator")
    generator = generator_mesh.spawn("generator", GeneratorActor)
    await generator.setup.call()

    trainer_mesh = host.spawn_procs(per_host={"gpus": 4}, name="trainer")
    trainer = trainer_mesh.spawn("trainer", TrainerActor)
    await trainer.setup.call()

    reward_mesh = host.spawn_procs(per_host={"procs": 1}, name="reward")
    reward_actor = reward_mesh.spawn("reward", RewardActor)
    await reward_actor.setup.call()

    actors = [generator, trainer, reward_actor]
    logger.info("Actors spawned.")

    # Dummy prompts for the generator.
    prompts = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # ---- Training loop ----
    async def run_training():
        for step in range(1, NUM_STEPS + 1):
            set_step(step)
            clear_step_tags()
            for actor in actors:
                await actor.set_step.call(step)

            # Generate completions.
            gen_results = await generator.generate.call(prompts)
            tokens, labels, loss_mask = next(iter(gen_results.values()))

            # Score is not used anywhere. This is a dummy call to demonstrate
            # multi-actor observability.
            with record_span("rl_time/scoring_s", EventType.RL_SCORING):
                reward_results = await reward_actor.score.call(
                    [tokens[i].float() for i in range(4)]
                )
                rewards = next(iter(reward_results.values()))

            with record_span("rl_time/training_s", EventType.FWD_BWD):
                loss_results = await trainer.train_step.call(
                    tokens, labels, loss_mask
                )
                loss = next(iter(loss_results.values()))

            reward_mean = sum(rewards) / len(rewards)
            record_event({"train.loss": loss, "reward_mean": reward_mean})
            logger.info(
                f"step: {step}  loss: {loss:8.5f}  reward_mean: {reward_mean:8.5f}"
            )

    await run_training()

    # ---- Cleanup ----
    await trainer.teardown.call()
    logger.info(f"Done in {time.time() - t0:.1f}s. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
