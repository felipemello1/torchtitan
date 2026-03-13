# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This is not a production training recipe. It is just dummy incomplete code
to demonstrate observability APIs.

Toy RL with Monarch actors: RollouterActor produces completions,
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
from dataclasses import dataclass

import torch
from monarch.actor import Actor, current_rank, endpoint, this_host
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.experiments.observability.toy_spmd import (
    BATCH_SIZE,
    DP_SIZE,
    SEQ_LEN,
    setup_data,
    ToyTrainer,
    VOCAB_SIZE,
)
from torchtitan.observability import (
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


# ---------------------------------------------------------------------------
# RolloutOutput
# ---------------------------------------------------------------------------


@dataclass
class RolloutOutput:
    """One prompt+completion pair from a rollout.

    Token fields and training tensors are used for training.
    Text fields are for logging and human inspection only — in a real
    pipeline they would be populated by tokenizer.decode().
    """

    prompt_tokens: list[int]
    completion_tokens: list[int]
    prompt_text: str
    completion_text: str
    reward: float | None = None
    # Training tensors (one sample, not batched)
    tokens: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None

    def to_logging_dict(self) -> dict:
        """Convert to a dict for logging. Only text + reward."""
        d = {"prompt": self.prompt_text, "completion": self.completion_text}
        if self.reward is not None:
            d["reward"] = self.reward
        return d


def rollouts_to_train_batch(
    rollouts: list[RolloutOutput],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack individual rollout tensors into a training batch.

    Returns:
        (tokens, labels, loss_mask), each of shape (batch_size, seq_len).
    """
    return (
        torch.stack([r.tokens for r in rollouts]),
        torch.stack([r.labels for r in rollouts]),
        torch.stack([r.loss_mask for r in rollouts]),
    )


def filter_top_bottom(
    records: list[dict], key: str = "reward", k: int = 1
) -> list[dict]:
    """Keep top-k and bottom-k records by a key.

    If fewer than 2*k records, returns all records.
    """
    sorted_recs = sorted(records, key=lambda r: r.get(key, 0))
    k = min(k, len(sorted_recs) // 2) if sorted_recs else 0
    if k == 0:
        return sorted_recs
    return sorted_recs[:k] + sorted_recs[-k:]


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------


class RollouterActor(Actor):
    """Dummy rollouter that returns fixed data as if it were generated."""

    @endpoint
    async def setup(self):
        dataset = setup_data(batch_size=DP_SIZE * BATCH_SIZE)
        self.dataset = dataset

    @endpoint
    async def set_step(self, step: int):
        """Receive step from controller."""
        set_step(step)

    @endpoint
    async def do_rollouts(self, prompts: torch.Tensor) -> list[RolloutOutput]:
        """Produce rollouts. Each carries one sample's tensors and dummy text.

        This is a dummy — not a real generation loop. In a real pipeline,
        the model would generate completions and the tokenizer would
        decode them into text.
        """
        rollouts = [
            RolloutOutput(
                prompt_tokens=self.dataset.tokens[i].tolist(),
                completion_tokens=self.dataset.tokens[i].tolist(),
                prompt_text=f"What is {i}+{i}?",
                completion_text=f"The answer is {i + i}.",
                tokens=self.dataset.tokens[i],
                labels=self.dataset.labels[i],
                loss_mask=self.dataset.loss_mask[i],
            )
            for i in range(len(self.dataset.tokens))
        ]
        return rollouts


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
        self.dp_rank = mesh["dp"].get_local_rank()
        self.trainer = ToyTrainer(self.device, mesh["dp"], mesh["tp"], OUTPUT_DIR)

    @endpoint
    async def set_step(self, step: int):
        """Receive step from controller. Sets step on trainer."""
        self.trainer.step = step
        self.trainer.metrics_processor.set_step(step)

    @endpoint
    async def train_step(self, tokens, labels, loss_mask) -> float:
        """Train one step on generated completions. Returns loss value."""
        # Slice this DP rank's shard from the full batch.
        start = self.dp_rank * BATCH_SIZE
        end = start + BATCH_SIZE
        tokens = tokens[start:end].to(self.device)
        labels = labels[start:end].to(self.device)
        loss_mask = loss_mask[start:end].to(self.device)

        # Train step
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
    async def score(self, rollouts: list[RolloutOutput]) -> list[RolloutOutput]:
        """Score rollouts. Fills in reward field and returns them."""
        with record_span("rl_time/scoring_s", EventType.RL_SCORING):
            for rollout in rollouts:
                rollout.reward = 1.0  # dummy constant reward
        return rollouts


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


async def main():
    t0 = time.time()
    logger.info(f"Toy RL: {NUM_STEPS} steps")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    init_observability(source="controller", output_dir=OUTPUT_DIR, rank=0)

    # ---- Setup ----
    host = this_host()

    rollouter_mesh = host.spawn_procs(per_host={"procs": 1}, name="rollouter")
    rollouter = rollouter_mesh.spawn("rollouter", RollouterActor)
    await rollouter.setup.call()

    trainer_mesh = host.spawn_procs(per_host={"gpus": 4}, name="trainer")
    trainer = trainer_mesh.spawn("trainer", TrainerActor)
    await trainer.setup.call()

    reward_mesh = host.spawn_procs(per_host={"procs": 1}, name="reward")
    reward_actor = reward_mesh.spawn("reward", RewardActor)
    await reward_actor.setup.call()

    actors = [rollouter, trainer, reward_actor]
    logger.info("Actors spawned.")

    # Dummy prompts for the rollouter.
    prompts = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # ---- Training loop ----
    async def run_training():
        for step in range(1, NUM_STEPS + 1):
            set_step(step)
            for actor in actors:
                await actor.set_step.call(step)

            with record_span("rl_time/generate_s", EventType.FETCHING_BATCH):
                result = await rollouter.do_rollouts.call(prompts)
                rollouts = next(iter(result.values()))

            with record_span("rl_time/scoring_s", EventType.RL_SCORING):
                result = await reward_actor.score.call(rollouts)
                rollouts = next(iter(result.values()))

            tokens, labels, loss_mask = rollouts_to_train_batch(rollouts)

            with record_span("rl_time/training_s", EventType.FWD_BWD):
                result = await trainer.train_step.call(tokens, labels, loss_mask)
                loss = next(iter(result.values()))

            record_event({"train.loss": loss})
            logger.info(
                f"step: {step}  loss: {loss:8.5f}  reward_mean: {reward_mean:8.5f}"
            )

    await run_training()

    # ---- Cleanup ----
    await trainer.teardown.call()
    logger.info(f"Done in {time.time() - t0:.1f}s. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
