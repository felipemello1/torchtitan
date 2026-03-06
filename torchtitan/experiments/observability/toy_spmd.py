# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy SPMD Training Example

Minimal SPMD training loop with:
- TP
- per layer compile
- FSDP2.

Serves as a testbed for the new observability features.

The model (TinyModel) is just some dummy MLPBlock with multiple "heads" so
we can test logging heads and layers within compiled regions.

Each rank gets a different number of valid tokens (via loss_mask) to exercise
weighted metric reduction.

Run:
    torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd
"""

import os
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
    RowwiseParallel,
)
from torchtitan.distributed.utils import clip_grad_norm_
from torchtitan.observability import (
    EventType,
    init_observability,
    MetricsProcessor,
    record_event,
    record_span,
)

# ---- Config ----
NUM_STEPS = 20
D_MODEL = 64
HIDDEN_DIM = 128
N_HEADS = 3
VOCAB_SIZE = 32
SEQ_LEN = 16
BATCH_SIZE = 8
LR = 1e-3
IGNORE_INDEX = -100
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "toy_spmd")


# ---- Data ----
class RepeatDataset:
    """Yields the same batch every step so we can overfit and se loss decrease."""

    def __init__(
        self, tokens: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor
    ):
        self.tokens = tokens
        self.labels = labels
        self.loss_mask = loss_mask

    def __iter__(self):
        while True:
            yield self.tokens, self.labels, self.loss_mask


# ---- Model ----
class MLPBlock(nn.Module):
    """Dummy MLP block with multiple projections (heads) so we can test
    logging on ModuleDict"""

    def __init__(self, d_model: int, hidden_dim: int, n_heads: int = N_HEADS):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict(
            {str(i): nn.Linear(d_model, hidden_dim, bias=False) for i in range(n_heads)}
        )
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        parts = []
        for head in self.heads.values():
            parts.append(F.silu(head(h)))
        return x + self.out_proj(sum(parts))


class TinyModel(nn.Module):
    """Dummy 3-layer model with embedding and output head."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = 3,
    ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleDict(
            {str(i): MLPBlock(d_model, hidden_dim) for i in range(n_layers)}
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h)
        h = self.norm(h)
        return self.output(h)


# ---- Trainer ----
class ToyTrainer:
    """Minimal trainer with TP + compile + FSDP.

    Structure tries to mirrors Trainer in torchtitan/trainer.py:
    - __init__: build model, apply parallelism, create optimizer
    - batch_generator: wraps dataloader, yields batches
    - train_step: one forward/backward/optimizer step
    - train: owns the training loop
    """

    def __init__(self, device, dp_mesh, tp_mesh, output_dir, dataloader):
        self.device = device
        self.rank = dist.get_rank()
        self.output_dir = output_dir
        self.dataloader = dataloader
        self.step = 0

        # Initialize observability (JSONL file handlers) before MetricsProcessor.
        init_observability(source="trainer", output_dir=output_dir, rank=self.rank)
        self.metrics_processor = MetricsProcessor(
            MetricsProcessor.Config(), dump_folder=output_dir, rank=self.rank
        )

        model = TinyModel().to(device)
        self._apply_tp(model, tp_mesh)
        self._apply_compile(model)
        self._apply_fsdp(model, dp_mesh)
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    @staticmethod
    def _replicate_params(module: nn.Module, tp_mesh) -> None:
        """Wrap non-TP-parallelized params (LayerNorm) as Replicate DTensors
        on the TP mesh so it works with FSDP"""
        for p_name, param in module.named_parameters():
            replicated = nn.Parameter(
                DTensor.from_local(param, tp_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated)

    def _apply_tp(self, model, tp_mesh):
        """Apply tensor parallelism. Embeddings and output use TP plans;
        remaining params are wrapped as Replicate DTensors."""
        parallelize_module(
            model,
            tp_mesh,
            {
                "tok_embeddings": RowwiseParallel(
                    input_layouts=Replicate(), use_local_output=False
                ),
                "output": ColwiseParallel(
                    output_layouts=Shard(-1), use_local_output=False
                ),
            },
        )
        self._replicate_params(model.norm, tp_mesh)
        for layer in model.layers.values():
            parallelize_module(
                layer,
                tp_mesh,
                {"out_proj": RowwiseParallel(use_local_output=False)},
            )
            for head_name in layer.heads:
                parallelize_module(
                    layer,
                    tp_mesh,
                    {f"heads.{head_name}": ColwiseParallel(use_local_output=False)},
                )
            self._replicate_params(layer.norm, tp_mesh)

    def _apply_compile(self, model):
        """Per-layer torch.compile"""
        for layer_id, block in model.layers.named_children():
            model.layers.register_module(layer_id, torch.compile(block, fullgraph=True))

    def _apply_fsdp(self, model, dp_mesh):
        """FSDP2 wrapping. Applied last (after TP and compile)."""
        fully_shard(model.tok_embeddings, mesh=dp_mesh)
        for layer in model.layers.values():
            fully_shard(layer, mesh=dp_mesh)
        fully_shard([model.norm, model.output], mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)

    def batch_generator(self, data_iterable):
        """Wraps a dataloader into an iterator."""
        data_iterator = iter(data_iterable)
        while True:
            yield next(data_iterator)

    def compute_loss(self, logits, labels, loss_mask):
        """Compute loss using loss_parallel + ignore_index.

        Returns (loss_sum, valid_tokens). The caller divides for backward.
        """
        masked_labels = labels.clone().flatten()
        masked_labels[loss_mask.flatten() == 0] = IGNORE_INDEX
        loss_sum = F.cross_entropy(
            logits.flatten(0, 1).float(),
            masked_labels,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        valid_tokens = (masked_labels != IGNORE_INDEX).sum()
        return loss_sum, valid_tokens

    def train_step(self, tokens, labels, loss_mask):
        """One training step. Returns (loss, grad_norm)."""
        with loss_parallel():
            with record_span("Forward/Backward", EventType.FWD_BWD):
                logits = self.model(tokens)
                loss_sum, valid_tokens = self.compute_loss(logits, labels, loss_mask)
                loss = loss_sum / valid_tokens
                self.optimizer.zero_grad()
                loss.backward()
            with record_span("Optimizer", EventType.OPTIM):
                grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
        return loss, grad_norm

    def train(self, num_steps):
        """Full training loop. Mirrors Trainer.train structure."""
        data_iterator = self.batch_generator(self.dataloader)

        # header of prints
        if self.rank == 0:
            print(f"{'Step':>4}  {'Loss':>10}  {'GradNorm':>10}  {'Time(ms)':>10}")
            print("-" * 50)

        for step in range(1, num_steps + 1):
            self.step = step
            self.metrics_processor.set_step(step)

            t0 = time.perf_counter()
            tokens, labels, loss_mask = next(data_iterator)
            loss, grad_norm = self.train_step(tokens, labels, loss_mask)
            dt_ms = (time.perf_counter() - t0) * 1000

            # System metrics: point-in-time scalars logged to system JSONL
            record_event(
                {"train.loss": loss.item(), "train.grad_norm": grad_norm.item()}
            )

            if self.rank == 0:
                print(
                    f"{step:>4}  {loss.item():>10.4f}  {grad_norm.item():>10.4f}  {dt_ms:>10.1f}"
                )

    def close(self):
        self.metrics_processor.close()


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    assert world_size == 4, f"Requires 4 GPUs, got {world_size}"

    if rank == 0 and os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    dist.barrier()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    dp_rank = mesh["dp"].get_local_rank()

    if rank == 0:
        print(f"Toy SPMD: {world_size} GPUs, 2DPx2TP, {NUM_STEPS} steps")

    # --- setup data ---
    # Fixed data for overfitting.
    # Seed by dp_rank so TP peers get identical data
    torch.manual_seed(42 + dp_rank)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

    # Different valid token counts per DP group for weighted metric reduction.
    # DP group 0 (rank 0, 1): 4 valid tokens per sequence.
    # DP group 1 (rank 2, 3): 12 valid tokens per sequence.
    # TP peers within a group get the same loss_mask.
    valid_lengths = [4, 12]
    valid_len = valid_lengths[dp_rank % len(valid_lengths)]
    loss_mask = torch.zeros(BATCH_SIZE, SEQ_LEN, device=device)
    loss_mask[:, :valid_len] = 1.0

    dataloader = RepeatDataset(tokens, labels, loss_mask)

    # --- train ---
    trainer = ToyTrainer(device, mesh["dp"], mesh["tp"], OUTPUT_DIR, dataloader)

    trainer.train(NUM_STEPS)
    trainer.close()

    if rank == 0:
        from torchtitan.experiments.observability.visualize import to_chrome_trace

        sys_log_dir = os.path.join(OUTPUT_DIR, "system_logs")
        trace_path = os.path.join(OUTPUT_DIR, "trace.json")
        to_chrome_trace(sys_log_dir, trace_path)
        print(f"\nDone. Output: {OUTPUT_DIR}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
