# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy SPMD Training Example — Observability Baseline

Minimal SPMD training loop. Establishes the model and trainer patterns
that later commits will instrument with observability.

ToyTrainer follows key TorchTitan patterns:
- Per-layer torch.compile
- TP → compile → FSDP application order
- 2D DeviceMesh (DP × TP) with DTensor parameters

The model (TinyModel) uses MLPBlock with multiple "heads" (parallel linear
projections). This structure supports three compile-safety scenarios for
tensor metric collection:
- Scenario 1: per-layer metrics via for-loop in eager (over model.layers)
- Scenario 2: per-head metrics via for-loop inside compile (over block.heads)
- Scenario 3: merged metrics via same key without child_context

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
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torchtitan.distributed.utils import clip_grad_norm_

# ---- Config ----
NUM_STEPS = 20
D_MODEL = 64
HIDDEN_DIM = 128
N_HEADS = 3
VOCAB_SIZE = 32
SEQ_LEN = 16
BATCH_SIZE = 8
LR = 1e-3
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "toy_spmd")


# ---- Model ----
class MLPBlock(nn.Module):
    """MLP block with multiple parallel projections (heads) + output projection.

    Each head is an independent linear projection. Their outputs are summed and
    projected back to d_model. This gives us a natural for-loop over sub-modules
    inside the compiled forward, which demonstrates per-head metrics with
    child_context when observability is added.
    """

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
    """3-layer model with embedding and output head."""

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

    Parallelism application order follows TorchTitan (parallelize.py):
    TP first, then per-layer compile, then FSDP2.
    """

    def __init__(self, device, dp_mesh, tp_mesh, output_dir):
        self.device = device
        self.rank = dist.get_rank()
        self.output_dir = output_dir
        model = TinyModel().to(device)
        self._apply_tp(model, tp_mesh)
        self._apply_compile(model)
        self._apply_fsdp(model, dp_mesh)
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    @staticmethod
    def _replicate_params(module: nn.Module, tp_mesh) -> None:
        """Wrap non-TP-parallelized params as Replicate DTensors on the TP mesh.

        This is needed in our simplified TP setup where LayerNorm and other
        non-parallelized params must still be DTensors on the same mesh so
        FSDP2 sees a consistent parameter space. TorchTitan's full TP plan
        handles this via the model-specific parallelize functions.
        """
        for p_name, param in module.named_parameters():
            replicated = nn.Parameter(
                DTensor.from_local(param, tp_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated)

    def _apply_tp(self, model, tp_mesh):
        """Apply tensor parallelism. Embeddings and output use TP plans;
        remaining params are wrapped as Replicate DTensors."""
        parallelize_module(
            model, tp_mesh,
            {
                "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), use_local_output=False),
                "output": ColwiseParallel(output_layouts=Replicate(), use_local_output=False),
            },
        )
        self._replicate_params(model.norm, tp_mesh)
        for layer in model.layers.values():
            parallelize_module(
                layer, tp_mesh,
                {"out_proj": RowwiseParallel(use_local_output=False)},
            )
            # TP-parallelize each head
            for head_name in layer.heads:
                parallelize_module(
                    layer, tp_mesh,
                    {f"heads.{head_name}": ColwiseParallel(use_local_output=False)},
                )
            self._replicate_params(layer.norm, tp_mesh)

    def _apply_compile(self, model):
        """Per-layer torch.compile (same as TorchTitan default)."""
        for layer_id, block in model.layers.named_children():
            model.layers.register_module(layer_id, torch.compile(block, fullgraph=True))

    def _apply_fsdp(self, model, dp_mesh):
        """FSDP2 wrapping. Applied last (after TP and compile)."""
        fully_shard(model.tok_embeddings, mesh=dp_mesh)
        for layer in model.layers.values():
            fully_shard(layer, mesh=dp_mesh)
        fully_shard([model.norm, model.output], mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)

    def train_step(self, tokens, labels, loss_mask, step):
        """One training step. Returns (loss, grad_norm, dt_ms).

        loss_mask has different valid token counts per rank to exercise
        weighted metric reduction across ranks.
        """
        t0 = time.perf_counter()
        logits = self.model(tokens)
        # FSDP2 wraps all outputs as DTensors. Convert to local tensor for plain
        # loss computation with loss_mask. For large vocabs, use loss_parallel()
        # + ignore_index instead (see torchtitan/components/loss.py).
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        per_token_loss = F.cross_entropy(
            logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="none"
        )
        loss = (per_token_loss * loss_mask.flatten()).sum() / loss_mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        dt_ms = (time.perf_counter() - t0) * 1000
        return loss, grad_norm, dt_ms

    def train(self, tokens, labels, loss_mask, num_steps):
        """Full training loop (overfitting to a single batch)."""
        if self.rank == 0:
            print(f"{'Step':>4}  {'Loss':>10}  {'GradNorm':>10}  {'Time(ms)':>10}")
            print("-" * 50)
        for step in range(1, num_steps + 1):
            loss, grad_norm, dt_ms = self.train_step(tokens, labels, loss_mask, step)
            if self.rank == 0:
                print(f"{step:>4}  {loss.item():>10.4f}  {grad_norm.item():>10.4f}  {dt_ms:>10.1f}")

    def close(self):
        """Cleanup. Subclasses override to close writers, profilers, etc."""
        pass


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
    trainer = ToyTrainer(device, mesh["dp"], mesh["tp"], OUTPUT_DIR)

    if rank == 0:
        print(f"Toy SPMD: {world_size} GPUs, 2DP×2TP, {NUM_STEPS} steps")

    # Fixed data for overfitting (same batch every step, like SPMD testbed)
    torch.manual_seed(42 + rank)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

    # Different valid token counts per rank for weighted metric reduction.
    # Rank 0: 4 valid tokens, rank 1: 8, rank 2: 12, rank 3: all 16.
    valid_lengths = [4, 8, 12, SEQ_LEN]
    valid_len = valid_lengths[rank % len(valid_lengths)]
    loss_mask = torch.zeros(BATCH_SIZE, SEQ_LEN, device=device)
    loss_mask[:, :valid_len] = 1.0

    trainer.train(tokens, labels, loss_mask, NUM_STEPS)
    trainer.close()

    if rank == 0:
        print(f"\nDone. Output: {OUTPUT_DIR}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
