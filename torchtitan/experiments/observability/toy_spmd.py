# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy SPMD Training Example — Observability Baseline (PR0)

ToyTrainer mirrors TorchTitan's Trainer pattern:
- __init__: builds model, applies TP/compile/FSDP, creates optimizer
- train_step: one step of forward/backward/optimizer
- train: full training loop
- close: cleanup

Run:
    torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py
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
    RowwiseParallel,
    parallelize_module,
)
from torchtitan.distributed.utils import clip_grad_norm_

# ---- Config ----
NUM_STEPS = 20
D_MODEL = 64
HIDDEN_DIM = 128
VOCAB_SIZE = 32
SEQ_LEN = 16
BATCH_SIZE = 8
LR = 1e-3
OUTPUT_DIR = '/tmp/toy_spmd_output'


# ---- Model ----
class TransformerBlock(nn.Module):
    """Minimal transformer block: attention-free, just MLP + norm + residual."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.w1(h))
        return x + self.w2(h)


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
            {str(i): TransformerBlock(d_model, hidden_dim) for i in range(n_layers)}
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h)
        h = self.norm(h)
        return self.output(h)


# ---- Trainer (mirrors TorchTitan Trainer pattern) ----
class ToyTrainer:
    """Minimal trainer with TP + compile + FSDP, matching TorchTitan's structure."""

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
        """Put params on the TP mesh as Replicate DTensors."""
        for p_name, param in module.named_parameters():
            replicated = nn.Parameter(
                DTensor.from_local(param, tp_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated)

    def _apply_tp(self, model, tp_mesh):
        """Apply TP to ALL params so they share the same 2D mesh."""
        parallelize_module(
            model, tp_mesh,
            {
                "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), use_local_output=False),
                "output": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
            },
        )
        self._replicate_params(model.norm, tp_mesh)
        for layer in model.layers.values():
            parallelize_module(
                layer, tp_mesh,
                {"w1": ColwiseParallel(use_local_output=False), "w2": RowwiseParallel(use_local_output=False)},
            )
            self._replicate_params(layer.norm, tp_mesh)

    def _apply_compile(self, model):
        """Per-layer torch.compile."""
        for layer_id, block in model.layers.named_children():
            model.layers.register_module(layer_id, torch.compile(block, fullgraph=True))

    def _apply_fsdp(self, model, dp_mesh):
        """FSDP2. Order: TP → compile → FSDP."""
        fully_shard(model.tok_embeddings, mesh=dp_mesh)
        for layer in model.layers.values():
            fully_shard(layer, mesh=dp_mesh)
        fully_shard([model.norm, model.output], mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)

    def train_step(self, tokens, labels, step):
        """One training step. Returns (loss, grad_norm)."""
        t0 = time.perf_counter()
        logits = self.model(tokens)
        with loss_parallel():
            loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1))
            self.optimizer.zero_grad()
            loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        dt_ms = (time.perf_counter() - t0) * 1000
        return loss, grad_norm, dt_ms

    def train(self, tokens, labels, num_steps):
        """Full training loop."""
        if self.rank == 0:
            print(f"{'Step':>4}  {'Loss':>10}  {'GradNorm':>10}  {'Time(ms)':>10}")
            print("-" * 50)
        for step in range(1, num_steps + 1):
            loss, grad_norm, dt_ms = self.train_step(tokens, labels, step)
            if self.rank == 0:
                print(f"{step:>4}  {loss.item():>10.4f}  {grad_norm.item():>10.4f}  {dt_ms:>10.1f}")

    def close(self):
        """Cleanup. Override in subclasses to close writers."""
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

    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    trainer = ToyTrainer(device, mesh["dp"], mesh["tp"], OUTPUT_DIR)

    if rank == 0:
        print(f"Toy SPMD: {world_size} GPUs, 2DP×2TP, {NUM_STEPS} steps")

    torch.manual_seed(42 + rank)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

    trainer.train(tokens, labels, NUM_STEPS)
    trainer.close()

    if rank == 0:
        print(f"\nDone. Output: {OUTPUT_DIR}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
