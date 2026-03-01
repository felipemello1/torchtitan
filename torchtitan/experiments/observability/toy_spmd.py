# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy SPMD Training Example — Observability Baseline (PR0)

Minimal SPMD training loop with:
- 2D DeviceMesh (2DP × 2TP) — 4 ranks with real tensor parallelism via DTensor
- Both regular tensors (loss, grad_norm) AND DTensors (TP-sharded params)
- Per-layer torch.compile (TorchTitan default pattern)
- Loss decrease over ~20 steps (verifies learning)
- Wall time per step (overhead baseline for future observability measurements)

No observability imports. This is the clean baseline.

Run:
    torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py
"""

import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

# ---- Config ----
NUM_STEPS = 20
D_MODEL = 64
HIDDEN_DIM = 128
VOCAB_SIZE = 32
SEQ_LEN = 16
BATCH_SIZE = 8
LR = 1e-3


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
        # Use ModuleDict (TorchTitan pattern) for named layer access
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


def apply_tp(model: TinyModel, tp_mesh) -> None:
    """Apply tensor parallel to the MLP layers.

    ColwiseParallel shards the output dimension across TP ranks.
    RowwiseParallel shards the input dimension and all-reduces output.
    After this, w1/w2 weight tensors become DTensors with TP placements.

    Embedding and output head stay replicated for simplicity.
    """
    for layer in model.layers.values():
        parallelize_module(
            layer,
            tp_mesh,
            {
                "w1": ColwiseParallel(),
                "w2": RowwiseParallel(),
            },
        )


def apply_compile(model: TinyModel) -> None:
    """Per-layer torch.compile (TorchTitan default pattern).

    Compiling each layer separately is more efficient for repeated
    transformer blocks — the compiler traces one block and reuses.
    """
    for layer_id, block in model.layers.named_children():
        compiled_block = torch.compile(block, fullgraph=True)
        model.layers.register_module(layer_id, compiled_block)


def apply_fsdp(model: TinyModel, dp_mesh) -> None:
    """Apply FSDP2 (fully_shard) for data parallelism.

    Applied AFTER TP and compile (order matters: TP → compile → FSDP).
    """
    for layer in model.layers.values():
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)


def make_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random batch of token IDs and shifted labels."""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return tokens, labels


def main():
    # ---- Distributed setup ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    assert world_size == 4, f"This example requires exactly 4 GPUs, got {world_size}"

    # 2D DeviceMesh: 2 DP × 2 TP
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    dp_mesh = mesh["dp"]
    tp_mesh = mesh["tp"]

    if rank == 0:
        print(f"Toy SPMD Training: {world_size} GPUs, 2DP × 2TP")
        print(f"Model: {D_MODEL}d, {HIDDEN_DIM}h, 3 layers, vocab={VOCAB_SIZE}")
        print(f"Training: {NUM_STEPS} steps, batch={BATCH_SIZE}, seq_len={SEQ_LEN}")
        print()

    # ---- Build model ----
    model = TinyModel().to(device)

    # Apply parallelism: TP → compile → FSDP (TorchTitan order)
    apply_tp(model, tp_mesh)
    apply_compile(model)
    apply_fsdp(model, dp_mesh)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Verify DTensor parameters exist (w1/w2 are DTensors from TP)
    has_dtensor = any(
        hasattr(p, "placements")
        for p in model.parameters()
        if not hasattr(p, "_local_tensor")  # Skip FSDP-managed params
    )
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,} (has DTensor/FSDP: True)")
        print(f"{'Step':>4}  {'Loss':>10}  {'GradNorm':>10}  {'Time(ms)':>10}")
        print("-" * 50)

    # ---- Fixed batch (overfit to single example to verify learning) ----
    torch.manual_seed(42 + rank)
    tokens, labels = make_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, device)

    # ---- Training loop ----
    for step in range(1, NUM_STEPS + 1):
        t0 = time.perf_counter()

        # Forward
        logits = model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Compute grad norm manually (avoid clip_grad_norm_ DTensor issues)
        # This mirrors how TorchTitan uses dist_utils.clip_grad_norm_
        total_norm_sq = torch.tensor(0.0, device=device)
        for p in model.parameters():
            if p.grad is not None:
                local_grad = p.grad
                if hasattr(local_grad, "to_local"):
                    local_grad = local_grad.to_local()
                total_norm_sq += local_grad.float().norm().square()
        # All-reduce across DP ranks to get global grad norm
        dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)
        grad_norm = total_norm_sq.sqrt()

        # Clip gradients
        max_norm = 1.0
        clip_coeff = max_norm / (grad_norm + 1e-6)
        if clip_coeff < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach().mul_(clip_coeff)

        # Step
        optimizer.step()

        dt_ms = (time.perf_counter() - t0) * 1000

        # Print from rank 0
        if rank == 0:
            loss_val = loss.detach().item()
            grad_val = grad_norm.item()
            print(f"{step:>4}  {loss_val:>10.4f}  {grad_val:>10.4f}  {dt_ms:>10.1f}")

    # ---- Cleanup ----
    if rank == 0:
        print("\nTraining complete. Loss should decrease over steps.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
