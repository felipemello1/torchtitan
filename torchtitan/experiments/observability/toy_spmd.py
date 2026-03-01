# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy SPMD Training Example — PR0 + PR1

ToyTrainer mirrors TorchTitan's Trainer pattern:
- __init__: builds model, applies TP/compile/FSDP, creates optimizer
- train_step: one step of forward/backward/optimizer
- train: full training loop
- close: cleanup

Run:
    torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py
"""

from dataclasses import dataclass
import json
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

# PR1: System metrics
from torchtitan.observability import (
    add_step_tag, clear_step_tags, EventType, init_observability,
    record_event, record_span, set_step,
)

# PR2: Tensor metrics
from torchtitan.observability import (
    child_context, InvocationContext, MaxTMetric, MeanTMetric,
    record_tensor_metric, replicate_to_host,
)

# PR3: Logging boundary + backends
from torchtitan.observability import (
    CompositeSummaryWriter, EveryNSteps,
)

# PR4: CPU metrics + aggregation
from torchtitan.observability import (
    DefaultAggregator, log_reduced_metrics, MeanMetric, MaxMetric, record_metric,
)

# PR5: Profiling
from torchtitan.observability import profile_annotation

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
        w2_out = self.w2(h)
        # PR2: Partial DTensor metric — exercises deferred reduction
        record_tensor_metric("w2_partial", MeanTMetric(sum=w2_out.abs().mean(), weight=1))
        return x + w2_out


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


@dataclass
class LoggingConfig:
    log_freq: int = 5
    enable_tensorboard: bool = False
    tb_log_dir: str = "tb"
    enable_wandb: bool = False
    enable_console: bool = True


# ---- Trainer (mirrors TorchTitan Trainer pattern) ----
class ToyTrainer:
    """Minimal trainer with TP + compile + FSDP, matching TorchTitan's structure."""

    def __init__(self, device, dp_mesh, tp_mesh, output_dir,
                 logging_cfg: LoggingConfig = LoggingConfig()):
        self.device = device
        self.rank = dist.get_rank()
        self.output_dir = output_dir
        model = TinyModel().to(device)
        self._apply_tp(model, tp_mesh)
        self._apply_compile(model)
        self._apply_fsdp(model, dp_mesh)
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        init_observability(source="trainer", output_dir=output_dir, rank=self.rank)
        self._setup_logging(logging_cfg)

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

    def _setup_logging(self, cfg: LoggingConfig):
        """Configure logging schedule and optional writer/aggregator.

        The log_schedule always exists — it gates replicate_to_host.
        Writer and aggregator are only created if there are backends.
        """
        self.log_schedule = EveryNSteps(every_n=cfg.log_freq, additional_steps={1})
        writers = {}
        if cfg.enable_console:
            from torchtitan.observability import LoggingSummaryWriter
            writers["console"] = LoggingSummaryWriter()
        # Add more backends here (tensorboard, wandb) as needed
        if writers:
            self.writer = CompositeSummaryWriter(writers=writers)
            self.writer.open()
            self.aggregator = DefaultAggregator(
                experiment_log_dir=os.path.join(self.output_dir, 'experiment_logs')
            )
        else:
            self.writer = None
            self.aggregator = None

    def train_step(self, tokens, labels, step):
        """One training step. Returns (loss, grad_norm, dt_ms, ctx)."""
        t0 = time.perf_counter()
        set_step(step)
        clear_step_tags()
        with InvocationContext() as ctx:
            with record_span("Forward/Backward", EventType.FWD_BWD):
                logits = self.model(tokens)
                with loss_parallel():
                    loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1))
                    record_tensor_metric("loss", MeanTMetric(sum=loss, weight=torch.tensor(1.0, device=self.device)))
                    self.optimizer.zero_grad()
                    with profile_annotation("backward"):
                        loss.backward()
            with record_span("Optimizer", EventType.OPTIM):
                with profile_annotation("optimizer"):
                    grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            record_tensor_metric("grad_norm", MeanTMetric(mean=grad_norm, weight=torch.tensor(1.0, device=self.device)))
            with child_context("layer_0"):
                w1 = list(self.model.layers.values())[0].w1.weight
                record_tensor_metric("w1_norm", MeanTMetric(mean=w1.float().norm(), weight=torch.tensor(1.0, device=self.device)))
        dt_ms = (time.perf_counter() - t0) * 1000
        record_metric("learning_rate", MeanMetric(value=LR))
        record_metric("step_time_ms", MeanMetric(value=dt_ms))
        if self.log_schedule(step):
            scalars = replicate_to_host(ctx.summaries())
            log_reduced_metrics(scalars)
            add_step_tag("logging")
            record_event({"train.loss": loss.item(), "train.grad_norm": grad_norm.item()})
            if self.writer and self.rank == 0:
                aggregated = self.aggregator.collect_and_aggregate(step=step)
                self.writer(step=step, values=aggregated)
            if self.rank == 0:
                with open(os.path.join(self.output_dir, f"scalars_step_{step}.json"), "w") as f:
                    json.dump(scalars, f)
        return loss, grad_norm, dt_ms, ctx

    def train(self, tokens, labels, num_steps):
        """Full training loop."""
        if self.rank == 0:
            print(f"{'Step':>4}  {'Loss':>10}  {'GradNorm':>10}  {'Time(ms)':>10}")
            print("-" * 50)
        losses = []
        for step in range(1, num_steps + 1):
            loss, grad_norm, dt_ms, ctx = self.train_step(tokens, labels, step)
            losses.append(loss.item())
            if self.rank == 0:
                print(f"{step:>4}  {loss.item():>10.4f}  {grad_norm.item():>10.4f}  {dt_ms:>10.1f}")
        if self.rank == 0:
            with open(os.path.join(self.output_dir, "losses.json"), "w") as f:
                json.dump(losses, f)

    def close(self):
        """Close writer if present."""
        if self.writer:
            self.writer.close()


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
    trainer = ToyTrainer(device, mesh["dp"], mesh["tp"], OUTPUT_DIR,
                         logging_cfg=LoggingConfig(log_freq=5, enable_console=True))

    if rank == 0:
        print(f"Toy SPMD: {world_size} GPUs, 2DP×2TP, {NUM_STEPS} steps")

    torch.manual_seed(42 + rank)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

    trainer.train(tokens, labels, NUM_STEPS)
    trainer.close()

    # Generate Chrome Trace for visualization (rank 0 only)
    if rank == 0:
        from torchtitan.experiments.observability.visualize import to_chrome_trace

        sys_log_dir = os.path.join(OUTPUT_DIR, "system_logs")
        trace_path = os.path.join(OUTPUT_DIR, "trace.json")
        to_chrome_trace(sys_log_dir, trace_path)
        print(f"\nDone. Output: {OUTPUT_DIR}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
