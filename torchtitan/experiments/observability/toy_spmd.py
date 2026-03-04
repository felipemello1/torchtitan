# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toy SPMD Training Example — Logging Boundary + Backends

SPMD training with system metrics, compile-safe tensor metrics, and
logging backends (EveryNSteps schedule, CompositeSummaryWriter).

record_event and record_span always log to JSONL (every step).
Only replicate_to_host and backend writes are gated by the logging schedule.

Run:
    torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd
"""

from dataclasses import dataclass
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

from torchtitan.observability import (
    child_context,
    clear_step_tags,
    CompositeSummaryWriter,
    EventType,
    EveryNSteps,
    init_observability,
    LoggingSummaryWriter,
    MeanTMetric,
    record_event,
    record_span,
    record_tensor_metric,
    replicate_to_host,
    set_step,
    TensorBoardSummaryWriter,
    TensorMetricContext,
    WandbSummaryWriter,
)

from torchtitan.observability import (
    DefaultAggregator, log_reduced_metrics, MeanMetric, record_metric,
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "toy_spmd")


@dataclass
class LoggingConfig:
    """Configuration for logging backends and schedule."""
    log_freq: int = 5
    enable_tensorboard: bool = False
    tb_log_dir: str = "tb"
    enable_wandb: bool = False
    enable_console: bool = True


# ---- Model ----
class MLPBlock(nn.Module):
    """MLP block with multiple parallel projections (heads) + output projection.

    Demonstrates Scenarios 2 and 3 for compile-safe metrics:
    - Scenario 2: for-loop over heads INSIDE compile with child_context per head
    - Scenario 3: same "abs_mean" key WITHOUT child_context → merge_() accumulates
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
        for name, head in self.heads.items():
            with child_context(name):
                out = F.silu(head(h))
                record_tensor_metric("abs_mean", MeanTMetric(sum=out.abs().mean(), weight=1))
                parts.append(out)
            record_tensor_metric("abs_mean", MeanTMetric(sum=out.abs().mean(), weight=1))
        return x + self.out_proj(sum(parts))


class TinyModel(nn.Module):
    """3-layer model with embedding and output head.

    Scenario 1: per-layer metrics in eager mode — the for-loop over
    self.layers is in eager (outside compile), so child_context("0") etc.
    create separate metric namespaces per layer.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, hidden_dim=HIDDEN_DIM, n_layers=3):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleDict({str(i): MLPBlock(d_model, hidden_dim) for i in range(n_layers)})
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        h = self.tok_embeddings(tokens)
        for name, layer in self.layers.items():
            with child_context(name):
                h = layer(h)
        h = self.norm(h)
        return self.output(h)


# ---- Trainer ----
class ToyTrainer:
    """Minimal trainer with TP + compile + FSDP and logging backends.

    Parallelism application order follows TorchTitan (parallelize.py):
    TP first, then per-layer compile, then FSDP2.
    """

    def __init__(self, device, dp_mesh, tp_mesh, output_dir, logging_cfg=LoggingConfig()):
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
    def _replicate_params(module, tp_mesh):
        """Wrap non-TP-parallelized params as Replicate DTensors on the TP mesh.

        Needed so FSDP2 sees a consistent DTensor parameter space.
        TorchTitan's full TP plan handles this via model-specific parallelize functions.
        """
        for p_name, param in module.named_parameters():
            module.register_parameter(
                p_name, nn.Parameter(DTensor.from_local(param, tp_mesh, [Replicate()], run_check=False))
            )

    def _apply_tp(self, model, tp_mesh):
        parallelize_module(model, tp_mesh, {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), use_local_output=False),
            "output": ColwiseParallel(output_layouts=Replicate(), use_local_output=False),
        })
        self._replicate_params(model.norm, tp_mesh)
        for layer in model.layers.values():
            parallelize_module(layer, tp_mesh, {"out_proj": RowwiseParallel(use_local_output=False)})
            for head_name in layer.heads:
                parallelize_module(layer, tp_mesh, {f"heads.{head_name}": ColwiseParallel(use_local_output=False)})
            self._replicate_params(layer.norm, tp_mesh)

    def _apply_compile(self, model):
        for layer_id, block in model.layers.named_children():
            model.layers.register_module(layer_id, torch.compile(block, fullgraph=True))

    def _apply_fsdp(self, model, dp_mesh):
        fully_shard(model.tok_embeddings, mesh=dp_mesh)
        for layer in model.layers.values():
            fully_shard(layer, mesh=dp_mesh)
        fully_shard([model.norm, model.output], mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)

    def _setup_logging(self, cfg):
        """Set up logging schedule, backends, and aggregator.

        The schedule gates replicate_to_host (expensive) and backend writes.
        System JSONL (record_span, record_event) is NOT gated.
        """
        self.log_schedule = EveryNSteps(every_n=cfg.log_freq, additional_steps={1})
        writers = {}
        if cfg.enable_console:
            writers["console"] = LoggingSummaryWriter()
        if cfg.enable_tensorboard:
            writers["tb"] = TensorBoardSummaryWriter.Config(
                log_dir=os.path.join(self.output_dir, cfg.tb_log_dir),
            ).build()
        if cfg.enable_wandb:
            writers["wandb"] = WandbSummaryWriter.Config().build()
        if writers:
            self.writer = CompositeSummaryWriter(writers=writers)
            self.writer.open()
            self.aggregator = DefaultAggregator(
                experiment_log_dir=os.path.join(self.output_dir, 'experiment_logs')
            )
        else:
            self.writer = None
            self.aggregator = None

    def train_step(self, tokens, labels, loss_mask, step):
        t0 = time.perf_counter()
        set_step(step)
        clear_step_tags()

        with TensorMetricContext() as ctx:
            with record_span("Forward/Backward", EventType.FWD_BWD):
                logits = self.model(tokens)
                if isinstance(logits, DTensor):
                    logits = logits.full_tensor()
                per_token_loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="none")
                loss = (per_token_loss * loss_mask.flatten()).sum() / loss_mask.sum()
                record_tensor_metric("loss", MeanTMetric(sum=loss * loss_mask.sum(), weight=loss_mask.sum()))
                self.optimizer.zero_grad()
                loss.backward()
            with record_span("Optimizer", EventType.OPTIM):
                grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        dt_ms = (time.perf_counter() - t0) * 1000
        # CPU metrics: always log to experiment JSONL (every step)
        record_metric("learning_rate", MeanMetric(sum=LR))
        record_metric("step_time_ms", MeanMetric(sum=dt_ms))

        # System metrics: always log to system JSONL (every step)
        record_event({"train.loss": loss.item(), "train.grad_norm": grad_norm.item()})

        # Tensor metrics + aggregation: gated by schedule (expensive)
        if self.log_schedule(step):
            scalars = replicate_to_host(ctx.summaries())
            log_reduced_metrics(scalars)
            if self.aggregator and self.rank == 0:
                aggregated = self.aggregator.collect_and_aggregate(step=step)
                if self.writer:
                    self.writer(step=step, values=aggregated)
            if self.rank == 0 and scalars:
                keys = ", ".join(f"{k}={v:.4f}" for k, v in sorted(scalars.items())[:5])
                print(f"  [tensor metrics] {keys} ...")

        return loss, grad_norm, dt_ms

    def train(self, tokens, labels, loss_mask, num_steps):
        if self.rank == 0:
            print(f"{'Step':>4}  {'Loss':>10}  {'GradNorm':>10}  {'Time(ms)':>10}")
            print("-" * 50)
        for step in range(1, num_steps + 1):
            loss, grad_norm, dt_ms = self.train_step(tokens, labels, loss_mask, step)
            if self.rank == 0:
                print(f"{step:>4}  {loss.item():>10.4f}  {grad_norm.item():>10.4f}  {dt_ms:>10.1f}")

    def close(self):
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
    trainer = ToyTrainer(device, mesh["dp"], mesh["tp"], OUTPUT_DIR, LoggingConfig(log_freq=5, enable_console=True))

    if rank == 0:
        print(f"Toy SPMD: {world_size} GPUs, 2DP×2TP, {NUM_STEPS} steps")

    torch.manual_seed(42 + rank)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
    valid_lengths = [4, 8, 12, SEQ_LEN]
    valid_len = valid_lengths[rank % len(valid_lengths)]
    loss_mask = torch.zeros(BATCH_SIZE, SEQ_LEN, device=device)
    loss_mask[:, :valid_len] = 1.0

    trainer.train(tokens, labels, loss_mask, NUM_STEPS)
    trainer.close()

    if rank == 0:
        from torchtitan.experiments.observability.visualize import to_chrome_trace
        to_chrome_trace(os.path.join(OUTPUT_DIR, "system_logs"), os.path.join(OUTPUT_DIR, "trace.json"))
        print(f"\nDone. Output: {OUTPUT_DIR}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
