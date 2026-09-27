# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-place reduce-scatter (``fsdp_reduce_scatter.install()``) vs FSDP2's copy-in path, CPU/gloo.

Each case trains the same model twice for two steps and compares every rank's sharded gradients:
once with FSDP's default reduce-scatter (in place when the gradients already have the reduce
dtype) and once with a ``DefaultReduceScatter`` subclass, which keeps the copy-in path.
"""

import os
import traceback
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultReduceScatter
from torch.distributed.tensor import Shard

from torchtitan.experiments.fp32_weight_grads import fsdp_reduce_scatter

NUM_FSDP_GROUPS = 3
NUM_STEPS = 2
NUM_MICROBATCHES = 2


class CopyInReduceScatter(DefaultReduceScatter):
    """Same collective as the default; being a subclass keeps FSDP on the copy-in path."""


@dataclass
class Case:
    name: str
    mesh_shape: tuple[int, ...]
    fp32_grads: bool = True
    divide_factor: float | None = None
    no_sync_first_microbatch: bool = False


CASES = [
    Case("fp32 grads, 2 ranks", (2,)),
    Case("fp32 grads, 4 ranks, uneven + Shard(1) params", (4,)),
    Case("fp32 grads, HSDP 2x2", (2, 2)),
    Case(
        "fp32 grads, 2 ranks, no-sync first microbatch",
        (2,),
        no_sync_first_microbatch=True,
    ),
    Case("fp32 grads, 2 ranks, divide factor 2", (2,), divide_factor=2.0),
    Case("fp32 grads, 1 rank, divide factor 2", (1,), divide_factor=2.0),
    Case("bf16 grads (copy-in fallback), 2 ranks", (2,), fp32_grads=False),
]


def _build_model(mesh, case: Case, copy_in: bool) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(16, 32, bias=False),
        nn.LayerNorm(32),
        nn.Linear(32, 30, bias=False),
        # dim-0 of 30 is uneven over 4 ranks: FSDP pads it
        nn.LayerNorm(30),
        nn.Linear(30, 32, bias=False),
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    # Shard(1): FSDP reorders the gradient into dim-0 chunks before reduce-scatter
    fully_shard(
        model[2],
        mesh=mesh,
        mp_policy=mp_policy,
        shard_placement_fn=lambda param: Shard(1),
    )
    fully_shard(model[3], mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    fsdp_modules = (model[2], model[3], model)
    for module in fsdp_modules:
        if copy_in:
            module.set_custom_reduce_scatter(CopyInReduceScatter())
        if case.divide_factor is not None:
            module.set_gradient_divide_factor(case.divide_factor)

    if case.fp32_grads:

        def accumulate_fp32(module, args):
            for param in module.parameters():
                param.grad_dtype = torch.float32

        for module in fsdp_modules:
            module.register_forward_pre_hook(accumulate_fp32)
    return model


def _train(model: nn.Module, case: Case, rank: int) -> dict[str, torch.Tensor]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(NUM_STEPS):
        for microbatch in range(NUM_MICROBATCHES):
            if case.no_sync_first_microbatch:
                model.set_requires_gradient_sync(microbatch == NUM_MICROBATCHES - 1)
            torch.manual_seed(100 * step + 10 * microbatch + rank)
            model(torch.randn(8, 16)).float().pow(2).mean().backward()
        grads = {
            name: param.grad.to_local().clone()
            for name, param in model.named_parameters()
        }
        optimizer.step()
        optimizer.zero_grad()
    return grads


def _worker(rank: int, world_size: int, case: Case, port: int, errors) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        fsdp_reduce_scatter.install()
        mesh_dim_names = (
            ("replicate", "shard") if len(case.mesh_shape) == 2 else ("shard",)
        )
        mesh = init_device_mesh("cpu", case.mesh_shape, mesh_dim_names=mesh_dim_names)

        in_place_calls = []
        in_place = fsdp_reduce_scatter.foreach_reduce_scatter_in_place

        def counting(unsharded_grads, *args, **kwargs):
            in_place_calls.append(len(unsharded_grads))
            return in_place(unsharded_grads, *args, **kwargs)

        fsdp_reduce_scatter.foreach_reduce_scatter_in_place = counting
        grads = {}
        for copy_in in (True, False):
            in_place_calls.clear()
            grads[copy_in] = _train(_build_model(mesh, case, copy_in), case, rank)
            if copy_in:
                assert (
                    not in_place_calls
                ), f"copy-in run went in place: {in_place_calls}"

        for name, copy_in_grad in grads[True].items():
            in_place_grad = grads[False][name]
            assert in_place_grad.dtype == copy_in_grad.dtype == torch.float32
            assert torch.equal(
                in_place_grad, copy_in_grad
            ), f"{name}: max diff {(in_place_grad - copy_in_grad).abs().max().item():.3e}"
        synced_microbatches = 1 if case.no_sync_first_microbatch else NUM_MICROBATCHES
        expected = (
            NUM_FSDP_GROUPS * synced_microbatches * NUM_STEPS if case.fp32_grads else 0
        )
        assert (
            len(in_place_calls) == expected
        ), f"{len(in_place_calls)} in-place calls, expected {expected}"
    except Exception:
        errors.put(f"rank {rank}:\n{traceback.format_exc()}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_in_place_reduce_scatter_matches_copy_in_bitwise(case):
    world_size = 1
    for size in case.mesh_shape:
        world_size *= size
    errors = mp.get_context("spawn").SimpleQueue()
    port = 29600 + CASES.index(case)
    mp.spawn(_worker, args=(world_size, case, port, errors), nprocs=world_size)
    failures = []
    while not errors.empty():
        failures.append(errors.get())
    assert not failures, "\n".join(failures)


def test_install_rejects_a_different_foreach_reduce(monkeypatch):
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    def foreach_reduce(*args, **kwargs):
        raise AssertionError("not PyTorch's foreach_reduce")

    monkeypatch.setattr(_fsdp_collectives, "foreach_reduce", foreach_reduce)
    with pytest.raises(RuntimeError, match="only supports the PyTorch"):
        fsdp_reduce_scatter.install()
