# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer-state CPU offload: bitwise parity with the resident optimizer under FSDP2.

Every assertion on values is ``torch.equal``: the offloaded step runs the same fused kernel on
the same tensors, only from a different storage location.
"""

import os
import shutil

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizerStateOffloadConfig,
    ParamGroupConfig,
)
from torchtitan.components.optimizer.offload import OptimizerStateOffloader
from torchtitan.distributed import utils as dist_utils


pytestmark = pytest.mark.multi_gpu

_DIM = 256
_MOMENT_KEYS = ("exp_avg", "exp_avg_sq")


class _Model(nn.Module):
    """Four small blocks, one wide block (an oversized chunk at 1 MiB), one unused block (no grad)."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(_DIM, _DIM) for _ in range(4)])
        self.wide = nn.Linear(_DIM, 8 * _DIM, bias=False)
        self.unused = nn.Linear(_DIM, _DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = torch.tanh(block(x))
        return self.wide(x)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class TestOptimizerStateOffload(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _build(
        self,
        *,
        offload: OptimizerStateOffloadConfig | None,
        implementation: str = "fused",
        seed: int = 0,
    ) -> tuple[_Model, OptimizersContainer]:
        torch.manual_seed(seed)
        model = _Model().cuda()
        mesh = self.build_device_mesh()
        for module in (*model.blocks, model.wide, model.unused):
            fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)
        config = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=r"\.bias$",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-2, "weight_decay": 0.0},
                ),
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name="AdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1},
                ),
            ],
            implementation=implementation,
            optimizer_state_offload=offload,
        )
        return model, config.build(model_parts=[model])

    def _train(
        self, model: _Model, optimizers: OptimizersContainer, *, steps: int, seed: int
    ) -> None:
        """`steps` optimizer steps, each accumulating two microbatches and clipping at 0.5."""
        params = list(model.parameters())
        for step in range(steps):
            for microbatch in range(2):
                torch.manual_seed(seed + 100 * step + microbatch)
                x = torch.randn(8, _DIM, device="cuda")
                model(x).square().mean().backward()
            dist_utils.clip_grad_norm_(params, max_norm=0.5, foreach=True)
            optimizers.step()
            optimizers.zero_grad()

    def _assert_same(
        self, resident: OptimizersContainer, offloaded: OptimizersContainer
    ) -> None:
        for res_opt, off_opt in zip(resident.optimizers, offloaded.optimizers):
            for res_group, off_group in zip(res_opt.param_groups, off_opt.param_groups):
                for res_param, off_param in zip(
                    res_group["params"], off_group["params"]
                ):
                    assert torch.equal(_local(res_param), _local(off_param))
                    res_state, off_state = (
                        res_opt.state[res_param],
                        off_opt.state[off_param],
                    )
                    if not res_state:
                        # Never received a grad: torch AdamW has no state yet; ours is eager and untouched.
                        assert off_state["step"].item() == 0
                        assert all(
                            not _local(off_state[key]).any() for key in _MOMENT_KEYS
                        )
                        continue
                    for key in (*_MOMENT_KEYS, "step"):
                        assert torch.equal(
                            _local(res_state[key]).cpu(), _local(off_state[key]).cpu()
                        ), key

    def _assert_cpu_canonical(self, offloaded: OptimizersContainer) -> None:
        for optimizer in offloaded.optimizers:
            assert isinstance(optimizer, OptimizerStateOffloader)
            for param, state in optimizer.state.items():
                for key in _MOMENT_KEYS:
                    assert isinstance(state[key], DTensor)
                    assert state[key].device.type == "cpu"
                    assert _local(state[key]).is_pinned()
                    assert state[key].placements == param.placements
                assert state["step"].device.type == "cuda"

    @with_comms
    def test_parity_with_resident_optimizer_fp32_moments(self) -> None:
        self._check_parity_with_resident_optimizer("fused")

    @with_comms
    def test_parity_with_resident_optimizer_bf16_moments(self) -> None:
        self._check_parity_with_resident_optimizer("fused_opt_states_bf16")

    def _check_parity_with_resident_optimizer(self, implementation: str) -> None:
        resident_model, resident = self._build(
            offload=None, implementation=implementation
        )
        offloaded_model, offloaded = self._build(
            offload=OptimizerStateOffloadConfig(chunk_size_mb=1),
            implementation=implementation,
        )
        # 1 MiB chunks: the four 256x256 blocks pack several per chunk, the wide block is oversized.
        assert len(offloaded.optimizers[0]._chunks) >= 3
        self._train(resident_model, resident, steps=3, seed=1)
        self._train(offloaded_model, offloaded, steps=3, seed=1)
        self._assert_same(resident, offloaded)
        self._assert_cpu_canonical(offloaded)
        expected_dtype = (
            torch.bfloat16
            if implementation == "fused_opt_states_bf16"
            else torch.float32
        )
        for state in offloaded.optimizers[0].state.values():
            assert state["exp_avg"].dtype == expected_dtype
            assert _local(state["exp_avg"]).dtype == expected_dtype

    @with_comms
    def test_dense_gradients_use_packed_slab_copy_with_resident_parity(self) -> None:
        resident_model, resident = self._build(offload=None)
        offloaded_model, offloaded = self._build(
            offload=OptimizerStateOffloadConfig(chunk_size_mb=1)
        )

        # Give every parameter, including the normally-unused block, a gradient so the
        # offloader can move each chunk as one contiguous slice of its pinned slab.
        for resident_param, offloaded_param in zip(
            resident_model.parameters(), offloaded_model.parameters()
        ):
            resident_param.grad = torch.ones_like(resident_param)
            offloaded_param.grad = torch.ones_like(offloaded_param)
        resident.step()
        offloaded.step()

        self._assert_same(resident, offloaded)
        self._assert_cpu_canonical(offloaded)

    @with_comms
    def test_staging_memory_is_bounded_by_two_chunks(self) -> None:
        model, offloaded = self._build(
            offload=OptimizerStateOffloadConfig(chunk_size_mb=1)
        )
        offloader = offloaded.optimizers[0]
        self._train(model, offloaded, steps=1, seed=3)
        params = list(model.parameters())
        model(torch.randn(8, _DIM, device="cuda")).square().mean().backward()
        dist_utils.clip_grad_norm_(params, max_norm=0.5, foreach=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        offloaded.step()
        peak = torch.cuda.max_memory_allocated()
        after = torch.cuda.memory_allocated()
        max_chunk_numel = max(
            sum(_local(p).numel() for p in chunk) for chunk in offloader._chunks
        )
        two_slots = 2 * len(_MOMENT_KEYS) * max_chunk_numel * 4
        assert peak - before <= two_slots + (1 << 20), (peak - before, two_slots)
        assert after - before <= 1 << 20
        offloaded.zero_grad()

    @with_comms
    def test_dcp_round_trip_loads_in_place_and_training_continues_identically(
        self,
    ) -> None:
        checkpoint_dir = "/tmp/torchtitan_test_optimizer_state_offload"
        pre_step_checkpoint_dir = checkpoint_dir + "_before_first_step"
        if dist.get_rank() == 0:
            for path in (checkpoint_dir, pre_step_checkpoint_dir):
                shutil.rmtree(path, ignore_errors=True)
                os.makedirs(path)
        dist.barrier()

        # Saving before the first step works: state is allocated eagerly.
        model_a, opt_a = self._build(
            offload=OptimizerStateOffloadConfig(chunk_size_mb=1)
        )
        assert all(
            key.startswith(("state.", "param_groups.")) for key in opt_a.state_dict()
        )
        dcp.save(opt_a.state_dict(), checkpoint_id=pre_step_checkpoint_dir)
        self._train(model_a, opt_a, steps=2, seed=4)
        dcp.save(opt_a.state_dict(), checkpoint_id=checkpoint_dir)

        model_b, opt_b = self._build(
            offload=OptimizerStateOffloadConfig(chunk_size_mb=1)
        )
        self._train(model_b, opt_b, steps=1, seed=99)  # diverge, then load over it
        slab_ptrs = {
            id(p): _local(state["exp_avg"]).data_ptr()
            for p, state in opt_b.optimizers[0].state.items()
        }
        state_dict_b = opt_b.state_dict()
        dcp.load(state_dict_b, checkpoint_id=checkpoint_dir)
        opt_b.load_state_dict(state_dict_b)

        for p, state in opt_b.optimizers[0].state.items():
            assert _local(state["exp_avg"]).data_ptr() == slab_ptrs[id(p)]
        self._assert_cpu_canonical(opt_b)
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            sa, sb = opt_a.optimizers[0].state[pa], opt_b.optimizers[0].state[pb]
            for key in (*_MOMENT_KEYS, "step"):
                assert torch.equal(_local(sa[key]).cpu(), _local(sb[key]).cpu()), key

        # Continued training from the loaded state matches, given equal params.
        with torch.no_grad():
            for pa, pb in zip(model_a.parameters(), model_b.parameters()):
                _local(pb).copy_(_local(pa))
        self._train(model_a, opt_a, steps=1, seed=5)
        self._train(model_b, opt_b, steps=1, seed=5)
        self._assert_same(opt_a, opt_b)
        dist.barrier()
        if dist.get_rank() == 0:
            for path in (checkpoint_dir, pre_step_checkpoint_dir):
                shutil.rmtree(path, ignore_errors=True)

    @with_comms
    def test_rejects_unsupported_configurations(self) -> None:
        with pytest.raises(ValueError, match="fused"):
            self._build(offload=OptimizerStateOffloadConfig(), implementation="foreach")
        with pytest.raises(ValueError, match="positive"):
            OptimizerStateOffloadConfig(chunk_size_mb=0)
        cpu_param = nn.Parameter(torch.zeros(4))
        with pytest.raises(ValueError, match="GPU-resident"):
            OptimizerStateOffloader(
                torch.optim.AdamW([cpu_param], lr=1e-3, fused=True),
                chunk_size_mb=1,
                state_dtype=torch.float32,
            )
        initialized_param = nn.Parameter(torch.ones(4, device="cuda"))
        initialized_optimizer = torch.optim.AdamW(
            [initialized_param], lr=1e-3, fused=True
        )
        initialized_param.grad = torch.ones_like(initialized_param)
        initialized_optimizer.step()
        with pytest.raises(ValueError, match="fresh optimizer"):
            OptimizerStateOffloader(
                initialized_optimizer, chunk_size_mb=1, state_dtype=torch.float32
            )
        gpu_param = nn.Parameter(torch.zeros(4, device="cuda"))
        with pytest.raises(ValueError, match="amsgrad"):
            OptimizerStateOffloader(
                torch.optim.AdamW([gpu_param], lr=1e-3, fused=True, amsgrad=True),
                chunk_size_mb=1,
                state_dtype=torch.float32,
            )
