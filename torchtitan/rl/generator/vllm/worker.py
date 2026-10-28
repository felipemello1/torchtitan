# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-owned vLLM worker and model runner customizations."""

import functools
from collections import defaultdict
from collections.abc import Set
from contextlib import nullcontext

import numpy as np
from vllm.config import CUDAGraphMode, get_layers_from_vllm_config
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.utils.math_utils import round_up
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker import gpu_model_runner as vllm_gpu_model_runner
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from torchtitan.rl.generator.vllm.linear_attention import (
    TorchTitanLinearAttentionBackend,
)


class TorchTitanCudagraphDispatcher(CudagraphDispatcher):
    """Keep native FULL descriptors distinct for packed linear attention and fused decode.

    Currently installed only for TorchTitan's Attention Gym linear attention
    backends (GDN/KDA).
    vLLM at c6fa1f0 erases the uniform-decode discriminator in FULL mode:
    four one-token decodes collide with one four-token prefill. Attention Gym's
    different numerical paths need distinct keys. Delegate decode keys to the
    native FULL_DECODE_ONLY dispatcher until upstream supports FULL decode
    specialization; native capture, replay and graph ownership stay intact.
    The single-token key also covers fresh requests: initialization is GPU mask
    data, not another dispatch key or a reason to switch to packed arithmetic.

    This costs one additional graph capture per decode-specialized batch size
    and additional memory for the expanded graph pool.
    """

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
    ) -> None:
        super().initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)
        if cudagraph_mode == CUDAGraphMode.FULL:
            self.decode_dispatcher = CudagraphDispatcher(self.vllm_config)
            self.decode_dispatcher.initialize_cudagraph_keys(
                CUDAGraphMode.FULL_DECODE_ONLY, uniform_decode_query_len
            )
            for mode, descriptors in self.decode_dispatcher.get_capture_descs():
                for descriptor in descriptors:
                    self.add_cudagraph_key(mode, descriptor)

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        valid_modes: Set[CUDAGraphMode] | None = None,
        invalid_modes: Set[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        dispatch = super().dispatch
        if (
            self.keys_initialized
            and self.cudagraph_mode == CUDAGraphMode.FULL
            and uniform_decode
        ):
            dispatch = self.decode_dispatcher.dispatch
        return dispatch(
            num_tokens,
            uniform_decode,
            has_lora,
            num_active_loras,
            valid_modes,
            invalid_modes,
        )


class TorchTitanGPUModelRunner(GPUModelRunner):
    """V1 runner with sequence-parallel padding, linear attention (GDN/KDA) decode specialization, and
    GRPO samples of one prompt in adjacent batch rows."""

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)
        if any(
            layer.get_attn_backend() is TorchTitanLinearAttentionBackend
            for layer in get_layers_from_vllm_config(
                self.vllm_config, MambaBase
            ).values()
        ):
            assert not self.cudagraph_dispatcher.keys_initialized
            self.cudagraph_dispatcher = TorchTitanCudagraphDispatcher(self.vllm_config)

    def _determine_batch_execution_and_padding(self, num_tokens: int, *args, **kwargs):
        """Tell the model how many of this step's tokens are real, before padding.

        vLLM calls this once before every forward: from ``execute_model`` for a
        real step, and from ``_dummy_run`` for profiling, warm-up, CUDA-graph
        capture and an idle DP rank's dummy batch (whose rows all count as
        real). ``num_tokens`` is the unpadded count; vLLM applies every padding
        source inside this call (TP rounding, CUDA-graph capture size, DP
        equalization). The model turns the count into a padding mask, which keeps
        those rows off Dist-MoE's network; graph replay reads the value published
        for the real step.
        """
        self.get_model().set_num_valid_tokens(num_tokens)
        return super()._determine_batch_execution_and_padding(
            num_tokens, *args, **kwargs
        )

    def _may_reorder_batch(self, scheduler_output: SchedulerOutput) -> None:
        """Reorder rows as vLLM does, then put requests that share a cached prompt in adjacent rows.

        The samples of a GRPO group share the prompt's KV blocks (prefix cache), but vLLM refills the
        rows of finished requests with other requests, which scatters a group over the batch (median
        3.5 runs per group at 8 samples, 5.8 at 16). Decode attention reads rows in order, so adjacent
        samples find the prompt in L2. Qwen3-8B, 64 prompts x 8 samples, 4k-token prompts: decode
        step 17.3 -> 15.2 ms; outputs bitwise equal.
        """
        super()._may_reorder_batch(scheduler_output)
        batch = self.input_batch
        num_rows = batch.num_reqs
        if self.reorder_batch_threshold is not None:
            # vLLM moved the decode rows to the front (same test as its reorder); group only
            # those, so its decode / prefill regions stay intact.
            scheduled = np.array(
                [scheduler_output.num_scheduled_tokens[r] for r in batch.req_ids]
            )
            is_decode = (scheduled <= self.reorder_batch_threshold) & (
                batch.num_computed_tokens_cpu[:num_rows]
                >= batch.num_prompt_tokens[:num_rows]
            )
            num_rows = num_rows if is_decode.all() else int(is_decode.argmin())
        if self._full_attention_kv_cache_group is not None:
            group_rows_by_prompt(
                batch,
                block_table=batch.block_table[self._full_attention_kv_cache_group],
                num_rows=num_rows,
            )

    @functools.cached_property
    def _full_attention_kv_cache_group(self) -> int | None:
        """Index of the first full-attention KV cache group, whose blocks identify the prompt (None: no
        such group, rows keep vLLM's order)."""
        return next(
            (
                i
                for i, group in enumerate(self.kv_cache_config.kv_cache_groups)
                if isinstance(group.kv_cache_spec, FullAttentionSpec)
            ),
            None,
        )

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        enable_dense_sp = self.compilation_config.pass_config.enable_sp and tp_size > 1
        enable_expert_sp = (
            self.vllm_config.parallel_config.enable_expert_parallel and tp_size > 1
        )
        if enable_dense_sp or enable_expert_sp:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens


def group_rows_by_prompt(
    batch: InputBatch, *, block_table: BlockTable, num_rows: int
) -> None:
    """Swap rows ``[0, num_rows)`` of vLLM's persistent batch so requests that share a cached prompt
    (the samples of one GRPO group) sit in adjacent rows.

    A row's key is its prompt's last cached block: the last whole block before the prompt's final
    token, which every sample hits in the prefix cache (vLLM recomputes the final token). Prompts that
    share only a system prompt get different keys; a prompt shorter than two blocks shares nothing.
    Each group gets one segment, ordered by the group's median row. Rows already inside their segment
    stay; the others swap into the free slots of their segment, so a step that refilled a few rows
    moves a few rows. Swaps go through vLLM's ``swap_states``, as in vLLM's own reorder.

    Example (letters = key of each row):
        before   A A B A C B C B
        after    A A A B B B C C     (2 swaps: rows 2 <-> 3, 4 <-> 7)
    """
    if num_rows < 2:
        return
    rows = np.arange(num_rows)
    last_cached_block = (
        batch.num_prompt_tokens[:num_rows] - 1
    ) // block_table.block_size - 1
    key = np.where(
        last_cached_block >= 0,
        block_table.block_table.np[rows, np.maximum(last_cached_block, 0)],
        -1 - rows,
    )
    run_starts = np.flatnonzero(key[1:] != key[:-1]) + 1
    run_keys = key[np.concatenate(([0], run_starts))].tolist()
    if len(set(run_keys)) == len(run_keys):
        return  # every group is contiguous

    # Target layout: one segment per group, in the order of the groups' median rows.
    _, group, size = np.unique(key, return_inverse=True, return_counts=True)
    rows_by_group = np.lexsort((rows, group))
    median_row = rows_by_group[np.cumsum(size) - size + size // 2]
    segment_order = np.argsort(median_row, kind="stable")
    slot_group = np.repeat(segment_order, size[segment_order])

    # Each row outside its segment takes a slot of its segment that holds another group's row.
    # A pair of rows sitting in each other's segments swaps directly (one swap places both).
    misplaced = np.flatnonzero(slot_group != group).tolist()
    # (group of the segment, group of the row in the slot) -> slots
    free_slots = defaultdict(list)
    for row in misplaced:
        free_slots[slot_group[row], group[row]].append(row)
    dest = {}
    for row in misplaced:
        pair = free_slots[group[row], slot_group[row]]
        if row not in dest and pair:
            other = pair.pop()
            free_slots[slot_group[row], group[row]].remove(row)
            dest[row], dest[other] = other, row
    free_by_group = defaultdict(list)
    for (segment_group, _), slots in free_slots.items():
        free_by_group[segment_group].extend(slots)
    for row in misplaced:
        if row not in dest:
            dest[row] = free_by_group[group[row]].pop()

    # Apply the permutation one cycle at a time, as vLLM's reorder_batch_to_split_decodes_and_prefills.
    for src in list(dest):
        dst = dest[src]
        while src != dst:
            batch.swap_states(src, dst)
            next_dst = dest[dst]
            dest[dst] = dst
            dst = next_dst


class TorchTitanGPUWorker(GPUWorker):
    """V1 worker that constructs :class:`TorchTitanGPUModelRunner`."""

    def _maybe_get_memory_pool_context(self, tag: str):
        # vLLM uses CuMem for model weights and the KV cache.
        # TorchTitan only transfers model weights over RDMA, so leave all other
        # allocations on PyTorch's default allocator.
        # The parent method checks enable_cumem_allocator and returns a no-op
        # context when it is disabled.
        if tag == "weights":
            return super()._maybe_get_memory_pool_context(tag)
        return nullcontext()

    def init_device(self):
        if self.use_v2_model_runner:
            raise ValueError(
                "TorchTitan's vLLM integration requires the V1 model runner"
            )

        # GPUWorker imports its runner class inside init_device and provides no
        # runner factory. Scope the class substitution to that construction.
        original_runner_cls = vllm_gpu_model_runner.GPUModelRunner
        vllm_gpu_model_runner.GPUModelRunner = TorchTitanGPUModelRunner
        try:
            super().init_device()
        finally:
            vllm_gpu_model_runner.GPUModelRunner = original_runner_cls
