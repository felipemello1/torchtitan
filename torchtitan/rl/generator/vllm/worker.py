# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-owned vLLM worker and model runner customizations."""

from collections.abc import Set
from contextlib import nullcontext

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.utils.math_utils import round_up
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.worker import gpu_model_runner as vllm_gpu_model_runner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker


class TorchTitanCudagraphDispatcher(CudagraphDispatcher):
    """Give decode batches their own FULL-graph keys, apart from prefill and mixed batches.

    vLLM at af4db2a erases the uniform-decode discriminator in FULL mode:
    four one-token decodes collide with one four-token prefill, and the shared
    graph is captured as the mixed batch, with ``max_query_len = num_tokens``.
    Decode needs its own key for two reasons:
      - Attention Gym linear attention (GDN/KDA) takes a different numerical path.
      - Softmax attention reads ``max_query_len`` on the host, and the graph
        freezes it at capture. FA4 sizes its grid from it: with ``num_tokens``
        instead of 1, decode takes 1.3-2.0x longer from 32 sequences at head
        dim 64 and from 64 at head dim 128 (GB300, 16-token pages).
    Delegate decode keys to the native FULL_DECODE_ONLY dispatcher until upstream
    supports FULL decode specialization; native capture, replay and graph
    ownership stay intact. Decode keys stop at ``max_num_seqs``; a larger decode
    batch (e.g. padded by DP sync to another rank's mixed batch) replays the
    mixed graph, as in stock FULL mode.
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
        if (
            self.keys_initialized
            and self.cudagraph_mode == CUDAGraphMode.FULL
            and uniform_decode
        ):
            # Ask without valid_modes: when only FULL is allowed (DP re-dispatch) and no
            # decode key fits, vLLM's dispatcher asserts instead of returning NONE.
            mode, descriptor = self.decode_dispatcher.dispatch(
                num_tokens,
                uniform_decode=True,
                has_lora=has_lora,
                num_active_loras=num_active_loras,
                invalid_modes=invalid_modes,
            )
            if mode == CUDAGraphMode.FULL and (
                valid_modes is None or mode in valid_modes
            ):
                return mode, descriptor
        return super().dispatch(
            num_tokens,
            uniform_decode,
            has_lora,
            num_active_loras,
            valid_modes,
            invalid_modes,
        )


class TorchTitanGPUModelRunner(GPUModelRunner):
    """V1 runner with sequence-parallel padding and FULL-graph decode specialization."""

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights)
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

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        enable_dense_sp = self.compilation_config.pass_config.enable_sp and tp_size > 1
        enable_expert_sp = (
            self.vllm_config.parallel_config.enable_expert_parallel and tp_size > 1
        )
        if enable_dense_sp or enable_expert_sp:
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens


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
