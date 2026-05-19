# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import contextlib
import logging
import math
import os
from dataclasses import dataclass, field

import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import (
    CompileConfig,
    Configurable,
    DebugConfig,
    ParallelismConfig,
)
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.actors.utils import (
    cuda_memory_stats,
    reset_cuda_peak_memory_stats,
)
from torchtitan.experiments.rl.models.vllm_engine import (
    build_torchtitan_vllm_engine_args,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.types import Completion
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import init_logger
from vllm import LLMEngine, SamplingParams
from vllm.config import CompilationConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind

logger = logging.getLogger(__name__)


def _prepare_generation_request_metrics(
    output: RequestOutput, *, prefix: str
) -> list[m.Metric]:
    """Prepare vLLM metrics from a RequestOutput.

    For `[num_prompts]` submitted prompts, vLLM returns `[num_prompts]`
    per parent `RequestOutput`s (one per `add_request` call), each carrying
    a single `RequestStateStats` on `.metrics`.
    """

    # TODO: Per-request fields here come from RequestOutput.metrics
    # (RequestStateStats). Engine-aggregate stats, such as KV-cache usage,
    # prefix-cache hit rate, preemptions, and batch occupancy, live in
    # SchedulerStats / IterationStats and require registering a
    # vllm.v1.metrics.loggers.StatLoggerBase via
    # LLMEngine.from_engine_args(..., stat_loggers=[...]).

    metric_values: dict[str, float] = {}
    if output.num_cached_tokens is not None:
        metric_values[f"{prefix}/num_cached_tokens"] = output.num_cached_tokens

    stats = output.metrics
    if stats is not None:
        metric_values[f"{prefix}/queue_time_ms"] = (
            stats.scheduled_ts - stats.queued_ts
        ) * 1000

        if stats.num_generation_tokens > 0:
            metric_values[f"{prefix}/time_to_first_token_ms"] = (
                stats.first_token_latency * 1000
            )
            metric_values[f"{prefix}/prefill_time_ms"] = (
                stats.first_token_ts - stats.scheduled_ts
            ) * 1000

        if stats.num_generation_tokens > 1:
            first_to_last_token_ms = (stats.last_token_ts - stats.first_token_ts) * 1000
            metric_values[f"{prefix}/decode_time_ms"] = first_to_last_token_ms
            metric_values[
                f"{prefix}/inter_token_latency_ms"
            ] = first_to_last_token_ms / (stats.num_generation_tokens - 1)

    # Emit each value with both Mean and Max aggregators.
    return [
        metric
        for key, value in metric_values.items()
        for metric in (m.Metric(key, m.Mean(value)), m.Metric(key, m.Max(value)))
    ]


@dataclass(kw_only=True, slots=True)
class VLLMCudagraphConfig:
    """CUDA graph capture settings for the vLLM inference engine.

    torch.compile is configured separately via ``CompileConfig`` at the
    ``RLTrainer`` level, shared by both trainer and generator.  Only CUDA
    graph capture, which is vLLM-specific, is controlled here.

    When enabled, vLLM captures the forward pass as a single CUDA graph
    ("full" mode).  "piecewise" modes are intentionally excluded: they
    require vLLM's whole-model torch.compile to split the graph around
    non-capturable ops, which conflicts with per-layer compile.
    """

    enable: bool = True
    """Whether to enable CUDA graph capture (vLLM "full" mode)."""

    # TODO: Validate CUDA graph capture with MoE / Expert Parallelism.
    # MoE routing produces dynamic shapes that may conflict with full
    # CUDA graph capture despite being torch.compile-compatible
    # post https://github.com/pytorch/torchtitan/pull/3142

    # TODO: Explore applying CUDA graph capture on the torchtitan trainer
    # side as well (not just the vLLM generator).
    # https://github.com/pytorch/torchtitan/issues/3175

    def get_vllm_compilation_config(
        self, *, max_num_seqs: int
    ) -> CompilationConfig | None:
        """Build a vLLM ``CompilationConfig``, or return ``None`` when
        CUDA graphs are disabled.

        ``max_num_seqs`` determines CUDA graph capture sizes: powers of
        2 from 1 up to ``max_num_seqs``, plus ``max_num_seqs`` itself
        if it isn't already a power of 2.
        """
        if not self.enable:
            return None
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        sizes = [1 << i for i in range(int(math.log2(max_num_seqs)) + 1)]
        if max_num_seqs not in sizes:
            sizes.append(max_num_seqs)
        return CompilationConfig(
            cudagraph_mode="full",
            mode=0,
            cudagraph_capture_sizes=sorted(sizes),
        )


class VLLMGenerator(Actor, Configurable):
    """
    Generates rollouts using vLLM engine.

    Maintains a vLLM engine synchronized with the Trainer via weight
    sync. ``generate()`` produces a flat list of Completions; reward
    and advantage computation live in the controller.

    Args:
        config: Generator-specific configuration.
        model_spec: TorchTitan model specification.
        model_path: Path to the HF model checkpoint.
        compile_config: Per-layer torch.compile config shared with the
            trainer so both sides compile identically.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration."""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        """Default sampling parameters for generation."""

        model_dtype: str = "bfloat16"
        """Model weight dtype passed to vLLM."""

        gpu_memory_limit: float = 0.9
        """vLLM ``gpu_memory_utilization`` fraction (0.0 to 1.0)."""

        cudagraph: VLLMCudagraphConfig = field(default_factory=VLLMCudagraphConfig)
        """CUDA graph capture settings for the vLLM engine."""

        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        """Controls whether the vLLM wrapper loads initial HF weights.
        In the RL loop this should stay disabled (default ``enable=False``)
        because weights arrive from TorchStore. For standalone inference,
        set ``enable=True`` and ``initial_load_in_hf=True``."""

        debug: DebugConfig = field(default_factory=DebugConfig)
        """Debug and determinism settings."""

        def __post_init__(self):
            # VLLMGenerator only supports TP. vLLM handles its own parallelism;
            # we only apply TP via the core parallelize function.
            p = self.parallelism
            if p.data_parallel_replicate_degree != 1:
                raise ValueError(
                    f"Generator does not support data parallel replication, "
                    f"got dp_replicate={p.data_parallel_replicate_degree}"
                )
            if p.pipeline_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support pipeline parallelism, "
                    f"got pp={p.pipeline_parallel_degree}"
                )
            if p.context_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support context parallelism, "
                    f"got cp={p.context_parallel_degree}"
                )
            if p.expert_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support expert parallelism, "
                    f"got ep={p.expert_parallel_degree}"
                )
            if p.enable_sequence_parallel:
                raise ValueError(
                    "Generator does not support sequence parallelism: "
                    "spmd_types erasure mode requires sequence length to be "
                    "evenly divisible by TP, which doesn't hold for inference "
                    "(uneven batches). Set enable_sequence_parallel=False."
                )
            if not p.disable_loss_parallel:
                raise ValueError(
                    "Generator requires disable_loss_parallel=True, "
                    f"got disable_loss_parallel={p.disable_loss_parallel}"
                )

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        model_path: str,
        compile_config: CompileConfig,
        max_num_seqs: int,
        output_dir: str,
    ):
        init_logger()
        sl.init_structured_logger(
            source="rl_generator",
            output_dir=output_dir,
            rank=current_rank().rank,
            enable=config.debug.enable_structured_logging,
        )
        sl.log_trace_instant("structured_logger_started")

        self.config = config
        self.model_spec = model_spec

        # max_num_seqs controls vLLM's maximum batch dimension, KV-cache
        # allocation, and CUDA graph capture sizes. The controller computes it
        # from its rollout and validation concurrency.
        self._max_num_seqs = max_num_seqs

        set_batch_invariance(config.debug.batch_invariant)

        self._set_determinism(config.debug)

        self.model_path = model_path

        engine_args = build_torchtitan_vllm_engine_args(
            config=config,
            model_spec=model_spec,
            model_path=model_path,
            compile_config=compile_config,
            checkpoint_config=config.checkpoint,
            max_num_seqs=self._max_num_seqs,
        )

        with sl.log_trace_span("vllm_init"):
            logger.info("Initializing LLMEngine from EngineArgs...")
            self._engine = LLMEngine.from_engine_args(engine_args)
            logger.info("vLLM rollout engine initialized")

        self._engine_lock = asyncio.Lock()
        self._pending_outputs: dict[str, asyncio.Future[RequestOutput]] = {}
        self._engine_driver_task: asyncio.Task[None] | None = None
        self._next_request_id = 0
        self.policy_version = 0

        logger.info("Generator initialized with vLLM engine")

    @staticmethod
    def _set_determinism(debug: DebugConfig) -> None:
        """Apply deterministic flags for the generator.

        The generator doesn't use torchtitan's ParallelDims, so we apply
        the deterministic flags directly instead of using set_determinism().
        """
        if debug.deterministic:
            torch.use_deterministic_algorithms(
                True, warn_only=debug.deterministic_warn_only
            )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        if debug.seed is not None:
            torch.manual_seed(debug.seed)

    def _get_model(self):
        """Access the model from the vLLM engine.
        Returns a VLLMModelWrapper instance.
        """
        return self._engine.model_executor.driver_worker.get_model()

    @endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        """Sync the structured-logger step counter from the controller."""
        sl.set_step(step, relative_step=relative_step)

    def _build_sampling_params(self, sampling_config: SamplingConfig) -> SamplingParams:
        return SamplingParams(
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            max_tokens=sampling_config.max_tokens,
            n=1,
            logprobs=1,
            output_kind=RequestOutputKind.FINAL_ONLY,
            stop_token_ids=list(sampling_config.stop_token_ids),
        )

    @staticmethod
    def _completion_from_sample(
        *,
        output: RequestOutput,
        sample_idx: int,
        policy_version: int,
    ) -> Completion:
        sample = output.outputs[sample_idx]
        if sample.logprobs is None:
            raise ValueError("vLLM did not return token logprobs")
        if len(sample.logprobs) != len(sample.token_ids):
            raise ValueError(
                "vLLM returned "
                f"{len(sample.logprobs)} logprob entries for "
                f"{len(sample.token_ids)} sampled tokens"
            )
        per_token_logprobs = []
        for token_id, logprob_dict in zip(
            sample.token_ids,
            sample.logprobs,
            strict=True,
        ):
            if token_id not in logprob_dict:
                raise ValueError(f"vLLM logprobs missing sampled token id {token_id}")
            per_token_logprobs.append(logprob_dict[token_id].logprob)
        return Completion(
            policy_version=policy_version,
            token_ids=sample.token_ids,
            token_logprobs=per_token_logprobs,
            finish_reason=sample.finish_reason,
        )

    def _ensure_engine_driver_locked(self) -> None:
        """Start the vLLM step driver while holding ``_engine_lock``."""
        task = self._engine_driver_task
        if task is None or task.done():
            reset_cuda_peak_memory_stats()
            self._engine_driver_task = asyncio.create_task(self._drive_engine())

    async def _drive_engine(self) -> None:
        """Step vLLM until all admitted requests finish.

        vLLM's sync ``LLMEngine`` needs one caller to drive ``step()``. Keeping
        that loop separate from request admission lets later ``generate`` calls
        add requests while earlier ones are still decoding, so vLLM can use its
        own continuous batching scheduler.
        """
        try:
            with sl.log_trace_span("engine_driver"):
                while True:
                    ready_outputs: list[
                        tuple[asyncio.Future[RequestOutput], RequestOutput]
                    ]
                    ready_outputs = []
                    pending_error: RuntimeError | None = None
                    pending_futures: list[asyncio.Future[RequestOutput]] = []

                    async with self._engine_lock:
                        if self._engine is None:
                            pending_error = RuntimeError(
                                "vLLM engine closed while generation requests "
                                "were active"
                            )
                            pending_futures = list(self._pending_outputs.values())
                            self._pending_outputs.clear()
                            self._engine_driver_task = None
                        elif self._engine.has_unfinished_requests():
                            with torch.no_grad():
                                step_outputs = self._engine.step()
                            for output in step_outputs:
                                future = self._pending_outputs.pop(
                                    str(output.request_id),
                                    None,
                                )
                                if future is None:
                                    pending_error = RuntimeError(
                                        "vLLM returned unknown request_id "
                                        f"{output.request_id!r}; expected one of "
                                        f"{sorted(self._pending_outputs)}"
                                    )
                                    pending_futures = [
                                        ready_future
                                        for ready_future, _ready_output in ready_outputs
                                    ]
                                    pending_futures.extend(
                                        self._pending_outputs.values()
                                    )
                                    self._pending_outputs.clear()
                                    self._engine_driver_task = None
                                    break
                                ready_outputs.append((future, output))
                        elif self._pending_outputs:
                            pending_error = RuntimeError(
                                "vLLM engine became idle with pending request_ids "
                                f"{sorted(self._pending_outputs)}"
                            )
                            pending_futures = list(self._pending_outputs.values())
                            self._pending_outputs.clear()
                            self._engine_driver_task = None
                        else:
                            self._engine_driver_task = None
                            return

                    if pending_error is not None:
                        for future in pending_futures:
                            if not future.done():
                                future.set_exception(pending_error)
                        return

                    for future, output in ready_outputs:
                        if not future.done():
                            future.set_result(output)

                    await asyncio.sleep(0)
        except Exception as exc:
            logger.exception("vLLM engine driver failed")
            async with self._engine_lock:
                pending_futures = list(self._pending_outputs.values())
                self._pending_outputs.clear()
                self._engine_driver_task = None
            for future in pending_futures:
                if not future.done():
                    future.set_exception(exc)

    def _admit_requests_locked(
        self,
        prompt_token_ids_batch: list[list[int]],
        *,
        request_ids: list[str] | None,
        sampling_config: SamplingConfig,
        loop: asyncio.AbstractEventLoop,
    ) -> tuple[list[str], list[asyncio.Future[RequestOutput]], int]:
        """Admit a generate batch while holding ``_engine_lock``.

        Request IDs must be unique among active vLLM requests. Futures are
        registered before ``add_request`` so the engine driver can resolve
        outputs immediately after admission. If vLLM partially accepts a batch
        and then raises, attempted external request IDs are aborted before the
        local pending map is cleared.
        """
        admitted_policy_version = self.policy_version
        if request_ids is None:
            start_id = self._next_request_id
            self._next_request_id += len(prompt_token_ids_batch)
            _request_ids = [
                str(start_id + idx) for idx in range(len(prompt_token_ids_batch))
            ]
        else:
            _request_ids = list(request_ids)
        logger.debug(
            f"{os.getpid()=} Generating start generate "
            f"(policy v{admitted_policy_version})..."
        )

        active_duplicates = [
            request_id
            for request_id in _request_ids
            if request_id in self._pending_outputs
        ]
        if active_duplicates:
            raise ValueError(
                "request_ids are already active in the vLLM engine: "
                f"{active_duplicates}"
            )

        sampling_params = self._build_sampling_params(sampling_config)
        # render_cmpl is vLLM's input-pipeline entry.
        # The tokenize step is a no-op for already-tokenized prompts. The
        # lower-level alternative is vllm.inputs.tokens_input; we use the
        # high-level API to stay resilient to vLLM internal changes.
        engine_inputs = self._engine.renderer.render_cmpl(
            [{"prompt_token_ids": ids} for ids in prompt_token_ids_batch]
        )
        request_futures: list[asyncio.Future[RequestOutput]] = []
        attempted_request_ids: list[str] = []
        try:
            for request_id, engine_input in zip(
                _request_ids,
                engine_inputs,
                strict=True,
            ):
                future = loop.create_future()
                self._pending_outputs[request_id] = future
                request_futures.append(future)
                attempted_request_ids.append(request_id)
                self._engine.add_request(
                    request_id=request_id,
                    prompt=engine_input,
                    params=sampling_params,
                )
        except Exception:
            if attempted_request_ids:
                self._engine.abort_request(attempted_request_ids, internal=False)
            for request_id in _request_ids:
                future = self._pending_outputs.pop(request_id, None)
                if future is not None and not future.done():
                    future.cancel()
            raise

        if request_futures:
            self._ensure_engine_driver_locked()

        return _request_ids, request_futures, admitted_policy_version

    @endpoint
    @sl.log_trace_span("generate")
    async def generate(
        self,
        prompt_token_ids_batch: list[list[int]],
        *,
        request_ids: list[str] | None = None,
        sampling_config: SamplingConfig | None = None,
        metrics_prefix: str = "generator",
    ) -> tuple[list[Completion], list[m.Metric]]:
        """Generate completions and generator metrics for tokenized prompts.

        Takes ``prompt_token_ids_batch`` as ``[num_prompts][prompt_tokens]``.
        Returns one completion per prompt plus generator metrics. GRPO sibling
        sampling is owned by ``rollout_group_size``.

        Args:
            prompt_token_ids_batch: Tokenized prompts shaped
                ``[num_prompts][prompt_tokens]``.
            request_ids: Optional vLLM request IDs matching
                ``prompt_token_ids_batch``. The controller scheduler passes
                rollout provenance IDs for structured debugging; direct callers
                can omit this and get stable numeric IDs.
            sampling_config: Optional per-call override for the generator's
                default SamplingConfig. The vLLM engine seed comes from
                ``config.debug.seed``; per-request sampling params do not
                override it, so same-prompt siblings can still diverge.
            metrics_prefix: Namespace prepended to every returned metric key
                (default ``"generator"``). Callers that need to keep streams
                separate, e.g. ``"validation/generator"``, can override it.
        """
        _sampling_config = (
            sampling_config if sampling_config is not None else self.config.sampling
        )
        if request_ids is not None and len(request_ids) != len(prompt_token_ids_batch):
            raise ValueError(
                "request_ids length must match prompt_token_ids_batch: "
                f"{len(request_ids)} != {len(prompt_token_ids_batch)}"
            )
        if request_ids is not None and len(set(request_ids)) != len(request_ids):
            raise ValueError(f"request_ids must be unique, got {request_ids}")

        loop = asyncio.get_running_loop()
        async with self._engine_lock:
            (
                _request_ids,
                request_futures,
                admitted_policy_version,
            ) = self._admit_requests_locked(
                prompt_token_ids_batch,
                request_ids=request_ids,
                sampling_config=_sampling_config,
                loop=loop,
            )

        all_outputs = await asyncio.gather(*request_futures)

        if self.policy_version != admitted_policy_version:
            raise RuntimeError(
                "generator policy_version changed during an active "
                "generate call; weight sync admission is broken"
            )

        # vLLM may return requests out of order; sort by the request IDs we
        # admitted so outputs line up with the input batch.
        request_order = {request_id: idx for idx, request_id in enumerate(_request_ids)}
        try:
            all_outputs.sort(key=lambda output: request_order[str(output.request_id)])
        except KeyError as exc:
            raise RuntimeError(
                f"vLLM returned unknown request_id {exc.args[0]!r}; "
                f"expected one of {_request_ids}"
            ) from exc

        completions: list[Completion] = []
        generation_metrics: list[m.Metric] = []
        output_token_counts: list[int] = []
        for output in all_outputs:
            generation_metrics.extend(
                _prepare_generation_request_metrics(output, prefix=metrics_prefix)
            )
            for sample_idx, sample in enumerate(output.outputs):
                output_token_counts.append(len(sample.token_ids))
                completions.append(
                    self._completion_from_sample(
                        output=output,
                        sample_idx=sample_idx,
                        policy_version=admitted_policy_version,
                    )
                )
        generation_metrics.append(
            m.Metric(
                f"{metrics_prefix}/output_tokens",
                m.Sum.from_list(output_token_counts),
            )
        )
        memory_stats = cuda_memory_stats()
        for key, value in memory_stats.items():
            metric_cls = m.Min if key.startswith("driver_free") else m.Max
            generation_metrics.append(
                m.Metric(
                    f"{metrics_prefix}/cuda_memory/{key}",
                    metric_cls(value),
                )
            )

        logger.debug(
            f"{os.getpid()=} Generating finish generate "
            f"(policy v{admitted_policy_version})..."
        )

        return completions, generation_metrics

    @endpoint
    @sl.log_trace_span("pull_model_state_dict")
    async def pull_model_state_dict(self, version: int) -> None:
        """Pull latest weights from TorchStore.

        When ``direct_rdma=True``, weights are read directly from the
        trainer's GPU memory via one-sided RDMA, bypassing StorageVolumes.
        When ``False``, data is fetched through StorageVolumes (which may
        themselves use RDMA as their transport internally).

        See ``push_model_state_dict`` for more details on the distinction.

        Args:
            version: New policy version number.
        """
        from monarch.rdma import is_rdma_available

        async with self._engine_lock:
            if self._pending_outputs or self._engine.has_unfinished_requests():
                raise RuntimeError(
                    "cannot pull new generator weights while generation requests "
                    "are active"
                )
            model_sd = self._get_model().model.state_dict()
            await ts.get_state_dict(
                "model_state_dict",
                user_state_dict=model_sd,
                strict=False,
                direct_rdma=is_rdma_available(),
            )
            self.policy_version = version
            # Invalidate the KV prefix cache so stale values computed with the
            # old weights are never reused for new generations.
            self._engine.reset_prefix_cache()
            logger.debug(
                f"{os.getpid()=} Generator pulled model state dict for policy v{version}"
            )

    @endpoint
    async def close(self) -> None:
        """Release the vLLM engine and distributed state.

        vLLM's sync ``LLMEngine`` (what we use) has no public ``shutdown``
        method; only the async ``AsyncLLM`` does. We tear it down by
        releasing the pieces that are available in this vLLM build:

        1. ``renderer.shutdown()`` — closes thread pools and the
           multimodal-processor cache.
        2. ``engine_core.shutdown()`` when present.

        The Monarch proc mesh owns the process lifetime after this endpoint
        returns. Calling vLLM's global ``cleanup_dist_env_and_memory`` from the
        actor can hang under ``external_launcher`` when this vLLM build lacks
        the nested worker shutdown method, so process-group teardown is left to
        ``mesh.stop`` on the controller side.
        """
        driver_task = self._engine_driver_task
        if driver_task is not None and not driver_task.done():
            driver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await driver_task

        async with self._engine_lock:
            close_error = RuntimeError("generator closed with active requests")
            for future in self._pending_outputs.values():
                if not future.done():
                    future.set_exception(close_error)
            self._pending_outputs.clear()
            if self._engine is not None:
                renderer = getattr(self._engine, "renderer", None)
                try:
                    try:
                        if renderer is not None:
                            renderer.shutdown()
                    finally:
                        try:
                            self._engine.engine_core.shutdown()
                        except AttributeError as exc:
                            if "shutdown" not in str(exc):
                                raise
                            logger.warning(
                                "vLLM engine_core.shutdown skipped because this vLLM "
                                "build is missing a nested shutdown method: %s",
                                exc,
                            )
                finally:
                    self._engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
