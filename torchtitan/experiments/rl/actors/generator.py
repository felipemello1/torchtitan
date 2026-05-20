# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM Monarch actor with a TBR-style continuous engine loop.

Public contract:

    generate(prompt_token_ids_batch, *, request_ids, sampling_config,
             metrics_prefix) -> (list[Completion], list[Metric])

One background ``_engine_loop`` per actor rank owns the vLLM engine.
``generate(...)`` enqueues per-request :class:`asyncio.Future` objects on
rank 0 and awaits them; the loop coalesces pending submits, broadcasts
the admission set to every TP rank, admits locally, then runs bounded
``engine.step()`` bursts and resolves request futures as outputs finish.
Continuous batching across ``generate`` calls falls out for free.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import gc
import logging
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.distributed as dist
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
from vllm.inputs import tokens_input
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind

logger = logging.getLogger(__name__)


# --- module-level helpers ----------------------------------------------------


def _prepare_generation_request_metrics(
    output: RequestOutput, *, prefix: str
) -> list[m.Metric]:
    """vLLM per-request metrics from a single ``RequestOutput``."""
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

    return [
        metric
        for key, value in metric_values.items()
        for metric in (m.Metric(key, m.Mean(value)), m.Metric(key, m.Max(value)))
    ]


@dataclass(kw_only=True, slots=True)
class VLLMCudagraphConfig:
    """CUDA graph capture settings for the vLLM inference engine."""

    enable: bool = True
    """Whether to enable CUDA graph capture (vLLM "full" mode)."""

    def get_vllm_compilation_config(
        self, *, max_num_seqs: int
    ) -> CompilationConfig | None:
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


# --- admission + lifecycle payloads (broadcast across TP ranks) --------------


_DEFAULT_MAX_STEPS_PER_ITERATION = 8

# vLLM finish reasons that mean the request did not produce usable output
# (matches TBR's per-request error surface; see RFC \xa77).
_FINISH_REASON_ERROR = frozenset({"error", "abort"})


@dataclass(slots=True)
class _Admission:
    """Decoded admission payload, identical on every TP rank.

    ``arrival_time`` is stamped once on rank 0 and broadcast so every TP
    rank passes the same value to ``engine.add_request``. FCFS scheduling
    ignores it, but priority scheduling uses it as a tiebreaker; sharing
    one value keeps ranks in lockstep if the policy ever changes.
    """

    prompt_token_ids_batch: list[list[int]]  # [num_prompts][prompt_tokens]
    request_ids: list[str]  # [num_prompts]
    sampling_params: SamplingParams
    arrival_time: float
    metrics_prefix: str


@dataclass(slots=True)
class _PendingRequest:
    """One ``generate`` request awaiting admission and a completion.

    The owning ``generate(...)`` coroutine holds ``future`` and the shared
    ``metrics_sink`` list; the engine loop populates them when this request
    finishes inside vLLM.
    """

    request_id: str
    prompt_token_ids: list[int]  # [prompt_tokens]
    sampling_params: SamplingParams
    future: asyncio.Future[Completion]
    metrics_prefix: str
    metrics_sink: list[m.Metric]
    admitted_policy_version: int = 0  # stamped at admission time on rank 0


@dataclass(slots=True)
class _EngineMessage:
    """One lifecycle directive broadcast from rank 0 to every TP rank.

    The engine loop only proceeds collectively, so admit, quiesce, resume,
    and shutdown all travel through the same gloo broadcast that ranks 1+
    are blocked on.
    """

    kind: Literal["admit", "quiesce", "resume", "shutdown"]
    admission: _Admission | None = None


def _encode_engine_message(msg: _EngineMessage) -> bytes:
    """Serialize an engine-loop message for cross-rank broadcast."""
    return pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)


def _decode_engine_message(buf: bytes) -> _EngineMessage:
    return pickle.loads(buf)


def _broadcast_engine_message_bytes(
    *,
    payload: bytes | None,
    rank: int,
    world_group: dist.ProcessGroup,
) -> bytes:
    """Broadcast a pickled engine message from rank 0 to all ranks.

    Two-step protocol: a uint64 size header followed by the payload. Uses
    gloo so this never contends with the NCCL stream that runs the model
    forward.
    """
    device = torch.device("cpu")
    if rank == 0:
        assert payload is not None, "rank 0 must provide a payload"
        size_tensor = torch.tensor([len(payload)], dtype=torch.int64, device=device)
        buf_tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    else:
        size_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        buf_tensor = None

    dist.broadcast(size_tensor, src=0, group=world_group)
    payload_size = int(size_tensor.item())

    if rank != 0:
        buf_tensor = torch.empty(payload_size, dtype=torch.uint8, device=device)

    assert buf_tensor is not None
    dist.broadcast(buf_tensor, src=0, group=world_group)
    return bytes(buf_tensor.numpy().tobytes())


# --- the actor ---------------------------------------------------------------


class VLLMGenerator(Actor, Configurable):
    """Generates rollouts using vLLM with a continuous TBR-style engine loop.

    A background ``_engine_loop`` runs on every TP rank from the first
    ``generate(...)`` until ``close()``. ``generate`` enqueues request
    futures on rank 0; the loop coalesces pending submits inside the
    actor, broadcasts the admission set, admits locally on every rank,
    and runs bounded ``engine.step()`` bursts. Outputs are dispatched
    back to per-request futures as they finish.

    Args:
        config: Generator-specific configuration.
        model_spec: TorchTitan model specification.
        model_path: Path to the HF model checkpoint.
        compile_config: Per-layer ``torch.compile`` config shared with trainer.
        max_num_seqs: vLLM batch dim, sized by the controller.
        output_dir: Structured-logger output directory.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration."""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        model_dtype: str = "bfloat16"
        gpu_memory_limit: float = 0.9
        cudagraph: VLLMCudagraphConfig = field(default_factory=VLLMCudagraphConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        debug: DebugConfig = field(default_factory=DebugConfig)
        max_steps_per_iteration: int = _DEFAULT_MAX_STEPS_PER_ITERATION
        """Max ``engine.step()`` calls per engine-loop iteration.

        Bounds how long the loop runs the engine before checking for new
        admissions. Matches TBR's ``one_step`` cap (sample_method.py:537-558).
        """

        def __post_init__(self):
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
                raise ValueError("Generator does not support sequence parallelism.")
            if not p.disable_loss_parallel:
                raise ValueError(
                    "Generator requires disable_loss_parallel=True, "
                    f"got disable_loss_parallel={p.disable_loss_parallel}"
                )
            if self.max_steps_per_iteration <= 0:
                raise ValueError(
                    "max_steps_per_iteration must be positive, "
                    f"got {self.max_steps_per_iteration}"
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

        if not dist.is_initialized():
            raise RuntimeError(
                "vLLM external_launcher did not initialize torch.distributed; "
                "cannot create the request-broadcast group."
            )
        self._tp_world_size = dist.get_world_size()
        self._tp_rank = dist.get_rank()
        if self._tp_world_size != self.config.parallelism.tensor_parallel_degree:
            raise RuntimeError(
                "vLLM distributed world size does not match configured "
                f"tensor_parallel_degree: world_size={self._tp_world_size}, "
                f"tensor_parallel_degree="
                f"{self.config.parallelism.tensor_parallel_degree}"
            )
        # Gloo PG: dedicated CPU process group for engine-loop messages so
        # broadcasts never contend with the NCCL stream running the model
        # forward (TBR sampler_base.py:820-833).
        self._world_group: dist.ProcessGroup = dist.new_group(backend="gloo")

        # TP ranks must admit requests in the same order. FCFS ignores
        # arrival_time and follows broadcast order. Priority scheduling
        # uses arrival_time as a heap key, so it needs a separate
        # deterministic ordering audit before enabling.
        scheduler_policy = getattr(
            self._engine.vllm_config.scheduler_config, "policy", "fcfs"
        )
        if scheduler_policy != "fcfs":
            raise RuntimeError(
                "VLLMGenerator currently supports only FCFS scheduling; got "
                f"scheduler_policy={scheduler_policy!r}."
            )

        # Engine-loop state. The loop is single-threaded per rank; rank 0
        # owns the message source (CV) while workers wait on the broadcast.
        self._cv = asyncio.Condition()
        self._pending_requests: list[_PendingRequest] = []
        self._pending_by_request_id: dict[str, _PendingRequest] = {}
        self._engine_loop_task: asyncio.Task[None] | None = None
        self._engine_loop_started = asyncio.Event()
        self._shutdown_requested = False
        self._quiesce_requested = False
        self._resume_requested = False
        self._quiesced_event = asyncio.Event()
        self._max_steps_per_iteration = config.max_steps_per_iteration

        self._next_request_id = 0
        self.policy_version = 0

        logger.info(
            f"Generator initialized: tp_rank={self._tp_rank}, "
            f"tp_world_size={self._tp_world_size}"
        )

    # --- determinism & helpers ---------------------------------------------

    @staticmethod
    def _set_determinism(debug: DebugConfig) -> None:
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
        return self._engine.model_executor.driver_worker.get_model()

    @endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
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

    def _validate_prompt_for_generation(
        self, prompt_token_ids: list[int]
    ) -> str | None:
        """Return ``None`` if the prompt is acceptable, otherwise an error string.

        Catches rank-independent failures (empty prompt, no room under
        ``max_model_len``, out-of-vocabulary token id) so rank 0 can drop
        them BEFORE the cross-rank broadcast. This matches TBR
        ``sample_method.py:923-955`` and protects the lockstep invariant:
        a broadcast that some ranks would reject in ``add_request`` would
        leave the TP world out of sync.
        """
        if not prompt_token_ids:
            return "decoder prompt cannot be empty"

        max_model_len = self._engine.model_config.max_model_len
        if len(prompt_token_ids) >= max_model_len:
            return (
                f"decoder prompt length {len(prompt_token_ids)} leaves no room "
                f"for generation under max_model_len={max_model_len}"
            )

        tokenizer = getattr(self._engine.input_processor, "tokenizer", None)
        if tokenizer is not None:
            model_vocab_size = self._engine.model_config.get_vocab_size()
            max_valid_token_id = max(
                getattr(tokenizer, "max_token_id", model_vocab_size - 1),
                model_vocab_size - 1,
            )
            max_input_id = max(prompt_token_ids)
            if max_input_id > max_valid_token_id:
                return (
                    f"token id {max_input_id} is out of vocabulary "
                    f"(max_valid_token_id={max_valid_token_id})"
                )
        return None

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
            sample.token_ids, sample.logprobs, strict=True
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

    # --- engine-loop infrastructure ---------------------------------------

    async def _ensure_engine_loop(self) -> None:
        """Start the single per-rank engine loop on first call."""
        if self._engine_loop_task is None:
            self._engine_loop_task = asyncio.create_task(self._engine_loop())
            self._engine_loop_started.set()

    def _admit_locally(self, admission: _Admission) -> None:
        """Run ``engine.add_request`` for every prompt in the admission set.

        Every rank executes this with bit-identical inputs (admission
        was just broadcast). vLLM's per-worker scheduler therefore
        admits the same set on every rank, so the next ``step()`` keeps
        TP ranks in lockstep.

        Note: vLLM mutates ``SamplingParams`` in place to attach
        ``_target_sampling_params``. Pass a fresh deep copy per request.
        """
        for request_id, prompt_token_ids in zip(
            admission.request_ids,
            admission.prompt_token_ids_batch,
            strict=True,
        ):
            self._engine.add_request(
                request_id=request_id,
                prompt=tokens_input(list(prompt_token_ids)),
                params=copy.deepcopy(admission.sampling_params),
                arrival_time=admission.arrival_time,
            )

    async def _broadcast_message(self, payload: bytes | None) -> _EngineMessage:
        """Collective broadcast of an engine message from rank 0.

        Runs the blocking CPU broadcast in a worker thread so the event
        loop can keep serving other endpoints (``sync_log_step``,
        ``close``) while the broadcast is in flight.
        """
        received = await asyncio.to_thread(
            _broadcast_engine_message_bytes,
            payload=payload,
            rank=self._tp_rank,
            world_group=self._world_group,
        )
        return _decode_engine_message(received)

    async def _next_rank0_message(self) -> _EngineMessage:
        """Rank-0-only: choose the next engine message based on actor state.

        Blocks on the condition variable until shutdown, quiesce, resume,
        a pending submit, or an unfinished engine request exists. Empty
        admits keep the workers stepping while requests are still in
        flight inside vLLM. Invalid prompts (oversized, OOV) are resolved
        with ``Completion.error`` here and never enter the broadcast.
        """
        async with self._cv:
            await self._cv.wait_for(
                lambda: (
                    self._shutdown_requested
                    or self._quiesce_requested
                    or self._resume_requested
                    or bool(self._pending_requests)
                    or self._engine.has_unfinished_requests()
                )
            )
            if self._shutdown_requested:
                return _EngineMessage(kind="shutdown")
            if self._quiesce_requested:
                self._quiesce_requested = False
                return _EngineMessage(kind="quiesce")
            if self._resume_requested:
                self._resume_requested = False
                return _EngineMessage(kind="resume")
            if not self._pending_requests:
                return _EngineMessage(
                    kind="admit", admission=self._empty_admission()
                )

            pending = self._pending_requests
            self._pending_requests = []
            admitted_policy_version = self.policy_version

        valid: list[_PendingRequest] = []
        for req in pending:
            error = self._validate_prompt_for_generation(req.prompt_token_ids)
            if error is None:
                req.admitted_policy_version = admitted_policy_version
                self._pending_by_request_id[req.request_id] = req
                valid.append(req)
                continue
            logger.warning(
                "rejected prompt request_id=%s prompt_tokens=%d: %s",
                req.request_id,
                len(req.prompt_token_ids),
                error,
            )
            sl.log_trace_scalar(
                {
                    "generator.prompt_validation_rejected": 1,
                    "generator.prompt_validation_prompt_tokens": len(
                        req.prompt_token_ids
                    ),
                }
            )
            req.metrics_sink.append(
                m.Metric(
                    f"{req.metrics_prefix}/prompt_validation_rejected",
                    m.Sum(1),
                )
            )
            if not req.future.done():
                req.future.set_result(
                    Completion(
                        policy_version=admitted_policy_version,
                        token_ids=[],
                        token_logprobs=[],
                        finish_reason=None,
                        error=error,
                    )
                )

        return _EngineMessage(
            kind="admit", admission=self._build_admission_from_valid(valid)
        )

    def _empty_admission(self) -> _Admission:
        # Used when rank 0 has unfinished requests but no new submits; the
        # broadcast keeps workers in lockstep for the next step burst.
        return _Admission(
            prompt_token_ids_batch=[],
            request_ids=[],
            sampling_params=self._build_sampling_params(self.config.sampling),
            arrival_time=time.time(),
            metrics_prefix="generator",
        )

    def _build_admission_from_valid(
        self, valid: list[_PendingRequest]
    ) -> _Admission:
        # ``valid`` has already been filtered against
        # ``_validate_prompt_for_generation``. An empty list is normal when
        # every pending request failed validation; we still build (and the
        # caller still broadcasts) an empty admission so workers stay in
        # lockstep on the engine-loop iteration count.
        if not valid:
            return self._empty_admission()
        return _Admission(
            prompt_token_ids_batch=[req.prompt_token_ids for req in valid],
            request_ids=[req.request_id for req in valid],
            sampling_params=valid[0].sampling_params,
            arrival_time=time.time(),
            metrics_prefix=valid[0].metrics_prefix,
        )

    async def _step_burst(self) -> list[RequestOutput]:
        """Run up to ``max_steps_per_iteration`` ``engine.step()`` calls.

        Yields back to the asyncio loop after each step so other endpoints
        can run between vLLM iterations. The loop body keeps stepping
        until either the cap is reached or no requests remain in flight.
        """
        outputs: list[RequestOutput] = []
        with sl.log_trace_span("engine_step_burst"):
            for _ in range(self._max_steps_per_iteration):
                if not self._engine.has_unfinished_requests():
                    break
                with torch.no_grad():
                    step_outputs = self._engine.step()
                outputs.extend(step_outputs)
                await asyncio.sleep(0)
        return outputs

    def _resolve_finished_outputs(self, outputs: list[RequestOutput]) -> None:
        """Rank-0-only: route finished outputs back to their request futures.

        vLLM may finish a request with ``finish_reason in {"error", "abort"}``
        when the engine couldn't produce a usable answer; those land as
        ``Completion.error`` so siblings keep running. Per-request decode
        errors (missing logprobs, bad sample shape) also map to
        ``Completion.error`` rather than raising into the engine loop --
        otherwise one bad request would tear down the actor.
        """
        for output in outputs:
            rid = str(output.request_id)
            pending = self._pending_by_request_id.pop(rid, None)
            if pending is None:
                # Output for a request we already resolved; ignore.
                continue
            pending.metrics_sink.extend(
                _prepare_generation_request_metrics(
                    output, prefix=pending.metrics_prefix
                )
            )
            sample = output.outputs[0]
            pending.metrics_sink.append(
                m.Metric(
                    f"{pending.metrics_prefix}/output_tokens",
                    m.Sum(len(sample.token_ids)),
                )
            )
            if pending.future.done():
                continue
            if sample.finish_reason in _FINISH_REASON_ERROR:
                pending.future.set_result(
                    Completion(
                        policy_version=pending.admitted_policy_version,
                        token_ids=list(sample.token_ids),
                        token_logprobs=[],
                        finish_reason=sample.finish_reason,
                        error=(
                            f"vLLM finished request_id={rid!r} with "
                            f"finish_reason={sample.finish_reason!r}"
                        ),
                    )
                )
                continue
            try:
                completion = self._completion_from_sample(
                    output=output,
                    sample_idx=0,
                    policy_version=pending.admitted_policy_version,
                )
            except Exception as exc:
                pending.future.set_result(
                    Completion(
                        policy_version=pending.admitted_policy_version,
                        token_ids=list(sample.token_ids),
                        token_logprobs=[],
                        finish_reason=sample.finish_reason,
                        error=f"completion build failed: {exc}",
                    )
                )
                continue
            pending.future.set_result(completion)

    async def _drain_engine_to_empty(self) -> None:
        """Step the engine until every in-flight request has finished.

        Used during quiesce so weight sync sees an idle engine. Finished
        outputs still resolve their request futures during the drain, so
        callers awaiting ``generate(...)`` see completions land before the
        weight swap proceeds.
        """
        with sl.log_trace_span("engine_drain"):
            while self._engine.has_unfinished_requests():
                with torch.no_grad():
                    step_outputs = self._engine.step()
                if self._tp_rank == 0:
                    self._resolve_finished_outputs(step_outputs)
                await asyncio.sleep(0)

    async def _engine_loop(self) -> None:
        """Single per-rank engine driver. Runs until ``shutdown`` is broadcast.

        Each iteration: rank 0 builds the next message and encodes it;
        every rank broadcasts; every rank handles the message identically.
        Lifecycle directives (quiesce/resume/shutdown) and admissions all
        flow through this one stream so workers stay in lockstep.
        """
        with sl.log_trace_span("engine_loop"):
            while True:
                payload: bytes | None = None
                if self._tp_rank == 0:
                    msg = await self._next_rank0_message()
                    payload = _encode_engine_message(msg)

                msg = await self._broadcast_message(payload)

                if msg.kind == "shutdown":
                    return
                if msg.kind == "quiesce":
                    await self._drain_engine_to_empty()
                    self._quiesced_event.set()
                    continue
                if msg.kind == "resume":
                    self._quiesced_event.clear()
                    continue

                assert msg.kind == "admit"
                if msg.admission is not None and msg.admission.request_ids:
                    self._admit_locally(msg.admission)

                outputs = await self._step_burst()
                if self._tp_rank == 0:
                    self._resolve_finished_outputs(outputs)

    # --- public endpoints --------------------------------------------------

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
        """Generate completions for tokenized prompts.

        Rank 0 enqueues one :class:`_PendingRequest` per prompt and waits
        on the per-request futures; the engine loop coalesces these with
        sibling submits, broadcasts the admission, and dispatches outputs
        back. Workers participate in the engine loop's collectives but
        return empty lists -- the controller indexes rank 0's
        ``ValueMesh`` entry via ``_get_rank_0_value``.

        Args:
            prompt_token_ids_batch: ``[num_prompts][prompt_tokens]``.
            request_ids: Optional vLLM request IDs (rank-0 supplied).
            sampling_config: Optional per-call sampling override.
            metrics_prefix: Metric key namespace for this call.

        Example::

            completions, metrics = await self._await_rank_0(
                generator.generate.call(
                    prompts, request_ids=ids, sampling_config=sc
                )
            )
        """
        await self._ensure_engine_loop()
        if self._tp_rank != 0:
            return [], []

        if request_ids is not None:
            if len(request_ids) != len(prompt_token_ids_batch):
                raise ValueError(
                    "request_ids length must match prompt_token_ids_batch: "
                    f"{len(request_ids)} != {len(prompt_token_ids_batch)}"
                )
            if len(set(request_ids)) != len(request_ids):
                raise ValueError(f"request_ids must be unique, got {request_ids}")
        else:
            start_id = self._next_request_id
            self._next_request_id += len(prompt_token_ids_batch)
            request_ids = [
                str(start_id + idx) for idx in range(len(prompt_token_ids_batch))
            ]

        sampling_cfg = (
            sampling_config if sampling_config is not None else self.config.sampling
        )
        sampling_params = self._build_sampling_params(sampling_cfg)

        loop = asyncio.get_running_loop()
        metrics_sink: list[m.Metric] = []
        pending: list[_PendingRequest] = []
        for request_id, prompt_token_ids in zip(
            request_ids, prompt_token_ids_batch, strict=True
        ):
            pending.append(
                _PendingRequest(
                    request_id=request_id,
                    prompt_token_ids=list(prompt_token_ids),
                    sampling_params=sampling_params,
                    future=loop.create_future(),
                    metrics_prefix=metrics_prefix,
                    metrics_sink=metrics_sink,
                )
            )

        reset_cuda_peak_memory_stats()
        async with self._cv:
            self._pending_requests.extend(pending)
            self._cv.notify_all()

        completions = await asyncio.gather(*(req.future for req in pending))

        memory_stats = cuda_memory_stats()
        for key, value in memory_stats.items():
            metric_cls = m.Min if key.startswith("driver_free") else m.Max
            metrics_sink.append(
                m.Metric(
                    f"{metrics_prefix}/cuda_memory/{key}",
                    metric_cls(value),
                )
            )

        return list(completions), metrics_sink

    @endpoint
    @sl.log_trace_span("pull_model_state_dict")
    async def pull_model_state_dict(self, version: int) -> None:
        """Pull latest weights from TorchStore after quiescing the engine.

        On rank 0, signals the engine loop to drain in-flight requests and
        broadcast a quiesce; every rank waits for its local engine loop
        to set ``_quiesced_event`` (rank 0 by its own request, workers by
        receiving the broadcast). All ranks then collectively pull the new
        weights, bump ``policy_version``, and reset the prefix cache. Rank
        0 finally signals resume so the loop clears the event.

        Args:
            version: New policy version number.
        """
        from monarch.rdma import is_rdma_available

        await self._ensure_engine_loop()

        if self._tp_rank == 0:
            async with self._cv:
                self._quiesce_requested = True
                self._cv.notify_all()

        await self._quiesced_event.wait()
        if self._engine.has_unfinished_requests():
            raise RuntimeError(
                "engine still has unfinished requests after quiesce drain"
            )

        # Catch the two failure modes most likely to be silent under
        # ``ts.get_state_dict(strict=False)``:
        #   1. Version regression / replay: the trainer must advance
        #      ``policy_version`` between syncs. Equal versions are only
        #      legal on the initial pull while the actor still has v0.
        #   2. Empty source state dict: the model wrapper returns no
        #      tensors -> we'd silently keep stale weights and not
        #      notice (one of the bugs the v7 ``_dedup_tied_tensors``
        #      story flagged; see actors/trainer.py:666-674).
        if version < self.policy_version:
            raise RuntimeError(
                f"policy_version regression: actor at v{self.policy_version}, "
                f"asked to pull v{version}"
            )
        if version == self.policy_version and self.policy_version != 0:
            raise RuntimeError(
                f"policy_version did not advance: actor at v{self.policy_version}, "
                "asked to pull the same version twice"
            )

        model_sd = self._get_model().model.state_dict()
        if not model_sd:
            raise RuntimeError(
                "generator model returned an empty state_dict; cannot "
                "perform weight sync"
            )
        await ts.get_state_dict(
            "model_state_dict",
            user_state_dict=model_sd,
            strict=False,
            direct_rdma=is_rdma_available(),
        )
        self.policy_version = version
        self._reset_engine_caches_after_weight_sync()

        if self._tp_rank == 0:
            async with self._cv:
                self._resume_requested = True
                self._cv.notify_all()
        logger.debug(
            f"{os.getpid()=} Generator pulled model state dict for v{version}"
        )

    def _reset_engine_caches_after_weight_sync(self) -> None:
        """Drop every vLLM cache that holds values computed under old weights.

        After a weight swap, prefix-cached prefill KV, mm-cached vision
        features, and encoder-cached hidden states are all stale. Reset
        whatever this vLLM build exposes (``reset_mm_cache`` /
        ``reset_encoder_cache`` are present on current builds but
        ``getattr``-guarded for forward compatibility), then GC so the
        freed allocations are released promptly.
        """
        self._engine.reset_prefix_cache()
        reset_mm_cache = getattr(self._engine, "reset_mm_cache", None)
        if reset_mm_cache is not None:
            reset_mm_cache()
        reset_encoder_cache = getattr(self._engine, "reset_encoder_cache", None)
        if reset_encoder_cache is not None:
            reset_encoder_cache()
        collected = gc.collect(generation=2)
        sl.log_trace_scalar(
            {
                "generator.weight_sync_cache_reset.gc_collected": collected,
            }
        )
        logger.debug(
            "generator cache reset after weight sync; gc_collected=%d", collected
        )

    @endpoint
    async def close(self) -> None:
        """Stop the engine loop and release the vLLM engine."""
        if self._engine_loop_task is not None:
            if self._tp_rank == 0:
                async with self._cv:
                    self._shutdown_requested = True
                    self._cv.notify_all()
            with contextlib.suppress(Exception):
                await self._engine_loop_task
            self._engine_loop_task = None

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
                            "vLLM engine_core.shutdown skipped on this "
                            "vLLM build: %s",
                            exc,
                        )
            finally:
                self._engine = None

        with contextlib.suppress(Exception):
            dist.destroy_process_group(self._world_group)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
