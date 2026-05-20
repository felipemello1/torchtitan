# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# UNTESTED -- REVIEW BEFORE LANDING.
# What changed vs experiments/rl/actors/generator.py @ v7 head:
#   1. Replaced the long-lived `_engine_driver_task` + `_engine_lock` race
#      with an in-coroutine step loop guarded by an `asyncio.Lock`
#      ("busy semaphore"), exactly one `generate` in flight per rank.
#   2. Every admission is preceded by a CPU broadcast of the prompt
#      payload over a gloo `_world_group` (TBR sampler_base.py:708-721),
#      so all TP ranks call `engine.add_request` with bit-identical args.
#   3. Non-rank-0 ranks return `([], [])`; the controller already drops
#      non-zero ranks via `_get_rank_0_value(result, gpus=0)`.
#   4. Public endpoint signatures unchanged. `pull_model_state_dict`,
#      `sync_log_step`, `close` keep their contracts.
#   5. Removed `_pending_outputs`, `_engine_driver_task`,
#      `_ensure_engine_driver_locked`, `_drive_engine`.

import asyncio
import contextlib
import copy
import logging
import math
import os
import pickle
import struct
import time
from dataclasses import dataclass, field

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
    """vLLM per-request metrics from a single `RequestOutput`.

    Identical to the v7 helper; only formatted here for completeness.
    """
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
    """CUDA graph capture settings for the vLLM inference engine.

    Identical to the v7 dataclass. Kept here so the file is a drop-in.
    """

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


# --- admission payload broadcast (TBR sampler_base.py:636-775 analog) --------


# Header is (num_prompts, payload_size_bytes); payload is one pickled tuple
# (prompt_token_ids_batch, request_ids, sampling_params_dict, metrics_prefix).
_ADMIT_HEADER_FMT = "<II"
_ADMIT_HEADER_SIZE = struct.calcsize(_ADMIT_HEADER_FMT)


@dataclass(slots=True)
class _Admission:
    """Decoded admission payload, identical on every TP rank.

    ``arrival_time`` is stamped once on rank 0 and broadcast so every TP
    rank passes the same value to ``engine.add_request``. FCFS scheduling
    ignores it, but priority scheduling uses it as a tiebreaker; sharing
    one value keeps ranks in lockstep if the policy ever changes.
    """

    prompt_token_ids_batch: list[list[int]]
    request_ids: list[str]
    sampling_params: SamplingParams
    arrival_time: float
    metrics_prefix: str


def _encode_admission(
    *,
    prompt_token_ids_batch: list[list[int]],
    request_ids: list[str],
    sampling_params: SamplingParams,
    arrival_time: float,
    metrics_prefix: str,
) -> bytes:
    """Serialize an admission payload for broadcast.

    Mirrors the TBR `_RequestWithDpRanks.broadcast_requests` shape but
    without the per-DP-rank routing fields (we have a single DP group).
    """
    payload = pickle.dumps(
        (
            [list(p) for p in prompt_token_ids_batch],
            list(request_ids),
            sampling_params,
            arrival_time,
            metrics_prefix,
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return payload


def _decode_admission(buf: bytes) -> _Admission:
    prompts, request_ids, sampling_params, arrival_time, metrics_prefix = pickle.loads(
        buf
    )
    return _Admission(
        prompt_token_ids_batch=prompts,
        request_ids=request_ids,
        sampling_params=sampling_params,
        arrival_time=arrival_time,
        metrics_prefix=metrics_prefix,
    )


def _broadcast_admission_bytes(
    *,
    payload: bytes | None,
    rank: int,
    world_group: dist.ProcessGroup,
) -> bytes:
    """Broadcast an admission payload from rank 0 to all ranks.

    Two-step protocol matches TBR `start_broadcast_requests`
    (sampler_base.py:708-721): a uint64 size header followed by the
    payload bytes. We use gloo so this never contends with the NCCL
    stream that runs the model forward.

    Args:
        payload: pickled admission bytes on rank 0; ``None`` on other ranks.
        rank: this process's rank within ``world_group``.
        world_group: gloo `ProcessGroup` covering every TP rank.
    """
    device = torch.device("cpu")
    if rank == 0:
        assert payload is not None, "rank 0 must provide an admission payload"
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
    """
    Generates rollouts using vLLM, with TBR-style cross-rank admission.

    Public contract is identical to v7's VLLMGenerator:

      generate(prompt_token_ids_batch, *, request_ids, sampling_config,
               metrics_prefix) -> (list[Completion], list[Metric])

    Internally, every TP rank holds a gloo broadcast group at startup;
    each ``generate`` call first ships the admission payload over that
    group, then every rank calls ``engine.add_request`` with the same
    decoded payload and runs ``engine.step()`` in lockstep until rank-0
    has all outputs.

    Args:
        config: Generator-specific configuration.
        model_spec: TorchTitan model specification.
        model_path: Path to the HF model checkpoint.
        compile_config: Per-layer torch.compile config shared with trainer.
        max_num_seqs: vLLM batch dim, sized by the controller.
        output_dir: structured-logger output directory.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration (unchanged from v7)."""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        model_dtype: str = "bfloat16"
        gpu_memory_limit: float = 0.9
        cudagraph: VLLMCudagraphConfig = field(default_factory=VLLMCudagraphConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        debug: DebugConfig = field(default_factory=DebugConfig)

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
            # vLLM external_launcher initializes torch.distributed across all
            # TP ranks during this call (vllm_engine.py:62-65). After it
            # returns, dist.is_initialized() is True on every rank.
            self._engine = LLMEngine.from_engine_args(engine_args)
            logger.info("vLLM rollout engine initialized")

        # Establish the TBR-style gloo world group.
        # TBR sampler_base.py:820-833 -- gloo so request fan-out never
        # contends with NCCL traffic that runs the model forward.
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
        # TODO: if NCCL_EAGER_INIT is on, use dist.split_group on the default
        # PG instead of new_group to avoid an extra rendezvous (mirrors
        # sampler_base.py:822-827). Skipped for now -- new_group is simpler
        # and our world size is small.
        self._world_group: dist.ProcessGroup = dist.new_group(backend="gloo")

        # FCFS scheduling is required: any other policy (e.g. priority)
        # uses ``arrival_time`` as a tiebreaker, and ``arrival_time`` is
        # populated from ``time.time()`` per rank inside
        # ``renderer.render_cmpl`` (vllm/renderers/base.py:927), which
        # diverges across ranks and would re-introduce the
        # admission-shape divergence this refactor exists to prevent.
        scheduler_policy = getattr(
            self._engine.vllm_config.scheduler_config, "policy", "fcfs"
        )
        if scheduler_policy != "fcfs":
            raise RuntimeError(
                "TBR-style generator requires FCFS scheduling; got "
                f"scheduler_policy={scheduler_policy!r}. Other policies "
                "use per-rank arrival_time as a tiebreaker and break the "
                "lockstep admission invariant."
            )

        # At-most-one generate in flight on THIS rank. Combined with the
        # collective broadcast in `generate`, this implies at-most-one
        # generate in flight across ALL ranks (no rank can move past the
        # broadcast without every rank participating).
        self._busy = asyncio.Lock()

        self._next_request_id = 0
        self.policy_version = 0

        logger.info(
            f"Generator initialized: tp_rank={self._tp_rank}, "
            f"tp_world_size={self._tp_world_size}"
        )

    # --- determinism & helpers (unchanged from v7) --------------------------

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

    # --- admission ---------------------------------------------------------

    def _prepare_admission_payload(
        self,
        prompt_token_ids_batch: list[list[int]],
        *,
        request_ids: list[str] | None,
        sampling_config: SamplingConfig,
        metrics_prefix: str,
    ) -> tuple[bytes, list[str], SamplingParams]:
        """Rank-0-only: prepare the bytes broadcast to every TP rank.

        Side effect: advances `self._next_request_id` if request_ids is
        ``None``, mirroring the v7 numbering. Worker ranks rebuild
        request_ids from the broadcast payload, so the counter is
        rank-0 authoritative -- this is fine because rank 0 is the
        only side that emits external IDs.
        """
        assert self._tp_rank == 0
        if request_ids is None:
            start_id = self._next_request_id
            self._next_request_id += len(prompt_token_ids_batch)
            request_ids = [
                str(start_id + idx) for idx in range(len(prompt_token_ids_batch))
            ]
        else:
            if len(request_ids) != len(prompt_token_ids_batch):
                raise ValueError(
                    "request_ids length must match prompt_token_ids_batch: "
                    f"{len(request_ids)} != {len(prompt_token_ids_batch)}"
                )
            if len(set(request_ids)) != len(request_ids):
                raise ValueError(f"request_ids must be unique, got {request_ids}")

        # TODO: pre-validate prompt lengths here, matching TBR
        # sample_method.py:923-955. Drop oversized prompts on rank 0
        # before broadcast so we never hand a doomed request to workers.

        sampling_params = self._build_sampling_params(sampling_config)
        payload = _encode_admission(
            prompt_token_ids_batch=prompt_token_ids_batch,
            request_ids=request_ids,
            sampling_params=sampling_params,
            arrival_time=time.time(),
            metrics_prefix=metrics_prefix,
        )
        return payload, request_ids, sampling_params

    def _broadcast_admission(self, payload: bytes | None) -> _Admission:
        """Collective: send the admission payload from rank 0 to all ranks.

        Mirrors TBR `_RequestWithDpRanks.broadcast_requests`
        (sampler_base.py:636-775). Every rank emerges with the same
        decoded admission.
        """
        payload_bytes = _broadcast_admission_bytes(
            payload=payload,
            rank=self._tp_rank,
            world_group=self._world_group,
        )
        return _decode_admission(payload_bytes)

    def _admit_locally(
        self,
        admission: _Admission,
    ) -> None:
        """Run vLLM `add_request` for every prompt in the broadcast set.

        Every rank executes this with bit-identical inputs (admission
        was just broadcast). vLLM's per-worker scheduler will therefore
        admit the same set on every rank, so the next `step()` produces
        matching all-reduce shapes. Passing ``tokens_input`` skips
        ``engine.renderer.render_cmpl`` (which stamps a per-rank
        ``arrival_time`` from ``time.time()``); rank 0 stamps one
        ``arrival_time`` and broadcasts it for all ranks.

        Note: vLLM's request processor mutates ``SamplingParams`` in
        place to attach ``_target_sampling_params``. Pass a fresh deep
        copy per request so the broadcast-cached instance survives
        clean for the next ``generate`` call on rank 0.
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

    # --- inline step loop --------------------------------------------------

    async def _step_until_drained(
        self,
        *,
        request_ids: list[str],
    ) -> dict[str, RequestOutput]:
        """Run `engine.step()` in lockstep until every admitted id finished.

        Replaces the v7 `_drive_engine` background task. Because this
        runs inside the held `_busy` semaphore (and the busy semaphore
        is taken under a collective `_broadcast_admission`), every rank
        enters this loop at the same logical point with the same
        admitted set. vLLM per-worker schedulers therefore produce
        matching collectives.

        Yields to the asyncio loop between steps so other endpoints can
        still be served (`sync_log_step`, `close`).

        Returns:
            dict mapping request_id -> final RequestOutput, populated
            on every rank. Workers' dicts are discarded by the caller.
        """
        outstanding = set(request_ids)
        finished: dict[str, RequestOutput] = {}

        with sl.log_trace_span("engine_step_loop"):
            while outstanding:
                if not self._engine.has_unfinished_requests():
                    # TBR semantics: this should never happen because we
                    # only enter the loop when we just admitted requests.
                    # If we see it, surface a clear error rather than
                    # spinning forever.
                    raise RuntimeError(
                        "vLLM engine became idle with outstanding "
                        f"request_ids {sorted(outstanding)}"
                    )

                with torch.no_grad():
                    step_outputs = self._engine.step()
                for output in step_outputs:
                    rid = str(output.request_id)
                    if rid in outstanding:
                        # vLLM only emits a RequestOutput for a given
                        # request once finished (RequestOutputKind.FINAL_ONLY).
                        finished[rid] = output
                        outstanding.discard(rid)

                # Yield so other actor endpoints can run between steps.
                # Mirrors v7 `_drive_engine` line 437 but no other coroutine
                # can interleave admission because we hold `_busy`.
                await asyncio.sleep(0)

        return finished

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

        Contract unchanged from v7. Internally, the call is collective
        across the generator TP mesh:

          1. Acquire the per-rank busy semaphore.
          2. Rank 0 encodes a pickled admission payload and broadcasts
             it on the gloo `_world_group`. Other ranks receive bytes.
          3. Every rank decodes the payload and calls `engine.add_request`
             with bit-identical args, then runs `engine.step()` in
             lockstep until rank 0's tracked request set drains.
          4. Rank 0 builds Completion + metrics from final outputs.
             Other ranks return ``([], [])`` -- the controller indexes
             rank 0's `ValueMesh` entry via ``_get_rank_0_value``.

        Args:
            prompt_token_ids_batch: ``[num_prompts][prompt_tokens]``.
            request_ids: Optional vLLM request IDs (rank-0 supplied).
            sampling_config: Optional per-call sampling override.
            metrics_prefix: Metric key namespace.

        Example:
            controller-side::

                completions, metrics = await self._await_rank_0(
                    generator.generate.call(
                        prompts, request_ids=ids, sampling_config=sc
                    )
                )
        """
        _sampling_config = (
            sampling_config if sampling_config is not None else self.config.sampling
        )

        async with self._busy:
            admitted_policy_version = self.policy_version

            if self._tp_rank == 0:
                # Pre-validate everything that can fail locally before
                # we broadcast bytes that workers would have to abort.
                if request_ids is not None and len(request_ids) != len(
                    prompt_token_ids_batch
                ):
                    raise ValueError(
                        "request_ids length must match prompt_token_ids_batch: "
                        f"{len(request_ids)} != {len(prompt_token_ids_batch)}"
                    )
                if request_ids is not None and len(set(request_ids)) != len(
                    request_ids
                ):
                    raise ValueError(f"request_ids must be unique, got {request_ids}")
                payload, resolved_request_ids, _ = self._prepare_admission_payload(
                    prompt_token_ids_batch,
                    request_ids=request_ids,
                    sampling_config=_sampling_config,
                    metrics_prefix=metrics_prefix,
                )
            else:
                payload = None
                resolved_request_ids = []  # filled in from broadcast on workers

            # Collective. Every rank must reach here for the broadcast
            # to make progress. The gloo group means a stuck NCCL stream
            # (e.g. from an unrelated hang) cannot block this step.
            admission = self._broadcast_admission(payload)
            if self._tp_rank == 0:
                # Sanity: workers must see the same ids we sent.
                if admission.request_ids != resolved_request_ids:
                    raise RuntimeError(
                        "request_ids broadcast round-trip mismatch: "
                        f"sent={resolved_request_ids} "
                        f"recv={admission.request_ids}"
                    )
            else:
                resolved_request_ids = admission.request_ids

            reset_cuda_peak_memory_stats()
            with sl.log_trace_span("engine_add_request"):
                self._admit_locally(admission)

            finished = await self._step_until_drained(
                request_ids=resolved_request_ids,
            )

            if self.policy_version != admitted_policy_version:
                raise RuntimeError(
                    "generator policy_version changed during an active "
                    "generate call; weight sync admission is broken"
                )

            if self._tp_rank != 0:
                # Workers contribute the collective; they don't shape outputs.
                return [], []

            # --- rank 0: build Completion + metrics ---
            try:
                all_outputs = [finished[rid] for rid in resolved_request_ids]
            except KeyError as exc:
                raise RuntimeError(
                    f"vLLM did not return output for request_id {exc.args[0]!r}; "
                    f"expected one of {resolved_request_ids}"
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

            return completions, generation_metrics

    @endpoint
    @sl.log_trace_span("pull_model_state_dict")
    async def pull_model_state_dict(self, version: int) -> None:
        """Pull latest weights from TorchStore.

        Public contract preserved. Must be called while no `generate`
        is in flight; we enforce this by taking `_busy`. The vLLM cache
        resets happen on every rank because `pull_model_state_dict` is
        already invoked via `generator.pull_model_state_dict.call(version)`
        (mirrors TBR snapshot.py:200-223 where reset_prefix_cache runs
        inside the GPU callback that loaded weights).

        Args:
            version: New policy version number.
        """
        from monarch.rdma import is_rdma_available

        async with self._busy:
            if self._engine.has_unfinished_requests():
                raise RuntimeError(
                    "cannot pull new generator weights while generation "
                    "requests are active"
                )
            model_sd = self._get_model().model.state_dict()
            await ts.get_state_dict(
                "model_state_dict",
                user_state_dict=model_sd,
                strict=False,
                direct_rdma=is_rdma_available(),
            )
            self.policy_version = version
            # Drop the prefix cache so values computed with stale weights
            # are never reused. Every rank runs this under the same
            # external trigger (the controller's .call), matching TBR.
            self._engine.reset_prefix_cache()
            logger.debug(
                f"{os.getpid()=} Generator pulled model state dict for v{version}"
            )

    @endpoint
    async def close(self) -> None:
        """Release the vLLM engine.

        Unchanged from v7 except that we no longer have a long-lived
        driver task to cancel. The Monarch proc mesh owns process
        lifetime after this returns.
        """
        async with self._busy:
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

        # Tear down our gloo group. Default PG is owned by vLLM external
        # launcher and gets reclaimed when mesh.stop kills the process,
        # mirroring the comment in v7 generator.close (lines 690-695).
        with contextlib.suppress(Exception):
            dist.destroy_process_group(self._world_group)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
