# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend A: raw `LLMEngine` (SPMD) + a hand-driven continuous-batching loop.

vLLM runs in `external_launcher` mode: one engine per TP-rank process, and
`engine.step()` is a collective. So every rank must `add_request` + `step` the
SAME set in the SAME order or the forward desyncs. One background `_engine_loop`
per rank enforces this: rank 0 is the only source of admissions and control
words, broadcast to all ranks over a CPU (gloo) group; every rank then admits
and steps in lockstep. Concurrent per-request `generate` calls coalesce because
the loop folds whatever is queued into the next admission (Monarch dispatches
the calls concurrently; the loop runs as a background task during their awaits).

Weight sync reuses worktree-37's drain-before-swap barrier, simplified to one
`_sync_requested` flag + one `_quiesced` event:

    rank0 generate ─submit→ _admit_q ─┐
                                      ▼
    rank0 _engine_loop: _next_control() ─"admit"/"drain"/"resume"/"shutdown"─►
        broadcast_object_list (gloo) ──► every rank handles the SAME word:
            admit  -> add_request all; bounded step() burst; rank0 resolves futures
            drain  -> step() until empty (collective, lockstep); set _quiesced
            resume -> clear _quiesced
            shutdown -> return
    pull_model_state_dict: rank0 sets _sync_requested → loop emits "drain" →
        every rank drains → _quiesced set → all ranks ts.get_state_dict
        (collective) → rank0 clears _sync_requested → loop emits "resume".
    No request is admitted while _sync_requested, so weights never swap mid-decode.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging

import torch
import torch.distributed as dist
import torchstore as ts
from monarch.actor import endpoint
from torchtitan.experiments.rl.actors.generators.base import (
    _prepare_generation_request_metrics,
    VLLMGeneratorBase,
)
from torchtitan.experiments.rl.actors.generators.types import SamplingConfig
from torchtitan.experiments.rl.models.vllm_registry import TORCHTITAN_CONFIG_FORMAT
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Completion
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.utils import has_cuda_capability
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)

_MAX_STEPS_PER_ITERATION = 8

# Control words broadcast from rank 0 to every TP rank each loop tick. Plain
# strings (not an enum/dataclass) — they only ever ride one broadcast.
_ADMIT, _DRAIN, _RESUME, _SHUTDOWN = "admit", "drain", "resume", "shutdown"


class LLMEngineGenerator(VLLMGeneratorBase):
    """SPMD `LLMEngine` generator with a continuous-batching engine loop.

    Example::

        completion = await gen.generate.call_one(
            [101, 102, 103], request_id="g0/sample=0/turn=0"
        )
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        engine_kwargs = dict(
            # ``model`` locates tokenizer assets + safetensors shards; the model
            # config comes from torchtitan's ModelSpec via
            # config_format=TORCHTITAN_CONFIG_FORMAT (no config.json read).
            model=self.model_path,
            trust_remote_code=True,
            config_format=TORCHTITAN_CONFIG_FORMAT,
            dtype=config.model_dtype,
            tensor_parallel_size=config.parallelism.tensor_parallel_degree,
            # Monarch already spawned one process per TP rank; external_launcher
            # tells vLLM to run one worker per process (no subprocess spawning).
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=config.gpu_memory_limit,
            enforce_eager=not config.cudagraph.enable,
            attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
            disable_log_stats=False,  # enables RequestOutput.metrics
            max_model_len=self.model_spec.model.rope.max_seq_len,
            max_num_seqs=self._max_num_seqs,
        )
        if not has_cuda_capability(9, 0):
            engine_kwargs["block_size"] = 256  # FA2 needs a multiple of 256
        compilation_config = config.cudagraph.get_vllm_compilation_config(
            max_num_seqs=self._max_num_seqs
        )
        if compilation_config is not None:
            engine_kwargs["compilation_config"] = compilation_config
        if config.debug.seed is not None:
            engine_kwargs["seed"] = config.debug.seed

        with sl.log_trace_span("vllm_init"):
            logger.info("Initializing LLMEngine from EngineArgs...")
            self._engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
            logger.info("vLLM rollout engine initialized")

        if not dist.is_initialized():
            raise RuntimeError(
                "vLLM external_launcher did not initialize torch.distributed; "
                "cannot create the request-broadcast group."
            )
        self._tp_rank = dist.get_rank()
        # Dedicated CPU (gloo) group for control broadcasts so they never
        # contend with the NCCL stream running the model forward.
        self._world_group = dist.new_group(backend="gloo")

        scheduler_policy = getattr(
            self._engine.vllm_config.scheduler_config, "policy", "fcfs"
        )
        if scheduler_policy != "fcfs":
            # FCFS makes admission order == broadcast order on every rank;
            # priority scheduling would reorder per rank and break TP lockstep.
            raise RuntimeError(
                f"LLMEngineGenerator requires FCFS scheduling; got {scheduler_policy!r}."
            )

        # Engine-loop state. Rank 0 owns the queue + futures (under the CV);
        # workers only react to broadcast control words.
        self._cv = asyncio.Condition()
        self._admit_q: list[tuple[str, list[int], SamplingConfig, str]] = []
        self._pending: dict[str, tuple[asyncio.Future[Completion], str]] = {}
        self._loop_task: asyncio.Task | None = None
        self._closing = False
        self._sync_requested = False  # rank 0 sets to request a weight-sync drain
        self._quiesced = asyncio.Event()  # set on every rank once its engine drained

    def _build_sampling_params(self, sampling: SamplingConfig) -> SamplingParams:
        """vLLM `SamplingParams` for one request. Always ``n=1``: `SamplingConfig.n`
        is the controller's group size, not vLLM's `n`."""
        return SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            n=1,
            stop_token_ids=list(sampling.stop_token_ids) or None,
            seed=self.config.debug.seed,
            logprobs=1,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

    # --- engine loop -------------------------------------------------------

    def _get_model(self):
        """Return the vLLM model wrapper owned by the driver worker."""
        return self._engine.model_executor.driver_worker.get_model()

    async def _ensure_engine_loop(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._engine_loop())

    async def _engine_loop(self) -> None:
        """Per-rank driver; runs until ``_SHUTDOWN`` is broadcast. On any crash,
        fail every outstanding future so awaiting callers don't hang forever."""
        try:
            with sl.log_trace_span("engine_loop"):
                while True:
                    payload = await self._next_control() if self._tp_rank == 0 else None
                    word, admits = await self._broadcast(payload)

                    if word == _SHUTDOWN:
                        return
                    if word == _DRAIN:
                        await self._drain_to_empty()
                        self._quiesced.set()
                        continue
                    if word == _RESUME:
                        self._quiesced.clear()
                        continue

                    if admits:
                        self._admit(admits)
                    with sl.log_trace_span("engine_step_burst"):
                        for _ in range(_MAX_STEPS_PER_ITERATION):
                            if not self._engine.has_unfinished_requests():
                                break
                            with torch.no_grad():
                                outputs = self._engine.step()
                            self._resolve(outputs)
                            await asyncio.sleep(0)
        except Exception as exc:
            logger.exception("engine loop crashed; failing pending requests")
            for future, _ in self._pending.values():
                if not future.done():
                    future.set_exception(exc)
            self._pending.clear()
            raise

    async def _next_control(
        self,
    ) -> tuple[str, list[tuple[str, list[int], SamplingConfig, str]] | None]:
        """Rank-0-only: choose the next control word + admission payload.

        Blocks until there is something to do: shutdown, a weight-sync
        drain/resume edge, a queued submit, or an unfinished engine (so
        in-flight requests keep stepping). No admission while a sync is pending.
        """
        async with self._cv:
            await self._cv.wait_for(
                lambda: self._closing
                or (self._sync_requested != self._quiesced.is_set())
                or (
                    not self._sync_requested
                    and (self._admit_q or self._engine.has_unfinished_requests())
                )
            )
            if self._closing:
                return (_SHUTDOWN, None)
            if self._sync_requested and not self._quiesced.is_set():
                return (_DRAIN, None)
            if not self._sync_requested and self._quiesced.is_set():
                return (_RESUME, None)
            admits = self._admit_q
            self._admit_q = []
        # An empty admit list is normal — workers must still step in lockstep
        # while in-flight requests finish; futures were registered in generate().
        return (_ADMIT, admits)

    async def _broadcast(
        self, payload
    ) -> tuple[str, list[tuple[str, list[int], SamplingConfig, str]] | None]:
        """Broadcast (word, admits) from rank 0 to every rank over gloo.

        `broadcast_object_list` pickles + size-headers internally; pinning it to
        the gloo group keeps it on CPU, off the NCCL forward stream.
        """
        box = list(payload) if self._tp_rank == 0 else [None, None]
        await asyncio.to_thread(
            dist.broadcast_object_list,
            box,
            src=0,
            group=self._world_group,
            device=torch.device("cpu"),
        )
        return box[0], box[1]

    def _admit(self, admits: list[tuple[str, list[int], SamplingConfig, str]]) -> None:
        """`add_request` every admitted prompt on this rank (bit-identical args
        on all ranks → schedulers stay in lockstep)."""
        engine_inputs = self._engine.renderer.render_cmpl(
            [{"prompt_token_ids": ids} for _, ids, _, _ in admits]
        )
        for (request_id, _, sampling, _), engine_input in zip(
            admits, engine_inputs, strict=True
        ):
            self._engine.add_request(
                request_id=request_id,
                prompt=engine_input,
                params=self._build_sampling_params(sampling),
            )
        if self._tp_rank == 0:
            self._metrics.append(
                m.Metric("generator/max_in_flight_requests", m.Max(len(self._pending)))
            )

    async def _drain_to_empty(self) -> None:
        """Step until every in-flight request finishes (weight-sync barrier)."""
        with sl.log_trace_span("engine_drain"):
            while self._engine.has_unfinished_requests():
                with torch.no_grad():
                    outputs = self._engine.step()
                self._resolve(outputs)
                await asyncio.sleep(0)

    def _resolve(self, outputs: list[RequestOutput]) -> None:
        """Resolve finished requests' futures (rank 0 holds them; a no-op on workers)."""
        for output in outputs:
            entry = self._pending.pop(str(output.request_id), None)
            if entry is None:
                continue
            future, metrics_prefix = entry
            self._metrics.extend(
                _prepare_generation_request_metrics(output, prefix=metrics_prefix)
            )
            if future.done():
                continue
            sample = output.outputs[0]
            self._metrics.append(
                m.Metric(
                    f"{metrics_prefix}/output_tokens", m.Sum(len(sample.token_ids))
                )
            )
            future.set_result(
                Completion(
                    policy_version=self.policy_version,
                    prompt_idx=0,
                    token_ids=list(sample.token_ids),
                    token_logprobs=[
                        next(iter(lp.values())).logprob for lp in sample.logprobs
                    ],
                    finish_reason=sample.finish_reason,
                )
            )

    # --- endpoints ---------------------------------------------------------

    @endpoint
    @sl.log_trace_span("generate")
    async def generate(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        sampling_config: SamplingConfig | None = None,
        metrics_prefix: str = "generator",
    ) -> Completion | None:
        """Generate one completion for a single prompt (n=1).

        Concurrent calls coalesce in the background engine loop. Rank 0 returns
        the `Completion`; other TP ranks return `None` (the controller reads rank 0).
        """
        await self._ensure_engine_loop()
        if self._tp_rank != 0:
            return None
        sampling = (
            sampling_config if sampling_config is not None else self.config.sampling
        )
        future: asyncio.Future[Completion] = asyncio.get_running_loop().create_future()
        async with self._cv:
            self._pending[request_id] = (future, metrics_prefix)
            self._admit_q.append(
                (request_id, list(prompt_token_ids), sampling, metrics_prefix)
            )
            self._cv.notify_all()
        return await future

    @endpoint
    @sl.log_trace_span("pull_model_state_dict")
    async def pull_model_state_dict(self, version: int) -> None:
        """Drain in-flight requests, pull new weights from TorchStore, resume.

        Draining before the swap keeps generation off stale weights and is
        correct even with requests in flight (ready for async producers).
        `direct_rdma=True` reads straight from the trainer's GPU memory.
        """
        from monarch.rdma import is_rdma_available

        await self._ensure_engine_loop()

        if self._tp_rank == 0:
            async with self._cv:
                self._sync_requested = True
                self._cv.notify_all()
        await self._quiesced.wait()
        if self._engine.has_unfinished_requests():
            raise RuntimeError("engine still has unfinished requests after drain")

        if version < self.policy_version or (
            version == self.policy_version and self.policy_version != 0
        ):
            raise RuntimeError(
                f"policy_version must advance: at v{self.policy_version}, "
                f"asked to pull v{version}"
            )

        model_sd = self._get_model().model.state_dict()
        if not model_sd:
            raise RuntimeError("generator model returned an empty state_dict")
        await ts.get_state_dict(
            "model_state_dict",
            user_state_dict=model_sd,
            strict=False,
            direct_rdma=is_rdma_available(),
        )
        for name, tensor in model_sd.items():
            if torch.isnan(tensor).any():
                raise RuntimeError(f"weight {name!r} contains NaNs after pull")
        self.policy_version = version
        # Stale prefix-cache KV was computed under the old weights.
        self._engine.reset_prefix_cache()
        gc.collect()

        if self._tp_rank == 0:
            async with self._cv:
                self._sync_requested = False
                self._cv.notify_all()

    @endpoint
    async def close(self) -> None:
        """Stop the engine loop, fail any stragglers, release the engine."""
        if self._loop_task is not None:
            if self._tp_rank == 0:
                async with self._cv:
                    self._closing = True
                    self._cv.notify_all()
            with contextlib.suppress(Exception):
                await self._loop_task
            self._loop_task = None
        for future, _ in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("generator closed"))
        self._pending.clear()

        if self._engine is not None:
            renderer = getattr(self._engine, "renderer", None)
            try:
                if renderer is not None:
                    renderer.shutdown()
            finally:
                # external_launcher builds may not expose engine_core.shutdown;
                # Monarch owns the process group and tears workers down on stop.
                with contextlib.suppress(AttributeError):
                    self._engine.engine_core.shutdown()
            self._engine = None
