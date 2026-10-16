# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serve a TorchTitan `GenerateFn` through vLLM's own HTTP app."""

from __future__ import annotations

import asyncio
import itertools
import logging
import socket
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, TYPE_CHECKING

import uvicorn
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launchers.api_server.app_state import init_app_state
from vllm.entrypoints.launchers.app import build_app
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.entrypoints.launchers.launcher import NoSignalServer
from vllm.exceptions import GracefulHTTPError, VLLMError, VLLMValidationError
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers import renderer_from_config
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.input_processor import InputProcessor

from torchtitan.config import Configurable
from torchtitan.rl.rollout.types import Completion, GenerateFn

if TYPE_CHECKING:
    from torchtitan.rl.generator.generator import SamplingConfig

logger = logging.getLogger(__name__)

GROUP_ID_KEY = "torchtitan_group_id"
"""Key of a request's rollout group id in `extra_args` (`vllm_xargs` for completions/chat)."""

_MODEL_NAME = "torchtitan"

_DEFAULT_SAMPLING_PARAMS = SamplingParams()

# Fields a client may set, then the ones vLLM's routes always set; any other public field
# that differs from its default is a 400.
_ALLOWED_SAMPLING_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "stop_token_ids",
        "logprobs",
        "skip_special_tokens",
        "routed_experts_prompt_start",
        "detokenize",
        "extra_args",
        "output_kind",
        "skip_clone",
    }
)


class GenerateFnEngineClient(EngineClient):
    """vLLM's `EngineClient` over a TorchTitan `GenerateFn`.

    vLLM's app calls `generate` for token-in generate, completions and chat. Each request
    becomes one `GenerateFn` call and one final output, so a stream yields one chunk.
    Everything else the app can reach is a stub that raises.

    Args:
        generate_fn: Runs one generation; owns abort on cancellation.
        vllm_config: Serving config with a tokenizer, for rendering and decoding.
        stop_token_ids: Stops for requests that set none, e.g. the renderer's.
        is_dead: Whether the engine behind `generate_fn` can no longer serve.
    """

    is_running = True
    is_stopped = False

    def __init__(
        self,
        *,
        generate_fn: GenerateFn,
        vllm_config: VllmConfig,
        stop_token_ids: Sequence[int],
        is_dead: Callable[[], bool],
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.renderer = renderer_from_config(vllm_config)
        self.input_processor = InputProcessor(vllm_config, self.renderer)
        self._generate_fn = generate_fn
        self._stop_token_ids = list(stop_token_ids)
        self._is_dead = is_dead
        self._request_counter = itertools.count()
        self.inflight: set[asyncio.Task] = set()
        """Tasks running a `generate_fn` call; `VLLMHttpServer.close` cancels them."""

    def generate(
        self,
        prompt: Any,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Check one request, then return the generator vLLM iterates for its output.

        A plain `def`, not `async def`, so a bad request fails before any bytes go out:

            bad request   -> raises here         -> 400, even with stream=true
            good request  -> returns the stream  -> generate_fn(request_id="{session}/http={n}")

        Example:

            POST /inference/v1/generate, X-Session-ID: trace-7,
            {"token_ids": [1, 2, 3], "sampling_params": {"max_tokens": 4, "extra_args": {"torchtitan_group_id": 3}}}
            -> generate_fn([1, 2, 3], request_id="trace-7/http=17", group_id=3, routing_session_id="trace-7",
                           sampling_config=SamplingConfig(max_tokens=4, ..., stop_token_ids=<renderer's>))
        """
        if prompt.get("type") == "multimodal":
            raise VLLMValidationError("multimodal inputs are not supported")
        prompt_token_ids = list(prompt["prompt_token_ids"])
        group_id = (sampling_params.extra_args or {}).get(GROUP_ID_KEY)
        if isinstance(group_id, bool) or not isinstance(group_id, int):
            raise VLLMValidationError(
                f"{GROUP_ID_KEY} must be an integer in extra_args (token-in) or vllm_xargs"
            )
        if not session_id:
            raise VLLMValidationError("missing X-Session-ID header")
        max_model_len = self.model_config.max_model_len
        # vLLM needs room for one output token. Verifiers reads this wording as an overlong prompt.
        if len(prompt_token_ids) >= max_model_len:
            raise VLLMValidationError(
                f"This model's maximum context length is {max_model_len} tokens, and the "
                f"prompt has {len(prompt_token_ids)} tokens, leaving no room for an output token"
            )
        unsupported = sorted(
            name
            for name in SamplingParams.__struct_fields__
            if not name.startswith("_")
            and name not in _ALLOWED_SAMPLING_FIELDS
            and getattr(sampling_params, name)
            != getattr(_DEFAULT_SAMPLING_PARAMS, name)
        )
        if unsupported:
            raise VLLMValidationError(f"unsupported sampling parameters: {unsupported}")
        if sampling_params.logprobs not in (None, 0, 1):
            raise VLLMValidationError(
                "only the sampled token's logprob is returned (logprobs <= 1)"
            )
        sampling_config = _to_sampling_config(
            sampling_params, default_stop_token_ids=self._stop_token_ids
        )
        return self._generate(
            prompt_token_ids,
            sampling_params,
            request_id=request_id,
            # Fresh id: the engine loop fails on a repeated in-flight id, and clients can repeat
            # theirs (X-Request-Id).
            internal_request_id=f"{session_id}/http={next(self._request_counter)}",
            group_id=group_id,
            session_id=session_id,
            sampling_config=sampling_config,
        )

    async def _generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        *,
        request_id: str,
        internal_request_id: str,
        group_id: int,
        session_id: str,
        sampling_config: SamplingConfig,
    ) -> AsyncGenerator[RequestOutput, None]:
        task = asyncio.current_task()
        self.inflight.add(task)
        try:
            completion = await self._generate_fn(
                prompt_token_ids,
                request_id=internal_request_id,
                group_id=group_id,
                routing_session_id=session_id,
                sampling_config=sampling_config,
            )
        except (ValueError, VLLMError):
            raise  # vLLM answers these (400 or their own status) and keeps the connection
        except Exception as error:
            # vLLM's catch-all handler re-raises after its 500, which closes the connection.
            logger.exception("generation %s failed", internal_request_id)
            raise GracefulHTTPError(
                f"generation {internal_request_id} failed: {error!r}",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            ) from error
        finally:
            self.inflight.discard(task)
        _check_usable(completion, internal_request_id)
        yield self._to_request_output(
            completion, prompt_token_ids, sampling_params, request_id=request_id
        )

    def _to_request_output(
        self,
        completion: Completion,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        *,
        request_id: str,
    ) -> RequestOutput:
        """Wrap one TorchTitan `Completion` as vLLM's final output for it."""
        logprobs = None
        if sampling_params.logprobs is not None:
            logprobs = [
                {token_id: Logprob(logprob=logprob)}
                for token_id, logprob in zip(
                    completion.token_ids, completion.token_logprobs, strict=True
                )
            ]
        # The engine runs without a tokenizer, so text endpoints get their text here.
        text = ""
        if sampling_params.detokenize:
            text = self.renderer.get_tokenizer().decode(
                completion.token_ids,
                skip_special_tokens=sampling_params.skip_special_tokens,
            )
        output = CompletionOutput(
            index=0,
            text=text,
            token_ids=completion.token_ids,
            cumulative_logprob=None,
            logprobs=logprobs,
            finish_reason=completion.finish_reason,
        )
        return RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
            outputs=[output],
            finished=True,
        )

    @property
    def errored(self) -> bool:
        return self._is_dead()

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    async def check_health(self) -> None:
        if self._is_dead():
            raise EngineDeadError()

    async def is_tracing_enabled(self) -> bool:
        return False

    def _unsupported(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    # Not reachable from the routes we serve; fail loudly if a route ever calls them.
    encode = abort = notify_kv_transfer_request_rejected = do_log_stats = _unsupported
    start_profile = stop_profile = reset_mm_cache = reset_encoder_cache = _unsupported
    reset_prefix_cache = sleep = wake_up = is_sleeping = add_lora = _unsupported
    pause_generation = resume_generation = is_paused = shutdown = _unsupported


class VLLMHttpServer(Configurable):
    """vLLM's HTTP app (token-in generate, completions, chat, tokenize, metrics) over a `GenerateFn`.

    Clients must send a group id (`extra_args` / `vllm_xargs` `torchtitan_group_id`) and an
    `X-Session-ID` header. vLLM's routes build each response on this event loop: 74 ms of CPU
    for an 8K-token completion with logprobs, against 8.5 ms for a hand-built response.
    The process must set `VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1` before it builds a vLLM engine, or
    `start` raises (`VLLMGenerator` sets it).

    Example:

        server = VLLMHttpServer.Config().build()
        await server.start(generate_fn=fn, hf_assets_path=path, max_model_len=8192, stop_token_ids=[151645])
        server.url  # "http://127.0.0.1:41237"
        await server.close()  # cancels in-flight calls (their GenerateFn aborts), then stops
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Where the server listens and extra `vllm serve` arguments."""

        host: str = "127.0.0.1"
        """Interface to bind, and the address `url` advertises. The server has no authentication.

            "127.0.0.1"          this machine only; advertised as is
            "0.0.0.0", "::"      every interface, IPv4 and IPv6; advertised as this host's FQDN
            "localhost", "node"  the first address family the name resolves to; advertised as is
        """

        port: int = 0
        """Port to bind; zero picks a free one."""

        serve_args: list[str] = field(default_factory=list)
        """Extra `vllm serve` arguments for the app, e.g. `["--tool-call-parser", "hermes"]`;
        `["--tokens-only"]` skips decoding text on the token-in route."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client: GenerateFnEngineClient | None = None
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task | None = None
        self._bound_port: int | None = None

    @property
    def url(self) -> str:
        """Address clients use, e.g. `"http://127.0.0.1:41237"`."""
        host = self.config.host
        if host in ("0.0.0.0", "::"):
            host = socket.getfqdn()
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self._bound_port}"

    async def start(
        self,
        *,
        generate_fn: GenerateFn,
        hf_assets_path: str,
        max_model_len: int,
        stop_token_ids: Sequence[int],
        is_dead: Callable[[], bool],
    ) -> None:
        """Build the app over `generate_fn` and return once the server is listening.

        Args:
            generate_fn: Serves every request.
            hf_assets_path: HF checkpoint directory; only its config and tokenizer are read.
            max_model_len: Context length; longer prompts are a 400.
            stop_token_ids: Stops for requests that set none (the renderer's).
            is_dead: Reports the engine behind `generate_fn` dead to `/health` and requests.
        """

        def raise_context_limit(hf_config):
            # As the generator's engine config: max_model_len may exceed the checkpoint's
            # max_position_embeddings (in `text_config` for multimodal checkpoints).
            hf_config.get_text_config().max_position_embeddings = max_model_len
            return hf_config

        vllm_config = EngineArgs(
            model=hf_assets_path,
            max_model_len=max_model_len,
            served_model_name=_MODEL_NAME,
            # Neutral defaults: "auto" would apply the checkpoint's top_p/top_k to chat requests.
            generation_config="vllm",
            # Without this, /tokenize fetches image URLs for multimodal checkpoints.
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
            hf_overrides=raise_context_limit,
        ).create_engine_config()
        args = make_arg_parser(FlexibleArgumentParser()).parse_args(
            [
                "--model",
                hf_assets_path,
                "--served-model-name",
                _MODEL_NAME,
                *self.config.serve_args,
            ]
        )
        if args.api_key:
            raise ValueError(
                "--api-key is not supported: vLLM's key guards only some routes"
            )
        self._client = GenerateFnEngineClient(
            generate_fn=generate_fn,
            vllm_config=vllm_config,
            stop_token_ids=stop_token_ids,
            is_dead=is_dead,
        )
        app = build_app(args, ("generate",), vllm_config.model_config)
        if not any(
            getattr(route, "path", None) == "/inference/v1/generate"
            for route in app.routes
        ):
            raise RuntimeError(
                "vLLM's token-in route is missing: set VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1 "
                "before this process builds a vLLM engine"
            )
        await init_app_state(self._client, app.state, args, ("generate",))
        # Bind here: on a bind error uvicorn calls sys.exit, which unwinds the caller's loop.
        host = self.config.host
        dualstack = host in ("0.0.0.0", "::") and socket.has_dualstack_ipv6()
        bind_host = "::" if dualstack else host
        sock = socket.create_server(
            (bind_host, self.config.port),
            family=socket.getaddrinfo(bind_host, None, type=socket.SOCK_STREAM)[0][0],
            dualstack_ipv6=dualstack,
        )
        # NoSignalServer: uvicorn must not take the actor process's SIGINT/SIGTERM.
        # lifespan="off": vLLM's lifespan would freeze this process's GC heap.
        self._server = NoSignalServer(
            uvicorn.Config(
                app, log_level="warning", lifespan="off", timeout_graceful_shutdown=5
            )
        )
        # vLLM's error handlers read the server from app state.
        app.state.server = self._server
        self._serve_task = asyncio.create_task(self._server.serve(sockets=[sock]))
        self._bound_port = sock.getsockname()[1]
        while not self._server.started:
            if self._serve_task.done():
                self._serve_task.result()  # raises the startup error, if any
                raise RuntimeError("vLLM HTTP server exited before it started")
            await asyncio.sleep(0.01)

    async def close(self) -> None:
        """Cancel in-flight calls (each `GenerateFn` aborts its own), then stop serving."""
        for task in list(self._client.inflight):
            task.cancel()
        self._server.should_exit = True
        await self._serve_task
        self._client.renderer.shutdown()


def _to_sampling_config(
    sampling_params: SamplingParams, *, default_stop_token_ids: list[int]
) -> SamplingConfig:
    """Map vLLM's checked sampling parameters to TorchTitan's `SamplingConfig`."""
    from torchtitan.rl.generator.generator import SamplingConfig

    return SamplingConfig(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        max_tokens=sampling_params.max_tokens,
        seed=sampling_params.seed,
        # vLLM sends [] when the request sets none.
        stop_token_ids=list(sampling_params.stop_token_ids) or default_stop_token_ids,
    )


def _check_usable(completion: Completion | None, request_id: str) -> None:
    """Raise a 502 for an empty, inconsistent or aborted completion."""
    if completion is None:
        reason = "no completion"
    elif len(completion.token_ids) != len(completion.token_logprobs):
        reason = "token ids and logprobs have different lengths"
    elif completion.finish_reason not in ("stop", "length"):
        reason = f"finish reason {completion.finish_reason!r}"
    else:
        return
    raise GracefulHTTPError(
        f"generation {request_id} finished without a usable completion: {reason}",
        HTTPStatus.BAD_GATEWAY,
    )
