# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`VLLMHttpServer` over a fake `GenerateFn`: vLLM's routes, the request contract, abort."""

import asyncio
import gc
import json
import shutil
import signal
import socket
from pathlib import Path

import pytest

pytest.importorskip("vllm")
aiohttp = pytest.importorskip("aiohttp")

from torchtitan.rl.rollout.types import Completion  # noqa: E402
from torchtitan.rl.serving.http_server import VLLMHttpServer  # noqa: E402

_TOKENIZER_DIR = Path(__file__).parents[3] / "assets" / "tokenizer"
_STOPS = [128001]
_MAX_MODEL_LEN = 64


@pytest.fixture
def hf_assets_path(tmp_path, monkeypatch):
    """A Llama config and the test tokenizer: vLLM reads only these, no weights.

    `max_position_embeddings` is below `_MAX_MODEL_LEN` and the generation config has
    `top_p`/`top_k`, as real checkpoints can; the server must override and ignore them.
    """
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(_TOKENIZER_DIR / name, tmp_path / name)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "max_position_embeddings": 16,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "generation_config.json").write_text('{"top_p": 0.95, "top_k": 20}')
    # Registers vLLM's token-in route; `VLLMGenerator` sets it before its engine starts.
    monkeypatch.setenv("VLLM_ENABLE_SCALE_OUT_ENDPOINTS", "1")
    return str(tmp_path)


class FakeGenerateFn:
    """Records calls; `hold` blocks until cancelled; `error` raises; `finish_reason` shapes the completion."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.hold = False
        self.error: Exception | None = None
        self.finish_reason = "stop"
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def __call__(
        self,
        prompt_token_ids,
        *,
        request_id,
        group_id,
        routing_session_id,
        sampling_config,
    ):
        self.calls.append(
            dict(
                prompt_token_ids=prompt_token_ids,
                request_id=request_id,
                group_id=group_id,
                routing_session_id=routing_session_id,
                sampling_config=sampling_config,
            )
        )
        self.entered.set()
        if self.error is not None:
            raise self.error
        if self.hold:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        return Completion(
            min_policy_version=7,
            max_policy_version=7,
            request_id=request_id,
            token_ids=[104, 105],
            token_logprobs=[-0.5, -0.25],
            finish_reason=self.finish_reason,
        )


def _token_in(group_id=3, stream=False, **sampling):
    sampling.setdefault("max_tokens", 4)
    if group_id is not None:
        sampling["extra_args"] = {"torchtitan_group_id": group_id}
    return {
        "model": "torchtitan",
        "token_ids": [1, 2, 3],
        "sampling_params": sampling,
        "stream": stream,
    }


def _chat(group_id=3, stream=False):
    body = {
        "model": "torchtitan",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }
    if group_id is not None:
        body["vllm_xargs"] = {"torchtitan_group_id": group_id}
    return body


_SESSION = {"X-Session-ID": "trace-7"}


async def _start(hf_assets_path, fake, dead=lambda: False, **config):
    server = VLLMHttpServer.Config(**config).build()
    await server.start(
        generate_fn=fake,
        hf_assets_path=hf_assets_path,
        max_model_len=_MAX_MODEL_LEN,
        stop_token_ids=_STOPS,
        is_dead=dead,
    )
    return server


def _run(hf_assets_path, test_body, dead=lambda: False):
    """Start a server over a `FakeGenerateFn`, run `test_body(base, session, fake)`, close."""

    async def run():
        fake = FakeGenerateFn()
        server = await _start(hf_assets_path, fake, dead)
        try:
            async with aiohttp.ClientSession() as session:
                await test_body(server.url, session, fake)
        finally:
            await server.close()

    asyncio.run(run())


async def _post(session, url, body, headers=_SESSION):
    async with session.post(url, json=body, headers=headers) as response:
        return response.status, await response.text()


def test_every_route_answers(hf_assets_path):
    async def body(base, session, fake):
        for path in ("/health", "/v1/models", "/metrics"):
            async with session.get(base + path) as response:
                assert response.status == 200, path
        assert (await _post(session, base + "/inference/v1/generate", _token_in()))[
            0
        ] == 200
        assert (await _post(session, base + "/v1/chat/completions", _chat()))[0] == 200
        completions = {
            "model": "torchtitan",
            "prompt": [1, 2, 3],
            "max_tokens": 4,
            "vllm_xargs": {"torchtitan_group_id": 3},
        }
        assert (await _post(session, base + "/v1/completions", completions))[0] == 200
        tokenize = {"model": "torchtitan", "prompt": "hi"}
        assert (await _post(session, base + "/tokenize", tokenize))[0] == 200
        # A stream yields the one final output as one chunk.
        status, text = await _post(
            session, base + "/inference/v1/generate", _token_in(stream=True)
        )
        assert status == 200 and "[DONE]" in text

    _run(hf_assets_path, body)


def test_token_in_returns_ids_and_one_logprob_per_token(hf_assets_path):
    async def run():
        fake = FakeGenerateFn()
        server = await _start(hf_assets_path, fake)
        tokenizer = server._client.renderer.get_tokenizer()
        decoded = []
        tokenizer.decode = lambda token_ids, **kwargs: decoded.append(token_ids) or ""
        async with aiohttp.ClientSession() as session:
            status, text = await _post(
                session,
                server.url + "/inference/v1/generate",
                _token_in(logprobs=1, detokenize=False),
            )
            assert decoded == []  # detokenize=False skips the decode
            await _post(session, server.url + "/inference/v1/generate", _token_in())
            assert decoded == [[104, 105]]  # the default still decodes
        await server.close()
        assert status == 200
        choice = json.loads(text)["choices"][0]
        assert choice["token_ids"] == [104, 105]
        assert [entry["logprob"] for entry in choice["logprobs"]["content"]] == [
            -0.5,
            -0.25,
        ]
        call = fake.calls[0]
        assert call["group_id"] == 3 and call["routing_session_id"] == "trace-7"
        assert call["sampling_config"].stop_token_ids == _STOPS  # the request set none

    asyncio.run(run())


def test_request_ids_are_assigned_by_the_server(hf_assets_path):
    # The engine loop fails on a repeated in-flight id, so a client's X-Request-Id never reaches it.
    async def body(base, session, fake):
        headers = {**_SESSION, "X-Request-Id": "dup"}
        url = base + "/inference/v1/generate"
        results = await asyncio.gather(
            *[_post(session, url, _token_in(), headers) for _ in range(2)]
        )
        assert [status for status, _ in results] == [200, 200]
        request_ids = [call["request_id"] for call in fake.calls]
        assert len(set(request_ids)) == 2 and all(
            rid.startswith("trace-7/http=") for rid in request_ids
        )

    _run(hf_assets_path, body)


@pytest.mark.parametrize(
    ("path", "request_body"),
    [
        ("/inference/v1/generate", _token_in()),
        ("/v1/chat/completions", _chat(stream=True)),
    ],
)
def test_client_disconnect_cancels_the_generation(hf_assets_path, path, request_body):
    async def body(base, session, fake):
        fake.hold = True
        request = asyncio.create_task(_post(session, base + path, request_body))
        await asyncio.wait_for(fake.entered.wait(), 10)
        request.cancel()  # aiohttp closes the socket: a real disconnect
        await asyncio.wait_for(fake.cancelled.wait(), 10)

    _run(hf_assets_path, body)


def test_close_cancels_in_flight_generations(hf_assets_path):
    async def run():
        fake = FakeGenerateFn()
        fake.hold = True
        server = await _start(hf_assets_path, fake)
        async with aiohttp.ClientSession() as session:
            pending = asyncio.create_task(
                _post(session, server.url + "/inference/v1/generate", _token_in())
            )
            await asyncio.wait_for(fake.entered.wait(), 10)
            await server.close()
            await pending
        assert fake.cancelled.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_bad_requests_are_400_before_any_output(hf_assets_path, stream):
    async def body(base, session, fake):
        url = base + "/inference/v1/generate"
        image = {
            "mm_hashes": {"image": ["abc"]},
            "mm_placeholders": {"image": [{"offset": 0, "length": 1}]},
        }
        cases = {
            "missing group": (_token_in(group_id=None, stream=stream), _SESSION),
            "missing session": (_token_in(stream=stream), {}),
            "unsupported field": (_token_in(stream=stream, top_k=20), _SESSION),
            "logprobs > 1": (_token_in(stream=stream, logprobs=2), _SESSION),
            "logprobs = -1 (all)": (_token_in(stream=stream, logprobs=-1), _SESSION),
            "top_p < 1": (_token_in(stream=stream, top_p=0.9), _SESSION),
            "multimodal": ({**_token_in(stream=stream), "features": image}, _SESSION),
        }
        for name, (request_body, headers) in cases.items():
            status, _ = await _post(session, url, request_body, headers)
            assert status == 400, name
        overlong = {
            **_token_in(stream=stream),
            "token_ids": list(range(_MAX_MODEL_LEN)),
        }
        status, text = await _post(session, url, overlong)
        assert status == 400 and "leaving no room for an output token" in text
        assert fake.calls == []

    _run(hf_assets_path, body)


def test_generation_errors_keep_the_connection(hf_assets_path):
    async def run():
        fake = FakeGenerateFn()
        server = await _start(hf_assets_path, fake)
        url = server.url + "/inference/v1/generate"
        # One pooled connection: an error status must leave it usable for the next request.
        connector = aiohttp.TCPConnector(limit=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            fake.finish_reason = "abort"
            assert (await _post(session, url, _token_in()))[0] == 502
            fake.error = RuntimeError("router failed")
            assert (await _post(session, url, _token_in()))[0] == 500
            fake.error = ValueError("rejected at admission")
            assert (await _post(session, url, _token_in()))[0] == 400
            fake.error, fake.finish_reason = None, "stop"
            assert (await _post(session, url, _token_in()))[0] == 200
        await server.close()

    asyncio.run(run())


def test_verifiers_request_passes_and_plain_chat_gets_neutral_defaults(hf_assets_path):
    async def body(base, session, fake):
        # What the Verifiers client sends on every turn.
        verifiers = _token_in(
            temperature=0.7,
            top_p=1.0,
            seed=3,
            stop_token_ids=[128009],
            logprobs=1,
            skip_special_tokens=False,
            routed_experts_prompt_start=2,
        )
        assert (await _post(session, base + "/inference/v1/generate", verifiers))[
            0
        ] == 200
        assert fake.calls[-1]["sampling_config"].stop_token_ids == [128009]
        # The checkpoint's generation config (top_p 0.95, top_k 20) is not applied.
        assert (await _post(session, base + "/v1/chat/completions", _chat()))[0] == 200
        sampling = fake.calls[-1]["sampling_config"]
        assert sampling.top_p == 1.0 and sampling.stop_token_ids == _STOPS

    _run(hf_assets_path, body)


def test_dead_engine_is_reported(hf_assets_path):
    async def body(base, session, fake):
        async with session.get(base + "/health") as response:
            assert response.status == 503
        status, text = await _post(
            session, base + "/inference/v1/generate", _token_in()
        )
        assert status == 500 and "EngineCore" in text and fake.calls == []

    _run(hf_assets_path, body, dead=lambda: True)


def test_start_keeps_signal_handlers_and_gc(hf_assets_path):
    async def run():
        handlers = [signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)]
        freeze_count = gc.get_freeze_count()
        server = await _start(hf_assets_path, FakeGenerateFn())
        assert [
            signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)
        ] == handlers
        assert (
            gc.get_freeze_count() == freeze_count
        )  # vLLM's lifespan would freeze the heap
        await server.close()

    asyncio.run(run())


def test_bind(hf_assets_path):
    async def run():
        with socket.create_server(("127.0.0.1", 0)) as busy:
            with pytest.raises(OSError):
                await _start(
                    hf_assets_path, FakeGenerateFn(), port=busy.getsockname()[1]
                )
        every_interface = await _start(hf_assets_path, FakeGenerateFn(), host="0.0.0.0")
        port = every_interface.url.rsplit(":", 1)[1]
        localhost = await _start(hf_assets_path, FakeGenerateFn(), host="localhost")
        async with aiohttp.ClientSession() as session:
            for url in (every_interface.url, f"http://127.0.0.1:{port}", localhost.url):
                async with session.get(url + "/health") as response:
                    assert response.status == 200, url
        await every_interface.close()
        await localhost.close()

    asyncio.run(run())


def test_nested_text_config_takes_the_context_limit(hf_assets_path, tmp_path):
    # Multimodal checkpoints keep max_position_embeddings in text_config; 16 is below the tests' 64.
    nested = tmp_path / "nested"
    shutil.copytree(hf_assets_path, nested)
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text", "max_position_embeddings": 16},
    }
    (nested / "config.json").write_text(json.dumps(config))

    async def run():
        server = await _start(str(nested), FakeGenerateFn())
        assert server._client.model_config.max_model_len == _MAX_MODEL_LEN
        await server.close()

    asyncio.run(run())


def test_start_needs_the_token_in_route(hf_assets_path, monkeypatch):
    monkeypatch.delenv("VLLM_ENABLE_SCALE_OUT_ENDPOINTS")
    with pytest.raises(RuntimeError, match="VLLM_ENABLE_SCALE_OUT_ENDPOINTS"):
        asyncio.run(_start(hf_assets_path, FakeGenerateFn()))


def test_api_key_is_rejected(hf_assets_path):
    # vLLM's key guards only some routes; any spelling its parser accepts is refused.
    with pytest.raises(ValueError, match="api-key"):
        asyncio.run(
            _start(hf_assets_path, FakeGenerateFn(), serve_args=["--api_key=secret"])
        )
