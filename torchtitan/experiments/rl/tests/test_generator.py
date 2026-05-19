# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``VLLMGenerator.generate``.

Exercises the endpoint in isolation by swapping in a fake vLLM engine —
no Monarch, no GPU, no real model. Covers the token-in / token-out
contract, the metric payload (timing math, edge cases, prefix override),
and the completion/metrics separation.
"""

import asyncio
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.actors import generator as generator_module
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.sampling import SamplingConfig


class _FakeRenderer:
    """Stub for vLLM's Renderer.render_cmpl.

    Mirrors the real shape: takes a list of ``{"prompt_token_ids": ids}``
    dicts and returns a list of typed ``TokensInput`` EngineInputs with
    ``type="token"`` plus a stamped ``arrival_time``.
    """

    def render_cmpl(self, prompts):
        return [
            {
                "type": "token",
                "prompt_token_ids": p["prompt_token_ids"],
                "arrival_time": 0.0,
            }
            for p in prompts
        ]


class _FakeEngine:
    def __init__(self, outputs):
        self.outputs = outputs
        self.add_requests = []
        self._stepped = False
        self.renderer = _FakeRenderer()

    def add_request(self, *args, **kwargs):
        self.add_requests.append((args, kwargs))

    def has_unfinished_requests(self):
        return not self._stepped

    def step(self):
        self._stepped = True
        return self.outputs


def _sample(*, index=0, token_ids=(10, 11), finish_reason="stop", logprobs=None):
    return SimpleNamespace(
        index=index,
        text="ok",
        token_ids=list(token_ids),
        logprobs=(
            logprobs
            if logprobs is not None
            else [{tok: SimpleNamespace(logprob=-0.1)} for tok in token_ids]
        ),
        finish_reason=finish_reason,
    )


def _request_output(
    *,
    request_id="0",
    prompt_token_ids=(1, 2),
    outputs=None,
    num_generation_tokens=4,
):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(prompt_token_ids),
        num_cached_tokens=0,
        metrics=SimpleNamespace(
            first_token_latency=0.012,
            queued_ts=1.0,
            scheduled_ts=1.005,
            first_token_ts=1.017,
            last_token_ts=1.047,
            num_generation_tokens=num_generation_tokens,
        ),
        outputs=list(outputs or [_sample()]),
    )


def _generator(outputs):
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator._engine = _FakeEngine(outputs)
    generator._engine_lock = asyncio.Lock()
    generator._pending_outputs = {}
    generator._engine_driver_task = None
    generator._next_request_id = 0
    generator.policy_version = 7
    generator.config = SimpleNamespace(
        sampling=SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4),
        debug=SimpleNamespace(seed=None),
    )
    return generator


def _run_generate(generator, tokenized_prompts, **kwargs):
    return asyncio.run(
        VLLMGenerator.generate._method(
            generator, tokenized_prompts, **kwargs
        )  # noqa: SLF001
    )


def test_generate_passes_token_prompt_to_vllm():
    generator = _generator([_request_output(prompt_token_ids=[1, 2, 3])])

    _run_generate(generator, [[1, 2, 3]])

    # add_request is invoked entirely with kwargs. The ``prompt`` kwarg
    # carries the renderer's output: a typed EngineInput (``type="token"``)
    # with the prompt token IDs and a stamped ``arrival_time`` — keeping us
    # off vLLM's deprecated raw-prompt path.
    (_args, kwargs) = generator._engine.add_requests[0]
    assert kwargs["request_id"] == "0"
    assert kwargs["prompt"]["type"] == "token"
    assert kwargs["prompt"]["prompt_token_ids"] == [1, 2, 3]
    assert "arrival_time" in kwargs["prompt"]


def test_generate_passes_request_ids_and_restores_input_order():
    generator = _generator(
        [
            _request_output(
                request_id="group:b",
                prompt_token_ids=[4],
                outputs=[_sample(token_ids=(40,))],
            ),
            _request_output(
                request_id="group:a",
                prompt_token_ids=[3],
                outputs=[_sample(token_ids=(30,))],
            ),
        ]
    )

    completions, _ = _run_generate(
        generator,
        [[3], [4]],
        request_ids=["group:a", "group:b"],
    )

    assert [
        kwargs["request_id"] for _args, kwargs in generator._engine.add_requests
    ] == ["group:a", "group:b"]
    assert [completion.token_ids for completion in completions] == [[30], [40]]


def test_generate_validates_request_ids():
    generator = _generator([_request_output()])

    with pytest.raises(ValueError, match="length must match"):
        _run_generate(generator, [[1], [2]], request_ids=["only-one"])

    with pytest.raises(ValueError, match="must be unique"):
        _run_generate(generator, [[1], [2]], request_ids=["dup", "dup"])


def test_generate_rejects_unknown_returned_request_id():
    generator = _generator([_request_output(request_id="unexpected")])

    with pytest.raises(RuntimeError, match="unknown request_id"):
        _run_generate(generator, [[1]], request_ids=["expected"])


def test_generate_rejects_unknown_returned_request_id_without_hanging_ready_output():
    async def run() -> None:
        generator = _generator(
            [
                _request_output(request_id="expected"),
                _request_output(request_id="unexpected"),
            ]
        )

        with pytest.raises(RuntimeError, match="unknown request_id"):
            await asyncio.wait_for(
                VLLMGenerator.generate._method(
                    generator,
                    [[1], [2]],
                    request_ids=["expected", "other"],
                ),
                timeout=1,
            )

    asyncio.run(run())


def test_generate_carries_finish_reason_and_metrics():
    output = _request_output(
        outputs=[
            _sample(index=0, token_ids=(10, 11), finish_reason="length"),
            _sample(index=1, token_ids=(12,), finish_reason="stop"),
        ]
    )
    generator = _generator([output])

    completions, generation_metrics = _run_generate(generator, [[1, 2]])

    assert [c.finish_reason for c in completions] == ["length", "stop"]
    assert [c.policy_version for c in completions] == [7, 7]
    assert not hasattr(completions[0], "metrics")
    aggregate = m.MetricsProcessor._aggregate_metrics(generation_metrics)
    assert aggregate["generator/output_tokens/sum"] == 3
    assert aggregate["generator/num_cached_tokens/mean"] == 0
    assert aggregate["generator/num_cached_tokens/max"] == 0
    assert aggregate["generator/time_to_first_token_ms/mean"] == 12
    assert aggregate["generator/time_to_first_token_ms/max"] == 12
    assert aggregate["generator/queue_time_ms/mean"] == pytest.approx(5)
    assert aggregate["generator/queue_time_ms/max"] == pytest.approx(5)
    assert aggregate["generator/prefill_time_ms/mean"] == pytest.approx(12)
    assert aggregate["generator/prefill_time_ms/max"] == pytest.approx(12)
    assert aggregate["generator/decode_time_ms/mean"] == pytest.approx(30)
    assert aggregate["generator/decode_time_ms/max"] == pytest.approx(30)
    assert aggregate["generator/inter_token_latency_ms/mean"] == pytest.approx(10)
    assert aggregate["generator/inter_token_latency_ms/max"] == pytest.approx(10)
    assert "generator/e2e_latency_ms/mean" not in aggregate


def test_generate_uses_sampled_token_logprob_not_first_dict_entry():
    output = _request_output(
        outputs=[
            _sample(
                token_ids=(10,),
                logprobs=[
                    {
                        9: SimpleNamespace(logprob=-9.0),
                        10: SimpleNamespace(logprob=-0.25),
                    }
                ],
            )
        ]
    )
    generator = _generator([output])

    [completion], _ = _run_generate(generator, [[1, 2]])

    assert completion.token_logprobs == [-0.25]


def test_generate_metrics_prefix_override_namespaces_keys():
    output = _request_output(
        outputs=[_sample(index=0, token_ids=(10, 11))],
    )
    generator = _generator([output])

    _, generation_metrics = _run_generate(
        generator, [[1, 2]], metrics_prefix="validation/generator"
    )

    metric_keys = {metric.key for metric in generation_metrics}
    assert "validation/generator/output_tokens" in metric_keys
    assert "validation/generator/queue_time_ms" in metric_keys
    assert all(key.startswith("validation/generator/") for key in metric_keys)


def test_decode_metrics_are_absent_for_single_generated_token():
    generator = _generator(
        [
            _request_output(
                outputs=[_sample(index=0, token_ids=(10,))],
                num_generation_tokens=1,
            )
        ]
    )

    _, generation_metrics = _run_generate(generator, [[1, 2]])

    metric_keys = {metric.key for metric in generation_metrics}
    assert "generator/prefill_time_ms" in metric_keys
    assert "generator/decode_time_ms" not in metric_keys
    assert "generator/inter_token_latency_ms" not in metric_keys


def test_generate_detects_policy_version_change_during_request():
    generator = _generator([_request_output()])
    step = generator._engine.step

    def mutate_version_during_step():
        generator.policy_version += 1
        return step()

    generator._engine.step = mutate_version_during_step

    with pytest.raises(RuntimeError, match="policy_version changed"):
        _run_generate(generator, [[1, 2]])


def test_generate_aborts_partially_admitted_requests_after_add_failure():
    class PartiallyFailingEngine(_FakeEngine):
        def __init__(self):
            super().__init__(outputs=[])
            self.active_request_ids: list[str] = []
            self.abort_requests: list[tuple[list[str], bool]] = []

        def add_request(self, *args, **kwargs):
            if kwargs["request_id"] == "bad":
                raise RuntimeError("add failed")
            super().add_request(*args, **kwargs)
            self.active_request_ids.append(kwargs["request_id"])

        def abort_request(self, request_ids, internal=False):
            request_ids = list(request_ids)
            self.abort_requests.append((request_ids, internal))
            for request_id in request_ids:
                self.active_request_ids.remove(request_id)

        def has_unfinished_requests(self):
            return bool(self.active_request_ids)

    generator = _generator([])
    generator._engine = PartiallyFailingEngine()

    with pytest.raises(RuntimeError, match="add failed"):
        _run_generate(generator, [[1], [2]], request_ids=["ok", "bad"])

    assert generator._engine.abort_requests == [(["ok"], True)]
    assert generator._engine.active_request_ids == []
    assert generator._pending_outputs == {}
    assert generator._engine_driver_task is None


def test_generate_admits_new_request_while_prior_request_is_decoding():
    async def run() -> None:
        first_step_seen = asyncio.Event()
        allow_finish = asyncio.Event()

        class ContinuousFakeEngine(_FakeEngine):
            def __init__(self):
                super().__init__(outputs=[])
                self.pending_request_ids: list[str] = []
                self.max_pending_during_step = 0

            def add_request(self, *args, **kwargs):
                super().add_request(*args, **kwargs)
                self.pending_request_ids.append(kwargs["request_id"])

            def has_unfinished_requests(self):
                return bool(self.pending_request_ids)

            def step(self):
                first_step_seen.set()
                self.max_pending_during_step = max(
                    self.max_pending_during_step,
                    len(self.pending_request_ids),
                )
                if not allow_finish.is_set():
                    return []
                request_id = self.pending_request_ids.pop(0)
                return [
                    _request_output(
                        request_id=request_id,
                        outputs=[_sample(token_ids=(int(request_id),))],
                    )
                ]

        generator = _generator([])
        generator._engine = ContinuousFakeEngine()

        first = asyncio.create_task(
            VLLMGenerator.generate._method(
                generator,
                [[1]],
                request_ids=["1"],
            )
        )
        await first_step_seen.wait()

        second = asyncio.create_task(
            VLLMGenerator.generate._method(
                generator,
                [[2]],
                request_ids=["2"],
            )
        )
        while len(generator._engine.add_requests) < 2:
            await asyncio.sleep(0)
        assert generator._engine.pending_request_ids == ["1", "2"]

        allow_finish.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result[0][0].token_ids == [1]
        assert second_result[0][0].token_ids == [2]
        assert generator._engine.max_pending_during_step == 2

    asyncio.run(run())


def test_pull_model_state_dict_rejects_while_generation_is_pending(monkeypatch):
    async def run() -> None:
        first_step_seen = asyncio.Event()
        allow_finish = asyncio.Event()
        get_state_dict_calls = []

        class ContinuousFakeEngine(_FakeEngine):
            def __init__(self):
                super().__init__(outputs=[])
                self.pending_request_ids: list[str] = []
                self.reset_prefix_cache_calls = 0

            def add_request(self, *args, **kwargs):
                super().add_request(*args, **kwargs)
                self.pending_request_ids.append(kwargs["request_id"])

            def has_unfinished_requests(self):
                return bool(self.pending_request_ids)

            def step(self):
                first_step_seen.set()
                if not allow_finish.is_set():
                    return []
                request_id = self.pending_request_ids.pop(0)
                return [
                    _request_output(
                        request_id=request_id,
                        outputs=[_sample(token_ids=(int(request_id),))],
                    )
                ]

            def reset_prefix_cache(self):
                self.reset_prefix_cache_calls += 1

        async def fake_get_state_dict(*args, **kwargs):
            get_state_dict_calls.append((args, kwargs))

        generator = _generator([])
        generator._engine = ContinuousFakeEngine()
        generator._get_model = lambda: SimpleNamespace(
            model=SimpleNamespace(state_dict=lambda: {})
        )
        monkeypatch.setattr(
            generator_module.ts,
            "get_state_dict",
            fake_get_state_dict,
        )
        import monarch.rdma as rdma

        monkeypatch.setattr(rdma, "is_rdma_available", lambda: False)

        generate_task = asyncio.create_task(
            VLLMGenerator.generate._method(
                generator,
                [[1]],
                request_ids=["1"],
            )
        )
        await first_step_seen.wait()

        with pytest.raises(RuntimeError, match="generation requests are active"):
            await VLLMGenerator.pull_model_state_dict._method(generator, version=8)

        allow_finish.set()
        completions, _metrics = await generate_task
        assert completions[0].token_ids == [1]

        await VLLMGenerator.pull_model_state_dict._method(generator, version=8)

        assert generator.policy_version == 8
        assert generator._engine.reset_prefix_cache_calls == 1
        assert len(get_state_dict_calls) == 1

    asyncio.run(run())
