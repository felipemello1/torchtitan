# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for engine-loop dispatch invariants."""

import asyncio
from types import SimpleNamespace

from torchtitan.experiments.rl.actors.generator import _PendingRequest, VLLMGenerator


def _sample(*, token_ids=(10, 11), finish_reason="stop"):
    return SimpleNamespace(
        token_ids=list(token_ids),
        logprobs=[{tok: SimpleNamespace(logprob=-0.1)} for tok in token_ids],
        finish_reason=finish_reason,
    )


def _request_output(*, request_id, outputs):
    return SimpleNamespace(
        request_id=request_id,
        num_cached_tokens=None,
        metrics=None,
        outputs=list(outputs),
    )


def _pending(*, request_id):
    return _PendingRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        sampling_params=SimpleNamespace(),
        future=asyncio.get_running_loop().create_future(),
        metrics_prefix="generator",
        metrics_sink=[],
        admitted_policy_version=7,
    )


def test_resolve_finished_outputs_returns_one_completion_per_request_id():
    """Two request IDs resolve to two completions in request-output order."""

    async def scenario():
        gen = VLLMGenerator.__new__(VLLMGenerator)
        p0 = _pending(request_id="g0:s0:t0")
        p1 = _pending(request_id="g0:s1:t0")
        gen._pending_by_request_id = {p0.request_id: p0, p1.request_id: p1}
        gen._resolve_finished_outputs(
            [
                _request_output(
                    request_id=p0.request_id,
                    outputs=[_sample(token_ids=(10,))],
                ),
                _request_output(
                    request_id=p1.request_id,
                    outputs=[_sample(token_ids=(20,))],
                ),
            ]
        )
        return await asyncio.gather(p0.future, p1.future)

    completions = asyncio.run(scenario())
    assert [c.token_ids for c in completions] == [[10], [20]]
    assert [c.policy_version for c in completions] == [7, 7]


def test_resolve_finished_outputs_maps_abort_finish_reason_to_completion_error():
    """vLLM `abort` becomes `Completion.error` so TokenEnv marks the rollout."""

    async def scenario():
        gen = VLLMGenerator.__new__(VLLMGenerator)
        pending = _pending(request_id="42")
        gen._pending_by_request_id = {"42": pending}
        gen._resolve_finished_outputs(
            [
                _request_output(
                    request_id="42",
                    outputs=[_sample(token_ids=(99,), finish_reason="abort")],
                )
            ]
        )
        return await pending.future

    completion = asyncio.run(scenario())
    assert completion.error is not None
    assert completion.token_ids == [99]
