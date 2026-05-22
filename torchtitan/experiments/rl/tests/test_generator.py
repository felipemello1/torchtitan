# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the engine-loop dispatch invariants added in PR 1.

End-to-end behavior is covered by the GPU smokes; these tests only pin
the two new shapes the engine loop has to honor:

- ``SamplingParams.n > 1`` returns every sibling completion (RFC §6
  cardinality), not just ``outputs[0]``;
- vLLM ``finish_reason in {"error", "abort"}`` becomes
  ``Completion.error`` so the controller can drop the sample.
"""

import asyncio
from types import SimpleNamespace

from torchtitan.experiments.rl.actors.generator import (
    _PendingRequest,
    VLLMGenerator,
)


def _sample(*, token_ids=(10, 11), finish_reason="stop", text="ok"):
    return SimpleNamespace(
        text=text,
        token_ids=list(token_ids),
        logprobs=[{tok: SimpleNamespace(logprob=-0.1)} for tok in token_ids],
        finish_reason=finish_reason,
    )


def _request_output(*, request_id, outputs):
    # ``metrics`` is None to skip the timing emission path — these tests
    # only care about which completions flow back to which futures.
    return SimpleNamespace(
        request_id=request_id,
        num_cached_tokens=None,
        metrics=None,
        outputs=list(outputs),
    )


def _pending(*, request_id, prompt_idx):
    return _PendingRequest(
        request_id=request_id,
        prompt_idx=prompt_idx,
        prompt_token_ids=[1, 2],
        sampling_params=SimpleNamespace(),
        future=asyncio.get_running_loop().create_future(),
        metrics_prefix="generator",
        metrics_sink=[],
        admitted_policy_version=7,
    )


def test_resolve_finished_outputs_flattens_n_siblings_in_input_order():
    """RFC §6 cardinality: n=3 over 2 prompts must yield 6 completions.

    The scalar-future shape used by the reference branch's final form
    silently dropped ``n - 1`` siblings per request when
    ``SamplingParams.n > 1``; catching that regression is the whole
    point of PR 1's temporary list-of-completions future.
    """

    async def scenario():
        gen = VLLMGenerator.__new__(VLLMGenerator)
        p0 = _pending(request_id="0", prompt_idx=0)
        p1 = _pending(request_id="1", prompt_idx=1)
        gen._pending_by_request_id = {"0": p0, "1": p1}
        gen._resolve_finished_outputs(
            [
                _request_output(
                    request_id="0",
                    outputs=[
                        _sample(token_ids=(10,), text="a"),
                        _sample(token_ids=(11,), text="b"),
                        _sample(token_ids=(12,), text="c"),
                    ],
                ),
                _request_output(
                    request_id="1",
                    outputs=[
                        _sample(token_ids=(20,), text="d"),
                        _sample(token_ids=(21,), text="e"),
                        _sample(token_ids=(22,), text="f"),
                    ],
                ),
            ]
        )
        per_prompt = await asyncio.gather(p0.future, p1.future)
        return [c for siblings in per_prompt for c in siblings]

    completions = asyncio.run(scenario())
    assert [c.prompt_idx for c in completions] == [0, 0, 0, 1, 1, 1]
    assert [c.text for c in completions] == ["a", "b", "c", "d", "e", "f"]


def test_resolve_finished_outputs_maps_abort_finish_reason_to_completion_error():
    """vLLM ``abort`` becomes ``Completion.error`` so the controller drops
    the sample instead of feeding empty text to the env."""

    async def scenario():
        gen = VLLMGenerator.__new__(VLLMGenerator)
        pending = _pending(request_id="42", prompt_idx=2)
        gen._pending_by_request_id = {"42": pending}
        gen._resolve_finished_outputs(
            [
                _request_output(
                    request_id="42",
                    outputs=[
                        _sample(token_ids=(99,), finish_reason="abort", text="")
                    ],
                )
            ]
        )
        return await pending.future

    completions = asyncio.run(scenario())
    assert len(completions) == 1
    assert completions[0].error is not None
    assert completions[0].prompt_idx == 2
