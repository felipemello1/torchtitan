# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import inspect
import random

import numpy as np
import pytest

pytest.importorskip("verifiers.v1.serve.delta")  # verifiers >= 0.3.2 streams traces

from torchtitan.rl.examples.verifiers.env_client import NumpyEpisodeAssembly
from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.delta import EpisodeAssembly, pack, unpack, TraceSummary


def _node(rng: random.Random, sampled: bool) -> dict:
    n = rng.randint(0, 40)
    return {
        "message": {"role": "assistant" if sampled else "tool", "content": "x"},
        "sampled": sampled,
        "token_ids": [rng.randrange(150_000) for _ in range(n)],
        "mask": [sampled and rng.random() < 0.8 for _ in range(n)],
        "is_content": [rng.random() < 0.5 for _ in range(n)],
        "logprobs": [-rng.random() for _ in range(n)] if sampled else [],
        "semantic_parents": [],
    }


def _deltas(rng: random.Random) -> list[dict]:
    """Two traces' deltas over 6 turns, plus a discarded trace, as the wire delivers them."""
    deltas = []
    for trace_id in ("a", "b", "gone"):
        deltas.append({"trace": trace_id, "open": {"id": trace_id}, "pending": []})
    for turn in range(6):
        for trace_id in ("a", "b"):
            delta = {"trace": trace_id, "nodes": [_node(rng, False), _node(rng, True)]}
            if turn:
                delta["links"] = {"0": [1]}
            delta["calls"] = [{"turn": turn}]
            deltas.append(delta)
        deltas.append({"trace": "a", "pending": [{"role": "user", "content": "p"}]})
    deltas.append({"trace": "gone", "discard": True})
    deltas.append({"trace": "b", "set": {"rewards": {"r": 1.0}}})
    # msgpack round trip: the values the client really decodes.
    return [unpack(pack(delta)) for delta in deltas]


def test_numpy_assembly_finishes_to_the_same_record() -> None:
    rng = random.Random(0)
    deltas = _deltas(rng)
    reference, numpy_assembly = EpisodeAssembly(), NumpyEpisodeAssembly()
    for delta in deltas:
        reference.apply(unpack(pack(delta)))
        numpy_assembly.apply(unpack(pack(delta)))

    held = numpy_assembly.traces["a"]["nodes"][-1]
    assert isinstance(held["token_ids"], np.ndarray)
    assert isinstance(held["logprobs"], np.ndarray)

    summaries = [
        TraceSummary(id=trace_id, nodes=12, calls=6) for trace_id in ("a", "b")
    ]
    head = {"version": 1}
    expected = reference.finish(head, summaries)
    actual = numpy_assembly.finish(head, summaries)
    assert actual == expected
    for trace in actual["traces"]:
        for node in trace["nodes"]:
            assert all(type(v) is int for v in node["token_ids"])
            assert all(type(v) is bool for v in node["mask"] + node["is_content"])
            assert all(type(v) is float for v in node["logprobs"])


# sha256 of `inspect.getsource(EnvClient.run)` in verifiers 0.3.2 to 0.4.0, which
# `NumpyEnvClient.run` copies.
_ENV_CLIENT_RUN_SHA256_PREFIX = "ac2fa8410a4e28af"


def test_env_client_run_is_the_copied_version() -> None:
    source = inspect.getsource(EnvClient.run)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    assert digest == _ENV_CLIENT_RUN_SHA256_PREFIX, (
        "verifiers' EnvClient.run changed. NumpyEnvClient.run "
        "(torchtitan/rl/examples/verifiers/env_client.py) is a copy of it with the assembly class "
        "swapped: re-check the copy against the new source, then update this hash."
    )
