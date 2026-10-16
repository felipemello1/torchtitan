# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verifiers' EnvClient, holding in-flight traces' per-token fields as numpy arrays."""

from __future__ import annotations

import numpy as np
from verifiers.v1.configs.client import ClientConfig
from verifiers.v1.episode import WireEpisode
from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.delta import EpisodeAssembly, unpack
from verifiers.v1.serve.types import RunRequest, RunResponse
from verifiers.v1.types import SamplingConfig

# Per-token node fields and their numpy dtypes. msgpack decodes ints as Python ints and
# floats as float64, so `tolist()` gives back the exact values.
_TOKEN_FIELDS = {
    "token_ids": np.int64,
    "mask": np.bool_,
    "is_content": np.bool_,
    "logprobs": np.float64,
}


class NumpyEpisodeAssembly(EpisodeAssembly):
    """`EpisodeAssembly` that stores each node's per-token lists as numpy arrays.

    A full GC visits every element of a Python list. An in-flight 30-turn SWE rollout holds
    ~200k of them (token ids, masks, logprobs); numpy arrays are one object with no references.
    `finish` returns the same record as `EpisodeAssembly.finish`, with lists again.

    Example:
        assembly = NumpyEpisodeAssembly()
        assembly.apply({"trace": "t", "open": {"id": "t"}, "nodes": [{"token_ids": [5, 6]}]})
        assembly.traces["t"]["nodes"][0]["token_ids"]  # -> array([5, 6])
    """

    def apply(self, delta: dict) -> None:
        trace = self.traces.get(delta["trace"])
        num_nodes = 0 if trace is None else len(trace["nodes"])
        super().apply(delta)
        trace = self.traces.get(delta["trace"])
        if trace is None:  # discarded
            return
        # `apply` appends copies of the delta's nodes, so the delta itself stays as received.
        for node in trace["nodes"][num_nodes:]:
            for field, dtype in _TOKEN_FIELDS.items():
                if isinstance(node.get(field), list):
                    node[field] = np.asarray(node[field], dtype=dtype)

    def finish(self, head: dict, summaries: list) -> dict:
        record = super().finish(head, summaries)
        for trace in record["traces"]:
            trace["nodes"] = [
                {
                    key: value.tolist()
                    if key in _TOKEN_FIELDS and isinstance(value, np.ndarray)
                    else value
                    for key, value in node.items()
                }
                for node in trace["nodes"]
            ]
        return record


class NumpyEnvClient(EnvClient):
    """`EnvClient` whose `run` assembles the streamed trace with `NumpyEpisodeAssembly`.

    The body is `EnvClient.run` (verifiers 0.3.2) with the assembly class swapped.
    """

    async def run(
        self,
        client: ClientConfig,
        model: str,
        sampling: SamplingConfig,
        task_data: dict,
        on_delta=None,
    ) -> WireEpisode:
        assembly = NumpyEpisodeAssembly()

        def apply(data: bytes) -> None:
            delta = unpack(data)
            assembly.apply(delta)
            if on_delta is not None:
                on_delta(delta)

        response = await self._request(
            RunRequest(task_data=task_data, client=client, model=model, sampling=sampling),
            RunResponse,
            on_delta=apply,
        )
        assert response.head is not None
        return await self._validate_episode(
            assembly.finish(response.head, response.traces)
        )
