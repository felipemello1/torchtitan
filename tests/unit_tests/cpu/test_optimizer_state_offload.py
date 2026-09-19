# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import pytest
import torch
from torchtitan.components.optimizer.offload import (
    _pack_params_by_state_bytes,
    _warn_if_numa_affinity_spans_nodes,
    OptimizerStateOffloadConfig,
)


def _params(*numels: int) -> list[torch.Tensor]:
    return [torch.empty(numel) for numel in numels]


def test_packer_keeps_whole_params_and_cuts_before_exceeding_target() -> None:
    # fp32 moments: 8 B per element; target 1 MiB = 131072 elements
    p = _params(60_000, 60_000, 60_000, 10)
    chunks = _pack_params_by_state_bytes(
        p, chunk_size_bytes=1 << 20, state_dtype=torch.float32
    )
    assert [[id(t) for t in chunk] for chunk in chunks] == [
        [id(p[0]), id(p[1])],
        [id(p[2]), id(p[3])],
    ]


def test_packer_bf16_moments_pack_twice_as_many() -> None:
    p = _params(60_000, 60_000, 60_000, 60_000)
    chunks = _pack_params_by_state_bytes(
        p, chunk_size_bytes=1 << 20, state_dtype=torch.bfloat16
    )
    assert [len(chunk) for chunk in chunks] == [4]


def test_packer_oversized_param_forms_its_own_chunk_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    p = _params(10, 500_000, 10)
    with caplog.at_level(logging.WARNING):
        chunks = _pack_params_by_state_bytes(
            p, chunk_size_bytes=1 << 20, state_dtype=torch.float32
        )
    assert [len(chunk) for chunk in chunks] == [1, 1, 1]
    assert "exceed chunk_size_mb" in caplog.text


def test_config_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError):
        OptimizerStateOffloadConfig(chunk_size_mb=0)


@pytest.mark.parametrize(("allowed_cpus", "warns"), [({0, 1}, False), ({0, 4}, True)])
def test_warns_when_cpu_affinity_spans_numa_nodes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    allowed_cpus: set[int],
    warns: bool,
) -> None:
    cpu_lists = []
    for node, contents in enumerate(("0-3", "4-7")):
        path = tmp_path / f"node{node}" / "cpulist"
        path.parent.mkdir()
        path.write_text(contents)
        cpu_lists.append(path)

    monkeypatch.setattr(Path, "glob", lambda _self, _pattern: iter(cpu_lists))
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: allowed_cpus)
    with caplog.at_level(logging.WARNING):
        _warn_if_numa_affinity_spans_nodes()
    assert ("multiple NUMA nodes" in caplog.text) is warns
