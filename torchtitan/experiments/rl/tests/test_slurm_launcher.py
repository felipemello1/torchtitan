# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

from torchtitan.experiments.rl import slurm_launcher


class _FakeSlurmJob:
    def __init__(self) -> None:
        self.apply_kwargs = None
        self.hostname_at_apply = None

    def apply(self, **kwargs) -> None:
        self.apply_kwargs = kwargs
        self.hostname_at_apply = os.environ.get("HOSTNAME")


@pytest.mark.parametrize("client_script", [None, "python -m example"])
def test_apply_omits_submission_hostname(monkeypatch, client_script) -> None:
    monkeypatch.setenv("HOSTNAME", "submission-node")
    job = _FakeSlurmJob()

    slurm_launcher._apply_without_inherited_hostname(
        job, client_script=client_script
    )

    assert job.hostname_at_apply is None
    assert os.environ["HOSTNAME"] == "submission-node"
    expected_kwargs = {} if client_script is None else {"client_script": client_script}
    assert job.apply_kwargs == expected_kwargs


def test_apply_restores_hostname_after_submission_failure(monkeypatch) -> None:
    class _FailingSlurmJob:
        def apply(self) -> None:
            assert "HOSTNAME" not in os.environ
            raise RuntimeError("submission failed")

    monkeypatch.setenv("HOSTNAME", "submission-node")

    with pytest.raises(RuntimeError, match="submission failed"):
        slurm_launcher._apply_without_inherited_hostname(_FailingSlurmJob())

    assert os.environ["HOSTNAME"] == "submission-node"
