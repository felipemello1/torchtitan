# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for resuming the RL controller's data stream from a checkpoint."""

import asyncio
import logging
import os
import pickle

import pytest

from torchtitan.rl.components.data_stream_state import _FILENAME, DataStreamState
from torchtitan.rl.rollout.rollouter import Rollouter


class _CountingDataset:
    """Yields 0, 1, 2, ...; its position is the next value."""

    def __init__(self) -> None:
        self.position = 0

    def __next__(self) -> int:
        value = self.position
        self.position += 1
        return value

    def state_dict(self) -> dict:
        return {"position": self.position}

    def load_state_dict(self, state_dict: dict) -> None:
        self.position = state_dict["position"]


class _Rollouter:
    """The two Rollouter methods DataStreamState calls, over one counting dataset."""

    def __init__(self) -> None:
        self.dataset = _CountingDataset()

    def state_dict(self) -> dict:
        return {"train": self.dataset.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.dataset.load_state_dict(state_dict["train"])


def _admit(state: DataStreamState, rollouter: _Rollouter, num_groups: int) -> list:
    async def read_sample() -> int:
        return next(rollouter.dataset)

    async def run() -> list:
        return [await state.admit_next(read_sample) for _ in range(num_groups)]

    return asyncio.run(run())


def test_resume_replays_untrained_prompts_then_continues_the_stream(tmp_path) -> None:
    # Run 1: 6 prompts admitted, the step-1 batch trained groups 0 and 1, checkpoint.
    state, rollouter = DataStreamState(), _Rollouter()
    assert _admit(state, rollouter, 6) == [(i, i) for i in range(6)]
    state.consume([0, 1])
    asyncio.run(state.save(str(tmp_path), rollouter))

    # Run 2 resumes from that checkpoint.
    resumed_state, resumed_rollouter = DataStreamState(), _Rollouter()
    assert resumed_state.load(str(tmp_path), resumed_rollouter)
    assert _admit(resumed_state, resumed_rollouter, 6) == [
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 7),
    ]


def test_save_during_replay_keeps_the_prompts_not_yet_replayed(tmp_path) -> None:
    state, rollouter = DataStreamState(), _Rollouter()
    _admit(state, rollouter, 4)
    asyncio.run(state.save(str(tmp_path / "step-1"), rollouter))

    resumed_state, resumed_rollouter = DataStreamState(), _Rollouter()
    resumed_state.load(str(tmp_path / "step-1"), resumed_rollouter)
    # Groups 0 and 1 are replayed and trained; 2 and 3 are still waiting to be replayed.
    _admit(resumed_state, resumed_rollouter, 2)
    resumed_state.consume([0, 1])
    asyncio.run(resumed_state.save(str(tmp_path / "step-2"), resumed_rollouter))

    final_state, final_rollouter = DataStreamState(), _Rollouter()
    final_state.load(str(tmp_path / "step-2"), final_rollouter)
    assert _admit(final_state, final_rollouter, 3) == [(2, 2), (3, 3), (4, 4)]


def test_resume_without_saved_state_restarts_the_stream(tmp_path, caplog) -> None:
    state, rollouter = DataStreamState(), _Rollouter()
    with caplog.at_level(logging.WARNING):
        assert not state.load(str(tmp_path), rollouter)
    assert "No RL data stream state" in caplog.text
    assert _admit(state, rollouter, 2) == [(0, 0), (1, 1)]


def test_save_replaces_the_file_atomically(tmp_path) -> None:
    state, rollouter = DataStreamState(), _Rollouter()
    _admit(state, rollouter, 1)
    asyncio.run(state.save(str(tmp_path), rollouter))
    asyncio.run(state.save(str(tmp_path), rollouter))
    assert sorted(os.listdir(tmp_path)) == [_FILENAME]


def test_load_rejects_an_unknown_version(tmp_path) -> None:
    with open(tmp_path / _FILENAME, "wb") as f:
        pickle.dump({"version": 999}, f)
    with pytest.raises(ValueError, match="version 999"):
        DataStreamState().load(str(tmp_path), _Rollouter())


def test_rollouter_restores_positions_and_skips_datasets_without_state(
    caplog,
) -> None:
    rollouter = Rollouter.__new__(Rollouter)
    rollouter._train_dataset = _CountingDataset()
    rollouter._validation_dataset = iter([0, 1, 2])  # no state_dict
    next(rollouter._train_dataset)
    saved = rollouter.state_dict()
    assert saved == {"train": {"position": 1}, "validation": None}

    resumed = Rollouter.__new__(Rollouter)
    resumed._train_dataset = _CountingDataset()
    resumed._validation_dataset = iter([0, 1, 2])
    with caplog.at_level(logging.WARNING):
        resumed.load_state_dict(saved)
    assert resumed.get_training_sample() == 1
    assert "validation dataset; it restarts" in caplog.text
