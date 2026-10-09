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

import types

import pytest

from torchtitan.rl.components.data_stream_state import (
    _FILENAME,
    DataStreamState,
    newest_step_with_data_stream_state,
)
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


def _bare_rollouter(
    train_dataset,
    validation_dataset,
    *,
    seed: int = 42,
    validation_config: str = "validation",
) -> Rollouter:
    """A Rollouter with only the fields ``state_dict`` / ``load_state_dict`` read."""
    rollouter = Rollouter.__new__(Rollouter)
    rollouter._config = types.SimpleNamespace(
        train_dataset=f"train(seed={seed})", validation_dataset=validation_config
    )
    rollouter._train_dataset = train_dataset
    rollouter._validation_dataset = validation_dataset
    return rollouter


def test_rollouter_restores_positions_and_skips_datasets_without_state(
    caplog,
) -> None:
    rollouter = _bare_rollouter(_CountingDataset(), iter([0, 1, 2]))  # no state_dict
    next(rollouter._train_dataset)
    saved = rollouter.state_dict()
    assert saved["train"] == {"position": 1}
    assert saved["validation"] is None

    resumed = _bare_rollouter(_CountingDataset(), iter([0, 1, 2]))
    with caplog.at_level(logging.WARNING):
        resumed.load_state_dict(saved)
    assert resumed.get_training_sample() == 1
    assert "validation dataset; it restarts" in caplog.text


def test_rollouter_warns_on_positions_saved_for_other_datasets(caplog) -> None:
    saved = _bare_rollouter(_CountingDataset(), iter([])).state_dict()
    resumed = _bare_rollouter(_CountingDataset(), iter([]), seed=7)
    with caplog.at_level(logging.WARNING):
        resumed.load_state_dict(saved)
    assert "dataset config differs from the checkpoint's" in caplog.text


def test_rollouter_restarts_validation_saved_for_other_datasets(caplog) -> None:
    """A position saved for another validation set would index the new set by the old
    set's rows; validation restarts instead, while training resumes."""
    rollouter = _bare_rollouter(_CountingDataset(), _CountingDataset())
    next(rollouter._train_dataset)
    next(rollouter._validation_dataset)
    saved = rollouter.state_dict()

    resumed = _bare_rollouter(
        _CountingDataset(), _CountingDataset(), validation_config="math_eval_suite"
    )
    with caplog.at_level(logging.WARNING):
        resumed.load_state_dict(saved)
    assert resumed.get_training_sample() == 1
    assert resumed.get_validation_sample() == 0
    assert "restarting validation from its first sample" in caplog.text

    # Same datasets: validation resumes too.
    unchanged = _bare_rollouter(_CountingDataset(), _CountingDataset())
    unchanged.load_state_dict(saved)
    assert unchanged.get_validation_sample() == 1


class _Checkpointer:
    """``_find_load_step`` / ``_create_checkpoint_id`` over ``step-N`` folders in ``folder``.

    A folder is a resumable trainer checkpoint when it has a ``.metadata`` file.
    """

    def __init__(self, folder) -> None:
        self.folder = str(folder)

    def _create_checkpoint_id(self, step: int) -> str:
        return os.path.join(self.folder, f"step-{step}")

    def _find_load_step(self, max_step: int | None = None) -> int:
        steps = [
            int(name.removeprefix("step-"))
            for name in os.listdir(self.folder)
            if os.path.isfile(os.path.join(self.folder, name, ".metadata"))
        ]
        steps = [s for s in steps if max_step is None or s <= max_step]
        return max(steps, default=-1)


def _write_step(folder, step: int, *, data_stream_state: bool) -> None:
    step_dir = folder / f"step-{step}"
    step_dir.mkdir()
    (step_dir / ".metadata").touch()
    if data_stream_state:
        (step_dir / _FILENAME).touch()


def test_resume_step_skips_steps_without_data_stream_state(tmp_path, caplog) -> None:
    _write_step(tmp_path, 2, data_stream_state=True)
    _write_step(tmp_path, 4, data_stream_state=True)
    _write_step(tmp_path, 6, data_stream_state=False)  # killed between the two saves
    with caplog.at_level(logging.WARNING):
        assert newest_step_with_data_stream_state(_Checkpointer(tmp_path)) == 4
    assert "resuming from step 4" in caplog.text


def test_resume_step_falls_back_to_newest_without_any_data_stream_state(
    tmp_path,
) -> None:
    checkpointer = _Checkpointer(tmp_path)
    assert newest_step_with_data_stream_state(checkpointer) == -1
    _write_step(tmp_path, 2, data_stream_state=False)
    _write_step(tmp_path, 4, data_stream_state=False)
    assert newest_step_with_data_stream_state(checkpointer) == 4
