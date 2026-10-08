# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from types import SimpleNamespace

from torchtitan.rl.components.data import IterableRLDataLoader, RLDataset
from torchtitan.rl.rollout.rollouter import Rollouter


class _ValidationConfig:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = values
        self.num_builds = 0

    def build(self):
        self.num_builds += 1
        return iter(self.values)


def _rollouter(validation_dataset: _ValidationConfig) -> Rollouter:
    rollouter = Rollouter.__new__(Rollouter)
    rollouter._config = SimpleNamespace(validation_dataset=validation_dataset)
    return rollouter


def test_validation_steps_bound_iterable_source() -> None:
    config = _ValidationConfig((1, 2, 3))
    rollouter = _rollouter(config)

    assert rollouter.get_validation_samples(2) == [1, 2]
    assert rollouter.get_validation_samples(5) == [1, 2, 3]
    assert config.num_builds == 2


def test_validation_minus_one_consumes_one_finite_pass() -> None:
    rollouter = _rollouter(_ValidationConfig((1, 2, 3)))

    assert rollouter.get_validation_samples(-1) == [1, 2, 3]


class _RecordsSolvedDataset(RLDataset):
    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.position = 0
        self.solved: list[int] = []

    def __iter__(self):
        return self

    def __next__(self) -> int:
        self.position += 1
        return self.position - 1

    def mark_solved(self, sample: int) -> None:
        self.solved.append(sample)

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state_dict) -> None:
        self.position = state_dict["position"]


def _training_rollouter() -> Rollouter:
    rollouter = Rollouter.__new__(Rollouter)
    rollouter._training_dataloader = IterableRLDataLoader.Config(
        dataset=_RecordsSolvedDataset.Config()
    ).build()
    return rollouter


def test_acknowledgement_passes_solved_samples_to_the_dataset() -> None:
    rollouter = _training_rollouter()
    assert [rollouter.get_training_sample() for _ in range(3)] == [
        (0, 0),
        (1, 1),
        (2, 2),
    ]
    rollouter.acknowledge_training_sample_ids([0, 1], solved_ids=[1])
    assert rollouter._training_dataloader._dataset.solved == [1]

    # After a resume, the unacknowledged sample 2 is replayed, then acknowledged as solved.
    resumed = _training_rollouter()
    resumed.load_state_dict(rollouter.state_dict())
    assert resumed.get_training_sample() == (2, 2)
    resumed.acknowledge_training_sample_ids([2], solved_ids=[2])
    assert resumed._training_dataloader._dataset.solved == [2]
