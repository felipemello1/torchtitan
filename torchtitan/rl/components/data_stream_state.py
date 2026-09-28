# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The controller's data state, saved next to each trainer checkpoint so a
resumed RL run continues its data stream instead of restarting it."""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.rl.rollout.rollouter import Rollouter

logger = logging.getLogger(__name__)

_FILENAME = "rl_data_stream_state.pkl"
_VERSION = 1


class DataStreamState:
    """Everything the controller needs to continue its data stream after a resume.

    The trainer checkpoint restores model, optimizer, LR scheduler and step. This
    restores the rest, from a file in the same ``step-N`` checkpoint folder:

    - the train and validation dataset positions (``Rollouter.state_dict``);
    - the next group id, so ids keep increasing across restarts;
    - the prompts admitted but not yet trained. They are replayed first, with
      their original group ids, and regenerated with the restored policy.

    Lifecycle:
        group_id, sample = await state.admit_next(read_sample)  # _data_input_loop
        state.consume(batch.group_ids)          # _trainer_loop, after the batch's optimizer step
        await state.save(step_dir, rollouter)   # when that optimizer step wrote a checkpoint
        state.load(step_dir, rollouter)         # setup_async, when resuming from step_dir

    Example (2 prompts per step, checkpoint at step 1):
        admit groups 0..5, train [0, 1]  -> save: next_group_id=6, pending=[2, 3, 4, 5]
        resume -> the data input loop admits 2, 3, 4, 5 again, then new samples as 6, 7, ...
    """

    def __init__(self) -> None:
        # Held while a sample is read and admitted, and while a snapshot is taken, so
        # a snapshot never counts a read sample (dataset position) it has not admitted.
        self._lock = asyncio.Lock()
        self._next_group_id = 0
        self._admitted: dict[int, object] = {}  # insertion (= admission) order
        self._replay: deque[tuple[int, object]] = deque()

    async def admit_next(
        self, read_sample: Callable[[], Awaitable[object]]
    ) -> tuple[int, object]:
        """Return the next ``(group_id, sample)`` to admit: a replayed prompt, else a new sample."""
        async with self._lock:
            if self._replay:
                group_id, sample = self._replay.popleft()
            else:
                sample = await read_sample()
                group_id = self._next_group_id
                self._next_group_id += 1
            self._admitted[group_id] = sample
            return group_id, sample

    def consume(self, group_ids: Iterable[int]) -> None:
        """Forget groups whose batch was trained; they are not replayed after a resume."""
        for group_id in group_ids:
            self._admitted.pop(group_id, None)

    async def save(self, step_dir: str, rollouter: Rollouter) -> None:
        """Write the data state into the trainer's ``step_dir``, atomically."""
        async with self._lock:
            state = {
                "version": _VERSION,
                "datasets": rollouter.state_dict(),
                "next_group_id": self._next_group_id,
                # Admitted groups come first: none were read while replays remained.
                "pending": [*self._admitted.items(), *self._replay],
            }
            payload = pickle.dumps(state)
        await asyncio.to_thread(_write_atomically, step_dir, payload)

    def load(self, step_dir: str, rollouter: Rollouter) -> bool:
        """Restore the data state saved in ``step_dir``; False if the checkpoint has none."""
        path = os.path.join(step_dir, _FILENAME)
        if not os.path.isfile(path):
            logger.warning(
                "No RL data stream state at %s; the datasets restart from their "
                "first samples and group ids from 0.",
                path,
            )
            return False
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state["version"] != _VERSION:
            raise ValueError(
                f"Unsupported RL data stream state version {state['version']} in "
                f"{path}; expected {_VERSION}."
            )
        rollouter.load_state_dict(state["datasets"])
        self._next_group_id = state["next_group_id"]
        self._admitted = {}
        self._replay = deque(state["pending"])
        logger.info(
            "Restored the RL data stream from %s: %d admitted prompts to replay, "
            "next group id %d.",
            path,
            len(self._replay),
            self._next_group_id,
        )
        return True


def _write_atomically(step_dir: str, payload: bytes) -> None:
    os.makedirs(step_dir, exist_ok=True)
    path = os.path.join(step_dir, _FILENAME)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
