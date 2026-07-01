# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Canonical asyncio-task names for the RL loop.

These name the long-lived tasks the controller and generator create, so the gantt
renderer can pin / group them by a stable identity instead of Python's auto-names
(``Task-9``). Kept import-free (just strings) so the hot training path can import
it without pulling in the renderer. The render policy that *uses* these lives in
``torchtitan/experiments/rl/gantt.py``.
"""

TRAINER_TASK_NAME = "trainer_loop"
BATCHER_TASK_NAME = "batcher_loop"
DATA_INPUT_TASK_NAME = "data_input_loop"
WEIGHT_SYNC_MANAGER_TASK_NAME = "weight_sync_manager"
VLLM_ENGINE_TASK_NAME = "vllm_engine"

ROLLOUT_WORKER_TASK_NAME_PREFIX = "rollout_worker"


def rollout_worker_task_name(worker_id: int) -> str:
    """Task name for the worker that owns active rollout buffer slot ``worker_id``.

    Example:

        rollout_worker_task_name(0)  # -> "rollout_worker_0"
    """
    return f"{ROLLOUT_WORKER_TASK_NAME_PREFIX}_{worker_id}"
