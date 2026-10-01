# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verifiers' Harbor taskset and env, with what Terminal-Bench on a remote sandbox needs.

Verifiers resolves a taskset ID by importing a top-level module, and a dotted ID
fails that lookup, so importing this module registers it under ``TASKSET_ID``.
Importing it also registers the agent-outside harness, whose ID Verifiers
resolves the same way.
"""

import re
import sys
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import (
    HarborConfig,
    HarborEnv,
    HarborTask,
    HarborTaskset,
)

from .agent_outside import AgentOutsideHarnessConfig, sandoq_task_context

TASKSET_ID = __name__.replace(".", "_").lower()
sys.modules.setdefault(TASKSET_ID, sys.modules[__name__])

_WORKDIR_DIRECTIVE = re.compile(r"\s*WORKDIR\s+(\S+)", re.IGNORECASE)


class TerminalTasksetConfig(HarborConfig):
    """Harbor taskset config; a local class so its module is the Verifiers plugin."""


class TerminalTaskset(HarborTaskset, vf.Taskset[HarborTask, TerminalTasksetConfig]):
    """Verifiers' Harbor taskset, with each task's image WORKDIR filled in."""

    config: TerminalTasksetConfig

    def load(self) -> Iterator[HarborTask]:
        # Harbor reads no workdir from a Terminal-Bench task.toml, so every task
        # would start in the runtime's default /app. Three of the 89 images use
        # another WORKDIR, and Sandoq fails a session whose workdir is missing.
        for task in super().load():
            workdir = task.data.workdir or image_workdir(Path(task.data.task_dir))
            yield HarborTask(
                task.data.model_copy(update={"workdir": workdir}), self.config.task
            )


class TerminalBenchEnv(HarborEnv):
    """Harbor's env, plus what the agent-outside harness and Sandoq need."""

    def _runs_local(self) -> bool:
        # The agent-outside loop calls the model from this process, so a remote
        # sandbox needs no tunnel back to it.
        return (
            isinstance(self.config.agent.harness, AgentOutsideHarnessConfig)
            or super()._runs_local()
        )

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        with sandoq_task_context(
            instance_id=task.data.name,
            requested_image=task.data.image,
            working_dir=task.data.workdir or self.config.agent.runtime.workdir,
        ):
            await super().run(task, agents)


def image_workdir(task_dir: Path) -> str | None:
    """Return the last WORKDIR in the task's Dockerfile, which its published image keeps.

    Example: ``WORKDIR /app`` followed by ``WORKDIR /app/dclm`` returns
    ``"/app/dclm"``; a task without ``environment/Dockerfile`` returns None.
    """
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return None
    workdirs = [
        match.group(1)
        for line in dockerfile.read_text(errors="replace").splitlines()
        if (match := _WORKDIR_DIRECTIVE.match(line))
    ]
    return workdirs[-1] if workdirs else None


# Verifiers discovers the env from this taskset plugin.
__all__ = ["TerminalTaskset", "TerminalBenchEnv"]
