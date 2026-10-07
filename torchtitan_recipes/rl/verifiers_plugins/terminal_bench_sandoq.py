# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verifiers plugin that runs pytorch/torchtitan#4942's Terminal-Bench recipe on Sandoq VMs.

The PR runs Verifiers' stock Terminus-2 program in a Docker container per rollout, where it calls
the model through the interception server. A Sandoq VM cannot connect back to the job, so
Terminus-2 runs in the env-server process with the same stock arguments and only its shell
commands (tmux) go to the VM through the Verifiers runtime. The taskset is the PR's
``TerminalTaskset`` plus what sandoq_provider needs: each task's image WORKDIR and the oci-runner
task context while the VM is leased.

A port of Jiani Wang's ``jw_tb_pr4942_sandoq`` (manifold
torchtrain_datasets/tree/jianiw/tb_sweep), plus ``install_shell_refresh``. The module is both the
harness and the taskset plugin. Verifiers imports plugin ids as top-level modules, so this
directory goes on PYTHONPATH:

    TerminalTasksetConfig(id=PLUGIN_ID, dataset=...)
    StockTerminusOutsideConfig(id=PLUGIN_ID)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shlex
import tempfile
import time
import traceback
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import verifiers.v1 as vf
from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from torchtitan.rl.examples.verifiers.terminal_bench.taskset import TerminalTaskset
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.tasksets.harbor import HarborEnv, HarborTask
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

PLUGIN_ID = __name__
# Each Sandoq rollout owns its VM, so one fixed directory cannot collide.
_SANDBOX_TMUX_DIR = "/tmp/vf-terminus-2"
_WORKDIR_DIRECTIVE = re.compile(r"\s*WORKDIR\s+(\S+)", re.IGNORECASE)
# Sandoq reaps a persistent shell unused for 1,800 s; refresh it with margin to spare.
_SHELL_REFRESH_IDLE_S = 1500.0


def sandbox_runtime() -> vf.PrimeConfig:
    """Verifiers' Prime runtime, which sandoq_provider serves with one VM per rollout."""
    if os.environ.get("VF_SANDBOX_PROVIDER") != "oci-runner":
        raise ValueError(f"{PLUGIN_ID} needs VF_SANDBOX_PROVIDER=oci-runner")
    # Each task's tests install pytest over the network; without host networking
    # every reward is a silent 0.
    if os.environ.get("OCI_RUNNER_TASK_NETWORK") != "host":
        raise ValueError(
            "VF_SANDBOX_PROVIDER=oci-runner needs OCI_RUNNER_TASK_NETWORK=host"
        )
    return vf.PrimeConfig(idle_timeout=3600)


class StockTerminusOutsideConfig(HarnessConfig):
    """Terminus-2 with Verifiers' stock constructor arguments, run in the env-server process."""

    interleaved_thinking: bool = False
    """Send each turn's reasoning back in the chat history. Set it with a thinking renderer:
    Verifiers keys assistant messages on their reasoning, so a history without it forks a new
    branch, a separate training sample holding the full context, at every turn."""


class RuntimeEnvironment:
    """Harbor environment whose shell commands run in a Verifiers runtime.

    Mirrors the ``LocalEnvironment`` in Verifiers' Terminus-2 program, with
    ``subprocess.run`` replaced by ``runtime.run``. ``user`` is ignored there too.
    """

    default_user = None
    session_id = "verifiers"

    def __init__(self, runtime: Runtime, env: dict[str, str]) -> None:
        self._runtime = runtime
        self._env = env

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = user
        if cwd is not None:
            command = f"cd {shlex.quote(cwd)} && {command}"
        result = await asyncio.wait_for(
            self._runtime.run(["sh", "-c", command], {**self._env, **(env or {})}),
            timeout_sec,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code
        )

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return (
            await self.exec(f"test -d {shlex.quote(path)}", user=user)
        ).return_code == 0


class StockTerminusOutsideHarness(Harness[StockTerminusOutsideConfig]):
    """Run Terminus-2 in this process; its tmux session lives in the rollout's VM."""

    APPENDS_SYSTEM_PROMPT = True
    NEEDS_CONTAINER = False

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("Terminus 2 does not support disabling tools")
        system_prompt, prompt = self.resolve_text_prompt(data)
        if prompt is None:
            raise ValueError("Terminus 2 requires a task prompt")
        try:
            return await self._run_terminus(
                runtime,
                endpoint=endpoint,
                secret=secret,
                model=ctx.model,
                system_prompt=system_prompt,
                prompt=prompt,
            )
        finally:
            try:
                await runtime.run(
                    [
                        "sh",
                        "-c",
                        'tmux kill-server >/dev/null 2>&1 || true; rm -rf "$TMUX_TMPDIR"',
                    ],
                    {"TMUX_TMPDIR": _SANDBOX_TMUX_DIR},
                )
            except Exception:
                logger.warning(
                    "failed to clean up Terminus 2 tmux server", exc_info=True
                )

    async def _run_terminus(
        self,
        runtime: Runtime,
        *,
        endpoint: str,
        secret: str,
        model: str,
        system_prompt: str | None,
        prompt: str,
    ) -> ProgramResult:
        """Failures become a nonzero ``ProgramResult``, like the program crashing in the sandbox."""
        environment = RuntimeEnvironment(runtime, {"TMUX_TMPDIR": _SANDBOX_TMUX_DIR})
        await environment.exec(f"mkdir -p -m 700 {_SANDBOX_TMUX_DIR}")
        # Terminus reads the sandbox-side log directory from this class attribute;
        # every rollout uses the same sandbox path.
        EnvironmentPaths.agent_dir = PurePosixPath(_SANDBOX_TMUX_DIR)
        with tempfile.TemporaryDirectory(prefix="vf-terminus-2-") as logs_dir:
            try:
                agent = Terminus2(
                    logs_dir=Path(logs_dir),
                    model_name=model,
                    api_base=endpoint,
                    llm_kwargs={"custom_llm_provider": "openai", "api_key": secret},
                    record_terminal_session=False,
                    interleaved_thinking=self.config.interleaved_thinking,
                )
                if system_prompt:
                    call = agent._llm.call

                    async def call_with_system_prompt(*args, message_history, **kwargs):
                        return await call(
                            *args,
                            message_history=[
                                {"role": "system", "content": system_prompt},
                                *message_history,
                            ],
                            **kwargs,
                        )

                    agent._llm.call = call_with_system_prompt
                await agent.setup(environment)
                await agent.run(prompt, environment, AgentContext())
            except Exception:  # noqa: BLE001 - reported like a crashed program
                return ProgramResult(
                    exit_code=1, stdout="", stderr=traceback.format_exc()
                )
        return ProgramResult(exit_code=0, stdout="", stderr="")


class SandoqTerminalTaskset(TerminalTaskset):
    """The PR's taskset, with each task's image WORKDIR filled in."""

    def load(self) -> Iterator[HarborTask]:
        # Without a task.toml workdir, use the image's last WORKDIR: the runtime
        # default (/app) is missing in some images, and a Sandoq VM fails then.
        for task in super().load():
            workdir = task.data.workdir or image_workdir(Path(task.data.task_dir))
            yield HarborTask(
                task.data.model_copy(update={"workdir": workdir}), self.config.task
            )


class SandoqTerminalBenchEnv(HarborEnv):
    """Harbor's env; Terminus-2 calls the model from this process, and each rollout's VM gets
    its task's image and workdir."""

    def _runs_local(self) -> bool:
        return (
            isinstance(self.config.agent.harness, StockTerminusOutsideConfig)
            or super()._runs_local()
        )

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        with oci_runner_task_context(
            instance_id=task.data.name,
            requested_image=task.data.image,
            working_dir=task.data.workdir or self.config.agent.runtime.workdir,
        ):
            await super().run(task, agents)


@contextlib.contextmanager
def oci_runner_task_context(**task_fields: object) -> Iterator[None]:
    """Give sandoq_provider this rollout's image and workdir while its VM is leased."""
    from sandoq_provider import install, registry

    install()
    install_shell_refresh()
    token = registry.bind_task_context(task_fields)
    try:
        yield
    finally:
        registry.reset_task_context(token)


def install_shell_refresh() -> None:
    """Make sandoq_provider replace a rollout's persistent shell before Sandoq reaps it for idling.

    Terminus-2's commands run as background jobs that never touch the shell, so without this the
    grading upload after ~30 min fails with "persistent shell was lost with HTTP 404". Idempotent.

    TODO: upstream into ram_prime_rl's oci_client, and ask Sandoq for a longer SHELL_IDLE_TTL.
    """
    from sandoq_provider import oci_client
    from sandoq_provider.pool import get_pool_client

    client_cls = oci_client.OCIRunnerAsyncSandboxClient
    if getattr(client_cls, "_shell_refresh_installed", False):
        return
    nested_exec_argv = client_cls._nested_exec_argv

    async def _nested_exec_argv(self, info, argv, **kwargs):
        last_used = info.metadata.get("shell_last_used")
        if (
            info.shell_id is not None
            and last_used is not None
            and time.monotonic() - last_used > _SHELL_REFRESH_IDLE_S
        ):
            # Stamp first: _initialize_shell re-enters this method with the new shell.
            info.metadata["shell_last_used"] = time.monotonic()
            info.shell_id = await self._create_shell(info)
            info.metadata["shell_id"] = info.shell_id
            if info.session_reuse:
                # The pool deletes the assignment's shell when it releases the VM.
                await asyncio.to_thread(
                    get_pool_client().update, info.session_id, shell_id=info.shell_id
                )
            await self._initialize_shell(info)
            # Verifiers routes only its own loggers, so INFO would not reach the job log.
            logger.warning(
                "sandoq: replaced the idle persistent shell of %s", info.session_id
            )
        try:
            return await nested_exec_argv(self, info, argv, **kwargs)
        finally:
            info.metadata["shell_last_used"] = time.monotonic()

    client_cls._nested_exec_argv = _nested_exec_argv
    client_cls._shell_refresh_installed = True


def image_workdir(task_dir: Path) -> str | None:
    """Return the last WORKDIR in the task's Dockerfile, which its published image keeps."""
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return None
    workdirs = [
        match.group(1)
        for line in dockerfile.read_text(errors="replace").splitlines()
        if (match := _WORKDIR_DIRECTIVE.match(line))
    ]
    return workdirs[-1] if workdirs else None


__all__ = [
    "StockTerminusOutsideHarness",
    "SandoqTerminalTaskset",
    "SandoqTerminalBenchEnv",
]
