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
import gzip
import json
import logging
import os
import re
import shlex
import tempfile
import time
import traceback
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import verifiers.v1 as vf
from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import ExecResult
from harbor.llms.base import ContextLengthExceededError
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.logger import logger as harbor_logger

from torchtitan.rl.examples.verifiers.terminal_bench.taskset import (
    TerminalTaskset,
    TerminalTasksetConfig,
)
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError, SandboxError
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.tasksets.harbor import HarborEnv, HarborTask
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

PLUGIN_ID = __name__
# Each Sandoq rollout owns its VM, so one fixed directory cannot collide.
_SANDBOX_TMUX_DIR = "/tmp/vf-terminus-2"
# Terminus-2 pipes its tmux pane here (`EnvironmentPaths.agent_dir / "terminus_2.pane"`).
_PANE_LOG = f"{_SANDBOX_TMUX_DIR}/terminus_2.pane"
_WORKDIR_DIRECTIVE = re.compile(r"\s*WORKDIR\s+(\S+)", re.IGNORECASE)
# Sandoq reaps a persistent shell unused for 1,800 s; refresh it with margin to spare.
_SHELL_REFRESH_IDLE_S = 1500.0
# Terminus-2 installs tmux with apt-get (120 s timeout) when the image has none, and the mirrors
# time out under load. Sandoq already streamed a static tmux into the container for the provider's
# shell; put a verified copy on PATH. If any step fails, Terminus-2 falls back to apt-get as before.
_USE_SANDOQ_TMUX = (
    "command -v tmux >/dev/null || { src=$(ls /tmp/.sandoq-shell/*/tmux 2>/dev/null | head -n 1); "
    '[ -n "$src" ] && cp "$src" /usr/local/bin/.tmux.tmp && /usr/local/bin/.tmux.tmp -V >/dev/null '
    "&& mv /usr/local/bin/.tmux.tmp /usr/local/bin/tmux; }"
)
# A rollout log keeps at most this many characters of each text field, its head and tail.
# Terminus-2 already caps each terminal output in its trajectory at 10,000 bytes.
_MAX_LOG_CHARS = 65536
# The log lines of the rollout whose launch is running in this asyncio task.
_launch_log_lines: ContextVar[list[str] | None] = ContextVar(
    "launch_log_lines", default=None
)
_LOG_FORMATTER = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")


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

    rollout_log_dir: str | None = None
    """Write each rollout's log to `<rollout_log_dir>/<trace id>.json.gz`; None writes none.
    See `StockTerminusOutsideHarness.cleanup`."""


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
        if "\0" in command:
            # Harbor's subprocess exec raises this too. Sandoq would answer HTTP 500, a
            # SandboxError the agent's retries rerun.
            raise ValueError("embedded null byte")
        # Sandoq keeps `setsid bash -lc '<argv>'` alive for the whole exec, so an agent's
        # `pkill -f x` would kill the exec whose keystrokes contain x. An env var is in no argv.
        result = await asyncio.wait_for(
            self._runtime.run(
                ["sh", "-c", 'eval "$TERMINUS_EXEC"'],
                {**self._env, **(env or {}), "TERMINUS_EXEC": command},
            ),
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

    def __init__(self, config: StockTerminusOutsideConfig) -> None:
        super().__init__(config)
        # Kept by each rollout's launch for its cleanup, by trace id.
        self._launch_logs: dict[str, LaunchLog] = {}

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
        return await self._run_terminus(
            runtime,
            trace,
            endpoint=endpoint,
            secret=secret,
            model=ctx.model,
            system_prompt=system_prompt,
            prompt=prompt,
        )

    async def _run_terminus(
        self,
        runtime: Runtime,
        trace: Trace,
        *,
        endpoint: str,
        secret: str,
        model: str,
        system_prompt: str | None,
        prompt: str,
    ) -> ProgramResult:
        """Raises ``SandboxError`` for a lost exec channel, which the agent's retries rerun on a
        fresh VM, and ``HarnessError`` for anything else Terminus-2 raised."""
        environment = RuntimeEnvironment(runtime, {"TMUX_TMPDIR": _SANDBOX_TMUX_DIR})
        await environment.exec(
            f"mkdir -p -m 700 {_SANDBOX_TMUX_DIR}; {_USE_SANDOQ_TMUX}"
        )
        # Terminus reads the sandbox-side log directory from this class attribute;
        # every rollout uses the same sandbox path.
        EnvironmentPaths.agent_dir = PurePosixPath(_SANDBOX_TMUX_DIR)
        launch_log = self._launch_logs[trace.id] = LaunchLog()
        with tempfile.TemporaryDirectory(prefix="vf-terminus-2-") as logs_dir:
            log_lines_token = _launch_log_lines.set(launch_log.terminus_log_lines)
            try:
                agent = Terminus2(
                    logs_dir=Path(logs_dir),
                    model_name=model,
                    api_base=endpoint,
                    llm_kwargs={"custom_llm_provider": "openai", "api_key": secret},
                    record_terminal_session=False,
                    interleaved_thinking=self.config.interleaved_thinking,
                    # A rollout ends at the context cap instead of Terminus-2 rewriting its history.
                    enable_summarize=False,
                )
                # trajectory.json records these kwargs; LiteLLM keeps its own copy of the key.
                del agent._llm_kwargs["api_key"]
                # A filter sees only its own logger's records, not a child's, so add it to each
                # logger Terminus-2 uses: its tmux session's, its own and LiteLLM's. Idempotent.
                for terminus_logger in (
                    harbor_logger,
                    agent.logger,
                    agent._llm._logger,
                ):
                    terminus_logger.addFilter(_keep_launch_record)
                call = agent._llm.call
                system = (
                    [{"role": "system", "content": system_prompt}]
                    if system_prompt
                    else []
                )

                async def call_with_sampled_history(*args, message_history, **kwargs):
                    if trace.stop_condition:
                        # Verifiers stopped the rollout (context or turn cap) and answers 400
                        # from now on; Terminus-2 retries any error but this one.
                        raise ContextLengthExceededError(
                            f"Verifiers stopped the rollout: {trace.stop_condition}"
                        )
                    return await call(
                        *args,
                        message_history=[
                            *system,
                            *restore_sampled_reasoning(message_history, trace),
                        ],
                        **kwargs,
                    )

                agent._llm.call = call_with_sampled_history
                await agent.setup(environment)
                # Harbor's default session recording runs the agent's shell under asciinema, so
                # an `exit` ends the recording, not the session. Nest a shell to match it.
                await agent._session.send_keys(
                    keys=["bash", "Enter"], min_timeout_sec=1.0
                )
                await agent._session.send_keys(keys=["clear", "Enter"])
                await agent.run(prompt, environment, AgentContext())
            except SandboxError:
                launch_log.harness_stderr = traceback.format_exc()
                raise
            except Exception as error:  # noqa: BLE001 - reported like a crashed program
                launch_log.harness_stderr = traceback.format_exc()
                # Not exit 1: Verifiers would probe `runtime.alive()` and call a container the agent
                # stopped (`pkill -9 -f sleep`) a SandboxError, which the retries would rerun.
                raise HarnessError(
                    f"harness {self.config.id!r} exited 1: "
                    f"{launch_log.harness_stderr.strip()[-2000:]}"
                ) from error
            finally:
                _launch_log_lines.reset(log_lines_token)
                # Missing when Terminus-2 failed before running, e.g. in tmux setup.
                trajectory_path = Path(logs_dir) / "trajectory.json"
                if trajectory_path.exists():
                    launch_log.trajectory = json.loads(trajectory_path.read_text())
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        """Write this rollout's log if `rollout_log_dir` is set, then stop its tmux server.

        Verifiers calls this after scoring, so test.sh still sees the agent's shell jobs
        (`python3 server.py &`), as under `harbor run`. It also calls it when leasing the VM or
        starting tmux failed, so those rollouts leave a log too. The log is gzipped JSON:

            {
                "verifiers_trace_id": "9f2c...",  # logs.verifiers_trace_id in rollout_samples.jsonl
                "task": ..., "task_name": "allenai-tmax/task_000000_c19dda5b",
                "reward": 0.0, "stop_condition": "error",
                "errors": [...],                   # trace.errors, as in Rollout.logs
                "timing": {"setup": {"start": ..., "end": ...}, "agent": {...}, ...},
                "tests": {"exit_code": 1, "stdout": "...", "stderr": ""},  # tests/test.sh
                "harness_stderr": "Traceback ...",  # what Terminus-2 raised; "" if nothing
                "terminus_log": "... WARNING harbor.utils.logger: Tool installation exceeded ...",
                "trajectory": {...},               # Terminus-2's trajectory.json
                "pane": "root@vm:/app# ls ...",    # the tmux pane log's last 64 KB; None if unreadable
            }
        """
        launch_log = self._launch_logs.get(trace.id)
        if self.config.rollout_log_dir is not None:
            # Before kill-server, whose rm -rf deletes the pane log.
            await self._write_rollout_log(trace, runtime, launch_log or LaunchLog())
        # Popped only now, so abort() repeats a cleanup cancelled while reading the pane. None:
        # launch never got to tmux, or an earlier cleanup already stopped it.
        if self._launch_logs.pop(trace.id, None) is None:
            return
        try:
            await runtime.run(
                [
                    "sh",
                    "-c",
                    'timeout 10 tmux kill-server >/dev/null 2>&1 || true; rm -rf "$TMUX_TMPDIR"',
                ],
                {"TMUX_TMPDIR": _SANDBOX_TMUX_DIR},
            )
        except Exception:
            logger.warning("failed to clean up Terminus 2 tmux server", exc_info=True)

    async def _write_rollout_log(
        self, trace: Trace, runtime: Runtime, launch_log: LaunchLog
    ) -> None:
        path = Path(self.config.rollout_log_dir) / f"{trace.id}.json.gz"
        # Verifiers' abort() calls cleanup again after a cancelled close(); keep the full log.
        if path.exists():
            return
        try:
            # The pane log holds every byte the terminal printed; keep its last 64 KB.
            result = await runtime.run(
                ["tail", "-c", str(_MAX_LOG_CHARS), _PANE_LOG], {}
            )
            pane = result.stdout if result.exit_code == 0 else None
        except Exception:  # noqa: BLE001 - e.g. the VM is gone
            pane = None
        record = {
            "verifiers_trace_id": trace.id,
            "task": trace.task.key,
            "task_name": trace.task.data.name,
            "reward": trace.reward,
            "stop_condition": trace.stop_condition,
            "errors": [error.model_dump(mode="json") for error in trace.errors],
            "timing": trace.timing.model_dump(mode="json"),
            "tests": trace.info.get("tests"),
            "harness_stderr": truncate_middle(launch_log.harness_stderr),
            "terminus_log": truncate_middle("\n".join(launch_log.terminus_log_lines)),
            "trajectory": launch_log.trajectory,
            "pane": pane,
        }
        # On the event loop, like Terminus-2's trajectory dump after every turn: ~3 ms per log,
        # 48 ms for T3's largest. Level 6 is as small as the default 9 and ~3x faster.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(record).encode(), compresslevel=6))


@dataclass
class LaunchLog:
    """What a rollout's `launch` keeps for its `cleanup` to write."""

    terminus_log_lines: list[str] = field(default_factory=list)
    """Records Terminus-2, its tmux session and LiteLLM logged during the launch."""
    harness_stderr: str = ""
    """The traceback of what Terminus-2 raised. Verifiers' error keeps its last 2,000 characters."""
    trajectory: dict | None = None
    """Terminus-2's trajectory.json."""


def _keep_launch_record(record: logging.LogRecord) -> bool:
    """Logging filter that copies a record into the running launch's log lines; drops nothing.

    A filter, not a handler, so the job log is unchanged: when no logger up the chain has a
    handler, Python prints warnings to stderr (`logging.lastResort`), and a handler stops that.
    """
    lines = _launch_log_lines.get()
    if lines is not None:
        lines.append(_LOG_FORMATTER.format(record))
    return True


def truncate_middle(text: str, max_chars: int = _MAX_LOG_CHARS) -> str:
    """Keep the head and tail of `text`, `max_chars` in all, as Terminus-2 does for terminal output.

    Example:
        truncate_middle("aaaaXXXXXXbbbb", max_chars=8)
        # -> "aaaa\\n[... 6 characters omitted ...]\\nbbbb"
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[... {omitted} characters omitted ...]\n{text[-half:]}"


def restore_sampled_reasoning(message_history: list[dict], trace: Trace) -> list[dict]:
    """Put the sampled reasoning back on assistant messages Terminus-2 re-sends without it.

    Terminus-2 re-sends a turn that hit max_tokens as its truncated text only. Verifiers keys
    assistant messages on their reasoning, so that copy no longer matches the sampled turn: the
    next prompt is re-rendered with every earlier turn's thinking stripped, and the rollout forks
    into a second training sample.

    Example:
        # sampled turn: content='{"analysis": "Wri', reasoning_content="I will write it."
        restore_sampled_reasoning([{"role": "assistant", "content": '{"analysis": "Wri'}], trace)
        # -> [{"role": "assistant", "content": '{"analysis": "Wri',
        #      "reasoning_content": "I will write it."}]
    """
    reasoning = {
        node.message.content or "": node.message.reasoning_content
        for node in trace.nodes
        if node.sampled and node.message.reasoning_content
    }
    return [
        {**message, "reasoning_content": reasoning[message["content"] or ""]}
        if message["role"] == "assistant"
        and "reasoning_content" not in message
        and (message["content"] or "") in reasoning
        else message
        for message in message_history
    ]


class SandoqHarborTask(HarborTask):
    """Harbor's task, which also keeps test.sh's exit code and output in `trace.info["tests"]`."""

    async def _graded(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        # The stock `_graded` runs test.sh with its one `run`, drops the output, and reads the
        # reward with `read`.
        async def run_and_keep(argv: list[str], env: dict[str, str]) -> ProgramResult:
            result = await runtime.run(argv, env)
            trace.info["tests"] = {
                "exit_code": result.exit_code,
                "stdout": truncate_middle(result.stdout),
                "stderr": truncate_middle(result.stderr),
            }
            return result

        grading_runtime = SimpleNamespace(run=run_and_keep, read=runtime.read)
        return await super()._graded(grading_runtime, trace)


# Verifiers' env server builds each rollout's task as this generic's task type; `load` only
# feeds the dataset.
class SandoqTerminalTaskset(
    TerminalTaskset, vf.Taskset[SandoqHarborTask, TerminalTasksetConfig]
):
    """The PR's taskset, with each task's image WORKDIR filled in."""

    def load(self) -> Iterator[SandoqHarborTask]:
        # Without a task.toml workdir, use the image's last WORKDIR: the runtime
        # default (/app) is missing in some images, and a Sandoq VM fails then.
        for task in super().load():
            workdir = task.data.workdir or image_workdir(Path(task.data.task_dir))
            yield SandoqHarborTask(
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
