# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the framework-agnostic Terminal-Bench environment.

This module imports nothing from ``torchtitan``, like the package it tests.
"""

import asyncio
import contextvars
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf
from openai.types.chat import ChatCompletion

from torchtitan_recipes.rl.terminal_bench import agent_outside, taskset
from torchtitan_recipes.rl.terminal_bench.agent_outside import (
    AgentOutsideHarness,
    AgentOutsideHarnessConfig,
    sandoq_task_context,
)
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.task import TaskData
from verifiers.v1.tasksets.harbor import (
    HarborEnv,
    HarborEnvConfig,
    taskset as harbor_taskset,
)


def _completion(*tool_calls: tuple[str, str]) -> ChatCompletion:
    """An assistant turn making these `(name, arguments)` tool calls, or none."""
    return ChatCompletion.model_validate(
        {
            "id": "completion",
            "object": "chat.completion",
            "created": 0,
            "model": "policy",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": None if tool_calls else "Done.",
                        "tool_calls": [
                            {
                                "id": f"call_{index}",
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                            for index, (name, arguments) in enumerate(tool_calls)
                        ]
                        or None,
                    },
                }
            ],
        }
    )


class _ScriptedOpenAI:
    """Stands in for `AsyncOpenAI`: answers with queued completions, records requests."""

    def __init__(self, completions: list[ChatCompletion]) -> None:
        self.completions = completions
        self.requests: list[list[dict]] = []
        self.client_kwargs: dict = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def __call__(self, **client_kwargs: object) -> "_ScriptedOpenAI":
        self.client_kwargs = client_kwargs
        return self

    async def __aenter__(self) -> "_ScriptedOpenAI":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def _create(
        self, *, model: str, messages: list[dict], tools: list[dict]
    ) -> ChatCompletion:
        self.requests.append(json.loads(json.dumps(messages)))
        return self.completions.pop(0)


class _FakeRuntime:
    """Records sandbox commands and answers each with the same result."""

    def __init__(self, result: ProgramResult, delay_sec: float = 0.0) -> None:
        self.result = result
        self.delay_sec = delay_sec
        self.commands: list[list[str]] = []

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append(argv)
        await asyncio.sleep(self.delay_sec)
        return self.result


def _run_agent_outside(
    runtime: _FakeRuntime,
    completions: list[ChatCompletion],
    monkeypatch: pytest.MonkeyPatch,
    **config: float,
) -> list[list[dict]]:
    """Run the harness against scripted model turns; return every model request."""
    client = _ScriptedOpenAI(completions)
    monkeypatch.setattr(agent_outside, "AsyncOpenAI", client)
    harness = AgentOutsideHarness(
        AgentOutsideHarnessConfig(id=agent_outside.HARNESS_ID, **config)
    )
    result = asyncio.run(
        harness.launch(
            SimpleNamespace(model="policy"),
            None,
            runtime,
            "http://127.0.0.1:1/v1",
            "secret",
            {},
            TaskData(prompt="Fix the build."),
        )
    )
    assert result.exit_code == 0
    assert not client.completions
    # No read timeout: a long turn must not be cut off and resent.
    assert client.client_kwargs["timeout"].read is None
    return client.requests


def test_agent_outside_runs_bash_calls_in_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(ProgramResult(exit_code=0, stdout="hi\n", stderr=""))
    requests = _run_agent_outside(
        runtime,
        [
            _completion(
                ("bash", '{"command": "echo hi"}'),
                ("python", "{}"),
                ("bash", "not json"),
            ),
            _completion(),
        ],
        monkeypatch,
    )

    assert runtime.commands == [["bash", "-lc", "echo hi"]]
    assert requests[0] == [{"role": "user", "content": "Fix the build."}]
    tool_results = [
        message["content"] for message in requests[1] if message["role"] == "tool"
    ]
    assert tool_results[0] == "stdout:\nhi\n\nstderr:\n\nexit_code: 0"
    assert tool_results[1].startswith("error: unknown tool 'python'")
    assert tool_results[2].startswith("error: invalid JSON arguments")


def test_agent_outside_keeps_the_exit_code_of_truncated_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(ProgramResult(exit_code=3, stdout="x" * 1000, stderr=""))
    requests = _run_agent_outside(
        runtime,
        [_completion(("bash", '{"command": "make"}')), _completion()],
        monkeypatch,
        max_tool_output_chars=40,
    )
    tool_result = requests[1][-1]["content"]
    assert tool_result.startswith("[output truncated]\n")
    assert tool_result.endswith("exit_code: 3")


def test_agent_outside_reports_a_command_timeout_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(ProgramResult(exit_code=0, stdout="", stderr=""), 10.0)
    requests = _run_agent_outside(
        runtime,
        [_completion(("bash", '{"command": "sleep 60"}')), _completion()],
        monkeypatch,
        command_timeout_sec=0.01,
    )
    assert requests[1][-1]["content"] == "error: command timed out after 0.01s"


@pytest.mark.parametrize("provider_selected", [True, False])
def test_sandoq_task_context_binds_only_when_the_provider_is_selected(
    provider_selected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if provider_selected:
        monkeypatch.setenv("VF_SANDBOX_PROVIDER", "oci-runner")
    else:
        monkeypatch.delenv("VF_SANDBOX_PROVIDER", raising=False)
    task_context: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
        "task_context", default=None
    )
    sandoq_provider = ModuleType("sandoq_provider")
    sandoq_provider.install = lambda: True
    sandoq_provider.registry = SimpleNamespace(
        bind_task_context=task_context.set,
        reset_task_context=task_context.reset,
    )
    monkeypatch.setitem(sys.modules, "sandoq_provider", sandoq_provider)

    with sandoq_task_context(requested_image="org/task:1", working_dir="/app"):
        inside = task_context.get()
    assert task_context.get() is None
    assert inside == (
        {"requested_image": "org/task:1", "working_dir": "/app"}
        if provider_selected
        else None
    )


def test_importing_the_harness_registers_its_verifiers_alias() -> None:
    assert sys.modules[agent_outside.HARNESS_ID] is agent_outside


def test_env_package_resolves_in_a_fresh_interpreter_without_torchtitan() -> None:
    """A Verifiers worker resolves the env and harness from one import, with torchtitan blocked.

    Other trainers import this package without TitanRL, and a spawned Verifiers
    worker shares no ``sys.modules`` with its parent.
    """
    environment = {
        "taskset": {"id": taskset.TASKSET_ID, "dataset": "org/tasks"},
        "agent": {"harness": {"id": agent_outside.HARNESS_ID}},
    }
    worker = f"""
import sys
sys.modules["torchtitan"] = None  # any torchtitan import now raises ImportError
import torchtitan_recipes.rl.terminal_bench.taskset
from verifiers.v1.utils.loaders import environment_class, load_harness, resolve_env_config

env_config = resolve_env_config({environment!r})
print(environment_class(env_config.taskset.id).__name__)
print(type(load_harness(env_config.agent.harness)).__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", worker], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split()[-2:] == ["TerminalBenchEnv", "AgentOutsideHarness"]


def _env(harness: vf.HarnessConfig) -> taskset.TerminalBenchEnv:
    """A `TerminalBenchEnv` with this harness, without loading tasks."""
    env = object.__new__(taskset.TerminalBenchEnv)
    env.config = HarborEnvConfig(
        agent=vf.AgentConfig(harness=harness, runtime=vf.PrimeConfig())
    )
    return env


def test_env_needs_no_tunnel_for_the_agent_outside_harness() -> None:
    harness = AgentOutsideHarnessConfig(id=agent_outside.HARNESS_ID)
    assert _env(harness)._runs_local()


@pytest.mark.parametrize("provider_selected", [True, False])
def test_env_hands_sandoq_the_task_image_and_workdir(
    provider_selected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider reads the rollout's image and workdir from a context variable."""
    env = _env(AgentOutsideHarnessConfig(id=agent_outside.HARNESS_ID))
    if provider_selected:
        monkeypatch.setenv("VF_SANDBOX_PROVIDER", "oci-runner")
    else:
        monkeypatch.delenv("VF_SANDBOX_PROVIDER", raising=False)
    task_context: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
        "task_context", default=None
    )
    sandoq_provider = ModuleType("sandoq_provider")
    sandoq_provider.install = lambda: True
    sandoq_provider.registry = SimpleNamespace(
        bind_task_context=task_context.set,
        reset_task_context=task_context.reset,
    )
    monkeypatch.setitem(sys.modules, "sandoq_provider", sandoq_provider)
    seen_by_harbor: list[dict | None] = []

    async def harbor_run(self: HarborEnv, task: object, agents: object) -> None:
        seen_by_harbor.append(task_context.get())

    monkeypatch.setattr(HarborEnv, "run", harbor_run)
    task = SimpleNamespace(
        data=TaskData(name="org/demo", image="org/demo:1", workdir="/home/user")
    )

    async def run_rollout() -> dict | None:
        await env.run(task, agents=None)
        return task_context.get()

    assert asyncio.run(run_rollout()) is None
    assert seen_by_harbor == [
        {
            "instance_id": "org/demo",
            "requested_image": "org/demo:1",
            "working_dir": "/home/user",
        }
        if provider_selected
        else None
    ]


def test_taskset_starts_each_task_in_its_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task.toml workdir wins; otherwise the image's last WORKDIR; otherwise None."""
    for name, toml_workdir, dockerfile in [
        ("declared", "/home/user", None),
        ("sanitize", None, "FROM debian\nWORKDIR /app\nworkdir /app/dclm\n"),
        ("no-dockerfile", None, None),
    ]:
        task_dir = tmp_path / name
        (task_dir / "tests").mkdir(parents=True)
        (task_dir / "tests" / "test.sh").write_text("exit 0\n")
        (task_dir / "instruction.md").write_text("Do it.\n")
        workdir_line = f'workdir = "{toml_workdir}"\n' if toml_workdir else ""
        (task_dir / "task.toml").write_text(
            f'[task]\nname = "org/{name}"\n\n'
            f'[environment]\ndocker_image = "org/{name}:1"\n{workdir_line}'
        )
        if dockerfile is not None:
            (task_dir / "environment").mkdir()
            (task_dir / "environment" / "Dockerfile").write_text(dockerfile)
    monkeypatch.setattr(harbor_taskset, "dataset_dir", lambda config: tmp_path)

    tasks = taskset.TerminalTaskset(
        taskset.TerminalTasksetConfig(dataset="org/demo")
    ).load()
    assert {task.data.name: task.data.workdir for task in tasks} == {
        "org/declared": "/home/user",
        "org/no-dockerfile": None,
        "org/sanitize": "/app/dclm",
    }


def test_importing_the_taskset_registers_its_verifiers_alias() -> None:
    assert sys.modules[taskset.TASKSET_ID] is taskset
