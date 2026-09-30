# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the shared agent-outside harness, with a scripted model and sandbox."""

import asyncio
import contextvars
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("verifiers")

from openai.types.chat import ChatCompletion

from torchtitan.rl.experiments.verifiers import agent_outside
from torchtitan.rl.experiments.verifiers.agent_outside import (
    AgentOutsideHarness,
    AgentOutsideHarnessConfig,
    sandoq_task_context,
)
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.task import TaskData


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
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def __call__(self, *, base_url: str, api_key: str) -> "_ScriptedOpenAI":
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
