# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers rollout integration."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers")

from aiohttp import ClientSession

from torchtitan.rl.examples.verifiers.generation_server import (
    _parse_sampling_config,
    GenerationServer,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.examples.verifiers.rollouter import (
    _trainable_token_spans,
    log_failed_rollout,
    verifiers_rollout_logs,
    VerifiersRollouter,
)
from torchtitan.rl.types import Completion


def test_trainable_token_spans() -> None:
    assert _trainable_token_spans([False, True, True, False, True]) == [
        (1, 3),
        (4, 5),
    ]


def test_verifiers_trace_preserves_generation_metadata() -> None:
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    node = SimpleNamespace(
        token_ids=[10, 11, 12, 13],
        mask=[False, False, True, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="Answer: $42$"),
    )
    trace = SimpleNamespace(
        nodes=[node],
        branches=[
            SimpleNamespace(
                nodes=[node],
                token_ids=[10, 11, 12, 13],
                logprobs=[0.0, 0.0, -0.2, -0.3],
            )
        ],
    )
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=trace,
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=3,
            max_policy_version=4,
            metrics=[],
        ),
        group_id=5,
        rollout_id=2,
    )

    assert len(turns) == 1
    assert turns[0].prompt_token_ids == [10, 11]
    assert turns[0].completion_token_ids == [12, 13]
    assert turns[0].completion_logprobs == [-0.2, -0.3]
    assert turns[0].completion_message == {
        "role": "assistant",
        "content": "Answer: $42$",
    }
    assert turns[0].min_policy_version == 3
    assert turns[0].max_policy_version == 4


def test_verifiers_multiturn_trace_matches_titanrl_rollout_structure() -> None:
    from verifiers.v1.types import AssistantMessage as VerifiersAssistantMessage

    first_node = SimpleNamespace(
        token_ids=[10, 11],
        mask=[False, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="first"),
    )
    second_node = SimpleNamespace(
        token_ids=[12, 13],
        mask=[False, True],
        sampled=True,
        message=VerifiersAssistantMessage(content="second"),
    )
    trace = SimpleNamespace(
        nodes=[first_node, second_node],
        branches=[
            SimpleNamespace(
                nodes=[first_node, second_node],
                token_ids=[10, 11, 12, 13],
                logprobs=[0.0, -0.1, 0.0, -0.2],
            )
        ],
    )
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=trace,
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=3,
            max_policy_version=8,
            metrics=[],
        ),
        group_id=5,
        rollout_id=2,
    )

    assert [turn.min_policy_version for turn in turns] == [3, 3]
    assert [turn.max_policy_version for turn in turns] == [8, 8]
    assert [turn.prompt_token_ids for turn in turns] == [[10], [10, 11, 12]]
    assert [turn.completion_token_ids for turn in turns] == [[11], [13]]
    assert [turn.completion_logprobs for turn in turns] == [[-0.1], [-0.2]]


def test_verifiers_trace_attaches_env_replies_to_the_preceding_turn() -> None:
    from verifiers.v1.types import (
        AssistantMessage as VerifiersAssistantMessage,
        UserMessage as VerifiersUserMessage,
    )

    def node(content: str, sampled: bool) -> SimpleNamespace:
        message = (
            VerifiersAssistantMessage(content=content)
            if sampled
            else VerifiersUserMessage(content=content)
        )
        return SimpleNamespace(
            token_ids=[0], mask=[sampled], sampled=sampled, message=message
        )

    # Two branches share the prompt and the first command, then diverge.
    task, first, out, second, other_out, other = (
        node("task", False),
        node("ls", True),
        node("a.txt", False),
        node("cat a.txt", True),
        node("No such file", False),
        node("pwd", True),
    )
    trace = SimpleNamespace(
        nodes=[task, first, out, second, other_out, other],
        branches=[
            SimpleNamespace(
                nodes=branch_nodes,
                token_ids=[0] * len(branch_nodes),
                logprobs=[0.0] * len(branch_nodes),
            )
            for branch_nodes in (
                [task, first, out, second],
                [task, first, other_out, other],
            )
        ],
    )
    turns = VerifiersRollouter.trace_to_rollout_turns(
        trace=trace,
        generation_metadata=VerifiersGenerationMetadata(
            min_policy_version=0, max_policy_version=0, metrics=[]
        ),
        group_id=0,
        rollout_id=0,
    )

    assert [turn.completion_message["content"] for turn in turns] == [
        "ls",
        "cat a.txt",
        "pwd",
    ]
    assert [turn.env_messages for turn in turns] == [
        [
            {"role": "user", "content": "a.txt"},
            {"role": "user", "content": "No such file"},
        ],
        [],
        [],
    ]


def test_verifiers_rollout_logs_keep_the_failure_reason(caplog) -> None:
    from verifiers.v1.trace import Error

    def span(seconds: float) -> SimpleNamespace:
        return SimpleNamespace(duration=seconds)

    trace = SimpleNamespace(
        id="trace-1",
        task=SimpleNamespace(key="task-1"),
        stop_condition="error",
        errors=[
            Error(
                type="HarnessError",
                message="agent timeout",
                traceback="Traceback\n  ...\nTimeoutError",
            )
        ],
        timing=SimpleNamespace(
            setup=span(5.0),
            agent=SimpleNamespace(
                duration=7200.0, model=span(6900.0), harness=span(300.0)
            ),
            scoring=span(0.0),
        ),
        calls=[
            SimpleNamespace(time=span(1801.0), error=None),
            SimpleNamespace(
                time=span(2.0), error=Error(type="APIError", message="502")
            ),
        ],
    )
    logs = verifiers_rollout_logs(SimpleNamespace(errors=[]), trace)

    assert logs["errors"] == [
        {
            "type": "HarnessError",
            "message": "agent timeout",
            "status_code": None,
            "traceback": "Traceback\n  ...\nTimeoutError",
        }
    ]
    assert (logs["agent_sec"], logs["model_sec"], logs["harness_sec"]) == (
        7200.0,
        6900.0,
        300.0,
    )
    assert logs["model_calls"] == 2
    assert logs["failed_model_calls"] == 1
    assert logs["slowest_model_call_sec"] == 1801.0

    with caplog.at_level(logging.WARNING):
        log_failed_rollout(logs, group_id=3, rollout_id=1)
        log_failed_rollout({**logs, "errors": []}, group_id=3, rollout_id=2)
    assert "error=HarnessError: agent timeout" in caplog.text
    assert "error=None: None" in caplog.text


def test_generation_server_forwards_token_request() -> None:
    async def run_test() -> None:
        received = []

        async def generate_fn(
            prompt_token_ids,
            *,
            request_id,
            group_id,
            routing_session_id=None,
            sampling_config=None,
        ):
            received.append(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "request_id": request_id,
                    "group_id": group_id,
                    "routing_session_id": routing_session_id,
                    "sampling_config": sampling_config,
                }
            )
            request_index = int(request_id.rsplit("=", 1)[1])
            return Completion(
                min_policy_version=7 - request_index,
                max_policy_version=8 + request_index,
                request_id=request_id,
                token_ids=[31, 32],
                token_logprobs=[-0.1, -0.2],
                finish_reason="stop",
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.get(f"{server.base_url}/models")
                assert response.status == 200
                assert await response.json() == {
                    "object": "list",
                    "data": [
                        {
                            "id": "torchtitan",
                            "object": "model",
                            "created": 0,
                            "owned_by": "torchtitan",
                            "max_model_len": 40960,
                        }
                    ],
                }
                for _ in range(2):
                    response = await session.post(
                        f"http://{server.host}:{server.port}/inference/v1/generate",
                        headers={"X-Session-ID": "group=1/rollout=2"},
                        json={
                            "token_ids": [10, 11],
                            "sampling_params": {
                                "temperature": 1.0,
                                "top_p": 1.0,
                                "max_tokens": 2,
                                "seed": 4,
                                "logprobs": 1,
                                "torchtitan_group_id": 1,
                                "stop_token_ids": [99],
                            },
                        },
                    )
                    assert response.status == 200
                    payload = await response.json()
            generation_metadata = server.pop_generation_metadata("group=1/rollout=2")
        finally:
            await server.close()

        assert [request["request_id"] for request in received] == [
            "group=1/rollout=2/request=0",
            "group=1/rollout=2/request=1",
        ]
        assert all(request["prompt_token_ids"] == [10, 11] for request in received)
        assert all(request["group_id"] == 1 for request in received)
        assert all(
            request["routing_session_id"] == "group=1/rollout=2" for request in received
        )
        assert all(request["sampling_config"].seed == 4 for request in received)
        assert payload["choices"][0]["token_ids"] == [31, 32]
        assert generation_metadata is not None
        assert generation_metadata.min_policy_version == 6
        assert generation_metadata.max_policy_version == 9

    asyncio.run(run_test())


def test_generation_server_rejects_aborted_generation() -> None:
    async def run_test() -> None:
        async def generate_fn(
            prompt_token_ids,
            *,
            request_id,
            group_id,
            routing_session_id=None,
            sampling_config=None,
        ):
            return Completion(
                min_policy_version=7,
                max_policy_version=7,
                request_id=request_id,
                token_ids=[],
                token_logprobs=[],
                finish_reason="abort",
            )

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={
                        "token_ids": [10, 11],
                        "sampling_params": {
                            "torchtitan_group_id": 1,
                            "stop_token_ids": [99],
                        },
                    },
                )
                assert response.status == 502
                payload = await response.json()
            generation_metadata = server.pop_generation_metadata("group=1/rollout=2")
        finally:
            await server.close()

        assert payload == {
            "error": "generation finished without a usable completion: abort"
        }
        assert generation_metadata is None

    asyncio.run(run_test())


def test_generation_server_requires_group_id() -> None:
    async def run_test() -> None:
        async def generate_fn(*args, **kwargs):
            raise AssertionError("generate_fn must not run without a group id")

        server = GenerationServer.Config(max_rollout_tokens=40960).build()
        server.set_generate_fn(generate_fn)
        await server.start()
        try:
            async with ClientSession() as session:
                response = await session.post(
                    f"http://{server.host}:{server.port}/inference/v1/generate",
                    headers={"X-Session-ID": "group=1/rollout=2"},
                    json={"token_ids": [10, 11], "sampling_params": {}},
                )
                assert response.status == 400
                payload = await response.json()
        finally:
            await server.close()

        assert "torchtitan_group_id" in payload["error"]

    asyncio.run(run_test())


def test_parse_sampling_config_requires_stop_token_ids() -> None:
    with pytest.raises(ValueError, match="stop_token_ids"):
        _parse_sampling_config({"temperature": 1.0})


def test_terminus_max_tokens_turn_stays_on_its_branch() -> None:
    """Terminus-2 re-sends a max_tokens turn without its reasoning; the plugin restores it so
    the next prompt still bridges from the sampled tokens instead of forking a new branch."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins.terminal_bench_sandoq import (
        restore_sampled_reasoning,
    )
    from verifiers.v1 import graph
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.dialects.chat import parse_message
    from verifiers.v1.trace import AgentInfo, Trace, TraceTask
    from verifiers.v1.types import AssistantMessage, Response, TurnTokens

    trace = Trace(
        task=TraceTask(type="Task", data={}), agent=AgentInfo(config=AgentConfig())
    )
    task = {"role": "user", "content": "Write a.py"}
    # Hit max_tokens after </think>: reasoning is set, content is partial JSON.
    truncated = AssistantMessage(
        content='{"analysis": "Wri', reasoning_content="I will write it."
    )
    graph.prepare_turn(trace, [parse_message(task)]).commit(
        Response(
            id="r0",
            created=0,
            model="torchtitan",
            message=truncated,
            finish_reason="length",
            tokens=TurnTokens(
                prompt_ids=[1, 2], completion_ids=[3, 4], completion_logprobs=[0.0] * 2
            ),
        )
    )
    # Terminus-2's history after its max_tokens re-prompt (terminus_2.py:1142-1143).
    history = [
        task,
        {"role": "assistant", "content": '{"analysis": "Wri'},
        {
            "role": "user",
            "content": "ERROR!! NONE of the actions you just requested ...",
        },
    ]

    def bridges(messages: list[dict]) -> bool:
        prompt = [parse_message(message) for message in messages]
        return graph.prepare_turn(trace, prompt).previous_token_ids() is not None

    assert not bridges(history)
    assert bridges(restore_sampled_reasoning(history, trace))


def test_sandoq_rollout_log_explains_a_tmux_failure(tmp_path, caplog) -> None:
    """A rollout whose tmux never starts leaves a log with Terminus-2's traceback and log lines."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.runtimes import ProgramResult

    class NoTmuxRuntime:
        async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
            return ProgramResult(exit_code=127, stdout="", stderr="tmux: not found")

    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    runtime = NoTmuxRuntime()

    async def rollout() -> ProgramResult:
        result = await _launch(harness, trace, runtime)
        sandoq.harbor_logger.warning("logged outside any launch")
        await harness.cleanup(trace, runtime)
        # Verifiers' abort() after a cancelled close() calls cleanup again.
        await harness.cleanup(trace, runtime)
        return result

    result = asyncio.run(rollout())
    log = _read_rollout_log(tmp_path, trace)

    assert result.exit_code == 1
    assert log["verifiers_trace_id"] == trace.id
    assert "Failed to start tmux session" in log["harness_stderr"]
    assert "ERROR harbor.utils.logger: Failed to install tmux" in log["terminus_log"]
    assert "outside any launch" not in log["terminus_log"]
    # The filter copies Harbor's records; they still reach the job log.
    assert "Failed to install tmux" in caplog.text
    assert log["trajectory"] is None
    assert log["tests"] is None


def test_sandoq_rollout_log_keeps_trajectory_and_test_output(
    tmp_path, monkeypatch
) -> None:
    """A finished rollout's log holds Terminus-2's trajectory, without the API key, and test.sh's
    capped output; the reward is the one the stock HarborTask reads."""
    pytest.importorskip("harbor")
    from harbor.models.trajectories import Step
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.runtimes import ProgramResult
    from verifiers.v1.tasksets.harbor import HarborTask
    from verifiers.v1.tasksets.harbor.taskset import HarborData

    sent_api_keys = []
    test_stdout = "apt-get update\n" + "x" * 100_000 + "\n1 failed"

    class Terminus2WithoutModel(_terminus2_without_model(sandoq)):
        async def run(self, instruction, environment, context) -> None:
            sent_api_keys.append(self._llm._build_base_kwargs()["api_key"])
            self._context = context
            self._trajectory_steps = [
                Step(
                    step_id=1,
                    timestamp="2026-10-07T12:00:00Z",
                    source="user",
                    message=instruction,
                )
            ]
            self._dump_trajectory()

    class VMRuntime:
        async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
            if argv == ["bash", "/tests/test.sh"]:
                return ProgramResult(exit_code=1, stdout=test_stdout, stderr="")
            return ProgramResult(exit_code=0, stdout="", stderr="")

        async def read(self, path: str, max_bytes: int) -> bytes:
            if path.endswith("reward.json"):
                raise OSError(path)
            return b"0"

    monkeypatch.setattr(sandoq, "Terminus2", Terminus2WithoutModel)
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    runtime = VMRuntime()
    data = HarborData(prompt="Write a.py")

    async def rollout() -> tuple[ProgramResult, float, float]:
        result = await _launch(harness, trace, runtime)
        reward = await sandoq.SandoqHarborTask(data)._graded(runtime, trace)
        stock_reward = await HarborTask(data)._graded(runtime, trace)
        trace.record_reward("solved", reward)
        await harness.cleanup(trace, runtime)
        return result, reward, stock_reward

    result, reward, stock_reward = asyncio.run(rollout())
    log = _read_rollout_log(tmp_path, trace)

    # The env server builds tasks as the taskset's task type.
    assert sandoq.SandoqTerminalTaskset.task_type() is sandoq.SandoqHarborTask
    assert reward == stock_reward == 0.0
    assert result.exit_code == 0
    assert log["harness_stderr"] == ""
    assert log["trajectory"]["steps"][0]["message"] == "Write a.py"
    assert log["trajectory"]["agent"]["extra"]["llm_kwargs"] == {
        "custom_llm_provider": "openai"
    }
    assert sent_api_keys == ["secret"]
    assert log["tests"]["exit_code"] == 1
    assert log["tests"]["stdout"].startswith("apt-get update\n")
    assert log["tests"]["stdout"].endswith("\n1 failed")
    assert "characters omitted" in log["tests"]["stdout"]


def test_sandoq_tmux_outlives_grading(tmp_path, monkeypatch) -> None:
    """launch leaves tmux running, so test.sh sees the agent's shell jobs as under `harbor run`;
    cleanup writes the log with the pane, then kills tmux."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.runtimes import ProgramResult

    monkeypatch.setattr(sandoq, "Terminus2", _terminus2_without_model(sandoq))
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    runtime = _RecordingRuntime(
        {
            "tail": ProgramResult(
                exit_code=0, stdout="$ python3 server.py &\n", stderr=""
            )
        }
    )

    asyncio.run(_launch(harness, trace, runtime))
    assert not any("kill-server" in " ".join(argv) for argv, _ in runtime.calls)

    asyncio.run(harness.cleanup(trace, runtime))
    commands = [" ".join(argv) for argv, _ in runtime.calls]
    pane_read = next(i for i, c in enumerate(commands) if c.startswith("tail"))
    kill = next(i for i, c in enumerate(commands) if "kill-server" in c)
    assert pane_read < kill
    assert _read_rollout_log(tmp_path, trace)["pane"] == "$ python3 server.py &\n"


def test_sandoq_exec_keeps_keystrokes_out_of_argv() -> None:
    """Each exec ships its command in an env var, so an agent's `pkill -f` pattern can't match
    the exec that typed it; the command still runs as written."""
    pytest.importorskip("harbor")
    import subprocess

    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq

    command = "tmux send-keys -t terminus-2 -- 'pkill -f server.py' Enter"
    runtime = _RecordingRuntime()
    environment = sandoq.RuntimeEnvironment(runtime, {"TMUX_TMPDIR": "/tmp/t"})
    asyncio.run(environment.exec(command, cwd="/app"))

    [(argv, env)] = runtime.calls
    assert "server.py" not in " ".join(argv)
    assert env == {"TMUX_TMPDIR": "/tmp/t", "TERMINUS_EXEC": f"cd /app && {command}"}
    # The wrapper runs a quoted, multi-line command unchanged.
    script = "cat <<'EOF'\nit's \"quoted\" $HOME\nEOF"
    output = subprocess.run(
        argv, env={"TERMINUS_EXEC": script}, capture_output=True, text=True
    ).stdout
    assert output == 'it\'s "quoted" $HOME\n'


def test_sandoq_session_nests_a_shell(tmp_path, monkeypatch) -> None:
    """After tmux setup the agent's shell is a child shell, as under Harbor's recording, so its
    first `exit` returns to the outer shell instead of ending the session."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq

    sessions = []
    terminus2 = _terminus2_without_model(sandoq)

    class Terminus2KeepingSession(terminus2):
        async def run(self, instruction, environment, context) -> None:
            sessions.append(self._session)

    monkeypatch.setattr(sandoq, "Terminus2", Terminus2KeepingSession)
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    result = asyncio.run(_launch(harness, trace, _RecordingRuntime()))

    assert result.exit_code == 0
    assert sessions[0].keys == [["bash", "Enter"], ["clear", "Enter"]]


def test_sandoq_exec_failure_raises_sandbox_error(tmp_path, monkeypatch) -> None:
    """A lost exec channel leaves launch as a SandboxError, which the agent's retries can rerun
    on a fresh VM; the rollout log still keeps the traceback."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq
    from verifiers.v1.errors import SandboxError

    class LostRuntime(_RecordingRuntime):
        async def run(self, argv, env):
            if env.get("TERMINUS_EXEC") == "ls":
                raise SandboxError("prime exec failed: uncertain transport failure")
            return await super().run(argv, env)

    class Terminus2Typing(_terminus2_without_model(sandoq)):
        async def run(self, instruction, environment, context) -> None:
            await environment.exec("ls")

    monkeypatch.setattr(sandoq, "Terminus2", Terminus2Typing)
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    runtime = LostRuntime()
    with pytest.raises(SandboxError):
        asyncio.run(_launch(harness, trace, runtime))
    asyncio.run(harness.cleanup(trace, runtime))
    assert (
        "uncertain transport failure"
        in _read_rollout_log(tmp_path, trace)["harness_stderr"]
    )


def test_sandoq_model_call_stops_with_the_rollout(tmp_path, monkeypatch) -> None:
    """Once Verifiers has stopped the rollout, the model call raises at once instead of letting
    Terminus-2 retry a request Verifiers will refuse."""
    pytest.importorskip("harbor")
    from harbor.llms.base import ContextLengthExceededError
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq as sandoq

    raised = []

    class Terminus2AtTheCap(_terminus2_without_model(sandoq)):
        async def run(self, instruction, environment, context) -> None:
            trace.stop_condition = "context_length"
            try:
                await self._llm.call(prompt="next turn", message_history=[])
            except ContextLengthExceededError as error:
                raised.append(str(error))
                raise

    monkeypatch.setattr(sandoq, "Terminus2", Terminus2AtTheCap)
    harness, trace = _sandoq_harness_and_trace(sandoq, tmp_path)
    result = asyncio.run(_launch(harness, trace, _RecordingRuntime()))

    assert raised == ["Verifiers stopped the rollout: context_length"]
    assert result.exit_code == 1


class _RecordingRuntime:
    """Runtime that records each `run` and answers by the command's first word."""

    def __init__(self, results: dict | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self._results = results or {}

    async def run(self, argv: list[str], env: dict[str, str]):
        from verifiers.v1.runtimes import ProgramResult

        self.calls.append((argv, env))
        return self._results.get(
            argv[0], ProgramResult(exit_code=0, stdout="", stderr="")
        )


def _terminus2_without_model(sandoq):
    """Terminus-2 that runs no model and no tmux; its session records the keys it is sent."""

    class Session:
        def __init__(self) -> None:
            self.keys: list = []

        async def send_keys(self, keys, **kwargs) -> None:
            self.keys.append(keys)

    class Terminus2WithoutModel(sandoq.Terminus2):
        async def setup(self, environment) -> None:
            self._session = Session()

        async def run(self, instruction, environment, context) -> None:
            pass

    return Terminus2WithoutModel


def _sandoq_harness_and_trace(sandoq, log_dir):
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.trace import AgentInfo, Trace, TraceTask

    config = sandoq.StockTerminusOutsideConfig(
        id=sandoq.PLUGIN_ID, rollout_log_dir=str(log_dir)
    )
    trace = Trace(
        task=TraceTask(type="Task", data={}), agent=AgentInfo(config=AgentConfig())
    )
    return sandoq.StockTerminusOutsideHarness(config), trace


async def _launch(harness, trace, runtime):
    from verifiers.v1.task import TaskData

    return await harness.launch(
        SimpleNamespace(model="policy"),
        trace,
        runtime,
        endpoint="http://127.0.0.1:1",
        secret="secret",
        mcp_urls={},
        data=TaskData(prompt="Write a.py"),
    )


def _read_rollout_log(log_dir, trace) -> dict:
    return json.loads(gzip.decompress((log_dir / f"{trace.id}.json.gz").read_bytes()))
