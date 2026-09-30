# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the Terminal-Bench Verifiers recipe."""

import ast
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

from torchtitan.config import ConfigLoader
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.rl.controller import Controller
from torchtitan.rl.experiments.verifiers.terminal_bench import agent_outside, taskset
from torchtitan.rl.experiments.verifiers.terminal_bench.agent_outside import (
    AgentOutsideHarness,
    AgentOutsideHarnessConfig,
)
from torchtitan.rl.experiments.verifiers.terminal_bench.harness import (
    TerminalBenchTerminusHarness,
    TerminalBenchTerminusHarnessConfig,
    terminus_program_source,
)
from torchtitan.rl.experiments.verifiers.terminal_bench.rollouter import (
    terminal_bench_rollouter_config,
)
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.serve import env_config_data
from verifiers.v1.task import TaskData
from verifiers.v1.tasksets.harbor import (
    HarborEnv,
    HarborEnvConfig,
    taskset as harbor_taskset,
)
from verifiers.v1.utils.loaders import (
    environment_class,
    load_harness,
    resolve_env_config,
)

TRAIN_DATASET = "org/train-tasks"
EVAL_DATASET = "terminal-bench/terminal-bench-2-1"


def test_terminus_program_keeps_coworker_xml_scaffold() -> None:
    source = terminus_program_source().replace("{version}", "0.22.0")
    ast.parse(source)
    assert source.count('parser_name="xml"') == 1
    assert source.count("enable_summarize=False") == 1
    assert source.count("max_turns=120") == 1


def test_agent_runs_inside_docker_and_verifier_uses_same_taskset() -> None:
    config = terminal_bench_rollouter_config(TRAIN_DATASET, EVAL_DATASET)
    environment = config.verifiers_env_server.environment

    assert isinstance(environment, HarborEnvConfig)
    assert isinstance(environment.agent.runtime, vf.DockerConfig)
    assert isinstance(environment.agent.harness, TerminalBenchTerminusHarnessConfig)
    assert environment.agent.harness.version == "0.22.0"
    assert environment.agent.max_turns == 120
    assert environment.agent.timeout.rollout == 7200
    assert environment.taskset == config.train_dataset.verifiers_taskset
    assert config.train_dataset.verifiers_taskset.dataset == TRAIN_DATASET
    assert config.validation_dataset.verifiers_taskset.dataset == EVAL_DATASET
    assert config.verifiers_env_server.local_taskset_module == taskset.__name__
    worker_config = resolve_env_config(env_config_data(environment))
    assert isinstance(worker_config.agent.harness, TerminalBenchTerminusHarnessConfig)
    assert isinstance(
        load_harness(worker_config.agent.harness), TerminalBenchTerminusHarness
    )


@pytest.mark.parametrize(
    ("sandbox", "harness_name"),
    [("docker", "TerminalBenchTerminusHarness"), ("sandoq", "AgentOutsideHarness")],
)
def test_worker_process_resolves_the_harness_from_a_fresh_interpreter(
    sandbox: str, harness_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env-server worker shares no ``sys.modules`` with the controller.

    It imports only the local taskset module and then rebuilds the environment
    config from JSON, so that one import must be enough to make the harness id
    resolvable. Resolving in the test process would pass regardless, because the
    controller side has already registered the alias there.
    """
    if sandbox == "sandoq":
        _select_sandoq(monkeypatch)
    config = terminal_bench_rollouter_config(
        TRAIN_DATASET, EVAL_DATASET, sandbox=sandbox
    )
    environment = json.dumps(env_config_data(config.verifiers_env_server.environment))
    worker = f"""
import json
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from verifiers.v1.utils.loaders import environment_class, load_harness, resolve_env_config

environment = json.loads({environment!r})
environment["taskset"]["id"] = register_local_taskset_alias(
    {config.verifiers_env_server.local_taskset_module!r}
)
env_config = resolve_env_config(environment)
print(environment_class(env_config.taskset.id).__name__)
print(type(load_harness(env_config.agent.harness)).__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", worker], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split()[-2:] == ["TerminalBenchEnv", harness_name]


def test_training_cannot_read_benchmark_as_training_data() -> None:
    with pytest.raises(ValueError, match="different datasets"):
        terminal_bench_rollouter_config(EVAL_DATASET, EVAL_DATASET)


def _terminal_bench_config(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> Controller.Config:
    monkeypatch.setenv("TERMINAL_BENCH_TRAIN_DATASET", TRAIN_DATASET)
    monkeypatch.setenv("TERMINAL_BENCH_EVAL_DATASET", EVAL_DATASET)
    return ConfigLoader().load(
        [
            "--module",
            "torchtitan.rl.experiments.verifiers.terminal_bench",
            "--config",
            name,
        ]
    )


def test_training_recipe_uses_separate_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _terminal_bench_config("rl_grpo_qwen35_9b_terminal_bench", monkeypatch)
    assert config.async_loop.num_training_steps == 100
    assert config.async_loop.num_samples_per_prompt == 32
    assert config.trainer.training.max_context_length == 65536
    assert config.trainer.training.dtype == "float32"
    assert config.trainer.training.mixed_precision_param == "bfloat16"
    assert config.trainer.training.mixed_precision_reduce == "float32"
    assert config.trainer.optim.optimizer.optimizers[0].fused
    assert config.async_loop.training_sample_builder.drop_zero_std_reward_groups
    assert isinstance(config.trainer.activation_checkpoint, FullAC.Config)
    assert config.trainer.checkpointer.interval == 20
    assert config.generator.cuda_graph.mode == "FULL_DECODE_ONLY"
    assert config.num_generators == 8
    assert config.generator.parallelism.data_parallel_degree == 1
    assert config.rollouter.train_dataset.verifiers_taskset.dataset == TRAIN_DATASET
    assert config.rollouter.validation_dataset.verifiers_taskset.dataset == EVAL_DATASET


def test_recipes_require_both_dataset_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TERMINAL_BENCH_TRAIN_DATASET", raising=False)
    monkeypatch.delenv("TERMINAL_BENCH_EVAL_DATASET", raising=False)
    with pytest.raises(KeyError, match="TERMINAL_BENCH_TRAIN_DATASET"):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan.rl.experiments.verifiers.terminal_bench",
                "--config",
                "rl_grpo_qwen35_9b_terminal_bench",
            ]
        )


def _num_kv_heads(model: object) -> int:
    for layer in model.layers:
        attention = getattr(layer, "attention", None)
        if hasattr(attention, "n_kv_heads"):
            return attention.n_kv_heads
    raise AssertionError("model has no full-attention layer")


def _num_experts(model: object) -> int | None:
    for layer in model.layers:
        moe = getattr(layer, "moe", None)
        if moe is not None:
            return moe.num_experts
    return None


@pytest.mark.parametrize(
    ("name", "trainer_gpus", "num_generators", "gpus_per_generator"),
    [
        ("rl_grpo_qwen35_9b_terminal_bench", 8, 8, 1),
        ("rl_grpo_qwen35_35b_a3b_terminal_bench", 8, 2, 4),
    ],
)
def test_recipe_layouts_fit_the_model(
    name: str,
    trainer_gpus: int,
    num_generators: int,
    gpus_per_generator: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each recipe's trainer and generator layouts respect the model's shape.

    The invariants are the ones a launch would otherwise trip over one at a
    time: tensor parallelism divides the KV heads in both roles, and for the MoE
    model expert parallelism divides the experts, is at least the trainer TP
    degree, and equals DP x TP in the generator. The GPU totals pin the intended
    16-GPU footprint.
    """
    config = _terminal_bench_config(name, monkeypatch)
    trainer = config.trainer.parallelism
    generator = config.generator.parallelism
    num_kv_heads = _num_kv_heads(config.model)
    num_experts = _num_experts(config.model)

    assert (
        trainer.data_parallel_replicate_degree
        * trainer.data_parallel_shard_degree
        * trainer.tensor_parallel_degree
        * trainer.context_parallel_degree
        == trainer_gpus
    )
    assert config.num_generators == num_generators
    assert generator.data_parallel_degree * generator.tensor_parallel_degree == (
        gpus_per_generator
    )
    assert num_kv_heads % trainer.tensor_parallel_degree == 0
    assert num_kv_heads % generator.tensor_parallel_degree == 0

    if num_experts is None:
        assert trainer.expert_parallel_degree == 1
        assert generator.expert_parallel_degree == 1
        assert config.generator.cuda_graph.mode == "FULL_DECODE_ONLY"
    else:
        assert num_experts % trainer.expert_parallel_degree == 0
        assert trainer.expert_parallel_degree >= trainer.tensor_parallel_degree
        assert (
            trainer.data_parallel_shard_degree * trainer.tensor_parallel_degree
        ) % trainer.expert_parallel_degree == 0
        assert generator.expert_parallel_degree == (
            generator.data_parallel_degree * generator.tensor_parallel_degree
        )
        assert num_experts % generator.expert_parallel_degree == 0
        # The standard MoE dispatcher reads split sizes back to the host, which
        # CUDA graph capture does not allow.
        assert config.generator.cuda_graph.mode == "NONE"


@pytest.mark.parametrize(
    "name",
    [
        "rl_grpo_qwen35_9b_terminal_bench",
        "rl_grpo_qwen35_35b_a3b_terminal_bench",
    ],
)
def test_recipes_share_the_loop_and_keep_fp32_master_weights(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every model trains on the same loop; only size-dependent settings differ.

    Master weights stay fp32 (the default): at a 1e-6 learning rate bf16
    parameters round most updates away.
    """
    config = _terminal_bench_config(name, monkeypatch)
    assert config.trainer.training.dtype == "float32"
    assert config.trainer.training.max_context_length == 65536
    assert config.async_loop.num_prompts_per_train_step == 8
    assert config.async_loop.num_samples_per_prompt == 32
    assert config.async_loop.num_training_steps == 100
    assert config.async_loop.training_sample_builder.drop_zero_std_reward_groups
    assert config.generator.sampling.max_tokens == 16384


def _select_sandoq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VF_SANDBOX_PROVIDER", "oci-runner")
    monkeypatch.setenv("OCI_RUNNER_TASK_NETWORK", "host")


def test_sandoq_runs_the_agent_outside_on_the_prime_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_sandoq(monkeypatch)
    config = terminal_bench_rollouter_config(
        TRAIN_DATASET, EVAL_DATASET, sandbox="sandoq"
    )
    env_server = config.verifiers_env_server
    agent = env_server.environment.agent

    assert isinstance(agent.runtime, vf.PrimeConfig)
    assert isinstance(agent.harness, AgentOutsideHarnessConfig)
    assert agent.max_turns == 120
    assert env_server.serve.pool.num_workers * env_server.serve.max_concurrent == 128
    assert (
        environment_class(env_server.environment.taskset.id) is taskset.TerminalBenchEnv
    )


@pytest.mark.parametrize(
    ("unset", "match"),
    [
        ("VF_SANDBOX_PROVIDER", "VF_SANDBOX_PROVIDER=oci-runner"),
        ("OCI_RUNNER_TASK_NETWORK", "OCI_RUNNER_TASK_NETWORK=host"),
    ],
)
def test_sandoq_requires_the_provider_and_host_network(
    unset: str, match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _select_sandoq(monkeypatch)
    monkeypatch.delenv(unset)
    with pytest.raises(ValueError, match=match):
        terminal_bench_rollouter_config(TRAIN_DATASET, EVAL_DATASET, sandbox="sandoq")


@pytest.mark.parametrize(
    "name",
    [
        "rl_grpo_qwen35_9b_terminal_bench",
        "rl_grpo_qwen35_35b_a3b_terminal_bench",
    ],
)
def test_recipes_select_the_sandbox_from_the_environment(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _select_sandoq(monkeypatch)
    monkeypatch.setenv("TERMINAL_BENCH_SANDBOX", "sandoq")
    config = _terminal_bench_config(name, monkeypatch)
    agent = config.rollouter.verifiers_env_server.environment.agent
    assert isinstance(agent.harness, AgentOutsideHarnessConfig)
    assert isinstance(agent.runtime, vf.PrimeConfig)


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
        AgentOutsideHarnessConfig(id=agent_outside.__name__, **config)
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


def _sandoq_env(monkeypatch: pytest.MonkeyPatch) -> taskset.TerminalBenchEnv:
    """A `TerminalBenchEnv` configured like the sandoq recipe, without loading tasks."""
    _select_sandoq(monkeypatch)
    config = terminal_bench_rollouter_config(
        TRAIN_DATASET, EVAL_DATASET, sandbox="sandoq"
    )
    env = object.__new__(taskset.TerminalBenchEnv)
    env.config = config.verifiers_env_server.environment
    return env


def test_env_needs_no_tunnel_for_the_agent_outside_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _sandoq_env(monkeypatch)._runs_local()


@pytest.mark.parametrize("provider_selected", [True, False])
def test_env_hands_sandoq_the_task_image_and_workdir(
    provider_selected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider reads the rollout's image and workdir from a context variable."""
    env = _sandoq_env(monkeypatch)
    if not provider_selected:
        monkeypatch.delenv("VF_SANDBOX_PROVIDER")
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
    task = SimpleNamespace(data=TaskData(name="org/demo", image="org/demo:1"))

    async def run_rollout() -> dict | None:
        await env.run(task, agents=None)
        return task_context.get()

    assert asyncio.run(run_rollout()) is None
    assert seen_by_harbor == [
        {
            "instance_id": "org/demo",
            "requested_image": "org/demo:1",
            "working_dir": "/app",
        }
        if provider_selected
        else None
    ]


def test_taskset_starts_each_task_in_its_image_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Harbor parses no workdir from task.toml, so the image's last WORKDIR wins."""
    for name, dockerfile in [
        ("sanitize", "FROM debian\nWORKDIR /app\nworkdir /app/dclm\n"),
        ("no-dockerfile", None),
    ]:
        task_dir = tmp_path / name
        (task_dir / "tests").mkdir(parents=True)
        (task_dir / "tests" / "test.sh").write_text("exit 0\n")
        (task_dir / "instruction.md").write_text("Do it.\n")
        (task_dir / "task.toml").write_text(
            f'[task]\nname = "org/{name}"\n\n'
            f'[environment]\ndocker_image = "org/{name}:1"\n'
        )
        if dockerfile is not None:
            (task_dir / "environment").mkdir()
            (task_dir / "environment" / "Dockerfile").write_text(dockerfile)
    monkeypatch.setattr(harbor_taskset, "dataset_dir", lambda config: tmp_path)

    tasks = taskset.TerminalTaskset(
        taskset.TerminalTasksetConfig(dataset="org/demo")
    ).load()
    assert {task.data.name: task.data.workdir for task in tasks} == {
        "org/no-dockerfile": None,
        "org/sanitize": "/app/dclm",
    }
