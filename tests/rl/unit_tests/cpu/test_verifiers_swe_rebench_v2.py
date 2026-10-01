# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the SWE-rebench V2 recipes, with a fake dataset and sandbox."""

import asyncio
import contextvars
import copy
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf
from datasets import Dataset

from torchtitan.config import apply_overrides, ConfigLoader
from torchtitan.config.validation import validate_model_training_config
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.rl.controller import Controller
from torchtitan.rl.experiments.verifiers.agent_outside import AgentOutsideHarnessConfig
from torchtitan.rl.experiments.verifiers.swe_rebench_v2 import taskset
from torchtitan.rl.experiments.verifiers.swe_rebench_v2.rollouter import (
    NUM_AGENT_TURNS,
    swe_rebench_v2_rollouter_config,
)
from verifiers.v1.envs.single_agent import SingleAgentEnv
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.serve import env_config_data
from verifiers.v1.utils.loaders import environment_class

MODULE = "torchtitan.rl.experiments.verifiers.swe_rebench_v2"
SMOKE = "rl_grpo_qwen35_9b_swe_rebench_v2_smoke"
MOE = "rl_grpo_qwen36_35b_a3b_swe_rebench_v2_dist_moe"


def _row(instance_id: str, repo: str, difficulty: str) -> dict:
    return {
        "FAIL_TO_PASS": ["tests/test_demo.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_demo.py::test_existing"],
        "base_commit": "abc123",
        "image_name": f"docker.io/swerebenchv2/{instance_id}:latest",
        "install_config": {"log_parser": "parse_log_pytest", "test_cmd": "pytest -rA"},
        "instance_id": instance_id,
        "language": "python",
        "meta": {
            "num_modified_files": 1,
            "num_modified_lines": 10,
            "llm_metadata": {"code": "A", "difficulty": difficulty},
        },
        "problem_statement": "Fix the demo.",
        "repo": repo,
        "test_patch": "diff --git a/tests/test_demo.py b/tests/test_demo.py\n",
    }


ROWS = [
    _row("org_a__repo_a-1", "org-a/repo-a", "easy"),
    _row("org_a__repo_a-2", "org-a/repo-a", "medium"),
    _row("org_b__repo_b-1", "org-b/repo-b", "easy"),
]


@pytest.fixture
def fake_dataset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        taskset, "load_dataset", lambda *args, **kwargs: Dataset.from_list(ROWS)
    )
    taskset._source_rows.cache_clear()
    yield
    taskset._source_rows.cache_clear()


def _load(**config: object) -> list[taskset.SWERebenchV2Task]:
    return list(
        taskset.SWERebenchV2Taskset(taskset.SWERebenchV2Config(id="swe", **config))
    )


def test_repository_level_split_is_disjoint(fake_dataset: None) -> None:
    train, evaluation = _load(partition="train"), _load(partition="eval")
    assert train and evaluation
    assert {task.data.repo for task in train}.isdisjoint(
        task.data.repo for task in evaluation
    )


def test_tasks_carry_the_repository_snapshot(fake_dataset: None) -> None:
    tasks = _load(partition="train") + _load(partition="eval")
    task = next(task for task in tasks if task.data.instance_id == "org_b__repo_b-1")
    assert task.data.image == "docker.io/swerebenchv2/org_b__repo_b-1:latest"
    assert task.data.workdir == "/repo-b"
    assert task.data.install_config["test_cmd"] == ["pytest -rA"]
    assert task.data.resources.cpu == 4


def test_difficulty_filter_keeps_only_easy(fake_dataset: None) -> None:
    tasks = _load(partition="train", difficulties=("easy",)) + _load(
        partition="eval", difficulties=("easy",)
    )
    assert {task.data.difficulty for task in tasks} == {"easy"}


def test_instance_ids_select_a_subset_and_reject_unknown_ids(
    fake_dataset: None,
) -> None:
    train_ids = [task.data.instance_id for task in _load(partition="train")]
    assert [
        task.data.instance_id
        for task in _load(partition="train", instance_ids=(train_ids[0],))
    ] == [train_ids[0]]
    with pytest.raises(ValueError, match="not in the selected partition"):
        _load(partition="train", instance_ids=("missing__repo-1",))


def test_a_changed_selection_size_fails(fake_dataset: None) -> None:
    with pytest.raises(ValueError, match="selection changed"):
        _load(partition="train", expected_num_tasks=1000)


def test_llm_metadata_accepts_the_dataset_list_shape() -> None:
    row = _row("org_a__repo_a-1", "org-a/repo-a", "medium")
    row["meta"]["llm_metadata"] = [row["meta"]["llm_metadata"]]
    assert taskset._metadata(row)[1]["difficulty"] == "medium"


def test_a_local_json_file_loads_without_a_hub_revision(tmp_path: Path) -> None:
    dataset_path = tmp_path / "rows.json"
    dataset_path.write_text(json.dumps(ROWS))
    taskset._source_rows.cache_clear()
    rows = taskset._source_rows(str(dataset_path), "train", "ignored-revision")
    taskset._source_rows.cache_clear()
    assert [row["instance_id"] for row in rows] == [row["instance_id"] for row in ROWS]


@pytest.fixture
def fake_log_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the SWE-rebench-V2 checkout's `lib.agent.log_parsers`."""
    log_parsers = ModuleType("lib.agent.log_parsers")
    log_parsers.NAME_TO_PARSER = {
        "fake": lambda test_output: {
            "tests/test_demo.py::test_fix [0.1s]": "PASSED",
            "tests/test_demo.py::test_existing": "FAILED",
        }
    }
    monkeypatch.setitem(sys.modules, "lib.agent.log_parsers", log_parsers)


@pytest.mark.parametrize(
    ("reward_mode", "expected"),
    [("f2p_regression", 0.5), ("dense_80_20", 0.8), ("official", 0.0)],
)
def test_reward_scores_fixed_tests_and_regressions(
    reward_mode: str, expected: float, fake_log_parser: None
) -> None:
    reward, report = taskset.SWERebenchV2Task.calculate_reward(
        test_output="log",
        info={
            "FAIL_TO_PASS": ["tests/test_demo.py::test_fix"],
            "PASS_TO_PASS": ["tests/test_demo.py::test_existing"],
            "install_config": {"log_parser": "fake"},
        },
        reward_mode=reward_mode,
    )
    assert reward == pytest.approx(expected)
    assert (report["f2p_fraction"], report["p2p_fraction"]) == (1.0, 0.0)


def test_reward_rejects_an_unknown_parser(fake_log_parser: None) -> None:
    with pytest.raises(ValueError, match="unsupported SWE-rebench V2 parser"):
        taskset.SWERebenchV2Task.calculate_reward(
            test_output="log",
            info={
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "install_config": {"log_parser": "missing"},
            },
        )


class _FakeRuntime:
    """Records sandbox commands and file writes; answers every command the same."""

    def __init__(self, result: ProgramResult) -> None:
        self.result = result
        self.commands: list[list[str]] = []
        self.writes: dict[str, bytes] = {}

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append(argv)
        return self.result

    async def write(self, path: str, data: bytes) -> None:
        self.writes[path] = data


def _task() -> taskset.SWERebenchV2Task:
    return taskset.SWERebenchV2Task(
        taskset.SWERebenchV2TaskData(
            idx=0,
            prompt="Fix it.",
            image="docker.io/swerebenchv2/example:latest",
            workdir="/repo-a",
            instance_id="org_a__repo_a-1",
            repo="org-a/repo-a",
            difficulty="easy",
            base_commit="abc123",
            test_patch="diff --git a/tests/test_demo.py b/tests/test_demo.py\n",
            install_config={"log_parser": "parse_log_pytest", "test_cmd": ["pytest"]},
            fail_to_pass=["tests/test_demo.py::test_fix"],
            pass_to_pass=[],
            reward_mode="f2p_regression",
        )
    )


@pytest.mark.parametrize(
    ("stdout", "ok"), [("abc123\n/repo-a\n", True), ("def456\n/repo-a\n", False)]
)
def test_setup_checks_the_image_is_at_the_base_commit(stdout: str, ok: bool) -> None:
    runtime = _FakeRuntime(ProgramResult(exit_code=0, stdout=stdout, stderr=""))
    if ok:
        asyncio.run(_task().setup(None, runtime))
    else:
        with pytest.raises(RuntimeError, match="image mismatch"):
            asyncio.run(_task().setup(None, runtime))


def test_grading_restores_tests_then_runs_the_test_command() -> None:
    runtime = _FakeRuntime(
        ProgramResult(
            exit_code=0, stdout="1 passed\nSWE_REBENCH_V2_TEST_EXIT=0\n", stderr=""
        )
    )
    output = asyncio.run(_task()._run_tests(runtime))
    assert "1 passed" in output
    assert runtime.writes["/tmp/swerebench-v2-test.patch"].startswith(b"diff --git")
    script = runtime.commands[0][-1]
    assert 'git checkout abc123 -- "$test_file"' in script
    assert "\n  pytest\n" in script


def test_grading_without_a_test_log_raises() -> None:
    runtime = _FakeRuntime(
        ProgramResult(exit_code=82, stdout="SWE_REBENCH_V2_APPLY_EXIT=1\n", stderr="")
    )
    with pytest.raises(RuntimeError, match="tests errored"):
        asyncio.run(_task()._run_tests(runtime))


def _select_sandoq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VF_SANDBOX_PROVIDER", "oci-runner")


def test_sandoq_runs_the_agent_outside_on_the_prime_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_sandoq(monkeypatch)
    config = swe_rebench_v2_rollouter_config(sandbox="sandoq", max_rollout_tokens=65536)
    env_server = config.verifiers_env_server
    agent = env_server.environment.agent
    assert isinstance(agent.runtime, vf.PrimeConfig)
    assert isinstance(agent.harness, AgentOutsideHarnessConfig)
    assert agent.max_turns == NUM_AGENT_TURNS
    assert env_server.serve.pool.num_workers * env_server.serve.max_concurrent == 128
    assert config.advantage.should_std_normalize
    assert config.train_dataset.verifiers_taskset.difficulties == ("easy",)
    assert config.validation_dataset.verifiers_taskset.partition == "eval"
    assert (
        environment_class(env_server.environment.taskset.id) is taskset.SWERebenchV2Env
    )


def test_docker_keeps_the_same_agent_on_a_local_container() -> None:
    config = swe_rebench_v2_rollouter_config(sandbox="docker", max_rollout_tokens=65536)
    agent = config.verifiers_env_server.environment.agent
    assert isinstance(agent.runtime, vf.DockerConfig)
    assert isinstance(agent.harness, AgentOutsideHarnessConfig)


def test_sandoq_requires_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VF_SANDBOX_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="VF_SANDBOX_PROVIDER=oci-runner"):
        swe_rebench_v2_rollouter_config(sandbox="sandoq", max_rollout_tokens=65536)


@pytest.mark.parametrize("sandbox", ["docker", "sandoq"])
def test_worker_process_resolves_the_env_and_harness_from_a_fresh_interpreter(
    sandbox: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env-server worker imports only the local taskset module, then the config."""
    _select_sandoq(monkeypatch)
    config = swe_rebench_v2_rollouter_config(sandbox=sandbox, max_rollout_tokens=65536)
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
    assert result.stdout.split()[-2:] == ["SWERebenchV2Env", "AgentOutsideHarness"]


def _swe_env(monkeypatch: pytest.MonkeyPatch) -> taskset.SWERebenchV2Env:
    """A `SWERebenchV2Env` configured like the sandoq recipe, without loading tasks."""
    _select_sandoq(monkeypatch)
    config = swe_rebench_v2_rollouter_config(sandbox="sandoq", max_rollout_tokens=65536)
    env = object.__new__(taskset.SWERebenchV2Env)
    env.config = config.verifiers_env_server.environment
    return env


def test_env_needs_no_tunnel_for_the_agent_outside_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _swe_env(monkeypatch)._runs_local()


def test_env_hands_sandoq_the_snapshot_to_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _swe_env(monkeypatch)
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
    seen_by_env: list[dict | None] = []

    async def single_agent_run(
        self: SingleAgentEnv, task: object, agents: object
    ) -> None:
        seen_by_env.append(task_context.get())

    monkeypatch.setattr(SingleAgentEnv, "run", single_agent_run)
    asyncio.run(env.run(_task(), agents=None))
    assert seen_by_env == [
        {
            "instance_id": "org_a__repo_a-1",
            "requested_image": "docker.io/swerebenchv2/example:latest",
            "working_dir": "/repo-a",
            "base_commit": "abc123",
        }
    ]


def _recipe(name: str) -> Controller.Config:
    return ConfigLoader().load(["--module", MODULE, "--config", name])


@pytest.mark.parametrize(
    ("name", "trainer_gpus", "generator_gpus", "context"),
    [(SMOKE, 4, 4, 65536), (MOE, 4, 4, 131072)],
)
def test_recipes_fit_one_4_gpu_host_per_role(
    name: str, trainer_gpus: int, generator_gpus: int, context: int
) -> None:
    config = _recipe(name)
    trainer = config.trainer.parallelism
    generator = config.generator.parallelism
    assert trainer.data_parallel_shard_degree * trainer.tensor_parallel_degree == (
        trainer_gpus
    )
    assert config.num_generators == 1
    assert generator.data_parallel_degree * generator.tensor_parallel_degree == (
        generator_gpus
    )
    assert config.trainer.training.max_context_length == context
    assert config.rollouter.generation_server.max_rollout_tokens == context
    assert config.generator.sampling.max_tokens == 16384
    assert config.trainer.training.dtype == "float32"
    assert config.renderer.renderers_config.name == "qwen3.5"
    assert config.renderer.renderers_config.thinking_retention == "all"
    assert not config.async_loop.training_sample_builder.drop_zero_std_reward_groups


def test_recipes_select_the_sandbox_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_sandoq(monkeypatch)
    monkeypatch.setenv("SWE_REBENCH_V2_SANDBOX", "sandoq")
    agent = _recipe(SMOKE).rollouter.verifiers_env_server.environment.agent
    assert isinstance(agent.runtime, vf.PrimeConfig)


def test_moe_recipe_uses_dist_moe_in_the_trainer_only() -> None:
    """The generator keeps stock MoE; the trainer's model copy gets Dist-MoE."""
    config = _recipe(MOE)
    config.to_dict()
    trainer = config.trainer
    assert trainer.dist_moe.device_scratch_capacity_factor == (
        trainer.parallelism.expert_parallel_degree
    )
    assert config.generator.override.imports == []
    assert config.generator.cuda_graph.mode == "NONE"
    num_moe_layers = len(list(config.model.traverse(RoutedExperts.Config)))

    # Prepare the trainer's model copy the way Trainer.__init__ does.
    model_config = copy.deepcopy(config.model)
    model_config.set_sharding_(trainer.parallelism)
    apply_overrides(trainer.override, model_config)
    validate_model_training_config(
        model_config,
        parallelism=trainer.parallelism,
        training=trainer.training,
        debug=trainer.debug,
        activation_checkpoint=trainer.activation_checkpoint,
        max_num_documents=config.async_loop.batcher.max_num_documents,
    )
    assert not list(model_config.traverse(RoutedExperts.Config))
    assert len(list(model_config.traverse(DistMoeRoutedExperts.Config))) == (
        num_moe_layers
    )
