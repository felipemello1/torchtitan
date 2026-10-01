# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the TitanRL Terminal-Bench recipes."""

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf

from torchtitan.config import ConfigLoader
from torchtitan.rl.controller import Controller
from torchtitan_recipes.rl.terminal_bench.agent_outside import AgentOutsideHarnessConfig
from torchtitan_recipes.rl.terminal_bench.taskset import TerminalBenchEnv
from torchtitan_recipes.rl.verifiers_terminal_bench import (
    _sandbox_runtime,
    _terminal_bench_rollouter_config,
    TERMINAL_BENCH_2_1_86,
    TMAX_1K,
)
from verifiers.v1.utils.loaders import environment_class


def _load(name: str) -> Controller.Config:
    return ConfigLoader().load(
        ["--module", "torchtitan_recipes.rl.verifiers_terminal_bench", "--config", name]
    )


def _select_oci_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VF_SANDBOX_PROVIDER", "oci-runner")
    monkeypatch.setenv("OCI_RUNNER_TASK_NETWORK", "host")


def test_recipe_trains_on_tmax_and_validates_on_terminal_bench(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _select_oci_runner(monkeypatch)
    config = _load("rl_grpo_qwen3_6_35b_a3b_terminal_bench")
    rollouter = config.rollouter
    agent = rollouter.verifiers_env_server.environment.agent
    serve = rollouter.verifiers_env_server.serve

    assert rollouter.train_dataset.verifiers_taskset.dataset == TMAX_1K
    assert rollouter.validation_dataset.verifiers_taskset.dataset == (
        TERMINAL_BENCH_2_1_86
    )
    assert not rollouter.train_dataset.shuffle
    assert config.async_loop.validation.num_samples == 86
    assert config.async_loop.validation.interval_steps == 25
    assert not config.async_loop.validation.greedy
    assert not config.async_loop.training_sample_builder.drop_zero_std_reward_groups
    assert config.async_loop.target_offpolicy_steps == 1
    assert isinstance(agent.harness, AgentOutsideHarnessConfig)
    assert isinstance(agent.runtime, vf.PrimeConfig)
    assert agent.max_turns == 30
    assert agent.harness.max_tool_output_chars == 16384
    assert serve.pool.num_workers * serve.max_concurrent == 384
    assert config.generator.sampling.max_tokens == 4096
    assert config.trainer.training.max_context_length == 65536
    assert rollouter.generation_server.max_rollout_tokens == 65536
    assert environment_class(rollouter.train_dataset.verifiers_taskset.id) is (
        TerminalBenchEnv
    )


def test_recipe_layout_fits_two_four_gpu_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trainer EP stays inside one host; four one-GPU generators fill the other host."""
    _select_oci_runner(monkeypatch)
    config = _load("rl_grpo_qwen3_6_35b_a3b_terminal_bench")
    trainer = config.trainer.parallelism
    generator = config.generator.parallelism

    assert trainer.data_parallel_shard_degree * trainer.tensor_parallel_degree == 4
    assert trainer.expert_parallel_degree == 4
    assert config.num_generators == 4
    assert generator.data_parallel_degree * generator.tensor_parallel_degree == 1
    assert generator.expert_parallel_degree == 1
    assert config.generator.cuda_graph.mode == "FULL"
    assert config.trainer.override.imports == [
        "torchtitan_recipes.overrides.dist_moe.dist_moe_routed_experts"
    ]


def test_training_cannot_read_validation_data() -> None:
    with pytest.raises(ValueError, match="different datasets"):
        _terminal_bench_rollouter_config(
            train_dataset=TERMINAL_BENCH_2_1_86,
            validation_dataset=TERMINAL_BENCH_2_1_86,
        )


def test_sandbox_runtime_follows_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VF_SANDBOX_PROVIDER", raising=False)
    assert isinstance(_sandbox_runtime(), vf.DockerConfig)
    _select_oci_runner(monkeypatch)
    assert isinstance(_sandbox_runtime(), vf.PrimeConfig)


def test_oci_runner_requires_host_network(monkeypatch: pytest.MonkeyPatch) -> None:
    _select_oci_runner(monkeypatch)
    monkeypatch.delenv("OCI_RUNNER_TASK_NETWORK")
    with pytest.raises(ValueError, match="OCI_RUNNER_TASK_NETWORK=host"):
        _sandbox_runtime()
