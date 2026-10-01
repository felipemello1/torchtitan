# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run Harbor tasks through Verifiers, one sandbox per rollout."""

import os
from dataclasses import dataclass
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig
from verifiers.v1.tasksets.harbor import HarborEnvConfig

from torchtitan.rl.examples.verifiers import (
    GenerationServer,
    RewardFromVerifiers,
    VerifiersEnvServer,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from torchtitan.rl.experiments.verifiers.agent_outside import (
    AgentOutsideHarnessConfig,
    HARNESS_ID as AGENT_OUTSIDE_HARNESS_ID,
)
from torchtitan.rl.experiments.verifiers.terminal_bench import harness
from torchtitan.rl.experiments.verifiers.terminal_bench.harness import (
    NUM_AGENT_TURNS,
    register_harness_alias,
    TerminalBenchTerminusHarnessConfig,
)
from torchtitan.rl.experiments.verifiers.terminal_bench.taskset import (
    TerminalTasksetConfig,
)
from torchtitan.rl.rubric import Rubric


class TerminalBenchRollouter(VerifiersRollouter):
    """Use Verifiers for agent execution and TitanRL for training orchestration."""

    @dataclass(kw_only=True, slots=True)
    class Config(VerifiersRollouter.Config):
        # Per-request timeout on the model call. The base default is 120 s,
        # which cannot be met here: a turn is allowed max_tokens=16384, and
        # finishing that inside 120 s needs a sustained 137 tok/s for one
        # sequence while its 31 group siblings share the same engine. A turn
        # that crosses the deadline raises APITimeoutError, which the harness
        # surfaces as a rollout with no turns and reward 0.0 -- identical to a
        # task the agent genuinely failed. Kept below the 7200 s rollout
        # timeout so a stuck request still loses to the rollout deadline.
        connection_timeout_sec: float = 1800.0


def terminal_bench_rollouter_config(
    train_dataset: str,
    validation_dataset: str,
    *,
    sandbox: Literal["docker", "sandoq"] = "docker",
) -> TerminalBenchRollouter.Config:
    """Select Harbor datasets by id and where each rollout's commands run.

    Args:
        train_dataset: Harbor dataset id to train on.
        validation_dataset: Harbor dataset id to validate on; must differ.
        sandbox: ``"docker"`` runs Terminus-2 inside a Docker container on the
            controller host. ``"sandoq"`` keeps a bash-tool agent loop on the
            controller and sends only its commands to a Sandoq Firecracker VM,
            for a controller that cannot run the task images.
    """
    if train_dataset == validation_dataset:
        raise ValueError(
            "Training and Terminal-Bench evaluation must use different datasets"
        )

    if sandbox == "docker":
        harness_config = TerminalBenchTerminusHarnessConfig(
            id=register_harness_alias(harness.__name__), version="0.22.0"
        )
        runtime = vf.DockerConfig()
        # Every container runs on the controller host.
        pool, max_concurrent = vf.StaticPoolConfig(num_workers=4), 4
    elif sandbox == "sandoq":
        # The Sandoq provider (`sandoq_provider`) serves `vf.PrimeConfig` only
        # when selected. Each task's test.sh installs pytest over the network, so
        # a VM without host networking silently scores every rollout 0.
        if os.environ.get("VF_SANDBOX_PROVIDER") != "oci-runner":
            raise ValueError("sandbox='sandoq' needs VF_SANDBOX_PROVIDER=oci-runner")
        if os.environ.get("OCI_RUNNER_TASK_NETWORK") != "host":
            raise ValueError(
                "sandbox='sandoq' needs OCI_RUNNER_TASK_NETWORK=host, or every "
                "reward is 0"
            )
        harness_config = AgentOutsideHarnessConfig(id=AGENT_OUTSIDE_HARNESS_ID)
        runtime = vf.PrimeConfig(idle_timeout=3600)
        # Rollouts only wait on the model and remote VMs; size
        # OCI_RUNNER_POOL_SIZE to this 8 x 16 = 128.
        pool, max_concurrent = vf.StaticPoolConfig(num_workers=8), 16
    else:
        raise ValueError(f"unknown sandbox {sandbox!r}")

    taskset_id = register_local_taskset_alias(TerminalTasksetConfig.__module__)
    return TerminalBenchRollouter.Config(
        train_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=TerminalTasksetConfig(
                id=taskset_id, dataset=train_dataset
            ),
            seed=42,
            shuffle=True,
        ),
        validation_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=TerminalTasksetConfig(
                id=taskset_id, dataset=validation_dataset
            ),
            seed=99,
            shuffle=False,
        ),
        verifiers_env_server=VerifiersEnvServer.Config(
            environment=HarborEnvConfig(
                agent=vf.AgentConfig(
                    harness=harness_config,
                    runtime=runtime,
                    max_turns=NUM_AGENT_TURNS,
                    timeout=AgentTimeoutConfig(
                        # Covers a cold image pull into a fresh sandbox.
                        setup=1500,
                        rollout=7200,
                        scoring=12000,
                    ),
                ),
            ),
            serve=vf.ServeConfig(
                pool=pool,
                max_concurrent=max_concurrent,
                address="tcp://127.0.0.1:0",
            ),
        ),
        rubric=Rubric.Config(
            reward_fns=[RewardFromVerifiers.Config(weight=1.0)],
            error_reward=0.0,
        ),
        generation_server=GenerationServer.Config(max_rollout_tokens=65536),
    )
