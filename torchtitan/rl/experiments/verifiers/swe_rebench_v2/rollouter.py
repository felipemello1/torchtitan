# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run SWE-rebench V2 through Verifiers, with the agent outside the sandbox."""

import os
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig

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
from torchtitan.rl.experiments.verifiers.swe_rebench_v2.taskset import (
    SWERebenchV2Config,
)
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rubric import Rubric

NUM_AGENT_TURNS = 30


def swe_rebench_v2_rollouter_config(
    *,
    sandbox: Literal["docker", "sandoq"],
    max_rollout_tokens: int,
) -> VerifiersRollouter.Config:
    """Train on the easy tasks of the training repositories; validate on held-out ones.

    The same bash-tool agent runs on the controller in both sandboxes; only where
    its commands and the grader run differs.

    Args:
        sandbox: ``"docker"`` runs each task container on the controller host.
            ``"sandoq"`` runs it in a Sandoq Firecracker VM, for a controller that
            cannot run the task images (they are amd64-only).
        max_rollout_tokens: Tokens in one rollout, prompt and all turns included.
    """
    if sandbox == "docker":
        runtime = vf.DockerConfig()
        # Every container runs on the controller host.
        pool, max_concurrent = vf.StaticPoolConfig(num_workers=4), 4
    elif sandbox == "sandoq":
        # The Sandoq provider (`sandoq_provider`) serves `vf.PrimeConfig` only when
        # selected; otherwise the runtime would call Prime Intellect's API. SWE
        # grading needs no network, so OCI_RUNNER_TASK_NETWORK keeps its `none`.
        if os.environ.get("VF_SANDBOX_PROVIDER") != "oci-runner":
            raise ValueError("sandbox='sandoq' needs VF_SANDBOX_PROVIDER=oci-runner")
        runtime = vf.PrimeConfig(idle_timeout=3600)
        # Rollouts only wait on the model and remote VMs; size
        # OCI_RUNNER_POOL_SIZE to this 8 x 16 = 128.
        pool, max_concurrent = vf.StaticPoolConfig(num_workers=8), 16
    else:
        raise ValueError(f"unknown sandbox {sandbox!r}")

    taskset_id = register_local_taskset_alias(SWERebenchV2Config.__module__)
    return VerifiersRollouter.Config(
        train_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=SWERebenchV2Config(
                id=taskset_id,
                partition="train",
                difficulties=("easy",),
                expected_num_tasks=1635,
            ),
            seed=42,
            shuffle=True,
        ),
        validation_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=SWERebenchV2Config(
                id=taskset_id, partition="eval", expected_num_tasks=442
            ),
            seed=99,
            shuffle=False,
        ),
        verifiers_env_server=VerifiersEnvServer.Config(
            environment=vf.SingleAgentEnvConfig(
                agent=vf.AgentConfig(
                    harness=AgentOutsideHarnessConfig(id=AGENT_OUTSIDE_HARNESS_ID),
                    runtime=runtime,
                    max_turns=NUM_AGENT_TURNS,
                    timeout=AgentTimeoutConfig(
                        # A task image is 1.2-1.9 GB and a fresh VM caches none.
                        setup=1500,
                        rollout=7200,
                        finalize=300,
                        scoring=1500,
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
        advantage=AdvantageEstimator.Config(should_std_normalize=True),
        generation_server=GenerationServer.Config(
            max_rollout_tokens=max_rollout_tokens
        ),
    )
