# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE-rebench V2 tasks for Verifiers: curriculum split, sandbox check and test reward.

Each task is a repository image (`docker.io/swerebenchv2/...`) at a base commit.
Grading restores the task's test files, applies its test patch, runs its test
command and parses the log with the project's parser from the SWE-rebench-V2
repository (`lib.agent.log_parsers`). That repository is not on PyPI, so its
checkout must be on `PYTHONPATH` wherever rollouts are graded.
"""

import hashlib
import json
import logging
import re
import shlex
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import verifiers.v1 as vf
from datasets import load_dataset
from verifiers.v1.envs.single_agent import SingleAgentEnv

from torchtitan.rl.experiments.verifiers.agent_outside import (
    AgentOutsideHarnessConfig,
    sandoq_task_context,
)

logger = logging.getLogger(__name__)

RewardMode = Literal["official", "f2p_regression", "dense_80_20"]

_TIMING_SUFFIXES = (
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
)


class SWERebenchV2Config(vf.TasksetConfig):
    """Which dataset rows become tasks: a curriculum filter, then a repository-level split."""

    dataset: str = "nebius/SWE-rebench-V2"
    """Hugging Face dataset id, or a local `.json`/`.parquet` file with the same rows."""

    split: str = "train"

    revision: str = "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0"
    """Hub revision; ignored for a local file."""

    partition: Literal["train", "eval"] = "train"
    """`eval` is every selected task of `heldout_repository_count` hashed repositories."""

    difficulties: tuple[Literal["easy", "medium"], ...] = ()
    """Keep only these difficulties within the partition; empty keeps all."""

    instance_ids: tuple[str, ...] = ()
    """Keep only these instances, e.g. for a smoke test; empty keeps all."""

    reward_mode: RewardMode = "f2p_regression"
    """Which score from `SWERebenchV2Task.calculate_reward` is the reward."""

    languages: tuple[str, ...] = ("python",)
    code_grade: str = "A"
    excluded_issue_labels: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5", "B6")
    curriculum_difficulties: tuple[Literal["easy", "medium"], ...] = ("easy", "medium")
    max_modified_files: int = 9
    max_modified_lines: int = 181
    heldout_seed: int = 20260811
    heldout_repository_count: int = 64
    unavailable_images: tuple[str, ...] = (
        "docker.io/swerebenchv2/dhi-mikeio:690-7882682",
    )
    unavailable_repositories: tuple[str, ...] = ("modin-project/modin",)
    unavailable_instances: tuple[str, ...] = (
        "aio-libs__aiohttp-10556",
        "keras-team__keras-19466",
        "openmdao__openmdao-3184",
        "pybamm-team__pybamm-4072",
        "pybamm-team__pybamm-4239",
        "qiskit__qiskit-12055",
        "wntrblm__nox-829",
    )

    expected_num_tasks: int | None = None
    """Fail when the selection size differs, e.g. after a dataset or filter change."""

    agent_system_prompt: str = (
        "You are a coding agent working in the root of a task repository. Use the "
        "bash tool to inspect and edit the repository, run focused tests, and "
        "implement the requested fix. The task image already contains the repository "
        "and its language-specific dependencies. Do not modify tests. Do not spend "
        "the entire turn budget only investigating: once you identify the likely root "
        "cause, make a concrete source-code edit and run focused tests to validate it."
    )


class SWERebenchV2TaskData(vf.TaskData):
    """One dataset row: the repository snapshot plus what grading needs."""

    instance_id: str
    repo: str
    difficulty: str
    base_commit: str
    test_patch: str
    install_config: dict[str, Any]
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    reward_mode: RewardMode

    def grader_info(self) -> dict[str, Any]:
        return {
            "FAIL_TO_PASS": self.fail_to_pass,
            "PASS_TO_PASS": self.pass_to_pass,
            "install_config": self.install_config,
        }


class SWERebenchV2Task(vf.Task[SWERebenchV2TaskData]):
    """Check the sandbox holds the task's snapshot, then grade the agent's edits."""

    NEEDS_CONTAINER = True

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del trace
        result = await runtime.run(
            ["bash", "-lc", 'printf \'%s\\n\' "$(git rev-parse HEAD)" "$(pwd -P)"'], {}
        )
        actual = result.stdout.splitlines()
        expected = [self.data.base_commit, self.data.workdir]
        if result.exit_code != 0 or actual != expected:
            raise RuntimeError(
                f"SWE-rebench V2 image mismatch for {self.data.instance_id}: "
                f"expected {expected!r}, got {actual!r}"
            )

    @vf.reward
    async def swe_rebench_v2_reward(
        self, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        test_output = await self._run_tests(runtime)
        reward, report = self.calculate_reward(
            test_output=test_output,
            info=self.data.grader_info(),
            reward_mode=self.data.reward_mode,
        )
        trace.info.update(
            swe_rebench_v2_test_output_tail=test_output[-4000:],
            swe_rebench_v2_test_report=report,
        )
        logger.info(
            "SWE-rebench V2 score: instance=%s reward=%.4f official_full_resolution=%s",
            self.data.instance_id,
            reward,
            report["official_full_resolution"],
        )
        return reward

    async def _run_tests(self, runtime: vf.Runtime) -> str:
        """Restore the held-out tests, run them, and return their raw output.

        Only files named by `test_patch` are reset to `base_commit`, so the agent's
        source edits stay. A failing test command is valid grader input; a patch or
        execution failure raises because there is no log to parse.
        """
        patch_path = "/tmp/swerebench-v2-test.patch"
        await runtime.write(patch_path, self.data.test_patch.encode())
        test_files = " ".join(
            shlex.quote(path) for path in _test_files(self.data.test_patch)
        )
        test_commands = "\n".join(self.data.install_config["test_cmd"])
        base_commit = shlex.quote(self.data.base_commit)
        command = f"""
set +e
set +u
cd {shlex.quote(self.data.workdir)} || exit 80
for test_file in {test_files}; do
  if git cat-file -e {base_commit}:"$test_file" 2>/dev/null; then
    git checkout {base_commit} -- "$test_file" || exit 81
  else
    rm -f -- "$test_file" || exit 81
  fi
done
git apply --verbose --3way --recount --ignore-space-change --whitespace=nowarn {shlex.quote(patch_path)}
apply_rc=$?
if [ "$apply_rc" -ne 0 ]; then
  printf '\\nSWE_REBENCH_V2_APPLY_EXIT=%s\\n' "$apply_rc"
  exit 82
fi
(
  set -e
  {test_commands}
)
test_rc=$?
printf '\\nSWE_REBENCH_V2_TEST_EXIT=%s\\n' "$test_rc"
true
"""
        result = await runtime.run(["bash", "-lc", command], {})
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.exit_code != 0 or "SWE_REBENCH_V2_TEST_EXIT=" not in output:
            raise RuntimeError(f"SWE-rebench V2 tests errored: {output[-2000:]}")
        return output

    @staticmethod
    def calculate_reward(
        *,
        test_output: str,
        info: dict[str, Any],
        reward_mode: RewardMode = "f2p_regression",
    ) -> tuple[float, dict[str, Any]]:
        """Score fixed tests (F2P) without hiding regressions (P2P).

        The task's log parser maps test names to statuses. `FAIL_TO_PASS` must turn
        green; `PASS_TO_PASS` must stay green; an empty category counts as 1.

        Example: F2P 1/1 passed and P2P 0/1 passed gives `f2p_regression`
        `1 * (0.5 + 0.5 * 0) = 0.5`, `dense_80_20` `0.8`, `official` `0.0`.
        """
        # Only grading needs the SWE-rebench-V2 checkout, so import it here.
        from lib.agent.log_parsers import NAME_TO_PARSER

        parser_name = info["install_config"]["log_parser"]
        parser = NAME_TO_PARSER.get(parser_name)
        if parser is None:
            raise ValueError(f"unsupported SWE-rebench V2 parser: {parser_name!r}")
        status_by_test = {
            _normalize_test_name(str(name)): str(status)
            for name, status in parser(test_output).items()
        }
        expected = {
            category: [_normalize_test_name(name) for name in info[category]]
            for category in ("FAIL_TO_PASS", "PASS_TO_PASS")
        }
        passed_actual = sorted(
            name for name, status in status_by_test.items() if status == "PASSED"
        )
        passed = set(passed_actual)
        fractions = {
            category: len(passed.intersection(names)) / len(names) if names else 1.0
            for category, names in expected.items()
        }
        expected_passed = sorted(
            set(expected["FAIL_TO_PASS"]) | set(expected["PASS_TO_PASS"])
        )
        official = passed_actual == expected_passed
        rewards = {
            "official": float(official),
            "f2p_regression": fractions["FAIL_TO_PASS"]
            * (0.5 + 0.5 * fractions["PASS_TO_PASS"]),
            "dense_80_20": 0.8 * fractions["FAIL_TO_PASS"]
            + 0.2 * fractions["PASS_TO_PASS"],
        }
        return rewards[reward_mode], {
            "official_full_resolution": official,
            "f2p_fraction": fractions["FAIL_TO_PASS"],
            "p2p_fraction": fractions["PASS_TO_PASS"],
            "passed_actual": passed_actual,
            "expected_passed": expected_passed,
            "parser": parser_name,
            "reward_mode": reward_mode,
        }


class SWERebenchV2Taskset(vf.Taskset[SWERebenchV2Task, SWERebenchV2Config]):
    """Load the pinned rows, keep the curriculum and the configured partition."""

    def load(self) -> Iterator[SWERebenchV2Task]:
        config = self.config
        selected = tuple(
            row
            for row in _source_rows(config.dataset, config.split, config.revision)
            if _matches_curriculum(row, config)
        )
        heldout = _heldout_repositories(
            [str(row["repo"]) for row in selected],
            count=config.heldout_repository_count,
            seed=config.heldout_seed,
        )
        rows = []
        for row in selected:
            partition = "eval" if row["repo"] in heldout else "train"
            _, llm = _metadata(row)
            if partition == config.partition and (
                not config.difficulties or llm.get("difficulty") in config.difficulties
            ):
                rows.append(row)

        if config.instance_ids:
            requested = set(config.instance_ids)
            rows = [row for row in rows if row["instance_id"] in requested]
            missing = requested - {str(row["instance_id"]) for row in rows}
            if missing:
                raise ValueError(
                    "requested SWE-rebench V2 instances are not in the selected "
                    f"partition: {sorted(missing)}"
                )
        elif (
            config.expected_num_tasks is not None
            and len(rows) != config.expected_num_tasks
        ):
            raise ValueError(
                f"pinned SWE-rebench V2 {config.partition} selection changed: "
                f"{len(rows)} != {config.expected_num_tasks}"
            )

        for index, row in enumerate(rows):
            install_config = dict(row["install_config"])
            test_cmd = install_config.get("test_cmd")
            install_config["test_cmd"] = (
                [test_cmd] if isinstance(test_cmd, str) else list(test_cmd)
            )
            _, llm = _metadata(row)
            data = SWERebenchV2TaskData(
                idx=index,
                name=row["instance_id"],
                prompt=row["problem_statement"],
                system_prompt=config.agent_system_prompt,
                image=row["image_name"],
                workdir=f"/{str(row['repo']).split('/', 1)[1]}",
                resources=vf.TaskResources(cpu=4, memory=16, disk=40),
                instance_id=row["instance_id"],
                repo=row["repo"],
                difficulty=str(llm.get("difficulty")),
                base_commit=row["base_commit"],
                test_patch=row["test_patch"],
                install_config=install_config,
                fail_to_pass=list(row["FAIL_TO_PASS"]),
                pass_to_pass=list(row["PASS_TO_PASS"]),
                reward_mode=config.reward_mode,
            )
            yield SWERebenchV2Task(data, config.task)


class SWERebenchV2Env(SingleAgentEnv):
    """Single-agent env, plus what the agent-outside harness and Sandoq need."""

    def _runs_local(self) -> bool:
        # The agent-outside loop calls the model from this process, so a remote
        # sandbox needs no tunnel back to it.
        return (
            isinstance(self.config.agent.harness, AgentOutsideHarnessConfig)
            or super()._runs_local()
        )

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        # Sandoq also checks the image's repository is at `base_commit`.
        with sandoq_task_context(
            instance_id=task.data.instance_id,
            requested_image=task.data.image,
            working_dir=task.data.workdir,
            base_commit=task.data.base_commit,
        ):
            await super().run(task, agents)


def _metadata(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a row's `meta` and its LLM-graded `llm_metadata` (a dict or a 1-list)."""
    meta = row.get("meta")
    if isinstance(meta, str):
        meta = json.loads(meta)
    meta = meta if isinstance(meta, dict) else {}
    llm = meta.get("llm_metadata")
    if isinstance(llm, list):
        llm = next((item for item in llm if isinstance(item, dict)), {})
    llm = llm if isinstance(llm, dict) else {}
    return meta, llm


def _matches_curriculum(row: dict[str, Any], config: SWERebenchV2Config) -> bool:
    meta, llm = _metadata(row)
    issues = llm.get("detected_issues")
    issues = issues if isinstance(issues, dict) else {}
    return (
        row.get("image_name") not in config.unavailable_images
        and row.get("repo") not in config.unavailable_repositories
        and row.get("instance_id") not in config.unavailable_instances
        and row.get("language") in config.languages
        and llm.get("code") == config.code_grade
        and not any(bool(issues.get(label)) for label in config.excluded_issue_labels)
        and llm.get("difficulty") in config.curriculum_difficulties
        and isinstance(meta.get("num_modified_files"), int)
        and meta["num_modified_files"] <= config.max_modified_files
        and isinstance(meta.get("num_modified_lines"), int)
        and meta["num_modified_lines"] <= config.max_modified_lines
    )


def _heldout_repositories(
    repositories: Sequence[str], *, count: int, seed: int
) -> set[str]:
    """Pick `count` repositories by seeded hash, always leaving one for training."""
    unique = sorted(set(repositories))
    ranked = sorted(
        unique, key=lambda repo: hashlib.sha256(f"{seed}:{repo}".encode()).digest()
    )
    return set(ranked[: min(count, len(unique) - 1)])


def _normalize_test_name(name: str) -> str:
    """Strip timing suffixes, e.g. `test_a [0.3s]` -> `test_a`."""
    for pattern in _TIMING_SUFFIXES:
        name = pattern.sub("", name)
    return name.strip()


@lru_cache(maxsize=4)
def _source_rows(dataset: str, split: str, revision: str) -> tuple[dict[str, Any], ...]:
    dataset_path = Path(dataset)
    if dataset_path.is_file():
        source = load_dataset(
            dataset_path.suffix.removeprefix("."),
            data_files=str(dataset_path),
            split=split,
            keep_in_memory=False,
        )
    else:
        source = load_dataset(
            dataset, split=split, revision=revision, keep_in_memory=False
        )
    return tuple(dict(row) for row in source)


def _test_files(test_patch: str) -> list[str]:
    files = re.findall(r"^diff --git a/.+ b/(.+)$", test_patch, flags=re.MULTILINE)
    if not files:
        raise ValueError("test patch contains no files")
    return list(dict.fromkeys(files))


# Verifiers discovers the env from this taskset plugin.
__all__ = ["SWERebenchV2Taskset", "SWERebenchV2Env"]
