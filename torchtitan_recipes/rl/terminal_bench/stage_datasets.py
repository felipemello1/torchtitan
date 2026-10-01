# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build the two Harbor task trees the Terminal-Bench recipes train and validate on.

Verifiers reads a Harbor dataset id from `$HOME/.cache/harbor/<id with "/" and "@" as "_">`
and downloads nothing when that directory exists. Neither id below is on the Harbor Hub;
this script builds both trees from public sources (needs internet, the `harbor` CLI,
`huggingface_hub` and `pyarrow`)::

    python -m torchtitan_recipes.rl.terminal_bench.stage_datasets --out ~/.cache/harbor
    # -> ~/.cache/harbor/tmax-mc1024_27de1c1b/tmax/<1,024 task dirs>
    #    ~/.cache/harbor/tb21-86_7131e437/terminal-bench-2-1/<86 task dirs>
"""

import argparse
import random
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# TMax-15K tasks from prime-envs' git-backed Harbor registry (prime-tasks @ 27de1c1b).
# Their task.toml names no image; the open-instruct release maps each task to a public
# Docker Hub image, and every task's setup creates /home/user (/app is often missing).
TMAX_DATASET = "tmax@2026-07-01"
TMAX_REGISTRY = "PrimeIntellect-ai/prime-envs@8797af59115e6a5af7b2834996f8559e35267d3a"
TMAX_IMAGES_REPO = "allenai/tmax-15k-open-instruct"
TMAX_IMAGES_REVISION = "7b090eca98bf351356bc1c64290c5c4a09f2f98c"
TMAX_WORKDIR = "/home/user"
# Short tasks are mostly solved by every rollout of a group (no reward variance);
# intricate ones expect 30-60 commands.
TMAX_COMPLEXITIES = ("moderate task", "complex task")
NUM_TMAX_TASKS = 1024
TMAX_OUTPUT = "tmax-mc1024_27de1c1b"

# Terminal-Bench 2.1, Harbor Hub revision 6: the tasks of harbor-framework/terminal-bench-2-1@7131e437
# (the git copy differs only in how sanitize-git-repo's test splits its fake token literals).
TERMINAL_BENCH_DATASET = "terminal-bench/terminal-bench-2-1@6"
# These three do not run in a remote Firecracker VM sandbox.
TERMINAL_BENCH_EXCLUDED = ("qemu-alpine-ssh", "qemu-startup", "protein-assembly")
TERMINAL_BENCH_OUTPUT = "tb21-86_7131e437"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path.home() / ".cache" / "harbor")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as work:
        stage_tmax(out=args.out, work=Path(work))
        stage_terminal_bench(out=args.out, work=Path(work))


def stage_tmax(*, out: Path, work: Path) -> None:
    """Write `NUM_TMAX_TASKS` seeded moderate and complex TMax tasks, each with its image and workdir."""
    harbor_download(TMAX_DATASET, work / "tmax", "--repo", TMAX_REGISTRY)
    parquet = hf_hub_download(
        TMAX_IMAGES_REPO,
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
        revision=TMAX_IMAGES_REVISION,
    )
    rows = pq.read_table(parquet, columns=["env_config"]).column("env_config")
    images = {row["task_id"]: row["image"] for row in rows.to_pylist()}

    task_dirs = sorted(toml.parent for toml in (work / "tmax").glob("tmax/*/task.toml"))
    pool = [
        task_dir for task_dir in task_dirs if complexity(task_dir) in TMAX_COMPLEXITIES
    ]
    selected = sorted(random.Random(0).sample(pool, NUM_TMAX_TASKS))

    output = out / TMAX_OUTPUT / "tmax"
    shutil.rmtree(output.parent, ignore_errors=True)
    for task_dir in selected:
        shutil.copytree(task_dir, output / task_dir.name)
        toml = output / task_dir.name / "task.toml"
        toml.write_text(
            set_environment(
                toml.read_text(),
                docker_image=images[task_dir.name],
                workdir=TMAX_WORKDIR,
            )
        )
    print(f"{TMAX_OUTPUT}: {len(selected)} of {len(pool)} moderate and complex tasks")


def stage_terminal_bench(*, out: Path, work: Path) -> None:
    """Write Terminal-Bench 2.1 without `TERMINAL_BENCH_EXCLUDED`."""
    harbor_download(TERMINAL_BENCH_DATASET, work / "terminal-bench")
    output = out / TERMINAL_BENCH_OUTPUT / "terminal-bench-2-1"
    shutil.rmtree(output.parent, ignore_errors=True)
    task_dirs = sorted((work / "terminal-bench" / "terminal-bench-2-1").iterdir())
    kept = [path for path in task_dirs if path.name not in TERMINAL_BENCH_EXCLUDED]
    for task_dir in kept:
        shutil.copytree(task_dir, output / task_dir.name)
    print(f"{TERMINAL_BENCH_OUTPUT}: {len(kept)} tasks")


def harbor_download(dataset: str, output: Path, *extra_args: str) -> None:
    subprocess.run(
        ["harbor", "download", dataset, "--export", "-o", str(output), *extra_args],
        check=True,
    )


def complexity(task_dir: Path) -> str:
    """Return a TMax task's complexity label, e.g. "moderate task (several commands ...)" -> "moderate task"."""
    metadata = tomllib.loads((task_dir / "task.toml").read_text()).get("metadata", {})
    return metadata.get("task_complexity", "").split("(")[0].strip()


def set_environment(task_toml: str, *, docker_image: str, workdir: str) -> str:
    """Return `task_toml` with `docker_image` and `workdir` set in its [environment] table.

    Example::

        set_environment('[environment]\\ncpus = 1\\n', docker_image="org/img:1", workdir="/home/user")
        # -> '[environment]\\ndocker_image = "org/img:1"\\nworkdir = "/home/user"\\ncpus = 1\\n'
    """
    text = re.sub(r"^\s*(docker_image|workdir)\s*=.*\n", "", task_toml, flags=re.M)
    text, count = re.subn(
        r"^\[environment\]\s*$",
        f'[environment]\ndocker_image = "{docker_image}"\nworkdir = "{workdir}"',
        text,
        count=1,
        flags=re.M,
    )
    if count != 1:
        raise ValueError("task.toml has no [environment] table")
    return text


if __name__ == "__main__":
    main()
