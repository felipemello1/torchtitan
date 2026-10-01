# Terminal-Bench environment

A bash-tool agent for Harbor terminal tasks (Terminal-Bench, TMax), packaged for
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers). It imports no torchtitan, so trainers
that ship their own torchtitan can run the same agent; that is why it lives outside `torchtitan/`.
TitanRL's recipes are in `torchtitan_recipes/rl/verifiers_terminal_bench.py`.

```text
trainer (TitanRL, prime-rl, ...)
  Verifiers env server (spawned workers)
    TerminalBenchEnv.run(task)                        taskset.py
      AgentOutsideHarness: the agent loop runs here    agent_outside.py
        ├─ chat completion ──> trainer's generator
        └─ bash tool call ──> runtime.run(["bash", "-lc", cmd]) ──> sandbox (Docker container or remote VM)
      Harbor grading: tests/test.sh in the same sandbox -> reward
```

- `agent_outside.py`: the harness only. It calls the model with one `bash` tool and runs each call in
  the sandbox; nothing is installed in the task container and the container never calls the model.
  Tool results keep their tail, with the exit code last.
- `taskset.py`: Verifiers' Harbor taskset and env. Each task starts in its `task.toml` workdir, else
  its image's last `WORKDIR`. With `VF_SANDBOX_PROVIDER=oci-runner`, the env hands the remote-sandbox
  provider the task's image and workdir.
- `stage_datasets.py`: builds the recipe's two datasets (below).

## Datasets

Verifiers reads a Harbor dataset id from `$HOME/.cache/harbor/<id with "/" and "@" as "_">` and
downloads nothing when that directory exists. The recipe's two ids are built, not downloaded:

- `tmax-mc1024@27de1c1b`: 1,024 moderate and complex TMax-15K tasks (`random.Random(0)` sample),
  from prime-envs' git-backed Harbor registry, each pointed at its public Docker Hub image
  (`hamishi740/swerl-tmax-v3:<tag>`, from the `allenai/tmax-15k-open-instruct` image map) and
  started in `/home/user`.
- `tb21-86@7131e437`: Terminal-Bench 2.1 (harbor-framework/terminal-bench-2-1@7131e437) without
  `qemu-alpine-ssh`, `qemu-startup` and `protein-assembly`.

```bash
pip install -r torchtitan_recipes/rl/terminal_bench/requirements.txt
python -m torchtitan_recipes.rl.terminal_bench.stage_datasets --out ~/.cache/harbor   # ~1 min
```

A launcher that stages the trees itself can ship each as one archive holding that top-level
directory, e.g. `tar --zstd -cf tmax-mc1024_27de1c1b.tar.zst -C ~/.cache/harbor tmax-mc1024_27de1c1b`,
and extract it into `$HOME/.cache/harbor` on the host that runs the env server.

## Run with TitanRL

Follow the [RL environment setup](../../../torchtitan/rl/README.md), then download the checkpoint:

```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3.6-35B-A3B \
  --local_dir torchtitan/rl/example_checkpoint --all
python -m torchtitan.rl.train --module torchtitan_recipes.rl.verifiers_terminal_bench \
  --config rl_grpo_qwen3_6_35b_a3b_terminal_bench
```

- Hardware: 8 SM100 GPUs (B200 / GB300): one 8-GPU host (trainer GPUs 0-3, generators 4-7) or two
  4-GPU hosts through a launcher. The trainer's Dist-MoE experts need SM100. The runs used FA4
  `4.0.0b34.dev10` (Dao-AILab `33985c6`).
- `dist_moe`: every torchtitan process on this branch imports it (the Dist-MoE adapter,
  pytorch/torchtitan#4541), and it is not public yet (`meta-pytorch/dist_moe` returns 404). Until
  #4541 makes the import optional, this branch runs only where `dist_moe` is installed.
- Without more settings, task containers run on this host through the `docker` CLI.
- A remote VM per rollout (Verifiers' `oci-runner` provider, not public yet) needs
  `VF_SANDBOX_PROVIDER=oci-runner`, `OCI_RUNNER_TASK_NETWORK=host` (tests install pytest at grading
  time; without host networking every reward is a silent 0), `OCI_RUNNER_TOKEN_FILE`, and
  `OCI_RUNNER_POOL_SIZE` >= the env server's rollouts in flight (16 workers x 24 = 384).
- vLLM engine stats (KV-cache usage, running requests) are logged to `<dump>/vllm_metrics/` when
  `OTEL_METRICS_EXPORTER=jsonl`.

## Use from another trainer

Import `taskset.py` in every env-server process before Verifiers resolves the env config; that import
registers the plugin ids `TASKSET_ID` and `HARNESS_ID` (Verifiers imports plugins by top-level module
name). Then build `HarborEnvConfig(agent=vf.AgentConfig(harness=AgentOutsideHarnessConfig(id=HARNESS_ID), ...))`
with `taskset=TerminalTasksetConfig(id=TASKSET_ID, dataset=...)`; `_terminal_bench_rollouter_config`
in the TitanRL recipe shows every budget the comparison uses.

## Troubleshooting

- Every reward is 0: the sandbox has no network at grading time (`OCI_RUNNER_TASK_NETWORK=host`).
- Every rollout fails with `OCI runner pool broker did not bind ... within 30s`: the runner token is
  missing or unreadable; its path is in the broker's traceback in the controller log.
