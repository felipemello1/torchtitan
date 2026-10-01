# Terminal-Bench environment

A bash-tool agent that solves Harbor terminal tasks (e.g. Terminal-Bench),
packaged for [Verifiers](https://github.com/PrimeIntellect-ai/verifiers). The package imports no
TorchTitan code, so any trainer can run the same agent; TitanRL's recipes are in
`torchtitan_recipes/rl/verifiers_terminal_bench.py`.

```text
trainer (TitanRL, prime-rl, ...)
  Verifiers env server (spawned workers)
    TerminalBenchEnv.run(task)                        taskset.py
      AgentOutsideHarness: the agent loop runs here    agent_outside.py
        ├─ chat completion ──> trainer's generator
        └─ bash tool call ──> runtime.run(["bash", "-lc", cmd]) ──> sandbox (Docker container or Sandoq VM)
      Harbor grading: tests/test.sh in the same sandbox -> reward
```

- `agent_outside.py`: `AgentOutsideHarness` calls the model with one `bash` tool and runs each call
  in the sandbox. Nothing is installed in the task container and the container never calls the
  model, so the sandbox can be a remote VM with no route back. Tool results keep their tail, with
  the exit code last.
- `taskset.py`: Verifiers' Harbor taskset and env. Each task starts in its `task.toml` workdir, else
  its image's last `WORKDIR`. With `VF_SANDBOX_PROVIDER=oci-runner`, the env hands the Sandoq
  provider the task's image and workdir.

Verifiers resolves taskset and harness ids by importing a top-level module, so importing
`taskset.py` registers both under `TASKSET_ID` and `HARNESS_ID`. A spawned worker must import it
before it resolves an env config (TitanRL's env server does this).

## Run with TitanRL

```bash
TERMINAL_BENCH_SANDBOX=docker python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_terminal_bench \
  --config rl_grpo_qwen3_6_35b_a3b_terminal_bench
```

- Datasets are Harbor ids, read from `$HOME/.cache/harbor/<id with "/" and "@" as "_">`; stage them
  with `harbor download <id> --export -o <that dir>` when the host has no internet.
- `TERMINAL_BENCH_SANDBOX=docker` runs task containers on the controller host through the `docker`
  CLI (Harbor tasks use `--network host`).
- `TERMINAL_BENCH_SANDBOX=sandoq` leases a Firecracker VM per rollout through the `sandoq_provider`
  from `ram_prime_rl` (on `PYTHONPATH`). It needs `VF_SANDBOX_PROVIDER=oci-runner`,
  `OCI_RUNNER_TASK_NETWORK=host` (tests install pytest at grading time; without host networking
  every reward is a silent 0), `OCI_RUNNER_TOKEN_FILE`, and `OCI_RUNNER_POOL_SIZE` >= the env
  server's rollouts in flight (8 workers x 16).
- A missing or unreadable runner token shows up only as `OCI runner pool broker did not bind ...
  within 30s` on every rollout; the token path is in the broker's traceback in the controller log.
