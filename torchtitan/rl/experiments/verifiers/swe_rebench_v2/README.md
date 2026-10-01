# SWE-rebench V2 with Verifiers and TitanRL

This experiment trains a Qwen3.5/3.6 coding agent on SWE-rebench V2 tasks. Each task
is a repository image at a base commit plus an issue to fix; the reward is how many of
the task's failing tests pass after the agent's edits, discounted by the tests it
broke. The agent is a bash-tool loop that runs on the controller
([`../agent_outside.py`](../agent_outside.py)) and only sends commands to the task
container, so the container never calls back to the generator.

```text
controller, env-server worker                          task container (docker or Sandoq VM)
  AgentOutsideHarness --chat--> GenerationServer
      | each bash tool call
      +-- runtime.run(["bash", "-lc", command]) -----> /<repo> at base_commit
                                                       (then the task's tests grade it)
```

## Run

Install the TorchTitan RL dependencies and this directory's `requirements.txt` in one
Python 3.12 environment, and put a checkout of
[SWE-rebench-V2](https://github.com/SWE-rebench/SWE-rebench-V2) at `c71902a8` on
`PYTHONPATH`: grading imports its log parsers (`lib.agent.log_parsers`), which are not
on PyPI.

```bash
git clone https://github.com/SWE-rebench/SWE-rebench-V2.git
git -C SWE-rebench-V2 checkout c71902a8cf8d2b725f63d51f199f4d3e56f68d2d
export PYTHONPATH=$PWD/SWE-rebench-V2:$PYTHONPATH

python -m torchtitan.rl.train \
  --module torchtitan.rl.experiments.verifiers.swe_rebench_v2 \
  --config rl_grpo_qwen35_9b_swe_rebench_v2_smoke
```

`SWE_REBENCH_V2_SANDBOX` picks where task containers run:

- `docker` (default): on the controller host, which needs Docker and an x86 CPU (the
  task images are amd64-only).
- `sandoq`: in a Sandoq Firecracker VM, for a controller that cannot run them. The
  `sandoq_provider` package from `ram_prime_rl` (`extensions/sandoq`, with its pinned
  `sandoq-client`) serves the `vf.PrimeConfig` runtime and reads its settings from the
  environment. Grading needs no network, so `OCI_RUNNER_TASK_NETWORK` stays `none`:

```bash
export SWE_REBENCH_V2_SANDBOX=sandoq
export VF_SANDBOX_PROVIDER=oci-runner  # checked when the recipe is built
export PYTHONPATH=/path/to/ram_prime_rl/extensions/sandoq:$PYTHONPATH
export OCI_RUNNER_ENVIRONMENT=oci-runner-firecracker-medium
export OCI_RUNNER_TOKEN_FILE=/path/to/firecracker-token  # mode 0600
export OCI_RUNNER_POOL_SIZE=128  # the recipe runs 8 x 16 rollouts at once
export OCI_RUNNER_ECR_REGISTRY=168653207203.dkr.ecr.us-east-2.amazonaws.com
export OCI_RUNNER_ECR_UCLOUD=ucloud  # or OCI_RUNNER_ECR_TOKEN_FILE=/path/to/ecr-password
```

A task image is 1.2-1.9 GB and a fresh VM caches none, so the first pull of each image
dominates setup; the setup timeout is 1,500 s.

## Data

Tasks come from `nebius/SWE-rebench-V2` at revision `475dd5e8` (32,079 rows), loaded
with `datasets`; with `HF_HUB_OFFLINE=1`, pre-populate the Hugging Face cache. The
taskset keeps Python tasks graded code quality A with no detected issues, at most 9
files and 181 lines changed, and easy or medium difficulty. It then holds out every task
of 64 repositories chosen by a seeded hash: training uses the 1,635 easy tasks of the
other repositories, validation the 442 held-out tasks. A changed dataset or filter
fails the pinned counts instead of silently training on different tasks.

The reward (`f2p_regression`) is `F2P_fraction * (0.5 + 0.5 * P2P_fraction)`, where F2P
are the tests the fix must turn green and P2P the tests that must stay green.

## Recipes

Both run 30 agent turns of up to 16,384 tokens, keep all thinking in the context, and
keep fp32 master weights. Qwen3.5 thinks before every tool call, and a turn cut off
mid-thought has no tool call and ends the rollout, so turns get 4x the 4,096 tokens of
earlier Qwen3 receipts.

- `rl_grpo_qwen35_9b_swe_rebench_v2_smoke`: Qwen3.5-9B, 65,536-token rollouts, 3 steps
  of 2 tasks x 8 samples. Trainer FSDP=2 x TP=2 on 4 GPUs; one TP=4 generator.
- `rl_grpo_qwen36_35b_a3b_swe_rebench_v2_dist_moe`: Qwen3.6-35B-A3B, 131,072-token
  rollouts, 100 steps of 4 tasks x 16 samples. The trainer (FSDP=2 x TP=2, EP=4, one
  host) uses Dist-MoE, which needs SM100+ GPUs; the generator (DP=2 x TP=2, EP=4, one
  host) keeps the stock experts with CUDA graphs off. Its memory fit at 131,072 tokens
  per microbatch has not been measured.

Validation is off (`num_samples=0`); the held-out tasks are configured for when it is
turned on.

## Known gaps

- Neither recipe has run end to end.
- A grader that cannot parse a test log raises, which ends that rollout as an error
  with reward 0.
- TitanRL does not read `Trace.info`, so the per-rollout test report stays in the
  env-server logs.
