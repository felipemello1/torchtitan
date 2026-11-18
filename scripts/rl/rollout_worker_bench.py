#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rerun three CPU results of `torchtitan/rl/rollout/ROLLOUT_WORKERS.md` on one Linux host, in ~3 min.

No GPU, model or sandbox: generation, tools and sandboxes are sleeps.

    threads    threads in a fresh process, with and without the rollout worker's bootstrap
    executor   how long renders wait for an executor thread while each sandbox cleanup blocks one
    gc         one full garbage collection of the worker's import heap, with and without gc.freeze()

The gc test's stand-in rollouts add less to a collection than real Verifiers rollouts, so it doesn't
rerun how the pause grows with rollouts per worker.

Example, from the repository root, all on one CPU socket and its memory:

    numactl --cpunodebind=0 --membind=0 python scripts/rl/rollout_worker_bench.py
    python scripts/rl/rollout_worker_bench.py --tests gc \\
        --gc-imports torchtitan.rl.actors.monarch.rollout_worker my_project.my_env
"""

import argparse
import asyncio
import gc
import importlib
import os
import random
import subprocess
import sys
import time

from torchtitan.rl.observability.metrics import (
    EventLoopLag,
    GcPause,
    Max,
    Metric,
    QueueWaitThreadPoolExecutor,
)

# Means over a Terminal-Bench training run's rollouts (an agent in a container), except `render_s`
# (CPU bench, ~12K-token prompts) and `sandbox_cleanup_s` (rootless Podman).
TERMINAL_BENCH_WORKLOAD = {
    "turns_per_rollout": 15.9,
    "first_prompt_tokens": 1194,
    "completion_tokens_per_turn": 568,
    "tool_output_tokens_per_turn": 529,
    # Time to first token 0.5 s + decode 19.9 s.
    "model_call_s": 20.4,
    "tool_call_s": 6.1,
    "sandbox_start_s": 13.0,
    "scoring_s": 6.0,
    # Chat messages -> token ids, on an executor thread. Median in the CPU bench.
    "render_s": 0.0022,
    # `docker rm --force` under rootless Podman: a 10 s stop timeout, then a kill.
    "sandbox_cleanup_s": 10.7,
}

# A fresh process counts its threads after each step. `bootstrap` first runs the
# environment-variable defaults that each rollout worker process sets at start
# (`OMP_NUM_THREADS=1`, unless it is already set).
_THREAD_COUNT_CHILD = """
import os, sys
if sys.argv[1] == "bootstrap":
    from torchtitan.rl._env import apply_rollout_worker_env_defaults
    apply_rollout_worker_env_defaults()
counts = [len(os.listdir("/proc/self/task"))]
import numpy
counts.append(len(os.listdir("/proc/self/task")))
import torch
counts.append(len(os.listdir("/proc/self/task")))
torch.randn(256, 256) @ torch.randn(256, 256)
counts.append(len(os.listdir("/proc/self/task")))
print(*counts, torch.get_num_threads())
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["threads", "executor", "gc"],
        default=["threads", "executor", "gc"],
        help="Which tests to run. They run in the order threads, executor, gc.",
    )
    parser.add_argument(
        "--executor-threads",
        type=int,
        nargs="+",
        default=[4, 32],
        help="executor test: executor sizes to run, one run each.",
    )
    parser.add_argument(
        "--executor-rollouts-per-worker",
        type=int,
        default=128,
        help="executor test: rollouts on the worker. The test has no groups, so all of them run.",
    )
    parser.add_argument(
        "--executor-speedup",
        type=float,
        default=10.0,
        help="executor test: divide every workload duration by this, so a rollout takes ~44 s.",
    )
    parser.add_argument(
        "--executor-window-s",
        type=float,
        default=60.0,
        help="executor test: seconds to measure, after a warm-up of a third of it (real time).",
    )
    parser.add_argument(
        "--gc-rollouts-per-worker",
        type=int,
        nargs="+",
        default=[64, 160, 480, 960],
        help="gc test: stand-in rollouts the worker holds, one full collection per value.",
    )
    parser.add_argument(
        "--gc-imports",
        nargs="+",
        default=[
            "torchtitan.rl.actors.monarch.rollout_worker",
            "torchtitan.rl.examples.verifiers.rollouter",
        ],
        help=(
            "gc test: modules a rollout worker imports, which set its starting heap. Without "
            "Verifiers installed, pass only torchtitan.rl.actors.monarch.rollout_worker."
        ),
    )
    args = parser.parse_args()
    if "gc" in args.tests:
        # Import first, so a missing package fails before the other tests run.
        for module in args.gc_imports:
            importlib.import_module(module)
    python_version = ".".join(map(str, sys.version_info[:3]))
    print(f"{len(os.sched_getaffinity(0))} CPUs, Python {python_version}")
    if "threads" in args.tests:
        run_threads_test()
    if "executor" in args.tests:
        for num_threads in args.executor_threads:
            asyncio.run(run_executor_test(num_threads, args=args))
    if "gc" in args.tests:
        run_gc_test(args)


def run_threads_test() -> None:
    """Print the threads of a fresh process after each step, without and with the worker's bootstrap."""
    env = {key: value for key, value in os.environ.items() if key != "OMP_NUM_THREADS"}
    print("\n== threads: a fresh process's threads after each step ==")
    print(
        "environment variables              start  import numpy  import torch  256x256 matmul  torch threads"
    )
    for bootstrap_mode, label in (
        ("unset", "OMP_NUM_THREADS unset"),
        ("bootstrap", "worker bootstrap (OMP=1)"),
    ):
        child = subprocess.run(
            [sys.executable, "-c", _THREAD_COUNT_CHILD, bootstrap_mode],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        (
            start,
            numpy_threads,
            torch_threads,
            matmul_threads,
            torch_num_threads,
        ) = child.stdout.split()
        print(
            f"{label:<34} {start:>5} {numpy_threads:>13} {torch_threads:>13} "
            f"{matmul_threads:>15} {torch_num_threads:>14}"
        )


async def run_executor_test(num_threads: int, *, args: argparse.Namespace) -> None:
    """Run a worker's rollouts on one event loop and print how long each render waits for an
    executor thread.

    Each step is a sleep. The render and the sandbox cleanup sleep on an executor thread. The rest
    sleep on the event loop:

        sandbox start -> turns: [render, model call, tool call] -> scoring -> sandbox cleanup

    A rollout that ends is replaced at once. The test has no groups, so every rollout on the
    worker is running. In TitanRL, `rollouts_per_worker` also counts finished rollouts that wait for
    their group, so fewer of them run.
    """
    workload = TERMINAL_BENCH_WORKLOAD
    speedup = args.executor_speedup
    rollouts_per_worker = args.executor_rollouts_per_worker
    # The worker's executor, as `RolloutWorkerActor.__init__` sets it: every `asyncio.to_thread`
    # call in the process runs on these threads, and it measures how long each call waits for one.
    executor = QueueWaitThreadPoolExecutor("rollout_worker", max_workers=num_threads)
    asyncio.get_running_loop().set_default_executor(executor)
    rng = random.Random(0)
    # When each render was submitted, and how long it waited once a thread started it.
    render_submitted_at: list[float] = []
    render_wait_s: dict[int, float] = {}
    rollout_end_times: list[float] = []

    async def render() -> None:
        render_index = len(render_submitted_at)
        render_submitted_at.append(time.perf_counter())

        def blocking_render() -> None:
            render_wait_s[render_index] = (
                time.perf_counter() - render_submitted_at[render_index]
            )
            time.sleep(workload["render_s"] / speedup)

        await asyncio.to_thread(blocking_render)

    async def run_rollouts_back_to_back() -> None:
        # The first rollout starts part-way through, so the first rollouts don't all end together.
        is_first = True
        while True:
            # Turns per rollout: exponential around the workload's mean, a rough stand-in for the spread.
            num_turns = max(
                1, round(rng.expovariate(1 / workload["turns_per_rollout"]))
            )
            first_turn = rng.randrange(num_turns) if is_first else 0
            if not is_first:
                await asyncio.sleep(workload["sandbox_start_s"] / speedup)
            is_first = False
            for _ in range(first_turn, num_turns):
                await render()
                await asyncio.sleep(workload["model_call_s"] / speedup)
                tool_s = workload["tool_call_s"] * rng.uniform(0.5, 1.5)
                await asyncio.sleep(tool_s / speedup)
            await asyncio.sleep(workload["scoring_s"] / speedup)
            # The sandbox cleanup holds an executor thread, as `docker rm --force` does.
            await asyncio.to_thread(time.sleep, workload["sandbox_cleanup_s"] / speedup)
            rollout_end_times.append(time.perf_counter())

    event_loop_lag = EventLoopLag("rollout_worker")
    rollouts = [
        asyncio.create_task(run_rollouts_back_to_back())
        for _ in range(rollouts_per_worker)
    ]
    # Warm up for a third of the window, then measure.
    await asyncio.sleep(args.executor_window_s / 3)
    event_loop_lag.flush()
    executor.flush()
    window_start = time.perf_counter()
    await asyncio.sleep(args.executor_window_s)
    window_end = time.perf_counter()
    max_lag_ms = _max_values(event_loop_lag.flush())["rollout_worker/event_loop_lag_ms"]
    executor_max_values = _max_values(executor.flush())
    event_loop_lag.close()
    for rollout in rollouts:
        rollout.cancel()
    await asyncio.gather(*rollouts, return_exceptions=True)

    # Report in the workload's own time: a wait of 1 s at speedup 10 is 10 s of Terminal-Bench time.
    # A render still waiting at the window's end counts with its wait so far.
    waits_ms = sorted(
        1000 * speedup * render_wait_s.get(render_index, window_end - submitted)
        for render_index, submitted in enumerate(render_submitted_at)
        if window_start <= submitted < window_end
    )
    ends_per_s = sum(window_start <= end < window_end for end in rollout_end_times) / (
        args.executor_window_s * speedup
    )
    mean_ms = sum(waits_ms) / len(waits_ms)
    p50_ms, p99_ms = waits_ms[len(waits_ms) // 2], waits_ms[int(0.99 * len(waits_ms))]
    model_call_ms = 1000 * workload["model_call_s"]
    print(
        f"\n== executor: {num_threads} threads, {rollouts_per_worker} running rollouts per worker, "
        f"each sandbox cleanup blocks a thread {workload['sandbox_cleanup_s']} s =="
    )
    print(
        f"rollouts ending per s: {ends_per_s:.2f} -> threads busy with cleanups (Little's law): "
        f"{ends_per_s * workload['sandbox_cleanup_s']:.1f} of {num_threads}"
    )
    print(
        f"render wait for a thread: mean {mean_ms:,.0f} ms, p50 {p50_ms:,.0f} ms, p99 {p99_ms:,.0f} ms "
        f"(mean = {100 * mean_ms / model_call_ms:.0f}% of a {model_call_ms:,.0f} ms model call)"
    )
    # The metric as the worker logs it, not scaled: a free thread's start-up takes real time, so
    # multiplying by `speedup` would inflate it. A wait behind the scaled cleanups reads `speedup`
    # times shorter than it would in Terminal-Bench time.
    print(
        "the worker's executor metrics (real time, not scaled): executor_queue_wait_ms/max "
        f"{executor_max_values['rollout_worker/executor_queue_wait_ms']:,.0f} ms (any job), "
        "num_executor_jobs_in_flight/max "
        f"{executor_max_values['rollout_worker/num_executor_jobs_in_flight']:.0f}"
    )
    print(f"event_loop_lag_ms/max: {max_lag_ms:.0f} ms (real time, not scaled)")


def run_gc_test(args: argparse.Namespace) -> None:
    """Time one full collection of a worker-like heap: the worker's imports, then stand-in rollouts.

    `main` imports `args.gc_imports` before any test runs.
    """
    print(
        f"\n== gc: one full collection, after importing {', '.join(args.gc_imports)} =="
    )
    print(
        "rollouts per worker   gc.collect() ms   gc_pause_ms/max   event_loop_lag_ms/max   tracked objects"
    )
    for freeze in (False, True):
        gc.collect()
        if freeze:
            print("-- gc.freeze() after the imports, as the worker does after setup --")
            gc.freeze()
        rng = random.Random(0)
        rollouts: list[dict[str, list]] = []
        for rollouts_per_worker in [0, *args.gc_rollouts_per_worker]:
            while len(rollouts) < rollouts_per_worker:
                rollouts.append(_make_stand_in_rollout(rng))
            collect_ms, gc_pause_ms, lag_ms = asyncio.run(_time_full_collection())
            print(
                f"{rollouts_per_worker:>19} {collect_ms:>17.0f} {gc_pause_ms:>17.0f} {lag_ms:>23.0f} "
                f"{len(gc.get_objects()):>17,}"
            )
        del rollouts
        if freeze:
            # Back to normal, so the frozen objects are freed at exit.
            gc.unfreeze()


def _make_stand_in_rollout(rng: random.Random) -> dict[str, list]:
    """Build one rollout's Python objects: each turn's messages and completion token ids, and the
    token ids of its next prompt. They add less to a full collection than real Verifiers rollouts.

    Example (2 turns, 568 completion tokens each):

        {"turns": [[assistant_message, tool_message, [1000, ..., 1567]], [...]],
         "prompt_token_ids": [1000, ..., 4387]}
    """
    workload = TERMINAL_BENCH_WORKLOAD
    # Turns held: uniform in 1..31, mean 16. A rough stand-in: running rollouts hold fewer turns,
    # and finished ones waiting for their group hold all of theirs.
    num_turns = rng.randint(1, 2 * round(workload["turns_per_rollout"]) - 1)
    tokens_per_turn = round(
        workload["completion_tokens_per_turn"] + workload["tool_output_tokens_per_turn"]
    )
    completion_tokens = round(workload["completion_tokens_per_turn"])
    turns = [
        [
            # ~4 characters per token.
            {"role": "assistant", "content": "x" * 4 * completion_tokens},
            {
                "role": "tool",
                "content": "y" * 4 * round(workload["tool_output_tokens_per_turn"]),
            },
            # Token ids over 256 are distinct int objects, as after a JSON parse.
            list(range(1000, 1000 + completion_tokens)),
        ]
        for _ in range(num_turns)
    ]
    prompt_tokens = round(workload["first_prompt_tokens"]) + num_turns * tokens_per_turn
    return {
        "turns": turns,
        "prompt_token_ids": list(range(1000, 1000 + prompt_tokens)),
    }


async def _time_full_collection() -> tuple[float, float, float]:
    """Return the ms of one full collection on a running loop, and what the worker's metrics
    `gc_pause_ms` and `event_loop_lag_ms` read for it."""
    # A first collection moves the new objects to the oldest generation, so the timed one walks
    # the same heap as a later automatic one.
    gc.collect()
    gc_pause = GcPause("rollout_worker")
    event_loop_lag = EventLoopLag("rollout_worker")
    await asyncio.sleep(0.2)
    began = time.perf_counter()
    # Blocks the loop, as an automatic full collection in a worker does.
    gc.collect()
    collect_ms = 1000 * (time.perf_counter() - began)
    await asyncio.sleep(0.2)
    gc_pause_ms = _max_values(gc_pause.flush())["rollout_worker/gc_pause_ms"]
    max_lag_ms = _max_values(event_loop_lag.flush())["rollout_worker/event_loop_lag_ms"]
    gc_pause.close()
    event_loop_lag.close()
    return collect_ms, gc_pause_ms, max_lag_ms


def _max_values(metrics: list[Metric]) -> dict[str, float]:
    """Return the value of each Max metric by key: what a step logs as `<key>/max` for one worker.

    Example: `EventLoopLag.flush()` -> {"rollout_worker/event_loop_lag_ms": 12.0}. Its Mean metric,
    `event_loop_lag_ms_per_s`, is left out.
    """
    return {
        metric.key: metric.value.value
        for metric in metrics
        if isinstance(metric.value, Max)
    }


if __name__ == "__main__":
    main()
