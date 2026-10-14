#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator microbenchmark: TorchTitan's ``VLLMGenerator`` vs prime-rl's vLLM, per decode and prefill shape.

Both sides run in process on one GPU, on Qwen3.5-35B-A3B cut to 4 layers [GDN, GDN, GDN, full attention],
with random weights and 30% of the GPU's memory:

    side    engine                                         MoE / attention kernels
    titan   TorchTitan VLLMGenerator (FULL CUDA graphs)    torch._grouped_mm / FA4
    prime   vLLM 0.31 offline LLM (prime-rl's runtime)     FlashInfer TRT-LLM MoE / FlashInfer

A hook on the engine's ``step()`` measures each shape:

    decode B x C   B requests with C random prompt tokens. Once all B decode: skip 2 steps, time 12 steps
                   unprofiled (wall_e2e_us per step), then profile 12 steps (kernel_us per step).
    prefill N@P    one request with P cached prefix tokens + N new tokens. Time it unprofiled, then profile a
                   twin request: kernel_us covers the prefill step + vLLM's GDN state-copy step. 2 repeats.

Subcommands:

    launch   set up the side's venv and env, then run ``measure`` in it; retries if vLLM's startup memory check
             fails because another process changed GPU memory
    measure  the in-process benchmark (``launch`` calls it); appends one JSON line per shape to --out
    report   kernel or wall table per shape, one column per tag, min over passes of each pass's median
    gap      kernel time by family (routed experts, attention, GDN, ...) for two tags, from --topk-out dumps

Paths below are absolute on our GB300 cluster; pass --tt / --titan-venv / --prime-root / --model elsewhere.

    TorchTitan venv   /storage/home/felipemello/libraries/fairtitan/runtimes/swe-20261013-verifiers/torchtitan/titan-rl
    prime-rl root     /checkpoint/pytorch/felipemello/runtimes/prime-latest-35b-next-runtime/aarch64/prime-rl
    model             /checkpoint/transformer2/oss_models/Qwen3.5-35B-A3B-Base (only config + tokenizer are read)

Compare two TorchTitan stacks (e.g. main vs a PR stack): one ``launch`` per worktree with ``--tt``, then ``report``.

    python scripts/rl/bench_generator.py launch --side titan --tt ~/tt-main  --tag main  --out res/main.jsonl
    python scripts/rl/bench_generator.py launch --side titan --tt ~/tt-stack --tag stack --out res/stack.jsonl
    python scripts/rl/bench_generator.py launch --side prime --tag prime --out res/prime.jsonl
    python scripts/rl/bench_generator.py report res/*.jsonl --metric kernel_us

Reproduce the selected-set tables (configs A-E, baseline, prime-rl), 2-3 passes each:

    for pass in 1 2 3; do
      for c in baseline A B C D E; do
        python scripts/rl/bench_generator.py launch --side titan --config $c --out res/$c.jsonl
      done
      python scripts/rl/bench_generator.py launch --side prime --tag prime --out res/prime.jsonl
    done
    python scripts/rl/bench_generator.py report res/*.jsonl --metric kernel_us --base A
    # Kernel families: one more pass with --topk-out kernels.jsonl on A and prime, then
    python scripts/rl/bench_generator.py gap kernels.jsonl --a A --b prime

Expected kernel_us (shared GB300, min of 3 passes):

    shape               baseline    A      prime
    decode B=1, 2k       1,845    1,026      883
    decode B=34, 2k      2,712    2,072    1,693
    prefill 8k + 0      15,598    9,646    9,148

The GPU is shared with other agents: keep runs short (``--shapes lite`` or ``--only``), repeat 2-3 passes, and
read the min. Each record's ``others`` field counts the other processes on the GPU during that shape.
prime-rl's wall time is not comparable (offline ``LLM.generate`` does per-token output work); compare its
kernel_us only. Its first launch JIT-builds FlashInfer's MoE module into /tmp/.flashinfer-abl (CPU, minutes).
"""

import argparse
import collections
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time

TITAN_VENV = "/storage/home/felipemello/libraries/fairtitan/runtimes/swe-20261013-verifiers/torchtitan/titan-rl"
PRIME_ROOT = "/checkpoint/pytorch/felipemello/runtimes/prime-latest-35b-next-runtime/aarch64/prime-rl"
MODEL = "/checkpoint/transformer2/oss_models/Qwen3.5-35B-A3B-Base"
TORCHSTORE = "/storage/home/felipemello/libraries/torchstore_pkgs/d7aeaba-revert-nixl"
DIST_MOE = "/storage/home/felipemello/libraries/dist_moe_src"
# TorchTitan worktrees of the selected-set runs: `ablate` adds ABL_OFF switches (47f36e0fd); `ablate307` adds #307.
ABLATE_TT = "/checkpoint/pytorch/felipemello/worktrees/ablate"
ABLATE307_TT = "/checkpoint/pytorch/felipemello/worktrees/ablate307"

LADDERS = {
    # vLLM's default sizes up to 1024 + prefill sizes up to max_num_batched_tokens.
    "full": [1, 2, 4]
    + list(range(8, 257, 8))
    + list(range(272, 1025, 16))
    + list(range(1536, 4097, 512))
    + list(range(6144, 16385, 2048)),
    # vLLM's default sizes only: prefills above 1024 tokens run without a graph.
    "small": [1, 2, 4] + list(range(8, 257, 8)) + list(range(272, 1025, 16)),
    # The old auto ladder for max_num_seqs=230 (the baseline).
    "old": [
        16,
        32,
        64,
        112,
        128,
        144,
        160,
        176,
        192,
        208,
        224,
        230,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    ],
}

# Each optimization PR on the `ablate` worktree, and how to turn it off: (ABL_OFF ids, compile regions to drop).
# ABL_OFF only exists on that worktree; dropping regions works on any TorchTitan.
ABLATION_SWITCHES = {
    "292": (["292"], ["moe_combine"]),  # gather-sum MoE combine
    "293": ([], ["moe_dispatch"]),  # counting-sort dispatch
    "294": (["294"], []),  # expert_counts region
    "297": (["297"], []),  # offsets into expert_counts
    "298": ([], ["residual_norm"]),  # residual add + ffn_norm region
    "299": ([], ["attention_output_gate"]),  # attention output gate region
    "300": (["300"], []),  # GDN derived-weight cache
    "303": (["303"], []),  # chunked mamba state copy
    "305": (["305"], []),  # V2 model runner
    "306": (["306"], []),  # lighter single-token GDN metadata
    "5343": ([], ["router"]),  # router compile region
}


def ablation_config(off: list[str], **extra) -> dict:
    """A `CONFIGS` entry with the PRs in ``off`` turned off on the `ablate` worktree."""
    abl_off = [x for pr in off for x in ABLATION_SWITCHES[pr][0]]
    regions = [r for pr in off for r in ABLATION_SWITCHES[pr][1]]
    return dict(tt=ABLATE_TT, abl_off=abl_off, drop_regions=regions, **extra)


_SELECTED_OFF = ["294", "297", "298", "299", "300", "305"]
CONFIGS = {
    # The --tt worktree as it is.
    "stack": {},
    "baseline": ablation_config(list(ABLATION_SWITCHES), ladder="old"),
    # A: #292, #293, #5343, #306, #303 + the full ladder, V1 runner.
    "A": ablation_config(_SELECTED_OFF),
    # B: A's runner PRs on Dist-MoE (#292 and #293 don't run under Dist-MoE).
    "B": ablation_config(_SELECTED_OFF, dist_moe=True),
    "C": dict(ablation_config(_SELECTED_OFF), tt=ABLATE307_TT),
    # D: every non-compile PR on the V1 runner; E: D + #307.
    "D": ablation_config(["305"]),
    "E": dict(ablation_config(["305"]), tt=ABLATE307_TT),
}
# Leave-one-out against the full stack: no292, no293, ...
CONFIGS.update({f"no{pr}": ablation_config([pr]) for pr in ABLATION_SWITCHES})

DECODE_SHAPES = {
    "full": [(b, c) for c in (2048, 8192, 32768) for b in (1, 8, 34, 64, 128)],
    "lite": [(1, 2048), (34, 8192), (128, 2048), (34, 32768), (128, 32768)],
}
# A multiple of both engines' prefix-cache block (1152 TitanRL, 1056 prime-rl), so the whole prefix is cached.
CACHED = 12672
PREFILL_SHAPES = {
    "full": [(n, p) for p in (0, CACHED) for n in (2048, 4096, 8192, 16384)]
    + [(4096, CACHED + 500)],
    "lite": [(2048, CACHED), (8192, 0), (16384, CACHED), (4096, CACHED + 500)],
}


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "launch":
        launch(args)
    elif args.command == "measure":
        measure(args)
    elif args.command == "report":
        report(args)
    else:
        gap(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "measure"):
        p = sub.add_parser(name)
        p.add_argument("--side", choices=("titan", "prime"), required=True)
        p.add_argument(
            "--config",
            default="stack",
            choices=sorted(CONFIGS),
            help="titan only; see CONFIGS",
        )
        p.add_argument(
            "--tt", help="TorchTitan worktree (titan); overrides the config's"
        )
        p.add_argument(
            "--tag", help="name of this column in `report` (default: the config name)"
        )
        p.add_argument(
            "--out", required=True, help="JSONL file; each launch appends one pass"
        )
        p.add_argument("--shapes", choices=("full", "lite"), default="full")
        p.add_argument(
            "--only",
            default="",
            help='re-measure shapes, e.g. "d:34:8192,p:4096:12672"',
        )
        p.add_argument(
            "--nsteps", type=int, default=12, help="decode steps timed and profiled"
        )
        p.add_argument("--prefill-reps", type=int, default=2)
        p.add_argument(
            "--topk-out",
            help="also append every kernel per shape (us per step) for `gap`",
        )
        p.add_argument(
            "--gpu-mem",
            type=float,
            default=0.3,
            help="fraction of GPU memory for the engine",
        )
        p.add_argument(
            "--fp32-head",
            action="store_true",
            help="prime: fp32 lm_head, as prime-rl's RL default",
        )
        p.add_argument("--gpu", default="0")
        p.add_argument("--retries", type=int, default=6)
        p.add_argument("--log", help="engine log (default: <out>.<tag>.log)")
        p.add_argument("--titan-venv", default=TITAN_VENV)
        p.add_argument("--prime-root", default=PRIME_ROOT)
        p.add_argument("--model", default=MODEL)
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p.add_argument(
        "--metric", choices=("kernel_us", "wall_e2e_us"), default="kernel_us"
    )
    p.add_argument(
        "--base", help="tag the delta columns compare against (default: the first tag)"
    )
    p = sub.add_parser("gap")
    p.add_argument("kernels", help="--topk-out file")
    p.add_argument("--a", required=True, help="tag")
    p.add_argument("--b", required=True, help="tag")
    p.add_argument(
        "--show", action="store_true", help="also list the top kernels of each family"
    )
    return parser


# ----------------------------------------------------------------------------------------------- launch


def launch(args: argparse.Namespace) -> None:
    """Run ``measure`` with the side's venv and env; retry until the pass records its startup line."""
    config = CONFIGS[args.config]
    tag = args.tag or (args.config if args.side == "titan" else "prime")
    log = args.log or f"{args.out}.{tag}.log"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpu,
        PYTHONUNBUFFERED="1",
        VLLM_NO_USAGE_STATS="1",
    )
    if args.side == "titan":
        tt = args.tt or config.get("tt") or os.getcwd()
        cuda = f"{args.titan_venv}/lib/python3.12/site-packages/nvidia/cu13"
        cache = f"/tmp/bench-gen-{args.gpu}"
        env.update(
            PATH=f"{cuda}/bin:{args.titan_venv}/bin:{env['PATH']}",
            CUDA_HOME=cuda,
            LIBRARY_PATH=f"{cuda}/lib",
            LD_LIBRARY_PATH=f"{cuda}/lib:{env.get('LD_LIBRARY_PATH', '')}",
            PYTHONPATH=f"{tt}:{TORCHSTORE}:{DIST_MOE}",
            TRITON_CACHE_DIR=f"{cache}/triton",
            TORCHINDUCTOR_CACHE_DIR=f"{cache}/inductor",
            NUMBA_CACHE_DIR="/tmp/numba-cache",
            MALLOC_ARENA_MAX="2",
            MALLOC_TRIM_THRESHOLD_="0",
            MALLOC_MMAP_THRESHOLD_="65536",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
            LOCAL_WORLD_SIZE="1",
            MASTER_ADDR="127.0.0.1",
            ABL_OFF=",".join(config.get("abl_off", [])),
        )
        python, cwd = f"{args.titan_venv}/bin/python", tt
    else:
        cuda = f"{args.prime_root}/.venv/lib/python3.12/site-packages/nvidia/cu13"
        # FlashInfer's JIT links against libcudart.so; the venv only ships libcudart.so.13.
        libfix = "/tmp/.flashinfer-abl/libfix"
        os.makedirs(libfix, exist_ok=True)
        if not os.path.exists(f"{libfix}/libcudart.so"):
            os.symlink(f"{cuda}/lib/libcudart.so.13", f"{libfix}/libcudart.so")
        env.update(
            PATH=f"{cuda}/bin:{args.prime_root}/.venv/bin:{env['PATH']}",
            CUDA_HOME=cuda,
            LIBRARY_PATH=libfix,
            FLASHINFER_WORKSPACE_BASE="/tmp/.flashinfer-abl",
            FLASHINFER_CUBIN_DIR="/storage/home/felipemello/.cache/flashinfer-prime/cubins",
            FLASHINFER_NO_DOWNLOAD="1",
            # The venv's nvcc (13.3) is newer than its CUDA headers (13.0).
            NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
            VLLM_USE_FLASHINFER_SAMPLER="0",
            VLLM_ENABLE_V1_MULTIPROCESSING="0",
            TRITON_CACHE_DIR="/tmp/triton-prefill",
            VLLM_CACHE_ROOT="/tmp/.vllm-cache-prefill",
            HF_HUB_OFFLINE="1",
            NCCL_CUMEM_ENABLE="0",
        )
        python, cwd = f"{args.prime_root}/.venv/bin/python", args.prime_root
    command = [
        python,
        os.path.abspath(__file__),
        "measure",
        *sys.argv[sys.argv.index("launch") + 1 :],
    ]
    if not args.tag:
        command += ["--tag", tag]
    for attempt in range(args.retries):
        env["MASTER_PORT"] = str(_free_port())
        started = _count_startup_lines(args.out)
        with open(log, "a") as log_file:
            subprocess.run(
                command, env=env, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT
            )
        if _count_startup_lines(args.out) > started:
            print(f"{tag}: done ({args.out}, log {log})")
            return
        print(f"{tag}: no startup line after attempt {attempt + 1}; see {log}")
    raise SystemExit(f"{tag}: failed {args.retries} times")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _count_startup_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    return sum('"kind": "startup"' in line for line in open(path))


# ---------------------------------------------------------------------------------------------- measure


def measure(args: argparse.Namespace) -> None:
    """Build the side's engine, measure every shape, and append one JSON line per shape to ``args.out``."""
    import asyncio

    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    started = time.time()
    if args.side == "titan":
        engine, desc = _build_titan_engine(args)
    else:
        engine, desc = _build_prime_engine(args)
    desc.update(side=args.side, tag=args.tag, run_id=run_id)
    startup_s = time.time() - started
    print("config", desc, flush=True)
    probe = StepProbe(engine)

    decode, prefill = DECODE_SHAPES[args.shapes], PREFILL_SHAPES[args.shapes]
    if args.only:
        wanted = [tuple(x.split(":")) for x in args.only.split(",") if x]
        decode = [(int(b), int(c)) for kind, b, c in wanted if kind == "d"]
        prefill = [(int(n), int(p)) for kind, n, p in wanted if kind == "p"]

    def emit(record: dict) -> None:
        record.update(desc)
        with open(args.out, "a") as f:
            f.write(json.dumps(record) + "\n")
        print("RESULT", json.dumps(record), flush=True)

    async def run_all() -> None:
        await engine.start()
        await engine.generate(rand_prompts(2, 512, seed=99), max_tokens=8, decode=True)
        for batch, ctx in decode:
            try:
                emit(await _measure_decode(engine, probe, batch, ctx, args))
            except Exception as e:  # noqa: BLE001 - one failing shape (e.g. KV too small) must not end the pass
                emit(dict(kind="decode", batch=batch, ctx=ctx, error=repr(e)[:300]))
            probe.armed = False
        for new, cached in prefill:
            for rep in range(args.prefill_reps):
                try:
                    emit(await _measure_prefill(engine, probe, new, cached, rep, args))
                except Exception as e:  # noqa: BLE001
                    emit(
                        dict(
                            kind="prefill",
                            new=new,
                            cached=cached,
                            rep=rep,
                            error=repr(e)[:300],
                        )
                    )
                probe.armed = False
        emit(dict(kind="startup", startup_s=round(startup_s, 1)))

    asyncio.run(run_all())
    os._exit(0)  # vLLM's engine threads keep the process alive otherwise


async def _measure_decode(engine, probe, batch: int, ctx: int, args) -> dict:
    """All ``batch`` requests prefill (16,384 tokens per step), then decode long enough to be timed together."""
    prefill_steps = -(-batch * ctx // 16384)
    probe.arm("decode", need=batch, want=args.nsteps, wall_n=args.nsteps)
    prompts = rand_prompts(batch, ctx, seed=batch * 100003 + ctx)
    await engine.generate(
        prompts, max_tokens=int(prefill_steps * 2.5) + 2 * args.nsteps + 30, decode=True
    )
    nsteps = len(probe.profiled_s)
    wall_us = 1e6 * probe.wall_s[-1] / len(probe.wall_s)
    _dump_kernels(probe.profiler, nsteps, f"d:{batch}:{ctx}", args)
    return dict(
        kind="decode",
        batch=batch,
        ctx=ctx,
        kernel_us=round(_kernel_us(probe.profiler, nsteps), 1),
        wall_e2e_us=round(wall_us, 1),
        tok_s=round(batch / (wall_us / 1e6), 1),
        copy_us=round(_copy_us(probe.profiler, nsteps), 1),
        steps=nsteps,
        **_other_gpu_processes(),
    )


async def _measure_prefill(
    engine, probe, new: int, cached: int, rep: int, args
) -> dict:
    """Time one request over a cached prefix unprofiled, then profile a twin request over the same prefix."""
    seed = new * 7 + cached * 13 + rep
    tokens = rand_prompts(1, cached + new, seed)[0]
    if cached:
        await engine.generate([tokens[:cached]], max_tokens=1)  # cache the prefix
    twin = tokens[:cached] + rand_prompts(1, new, seed + 1_000_003)[0]
    probe.arm("prefill", need=0, want=0, skip=0, wall_n=99)
    await engine.generate([twin], max_tokens=1)
    await probe.wait()
    wall_us = 1e6 * probe.wall_s[-1]
    # The prefill step, plus vLLM's follow-up step that copies the GDN state into its prefix-cache block.
    probe.arm("prefill", need=0, want=4, skip=0)
    await engine.generate([tokens], max_tokens=1)
    await probe.wait()
    _dump_kernels(probe.profiler, 1, f"p:{new}:{cached}", args)
    return dict(
        kind="prefill",
        new=new,
        cached=cached,
        rep=rep,
        kernel_us=round(_kernel_us(probe.profiler, 1), 1),
        wall_e2e_us=round(wall_us, 1),
        tok_s=round(new / (wall_us / 1e6), 1),
        copy_us=round(_copy_us(probe.profiler, 1), 1),
        steps=len(probe.profiled_s),
        **_other_gpu_processes(),
    )


class StepProbe:
    """Wraps the engine's ``step()`` to time a window of steps unprofiled, then profile the next steps.

    Timeline of one armed shape (decode; prefill skips nothing and ends when the request finishes):

        steps until all `need` requests decode | skip 2 | wall_n steps timed | `want` steps profiled
    """

    def __init__(self, engine) -> None:
        import torch

        self.torch = torch
        self.engine = engine
        self.original_step = engine.step_owner.step
        engine.step_owner.step = self.step
        self.armed, self.done, self.profiler = False, False, None

    def arm(
        self, mode: str, need: int, want: int, skip: int = 2, wall_n: int = 0
    ) -> None:
        self.armed, self.mode, self.need, self.want, self.skip, self.wall_n = (
            True,
            mode,
            need,
            want,
            skip,
            wall_n,
        )
        self.skipped, self.done, self.profiler = 0, False, None
        self.wall_s: list[
            float
        ] = []  # time since the window started, after each timed step
        self.profiled_s: list[float] = []

    async def wait(self) -> None:
        """Wait for the engine thread to end the window; stop a profiler the last step left running."""
        import asyncio

        for _ in range(50):
            if self.done:
                return
            await asyncio.sleep(0.05)
        if self.profiler is not None and not self.done:
            self.torch.cuda.synchronize()
            self.profiler.stop()
            self.done = True

    def step(self):
        torch = self.torch
        if self.armed and not self.done and self.profiler is None:
            ready = self.mode == "prefill" or self.engine.num_decoding() >= self.need
            if ready and self.skipped >= self.skip and len(self.wall_s) < self.wall_n:
                # Synchronize only at the window's two ends, so async scheduling still overlaps host and GPU work.
                if not self.wall_s:
                    torch.cuda.synchronize()
                    self.window_start = time.perf_counter()
                out = self.original_step()
                finished = self.mode == "prefill" and self.engine.num_active() == 0
                if len(self.wall_s) + 1 == self.wall_n or finished:
                    torch.cuda.synchronize()
                self.wall_s.append(time.perf_counter() - self.window_start)
                self.done = finished
                return out
            if ready and self.skipped >= self.skip:
                torch.cuda.synchronize()
                self.profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                )
                self.profiler.start()
            elif ready:
                self.skipped += 1
        step_start = time.perf_counter()
        out = self.original_step()
        if self.profiler is not None and not self.done:
            torch.cuda.synchronize()
            self.profiled_s.append(time.perf_counter() - step_start)
            finished = self.mode == "prefill" and self.engine.num_active() == 0
            if len(self.profiled_s) >= self.want or finished:
                self.profiler.stop()  # in this thread, where it started
                self.done = True
        return out


class _Engine:
    """What `measure` needs from either side: the object owning ``step()``, the scheduler, and generate."""

    def __init__(self, step_owner, engine_core, generate, start) -> None:
        self.step_owner, self._engine_core, self.generate, self.start = (
            step_owner,
            engine_core,
            generate,
            start,
        )

    def _scheduler(self):
        core = self._engine_core()
        return getattr(core, "engine_core", core).scheduler

    def num_decoding(self) -> int:
        """Requests that already emitted a token: every later step is pure decode for them."""
        return sum(1 for r in self._scheduler().running if r.num_output_tokens > 0)

    def num_active(self) -> int:
        scheduler = self._scheduler()
        return len(scheduler.running) + len(scheduler.waiting)


def _build_titan_engine(args):
    import asyncio
    import dataclasses
    import inspect

    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.qwen3_5 import build_model_config
    from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
    from torchtitan.rl.generator import (
        SamplingConfig,
        VLLMCudaGraphConfig,
        VLLMGenerator,
    )
    from torchtitan_recipes.rl.model_configs import build_rl_model_configs

    config = CONFIGS[args.config]
    parallelism = InferenceParallelismConfig(
        data_parallel_degree=1, tensor_parallel_degree=1
    )
    _, model = build_rl_model_configs(
        build_model_config,
        "35B-A3B",
        seq_len=65536,
        attn_backend="varlen",
        trainer_parallelism=ParallelismConfig(),
        generator_parallelism=parallelism,
    )
    model.layers = model.layers[:4]  # [GDN, GDN, GDN, full attention]
    drop = set(config.get("drop_regions", []))
    model.local_compile_regions = [
        r for r in model.local_compile_regions if r not in drop
    ]
    cuda_graph = dict(mode="FULL", capture_sizes=LADDERS[config.get("ladder", "full")])
    if config.get("vllm_compile"):
        cuda_graph["vllm_compile"] = True
    generator_config = VLLMGenerator.Config(
        model=model,
        model_dtype="bfloat16",
        gpu_memory_limit=args.gpu_mem,
        max_num_batched_tokens=16_384,
        parallelism=parallelism,
        cuda_graph=VLLMCudaGraphConfig(**cuda_graph),
        checkpointer=None,
        extra_vllm_engine_args={"watermark": 0.1},
        sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=16384),
    )
    if config.get("dist_moe"):
        from torchtitan.config.transform import apply_transforms
        from torchtitan.config.transform.dist_moe import DistMoeTransform
        from torchtitan.models.common.dist_moe import DistMoeRuntime

        generator_config.dist_moe_runtime = DistMoeRuntime.Config(
            scratch_capacity_factor=4.0, inference=True
        )
        generator_config = apply_transforms(generator_config, [DistMoeTransform()])
    # The constructor and generate() signatures differ across the TorchTitan stacks this script compares.
    init_kwargs = dict(
        model_path=args.model, max_num_seqs=230, output_dir=args.out + ".sl", rank=0
    )
    init_params = inspect.signature(VLLMGenerator.__init__).parameters
    generator = VLLMGenerator(
        generator_config, **{k: v for k, v in init_kwargs.items() if k in init_params}
    )
    sampling = dataclasses.replace(generator_config.sampling, stop_token_ids=[])
    generate_params = inspect.signature(generator.generate).parameters
    counter = [0]

    async def generate(
        prompts: list[list[int]], max_tokens: int, decode: bool = False
    ) -> None:
        counter[0] += 1

        async def one(i: int, prompt: list[int]):
            kwargs = dict(
                request_id=f"r{counter[0]}-{i}",
                group_id=0,
                routing_session_id=f"s{counter[0]}-{i}",
                sampling_config=dataclasses.replace(sampling, max_tokens=max_tokens),
            )
            return await generator.generate(
                prompt, **{k: v for k, v in kwargs.items() if k in generate_params}
            )

        await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))

    engine = _Engine(
        step_owner=generator._engine,
        engine_core=lambda: generator._engine.engine_core,
        generate=generate,
        start=generator.start_engine_loop,
    )
    desc = dict(
        config=args.config,
        tt=os.getcwd(),
        abl_off=os.environ.get("ABL_OFF", ""),
        regions=model.local_compile_regions,
        ladder=config.get("ladder", "full"),
        dist_moe=bool(config.get("dist_moe")),
    )
    return engine, desc


def _build_prime_engine(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    overrides = {
        "text_config": {
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        }
    }
    if args.fp32_head:
        overrides["head_dtype"] = "float32"
    llm = LLM(
        model=args.model,
        load_format="dummy",
        hf_overrides=overrides,
        max_model_len=65536,
        gpu_memory_utilization=args.gpu_mem,
        max_num_batched_tokens=16384,
        enable_prefix_caching=True,
        enable_return_routed_experts=True,
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )

    async def generate(
        prompts: list[list[int]], max_tokens: int, decode: bool = False
    ) -> None:
        # Decode asks for the sampled token's logprob, as the RL run; prefill requests ask for none.
        params = SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=True,
            temperature=1.0,
            logprobs=0 if decode else None,
        )
        llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False
        )

    async def start() -> None:
        pass

    engine = _Engine(
        step_owner=llm.llm_engine,
        engine_core=lambda: llm.llm_engine.engine_core,
        generate=generate,
        start=start,
    )
    return engine, dict(config="prime", fp32_head=args.fp32_head)


def rand_prompts(batch: int, length: int, seed: int) -> list[list[int]]:
    import torch

    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(1000, 200000, (length,), generator=generator).tolist()
        for _ in range(batch)
    ]


def _kernel_us(profiler, nsteps: int) -> float:
    return sum(e.self_device_time_total or 0.0 for e in profiler.key_averages()) / max(
        nsteps, 1
    )


def _copy_us(profiler, nsteps: int) -> float:
    """Kernel time per step of state and block copies (vLLM's batch_memcpy, the chunked copy, memcpy/memset)."""
    keys = ("memcpy", "copy_block", "mamba", "memset")
    events = profiler.key_averages()
    return sum(
        e.self_device_time_total or 0.0
        for e in events
        if any(k in e.key.lower() for k in keys)
    ) / max(nsteps, 1)


def _dump_kernels(profiler, nsteps: int, label: str, args) -> None:
    """With --topk-out: append the 400 longest kernels of this shape, as [name, us per step, calls per step]."""
    if not args.topk_out:
        return
    events = [
        (
            e.key,
            (e.self_device_time_total or 0.0) / max(nsteps, 1),
            e.count / max(nsteps, 1),
        )
        for e in profiler.key_averages()
        if e.self_device_time_total
    ]
    top = sorted(events, key=lambda t: -t[1])[:400]
    with open(args.topk_out, "a") as f:
        row = dict(
            label=label,
            tag=args.tag,
            top=[[k[:120], round(us, 1), round(calls, 2)] for k, us, calls in top],
        )
        f.write(json.dumps(row) + "\n")


def _other_gpu_processes() -> dict:
    """Other processes on this GPU and their memory: the GPU is shared with other agents' benchmarks."""
    query = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    lines = (
        subprocess.run(query, capture_output=True, text=True, timeout=10)
        .stdout.strip()
        .splitlines()
    )
    rows = [tuple(int(x) for x in line.split(",")) for line in lines if line.strip()]
    others = [mib for pid, mib in rows if pid != os.getpid()]
    return dict(others=len(others), others_mib=sum(others))


# ----------------------------------------------------------------------------------------------- report


def _shape_key(record: dict) -> tuple:
    if record["kind"] == "decode":
        return ("decode", record["batch"], record["ctx"])
    return ("prefill", record["new"], record["cached"])


def _shape_label(key: tuple) -> str:
    if key[0] == "decode":
        return f"decode B={key[1]}, {key[2] // 1024}k ctx"
    return f"prefill {key[1] // 1024}k new + {key[2]:,} cached"


def report(args: argparse.Namespace) -> None:
    """Print one row per shape and one column per tag: the min over passes of each pass's median."""
    # values[tag][shape][run_id] -> measurements of that pass (prefill has one per repeat)
    values = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )
    others = collections.defaultdict(list)
    for path in args.files:
        for line in open(path):
            record = json.loads(line)
            if (
                record.get("error")
                or record["kind"] == "startup"
                or args.metric not in record
            ):
                continue
            values[record["tag"]][_shape_key(record)][record["run_id"]].append(
                record[args.metric]
            )
            others[record["tag"]].append(record["others"])
    tags = list(values)
    base = args.base or tags[0]
    best = {
        tag: {
            shape: min(statistics.median(v) for v in runs.values())
            for shape, runs in shapes.items()
        }
        for tag, shapes in values.items()
    }
    shapes = sorted(
        {s for t in best.values() for s in t},
        key=lambda s: (s[0] != "decode", s[2], s[1]),
    )
    deltas = [t for t in tags if t != base]
    print(
        f"\n{args.metric} (min over passes of each pass's median); delta columns vs {base}\n"
    )
    print(
        "| " + " | ".join(["shape", *tags, *(f"{t} vs {base}" for t in deltas)]) + " |"
    )
    print("|---" * (1 + len(tags) + len(deltas)) + "|")
    for shape in shapes:
        cells = [f"{best[t][shape]:,.0f}" if shape in best[t] else "-" for t in tags]
        b = best[base].get(shape)
        pct = [
            f"{100 * (best[t][shape] - b) / b:+.1f}%" if b and shape in best[t] else "-"
            for t in deltas
        ]
        print(f"| {_shape_label(shape)} | " + " | ".join(cells + pct) + " |")
    passes = {t: len({r for runs in values[t].values() for r in runs}) for t in tags}
    print("\npasses: " + ", ".join(f"{t} {n}" for t, n in passes.items()))
    print(
        "other GPU processes per shape: "
        + ", ".join(f"{t} {min(v)}-{max(v)}" for t, v in others.items())
    )


# -------------------------------------------------------------------------------------------------- gap

# Kernel name -> family, first match wins; kernels matching none are GLUE.
GLUE = "norm/elementwise glue"
KERNEL_FAMILIES = [
    (
        "MoE routed experts",
        r"_17Gr|GroupProblem|prepare_grouped_gemm|grouped_gemm|moe_gemm|fused_moe|cutlass_moe|"
        r"moe::dev::finalize|expandInputRows|finalizeMoe|doActivation|moe_permute|moe_unpermute|moe_align|moe_sum|"
        r"dist_moe|_triton_swiglu|scale_and_sum|dispatch_routing|^bmm_|"
        r"fused_index_4|floor_divide_index|scatter_zeros_like|fused_cumsum_1|gather_index_index_put|"
        r"add_gather_mul_scatter_sum|_to_copy_add_index_mul_select_slice|arange_cumsum_eq_sum|"
        r"_scatter_gather_elementwise|DeviceScan|ReduceOp<long, at::native::func_wrapper_t<long|countsort|count_sort|histc",
    ),
    (
        "router/topk",
        r"moe::dev::routing|topkGating|topk_softmax|softmax_online|grouped_topk|routingRenormalize",
    ),
    (
        "shared expert",
        r"mul_sigmoid_0|shared_expert|mul_silu_slice|sigmoid_kernel_cuda",
    ),
    (
        "sampling/logprobs",
        r"cunn_SoftMax|log_softmax|ArgMax|distribution_elementwise|ge_sum|_ranks_kernel|"
        r"_gumbel_sample|sampl|top_k_top_p|logprob|_get_num_sampled|_scatter_num_accepted|_combine_sampled",
    ),
    (
        "attention",
        r"flash_attn|FlashAttention|flash_fwd|flashinfer::Batch|MergeStates|fmha|reshape_and_cache|"
        r"paged_attention|_fused_qk_rmsnorm_rope_gate|attn_fwd|trtllm_attn",
    ),
    (
        "GDN",
        r"delta_rule|short_conv|causal_conv|gated_delta|chunk_|l2norm|solve_tril|wy_repr|mamba|ssm|conv1d|"
        r"recompute_w_u|kkt|fla_|layer_norm_fwd_kernel|layer_norm_gated|_fwd_kernel_h|rms_norm__to_copy_mul_silu|"
        r"get_aligned_state_indices|gdn|post_conv",
    ),
    ("dense GEMMs", r"nvjet|gemm|Gemm|splitKreduce|cublas|xmma|cutlass3x|sgemm|matmul"),
    (
        "memcpy/other",
        r"Memcpy|Memset|memcpy|memset|_compute_slot_mappings|_gather_block_tables|_prepare_pos_seq_lens|"
        r"_prepare_rope_positions|_copy_page_indices|_post_update_kernel",
    ),
]


def kernel_family(name: str, calls_per_step: float) -> list[tuple[str, float]]:
    """[(family, share of this kernel's time)].

    Example:
        kernel_family("mul_silu_unbind_...", 8.0)       -> [("MoE routed experts", 8/9), ("shared expert", 1/9)]
        kernel_family("BinaryFunctor<float, ...>", 1.0) -> [("sampling/logprobs", 1.0)]  # once per step
    """
    if (
        "mul_silu_unbind" in name
    ):  # one SwiGLU kernel serves routed (top-8 x T rows) and shared (T rows) experts
        return [("MoE routed experts", 8 / 9), ("shared expert", 1 / 9)]
    if calls_per_step <= 1.01 and (
        "BinaryFunctor<float" in name or "_scatter_gather_elementwise" in name
    ):
        return [
            ("sampling/logprobs", 1.0)
        ]  # once per step on fp32 logits: temperature, Gumbel, logprob gather
    if calls_per_step <= 1.01 and (
        "sigmoid_kernel_cuda" in name or "mul_sigmoid" in name
    ):
        return [
            ("attention", 1.0)
        ]  # the one full-attention layer's output gate; shared-expert gates run 4x
    for family, pattern in KERNEL_FAMILIES:
        if re.search(pattern, name):
            return [(family, 1.0)]
    return [(GLUE, 1.0)]


def gap(args: argparse.Namespace) -> None:
    """Per shape, kernel time by family for tags A and B and each family's share of the gap."""
    dumps = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )  # tag -> label -> [kernel lists]
    for line in open(args.kernels):
        row = json.loads(line)
        dumps[row["tag"]][row["label"]].append(row["top"])
    order = [family for family, _ in KERNEL_FAMILIES] + [GLUE]
    for label in [label for label in dumps[args.a] if label in dumps[args.b]]:
        per_tag = {}
        for tag in (args.a, args.b):
            least_contended = min(
                dumps[tag][label], key=lambda top: sum(us for _, us, _ in top)
            )
            families, members = collections.defaultdict(float), collections.defaultdict(
                list
            )
            for name, us, calls in least_contended:
                for family, share in kernel_family(name, calls):
                    families[family] += us * share
                    members[family].append((us * share, calls, name))
            per_tag[tag] = (families, members)
        (fa, ma), (fb, mb) = per_tag[args.a], per_tag[args.b]
        total_a, total_b = sum(fa.values()), sum(fb.values())
        print(
            f"\n#### {label}: {args.a} {total_a:,.0f} us vs {args.b} {total_b:,.0f} us "
            f"(gap {total_a - total_b:+,.0f} us, {100 * (total_a - total_b) / total_b:+.1f}%)\n"
        )
        print(
            f"| family | {args.a} | {args.b} | delta | share of gap |\n|---|---|---|---|---|"
        )
        for family in order:
            a, b = fa.get(family, 0.0), fb.get(family, 0.0)
            print(
                f"| {family} | {a:,.0f} | {b:,.0f} | {a - b:+,.0f} | {100 * (a - b) / (total_a - total_b):+.0f}% |"
            )
        if args.show:
            for family in order:
                for tag, members in ((args.a, ma), (args.b, mb)):
                    for us, calls, name in sorted(
                        members.get(family, []), reverse=True
                    )[:8]:
                        print(
                            f"    {family[:14]:14s} {tag[:8]:8s} {us:7.1f} x{calls:5.2f} {name[:100]}"
                        )


if __name__ == "__main__":
    main()
