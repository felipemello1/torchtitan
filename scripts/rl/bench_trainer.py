#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer microbenchmark: fwd+bwd of 2 Qwen3.5 decoder blocks, TorchTitan's RL trainer vs prime-rl's modules.

One GPU, random bf16 weights, documents of 4,096 tokens packed into T tokens, full activation checkpointing:

    side    blocks                                                         compile
    titan   layers 2-3 of the RL trainer config ([GDN, full attention])     the config's local_compile regions
    prime   prime-rl's Qwen3_5Model, same 2 layers, its RL defaults:        per-block torch.compile inside AC
            FA4, gate_up + qkv fusions, frozen fp32 router

Each process runs one T: 3 warmup steps, 8 timed steps (CUDA events, median = wall_us), then 3 profiled steps
(kernel_us = CUDA kernel time per step) and the peak memory.

Subcommands:

    launch   set up the side's venv and env, then run ``measure`` once per --tokens value
    measure  the in-process benchmark (``launch`` calls it); appends one JSON line to --out
    report   kernel, wall or peak-memory table per T, one column per tag, min over passes

Paths below are absolute on our GB300 cluster; pass --tt / --titan-venv / --prime-root / --model elsewhere.

    TorchTitan venv   /storage/home/felipemello/libraries/fairtitan/runtimes/swe-20261013-verifiers/torchtitan/titan-rl
    prime-rl root     /checkpoint/pytorch/felipemello/runtimes/prime-latest-35b-next-runtime/aarch64/prime-rl
    model config      /checkpoint/transformer2/oss_models/Qwen3.5-35B-A3B-Base/config.json (prime side)

Compare two TorchTitan stacks (e.g. main vs a PR stack):

    python scripts/rl/bench_trainer.py launch --side titan --tt ~/tt-main  --tag main  --out res/train.jsonl
    python scripts/rl/bench_trainer.py launch --side titan --tt ~/tt-stack --tag stack --out res/train.jsonl
    python scripts/rl/bench_trainer.py launch --side prime --tag prime --out res/train.jsonl
    python scripts/rl/bench_trainer.py report res/train.jsonl --base main

Reproduce the selected-set training table (2-3 passes):

    for pass in 1 2 3; do
      for c in baseline A B C D E B0; do
        python scripts/rl/bench_trainer.py launch --side titan --config $c --out res/train.jsonl
      done
      python scripts/rl/bench_trainer.py launch --side prime --tag prime --out res/train.jsonl
    done
    python scripts/rl/bench_trainer.py report res/train.jsonl --base A

Expected kernel_us per fwd+bwd step (35B-A3B, shared GB300, min of 2-3 passes):

    T      baseline      A      prime
    4k       12,784    9,371   11,296
    16k      29,141   16,973   33,660
    32k      54,172   30,976   64,503

The GPU is shared with other agents: keep runs short, repeat 2-3 passes, and read the min. ``--flavor 27B``
benchmarks the dense 27B on the titan side (the prime side is 35B-A3B only). For a kernel-family breakdown,
add ``--topk-out`` and read it with ``bench_generator.py gap``.
"""

import argparse
import collections
import json
import os
import socket
import statistics
import subprocess
import sys

TITAN_VENV = "/storage/home/felipemello/libraries/fairtitan/runtimes/swe-20261013-verifiers/torchtitan/titan-rl"
PRIME_ROOT = "/checkpoint/pytorch/felipemello/runtimes/prime-latest-35b-next-runtime/aarch64/prime-rl"
MODEL = "/checkpoint/transformer2/oss_models/Qwen3.5-35B-A3B-Base"
TORCHSTORE = "/storage/home/felipemello/libraries/torchstore_pkgs/d7aeaba-revert-nixl"
DIST_MOE = "/storage/home/felipemello/libraries/dist_moe_src"
# TorchTitan worktrees of the selected-set runs: `ablate` adds ABL_OFF switches (47f36e0fd); `ablate307` adds #307.
ABLATE_TT = "/checkpoint/pytorch/felipemello/worktrees/ablate"
ABLATE307_TT = "/checkpoint/pytorch/felipemello/worktrees/ablate307"
DOC_TOKENS = 4096

# Compile regions of the baseline; the configs below add the optimization PRs' regions to it.
BASE_REGIONS = [
    "fused_binary_activation",
    "gated_rmsnorm",
    "offset_rmsnorm",
    "partial_rope",
    "shared_expert_gate",
    "fp32_to_bf16_split",
    "moe_aux_loss",
]
_SELECTED_REGIONS = BASE_REGIONS + ["moe_dispatch", "moe_combine", "router"]
_ALL_REGIONS = _SELECTED_REGIONS + [
    "expert_counts",
    "residual_norm",
    "attention_output_gate",
]
_ALL_OFF = ["292", "294", "297", "298", "299", "300", "303", "305", "306"]
_SELECTED_OFF = ["294", "297", "298", "299", "300", "305"]
# regions: local_compile regions (None: the config's own); abl_off: ABL_OFF switches on the `ablate` worktree.
CONFIGS = {
    # The --tt worktree as it is.
    "stack": dict(regions=None, abl_off=[]),
    "baseline": dict(tt=ABLATE_TT, regions=BASE_REGIONS, abl_off=_ALL_OFF),
    # A: #292, #293, #5343, #296 (training gather-sum), stock MoE.
    "A": dict(tt=ABLATE_TT, regions=_SELECTED_REGIONS, abl_off=_SELECTED_OFF),
    "B": dict(
        tt=ABLATE_TT,
        regions=BASE_REGIONS + ["router"],
        abl_off=_SELECTED_OFF,
        dist_moe=True,
    ),
    # B0: Dist-MoE without the PRs.
    "B0": dict(tt=ABLATE_TT, regions=BASE_REGIONS, abl_off=_ALL_OFF, dist_moe=True),
    "C": dict(tt=ABLATE307_TT, regions=_SELECTED_REGIONS, abl_off=_SELECTED_OFF),
    "D": dict(tt=ABLATE_TT, regions=_ALL_REGIONS, abl_off=["305"]),
    "E": dict(tt=ABLATE307_TT, regions=_ALL_REGIONS, abl_off=["305"]),
}


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "launch":
        launch(args)
    elif args.command == "measure":
        if args.side == "titan":
            measure_titan(args)
        else:
            measure_prime(args)
    else:
        report(args)


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
            "--out", required=True, help="JSONL file; one line per (pass, T)"
        )
        p.add_argument(
            "--tokens",
            default="4096,16384,32768",
            help="packed tokens per step; launch runs each",
        )
        p.add_argument(
            "--flavor", default="35B-A3B", help="titan model flavor, e.g. 27B"
        )
        p.add_argument(
            "--topk-out",
            help="also append every kernel (us per step) for `bench_generator.py gap`",
        )
        p.add_argument(
            "--prof-rows",
            type=int,
            default=25,
            help="rows of the printed profiler table",
        )
        p.add_argument("--gpu", default="0")
        p.add_argument("--titan-venv", default=TITAN_VENV)
        p.add_argument("--prime-root", default=PRIME_ROOT)
        p.add_argument("--model", default=MODEL)
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p.add_argument(
        "--metric", choices=("kernel_us", "wall_us", "peak_gib"), default="kernel_us"
    )
    p.add_argument(
        "--base", help="tag the delta columns compare against (default: the first tag)"
    )
    return parser


def launch(args: argparse.Namespace) -> None:
    """Run ``measure`` in the side's venv, one process per T (compile caches and peak memory stay per T)."""
    config = CONFIGS[args.config]
    tag = args.tag or (args.config if args.side == "titan" else "prime")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")
    if args.side == "titan":
        tt = args.tt or config.get("tt") or os.getcwd()
        cuda = f"{args.titan_venv}/lib/python3.12/site-packages/nvidia/cu13"
        cache = f"/tmp/bench-train-{args.gpu}"
        env.update(
            PATH=f"{cuda}/bin:{args.titan_venv}/bin:{env['PATH']}",
            CUDA_HOME=cuda,
            LIBRARY_PATH=f"{cuda}/lib",
            LD_LIBRARY_PATH=f"{cuda}/lib:{env.get('LD_LIBRARY_PATH', '')}",
            PYTHONPATH=f"{tt}:{TORCHSTORE}:{DIST_MOE}",
            TRITON_CACHE_DIR=f"{cache}/triton",
            TORCHINDUCTOR_CACHE_DIR=f"{cache}/inductor",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
            LOCAL_WORLD_SIZE="1",
            MASTER_ADDR="127.0.0.1",
            ABL_OFF=",".join(config["abl_off"]),
        )
        python, cwd = f"{args.titan_venv}/bin/python", tt
    else:
        cuda = f"{args.prime_root}/.venv/lib/python3.12/site-packages/nvidia/cu13"
        env.update(
            PATH=f"{cuda}/bin:{args.prime_root}/.venv/bin:{env['PATH']}",
            CUDA_HOME=cuda,
            TRITON_CACHE_DIR=f"/tmp/bench-train-prime-{args.gpu}/triton",
            HF_HUB_OFFLINE="1",
        )
        python, cwd = f"{args.prime_root}/.venv/bin/python", args.prime_root
    passthrough = sys.argv[sys.argv.index("launch") + 1 :]
    for tokens in args.tokens.split(","):
        env["MASTER_PORT"] = str(_free_port())
        command = [
            python,
            os.path.abspath(__file__),
            "measure",
            *passthrough,
            "--tokens",
            tokens,
        ]
        if not args.tag:
            command += ["--tag", tag]
        log = f"{args.out}.{tag}.{tokens}.log"
        with open(log, "a") as log_file:
            result = subprocess.run(
                command, env=env, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT
            )
        print(
            f"{tag} T={tokens}: {'done' if result.returncode == 0 else 'FAILED'} (log {log})"
        )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def measure_titan(args: argparse.Namespace) -> None:
    """fwd+bwd of layers 2-3 of TorchTitan's RL trainer model, FullAC per block, EP=1, no FSDP."""
    import torch
    import torch.distributed as dist

    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    import torch_remat as remat
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.distributed.local_compile import apply_local_compile
    from torchtitan.distributed.parallelism_context import ParallelismContext
    from torchtitan.models.qwen3_5 import build_model_config
    from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
    from torchtitan_recipes.rl.model_configs import build_rl_model_configs

    config = CONFIGS[args.config]
    tokens = int(args.tokens)
    model_config, _ = build_rl_model_configs(
        build_model_config,
        args.flavor,
        seq_len=65536,
        attn_backend="varlen",
        trainer_parallelism=ParallelismConfig(),
        generator_parallelism=InferenceParallelismConfig(
            data_parallel_degree=1, tensor_parallel_degree=1
        ),
    )
    model_config.layers = model_config.layers[2:4]  # [GDN, full attention]
    if config["regions"] is not None:
        model_config.local_compile_regions = list(config["regions"])
    regions = list(model_config.local_compile_regions)
    if config.get("dist_moe"):
        from torchtitan.config.transform.apply import transform_model_config_
        from torchtitan.config.transform.dist_moe import DistMoeTransform

        model_config = transform_model_config_(model_config, [DistMoeTransform()])

    parallelism = ParallelismContext(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        world_size=1,
        enable_sequence_parallel=False,
    )
    with parallelism.activate_spmd(), torch.device("meta"):
        model = model_config.build()
    model.to_empty(device="cuda")
    torch.manual_seed(0)
    with torch.no_grad(), parallelism.activate_spmd():
        model.init_states(buffer_device=None)
    model.to(torch.bfloat16)
    apply_local_compile([r for r in regions if r != "loss"])  # no loss runs here
    model.train()
    model.configure_remat_regions([], [])  # FullAC: no policy-saved activations
    runtime = None
    if config.get("dist_moe"):
        from torchtitan.models.common.dist_moe import DistMoeRuntime

        runtime = DistMoeRuntime.Config(scratch_capacity_factor=4.0).build(
            model_parts=[model],
            parallelism_context=parallelism,
            device=torch.device("cuda"),
            num_tokens_per_microbatch_per_dp_rank=tokens,
            pp_schedule=None,
        )

    generator = torch.Generator(device="cuda").manual_seed(tokens)
    x = torch.randn(
        tokens,
        model_config.dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    x.requires_grad_()
    grad_out = torch.randn(
        tokens,
        model_config.dim,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    positions = torch.arange(tokens, device="cuda") % DOC_TOKENS
    padding = torch.zeros(tokens, dtype=torch.bool, device="cuda")
    with parallelism.activate_spmd():
        metadata = model._get_attention_metadata(
            positions=positions.unsqueeze(0),
            padding_mask=None,
            max_context_length=65536,
        )
    blocks = list(model.layers.values())

    def fwd_bwd() -> None:
        for p in model.parameters():
            p.grad = None
        x.grad = None
        with parallelism.activate_spmd():
            h = x
            for i, block in enumerate(blocks):

                def run(h_, block=block):
                    return block(
                        h_,
                        metadata.get(block.attention_metadata_key),
                        positions,
                        padding_mask=padding,
                    )

                checkpoint = remat.checkpoint(
                    region_name=f"layers.{i}",
                    determinism_check="default",
                    preserve_rng_state=False,
                )
                h = checkpoint(run)(h)
        h.backward(grad_out)

    record = _time_fwd_bwd(fwd_bwd, args)
    record.update(
        regions=regions,
        abl_off=os.environ.get("ABL_OFF", ""),
        dist_moe=bool(config.get("dist_moe")),
    )
    _emit(record, args)
    if runtime is not None:
        runtime.close()


def measure_prime(args: argparse.Namespace) -> None:
    """fwd+bwd of 2 blocks of prime-rl's Qwen3_5Model with the RL run's settings."""
    import torch
    from prime_rl.configs.trainer import ActivationCheckpointConfig
    from prime_rl.trainer.activation_checkpointing import (
        get_activation_checkpoint_wrapper,
    )
    from prime_rl.trainer.model import (
        apply_fp32_moe_router,
        freeze_moe_router,
        mark_dynamic_int_args,
    )
    from prime_rl.trainer.models.fusions import apply_model_fusions
    from prime_rl.trainer.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5MoeTextConfig,
    )
    from prime_rl.trainer.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    tokens = int(args.tokens)
    hf_config = json.load(open(f"{args.model}/config.json"))["text_config"]
    hf_config.update(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        vocab_size=1024,
    )
    model_config = Qwen3_5MoeTextConfig(**hf_config)
    model_config._attn_implementation = "flash_attention_4"
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = Qwen3_5Model(model_config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                p.normal_(0.0, 0.02)
            elif "norm" in name:
                p.fill_(1.0)
            elif "A_log" in name or "dt_bias" in name:
                p.zero_()
    model.to(torch.bfloat16)
    apply_model_fusions(model, ["gate_up", "qkv"])
    holder = torch.nn.Module()  # prime-rl's helpers find the text model at <root>.model
    holder.model = model
    freeze_moe_router(holder)
    apply_fp32_moe_router(holder)
    wrap = get_activation_checkpoint_wrapper(ActivationCheckpointConfig())
    for name, block in list(model.layers.named_children()):
        model.layers.register_module(name, wrap(block))
    torch._dynamo.config.recompile_limit = 16
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.capture_scalar_outputs = True
    mark_dynamic_int_args()
    for layer in model.layers:
        # The RL run keeps AC eager and compiles the block inside it (the fp32 router is its own FSDP unit there).
        layer._checkpoint_wrapped_module.compile()
    model.train()

    generator = torch.Generator(device="cuda").manual_seed(tokens)
    hidden = model_config.hidden_size
    x = torch.randn(
        1, tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
    ).requires_grad_()
    grad_out = torch.randn(
        1, tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    position_ids = (torch.arange(tokens, device="cuda") % DOC_TOKENS).unsqueeze(0)
    seq_lens = torch.full(
        (tokens // DOC_TOKENS,), DOC_TOKENS, dtype=torch.long, device="cuda"
    )

    def fwd_bwd() -> None:
        for p in model.parameters():
            p.grad = None
        x.grad = None
        out = model(
            inputs_embeds=x, position_ids=position_ids, seq_lens=seq_lens
        ).last_hidden_state
        out.backward(grad_out)

    _emit(_time_fwd_bwd(fwd_bwd, args), args)


def _time_fwd_bwd(fwd_bwd, args: argparse.Namespace) -> dict:
    """3 warmup steps, 8 CUDA-event-timed steps, then 3 profiled steps; times in us per step."""
    import torch

    for _ in range(3):
        fwd_bwd()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    walls_us = []
    for _ in range(8):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        fwd_bwd()
        end.record()
        end.synchronize()
        walls_us.append(start.elapsed_time(end) * 1000)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
    with profiler:
        for _ in range(3):
            fwd_bwd()
        torch.cuda.synchronize()
    events = profiler.key_averages()
    print(
        events.table(sort_by="device_time_total", row_limit=args.prof_rows), flush=True
    )
    if args.topk_out:
        top = sorted(
            (e for e in events if e.self_device_time_total),
            key=lambda e: -e.self_device_time_total,
        )[:400]
        with open(args.topk_out, "a") as f:
            rows = [
                [
                    e.key[:120],
                    round(e.self_device_time_total / 3, 1),
                    round(e.count / 3, 2),
                ]
                for e in top
            ]
            f.write(
                json.dumps(dict(label=f"train:{args.tokens}", tag=args.tag, top=rows))
                + "\n"
            )
    kernel_us = sum(e.self_device_time_total for e in events) / 3
    return dict(
        tokens=int(args.tokens),
        kernel_us=round(kernel_us, 1),
        wall_us=round(statistics.median(walls_us), 1),
        wall_min_us=round(min(walls_us), 1),
        peak_gib=round(peak_gib, 2),
    )


def _emit(record: dict, args: argparse.Namespace) -> None:
    record.update(
        side=args.side,
        tag=args.tag,
        config=args.config,
        flavor=args.flavor,
        cwd=os.getcwd(),
    )
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    print("RESULT", json.dumps(record), flush=True)


def report(args: argparse.Namespace) -> None:
    """Print one row per T and one column per tag: the min over passes."""
    values = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )  # tag -> tokens -> passes
    for path in args.files:
        for line in open(path):
            record = json.loads(line)
            values[record["tag"]][record["tokens"]].append(record[args.metric])
    tags = list(values)
    base = args.base or tags[0]
    deltas = [t for t in tags if t != base]
    print(
        f"\n{args.metric} per fwd+bwd step (min over passes); delta columns vs {base}\n"
    )
    print(
        "| "
        + " | ".join(["packed tokens", *tags, *(f"{t} vs {base}" for t in deltas)])
        + " |"
    )
    print("|---" * (1 + len(tags) + len(deltas)) + "|")
    for tokens in sorted({t for per_tag in values.values() for t in per_tag}):
        best = {
            tag: min(values[tag][tokens]) if values[tag].get(tokens) else None
            for tag in tags
        }
        cells = [
            f"{best[t]:,.2f}"
            if args.metric == "peak_gib" and best[t]
            else f"{best[t]:,.0f}"
            if best[t]
            else "-"
            for t in tags
        ]
        b = best[base]
        pct = [
            f"{100 * (best[t] - b) / b:+.1f}%" if b and best[t] else "-" for t in deltas
        ]
        print(f"| {tokens // 1024}k | " + " | ".join(cells + pct) + " |")
    print(
        "\npasses per T: "
        + ", ".join(
            f"{t} " + "/".join(str(len(values[t][n])) for n in sorted(values[t]))
            for t in tags
        )
    )


if __name__ == "__main__":
    main()
