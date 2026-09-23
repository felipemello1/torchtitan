#!/usr/bin/env python3
"""Profile Qwen3.5-27B eager, regional-compile, and kernel-reference paths."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from torch.profiler import ProfilerActivity, profile, record_function

TensorFn = Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]
EPS = 1e-6


@dataclass
class Region:
    inputs: tuple[torch.Tensor, ...]
    grad_outputs: tuple[torch.Tensor, ...]
    implementations: dict[str, TensorFn]


def offset_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return ((1.0 + weight.float()) * x).to(dtype)


def gated_rmsnorm(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    x = (weight.float() * x).to(dtype)
    return (x * F.silu(gate.float())).to(dtype)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def eager_swiglu_from_packed(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.unbind(-2)
    return swiglu(gate, up)


def compiled_swiglu_from_packed(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.unbind(-2)
    return compiled_swiglu(gate, up)


def partial_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = rope_cache.shape[-1] // 2
    cache = rope_cache[positions].unsqueeze(1)
    cos, sin = cache[..., :rotary_dim], cache[..., rotary_dim:]

    query_rot = query[..., :rotary_dim].float()
    key_rot = key[..., :rotary_dim].float()
    half = rotary_dim // 2
    query_half = torch.cat((-query_rot[..., half:], query_rot[..., :half]), -1)
    key_half = torch.cat((-key_rot[..., half:], key_rot[..., :half]), -1)
    query_rot = (query_rot * cos + query_half * sin).to(query.dtype)
    key_rot = (key_rot * cos + key_half * sin).to(key.dtype)
    return (
        torch.cat((query_rot, query[..., rotary_dim:]), -1),
        torch.cat((key_rot, key[..., rotary_dim:]), -1),
    )


def residual_offset_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = x + residual
    return offset_rmsnorm(residual_out, weight), residual_out


def compiled(fn: TensorFn) -> TensorFn:
    return torch.compile(fn, fullgraph=True, dynamic=False)


compiled_swiglu = compiled(swiglu)


def random_tensor(
    shape: tuple[int, ...], generator: torch.Generator, *, grad: bool = True
) -> torch.Tensor:
    return torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, generator=generator
    ).requires_grad_(grad)


def build_regions(tokens: int) -> dict[str, Region]:
    generator = torch.Generator(device="cuda").manual_seed(42)

    offset_shape = (tokens, 24, 256)
    offset_inputs = (
        random_tensor(offset_shape, generator),
        random_tensor((256,), generator),
    )

    gated_shape = (tokens * 48, 128)
    gated_inputs = (
        random_tensor(gated_shape, generator),
        random_tensor(gated_shape, generator),
        random_tensor((128,), generator),
    )

    swiglu_shape = (tokens, 17_408)
    swiglu_inputs = (random_tensor((tokens, 2, 17_408), generator),)

    query_shape = (tokens, 24, 256)
    key_shape = (tokens, 4, 256)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    theta = 10_000_000.0
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64)
    )
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), -1)
    rope_cache = torch.cat((emb.cos(), emb.sin()), -1)
    rope_inputs = (
        random_tensor(query_shape, generator),
        random_tensor(key_shape, generator),
        rope_cache,
        positions,
    )

    residual_shape = (tokens, 5_120)
    residual_inputs = (
        random_tensor(residual_shape, generator),
        random_tensor(residual_shape, generator),
        random_tensor((5_120,), generator),
    )

    regions = {
        "offset_rmsnorm": Region(
            offset_inputs,
            (random_tensor(offset_shape, generator, grad=False),),
            {"eager": offset_rmsnorm, "compile": compiled(offset_rmsnorm)},
        ),
        "gdn_rmsnorm_gated": Region(
            gated_inputs,
            (random_tensor(gated_shape, generator, grad=False),),
            {"eager": gated_rmsnorm, "compile": compiled(gated_rmsnorm)},
        ),
        "swiglu": Region(
            swiglu_inputs,
            (random_tensor(swiglu_shape, generator, grad=False),),
            {
                "eager": eager_swiglu_from_packed,
                "compile": compiled_swiglu_from_packed,
            },
        ),
        "partial_rope": Region(
            rope_inputs,
            (
                random_tensor(query_shape, generator, grad=False),
                random_tensor(key_shape, generator, grad=False),
            ),
            {"eager": partial_rope, "compile": compiled(partial_rope)},
        ),
        "residual_ffn_rmsnorm": Region(
            residual_inputs,
            (
                random_tensor(residual_shape, generator, grad=False),
                random_tensor(residual_shape, generator, grad=False),
            ),
            {
                "eager": residual_offset_rmsnorm,
                "compile": compiled(residual_offset_rmsnorm),
            },
        ),
    }
    add_kernel_references(regions)
    return regions


def add_kernel_references(regions: dict[str, Region]) -> None:
    """Add installed reference kernels without making them dependencies."""
    try:
        from torchtitan.overrides.fused_swiglu import (
            _MAX_BLOCK_N,
            _SILU_AND_MUL_BLOCK_M,
            _silu_and_mul_backward_kernel,
            silu_and_mul_forward_kernel,
            silu_and_mul_op,
        )

        def triton_swiglu(gate_up):
            gate, up = gate_up.unbind(-2)
            return silu_and_mul_op(gate, up)

        class PackedSwiGLU(torch.autograd.Function):
            @staticmethod
            def forward(ctx, gate_up):
                gate, up = gate_up.unbind(-2)
                ctx.save_for_backward(gate_up)
                return silu_and_mul_forward_kernel(gate, up)

            @staticmethod
            def backward(ctx, grad_output):
                (gate_up,) = ctx.saved_tensors
                gate, up = gate_up.unbind(-2)
                grad_packed = torch.empty_like(gate_up)
                grad_gate, grad_up = grad_packed.unbind(-2)
                block_m = _SILU_AND_MUL_BLOCK_M
                block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(gate.shape[-1]))
                grid = (
                    triton.cdiv(gate.shape[0], block_m),
                    triton.cdiv(gate.shape[1], block_n),
                )
                _silu_and_mul_backward_kernel[grid](
                    grad_output,
                    gate,
                    up,
                    grad_gate,
                    grad_up,
                    gate,
                    NUM_ROWS=gate.shape[0],
                    NUM_COLS=gate.shape[1],
                    NUM_OFFSETS=0,
                    HAS_OFFSETS=False,
                    GRAD_OUT_ROW_STRIDE=grad_output.stride(0),
                    GRAD_OUT_COL_STRIDE=grad_output.stride(1),
                    GATE_ROW_STRIDE=gate.stride(0),
                    GATE_COL_STRIDE=gate.stride(1),
                    UP_ROW_STRIDE=up.stride(0),
                    UP_COL_STRIDE=up.stride(1),
                    GRAD_GATE_ROW_STRIDE=grad_gate.stride(0),
                    GRAD_GATE_COL_STRIDE=grad_gate.stride(1),
                    GRAD_UP_ROW_STRIDE=grad_up.stride(0),
                    GRAD_UP_COL_STRIDE=grad_up.stride(1),
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=8,
                )
                return grad_packed

        regions["swiglu"].implementations["triton"] = triton_swiglu
        regions["swiglu"].implementations["triton_packed"] = PackedSwiGLU.apply
    except ImportError:
        pass

    try:
        from torchtitan.overrides.offset_rmsnorm import triton_offset_rms_norm

        regions["offset_rmsnorm"].implementations["triton"] = lambda x, weight: (
            triton_offset_rms_norm(x, weight, EPS)
        )
    except ImportError:
        pass

    try:
        from fla.modules import FusedRMSNormGated
        from fla.modules.layernorm import rms_norm

        gated = FusedRMSNormGated(
            128,
            eps=EPS,
            activation="swish",
            device="cuda",
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            gated.weight.copy_(regions["gdn_rmsnorm_gated"].inputs[2])

        def fla_gated(x, gate, _weight):
            gated.weight.grad = None
            return gated(x, gate)

        regions["gdn_rmsnorm_gated"].implementations["fla"] = fla_gated
        regions["residual_ffn_rmsnorm"].implementations["fla"] = (
            lambda x, residual, weight: rms_norm(
                x, weight + 1.0, None, residual=residual, eps=EPS, prenorm=True
            )
        )
    except ImportError:
        pass

    try:
        from torchtitan.overrides.helion_rope import _apply_helion_cossin_rope

        regions["partial_rope"].implementations["helion"] = (
            lambda query, key, cache, positions: _helion_partial_rope(
                _apply_helion_cossin_rope, query, key, cache, positions
            )
        )
    except ImportError:
        pass


def _helion_partial_rope(apply_helion, query, key, cache, positions):
    rotary_dim = cache.shape[-1] // 2
    rotated = apply_helion(
        query[..., :rotary_dim], key[..., :rotary_dim], cache, positions
    )
    if rotated is None:
        raise RuntimeError("Helion rejected the Qwen3.5 text-only shape")
    query_rot, key_rot = rotated
    return (
        torch.cat((query_rot, query[..., rotary_dim:]), -1),
        torch.cat((key_rot, key[..., rotary_dim:]), -1),
    )


def run_once(region: Region, implementation: str) -> None:
    for tensor in region.inputs:
        tensor.grad = None
    outputs = region.implementations[implementation](*region.inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    torch.autograd.backward(outputs, region.grad_outputs)


def timed(region: Region, implementation: str) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_once(region, implementation)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def export_trace(
    region_name: str,
    region: Region,
    implementation: str,
    output_dir: Path,
) -> Path:
    trace = output_dir / f"{region_name}__{implementation}.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function(f"{region_name}/{implementation}/forward_backward"):
            run_once(region, implementation)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    compressed = trace.with_suffix(".json.gz")
    with trace.open("rb") as source, gzip.open(compressed, "wb") as target:
        target.write(source.read())
    trace.unlink()
    return compressed


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=40_960)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("region_traces"))
    parser.add_argument(
        "--regions",
        default="all",
        help="Comma-separated region names, or all",
    )
    parser.add_argument(
        "--implementations",
        default="eager,compile",
        help=(
            "Comma-separated implementations: "
            "eager,compile,triton,triton_packed,fla,helion"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.implementations.split(",")
    regions = build_regions(args.tokens)
    selected_regions = (
        set(regions) if args.regions == "all" else set(args.regions.split(","))
    )
    unknown_regions = selected_regions - set(regions)
    if unknown_regions:
        raise ValueError(f"Unknown regions: {sorted(unknown_regions)}")
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "dtype": "bfloat16",
        "tokens": args.tokens,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "regions": {},
    }

    for region_name, region in regions.items():
        if region_name not in selected_regions:
            continue
        region_result: dict[str, object] = {}
        result["regions"][region_name] = region_result
        for implementation in selected:
            if implementation not in region.implementations:
                continue
            try:
                for _ in range(args.warmups):
                    run_once(region, implementation)
                torch.cuda.synchronize()
                samples = [
                    timed(region, implementation) for _ in range(args.iterations)
                ]
                trace = export_trace(
                    region_name, region, implementation, args.output_dir
                )
                region_result[implementation] = {
                    **summarize(samples),
                    "trace": trace.name,
                }
            except Exception as error:  # noqa: BLE001 - record unsupported kernels
                region_result[implementation] = {
                    "error": f"{type(error).__name__}: {error}"
                }
                torch.cuda.empty_cache()
            print(
                region_name, implementation, region_result[implementation], flush=True
            )

    output = args.output_dir / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
