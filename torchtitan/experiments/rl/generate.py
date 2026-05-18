#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone vLLM CLI smoke for the RL stack.

Two modes:

1. Raw text: ``--prompt "Hello, my name is"`` (bypasses the renderer).
2. Chat messages via renderer: ``--message "user:What is 2+2?"`` (repeatable).
   The renderer converts the message list into token IDs, the response is
   parsed back into a structured ``ParsedResponse``.

The script talks to ``vLLM.LLMEngine`` directly (no Monarch actor), so it is
the cheapest way to verify the renderer ↔ generator wiring without spinning
up the full RL controller. The Monarch-actor path lands in PR3 with the env
migration.

Example::

    python torchtitan/experiments/rl/generate.py \\
        --config rl_grpo_qwen3_0_6b \\
        --message "system:You are concise." \\
        --message "user:What is 2+2?"
"""
from __future__ import annotations

import argparse
import os
from importlib import import_module

# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
# See also https://docs.vllm.ai/en/v0.8.3/design/multiprocessing.html#python-multiprocessing
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from renderers import Message, ParsedResponse, Renderer
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.logger import init_logger

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.experiments.rl.actors.generator import SamplingConfig
from torchtitan.experiments.rl.renderer import RendererConfig

logger = init_logger(__name__)


def generate_from_messages(
    *,
    engine: LLMEngine,
    renderer: Renderer,
    messages: list[Message],
    sampling: SamplingConfig,
    request_id: str = "0",
) -> ParsedResponse:
    """Render ``messages`` -> tokens, generate one completion, parse it back.

    The renderer's stop token IDs are passed to ``SamplingParams`` so generation
    halts at the assistant turn boundary (e.g. ``<|im_end|>`` on Qwen3),
    matching what the controller does in PR3 once it owns the rendering path.

    Returns the renderer's structured ``ParsedResponse`` (text + tool calls +
    reasoning chunks).
    """
    prompt_token_ids = list(renderer.render_ids(messages, add_generation_prompt=True))
    sampling_params = SamplingParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
        n=1,
        stop_token_ids=list(renderer.get_stop_token_ids()),
        logprobs=1,
    )
    engine.add_request(
        request_id, {"prompt_token_ids": prompt_token_ids}, sampling_params
    )
    outputs = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
    [out] = outputs
    response_token_ids = list(out.outputs[0].token_ids)
    return renderer.parse_response(response_token_ids)


def _parse_messages(items: list[str]) -> list[Message]:
    """Parse ``--message role:content`` repeated args into a message list."""
    messages: list[Message] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"--message must be 'role:content', got {item!r}")
        role, content = item.split(":", 1)
        messages.append({"role": role, "content": content})
    return messages


def _build_engine(config) -> LLMEngine:
    gen_config = config.generator
    model_path = config.hf_assets_path

    from torchtitan.experiments.rl.models.vllm_registry import (
        registry_to_vllm,
        VLLM_MODEL_NAME,
    )

    registry_to_vllm(
        config.model_spec,
        parallelism=gen_config.parallelism,
        compile_config=config.compile,
        checkpoint_config=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path=model_path,
        ),
    )

    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype=gen_config.model_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=not gen_config.cudagraph.enable,
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        attention_backend="CUSTOM",
        max_num_seqs=max(config.num_prompts_per_step * gen_config.sampling.n, 1),
    )
    vllm_compilation_config = gen_config.cudagraph.get_vllm_compilation_config(
        max_num_seqs=engine_kwargs["max_num_seqs"],
    )
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if gen_config.debug.seed is not None:
        engine_kwargs["seed"] = gen_config.debug.seed
    return LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="rl_grpo_qwen3_0_6b",
        help="Function name in torchtitan.experiments.rl.config_registry to load.",
    )
    parser.add_argument(
        "--renderer",
        default="auto",
        help="Renderer name passed to renderers.create_renderer. "
        "Defaults to 'auto' (chat-template inferred).",
    )
    parser.add_argument(
        "--prompt",
        help="Raw text prompt; bypasses the renderer. Mutually exclusive with --message.",
    )
    parser.add_argument(
        "--message",
        action="append",
        default=[],
        metavar="ROLE:CONTENT",
        help="Repeatable chat message in 'role:content' form. "
        "Goes through the renderer. Defaults to a single user 'Hello, who are you?'.",
    )
    args = parser.parse_args()

    if args.prompt and args.message:
        parser.error("Pass --prompt OR --message, not both.")

    config_registry = import_module("torchtitan.experiments.rl.config_registry")
    config = getattr(config_registry, args.config)()
    engine = _build_engine(config)
    sampling = config.generator.sampling

    if args.prompt is not None:
        sampling_params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
        )
        engine.add_request("0", args.prompt, sampling_params)
        outputs = []
        while engine.has_unfinished_requests():
            outputs.extend(engine.step())
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated text: {outputs[0].outputs[0].text!r}\n")
        return

    messages = (
        _parse_messages(args.message)
        if args.message
        else [{"role": "user", "content": "Hello, who are you?"}]
    )
    renderer = RendererConfig(name=args.renderer).build(
        model_path=config.hf_assets_path
    )
    parsed = generate_from_messages(
        engine=engine, renderer=renderer, messages=messages, sampling=sampling
    )
    print("\nMessages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print(f"\nParsed response:\n  content: {parsed.content!r}")
    if parsed.tool_calls:
        print(f"  tool_calls: {parsed.tool_calls}")
    if parsed.reasoning_content:
        print(f"  reasoning: {parsed.reasoning_content!r}")
    print()


if __name__ == "__main__":
    main()
