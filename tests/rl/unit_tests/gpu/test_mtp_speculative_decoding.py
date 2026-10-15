# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.generator.generator import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm import registry
from torchtitan.tools.utils import has_cuda_capability
from vllm import EngineArgs, LLMEngine, SamplingParams

MODEL_ENV = "TORCHTITAN_QWEN3_5_0_8B_HF_PATH"
NUM_SPECULATIVE_TOKENS = 3
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_mtp_matches_no_spec(tmp_path: Path) -> None:
    """MTP keeps no-spec logprobs and prefix hits, and accepts drafts.

    Each mode runs in its own engine::

        mtp: turn 1 (sampled) -> turn 2 = turn 1 + completion + follow-up
             -> 4 prompt pairs diverging on a block boundary (greedy)
        off: same turns, then rescores both modes' completions without spec
    """
    model = os.environ.get(MODEL_ENV)
    if not model:
        pytest.skip(f"set {MODEL_ENV} to a local Qwen3.5-0.8B checkpoint")
    root = Path(__file__).resolve().parents[4]
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        MODEL_ENV: str(Path(model).resolve()),
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "HF_HUB_OFFLINE": "1",
    }
    results = {}
    # The no-spec run rescores the MTP run's completions, so MTP runs first.
    for mode in ("mtp", "off"):
        output = tmp_path / f"{mode}.json"
        command = [sys.executable]
        command += "-m torch.distributed.run --standalone --nproc-per-node=1".split()
        command += [str(Path(__file__).resolve()), mode, str(tmp_path)]
        log_path = output.with_suffix(".log")
        with log_path.open("w") as log:
            completed = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if completed.returncode:
            pytest.fail(
                f"{mode} subprocess exited with {completed.returncode}:\n"
                + "\n".join(log_path.read_text(errors="replace").splitlines()[-120:]),
                pytrace=False,
            )
        results[mode] = json.loads(output.read_text())
    mtp, off = results["mtp"], results["off"]

    # Rejection sampling keeps the target's distribution and logprobs, so MTP's
    # completions rescore as close as no-spec's own (the numerical floor). 2x
    # absorbs noise; wrong draft state or drafter logprobs would be far above it.
    floor, spec = off["score"]["off"], off["score"]["mtp"]
    assert spec["num_tokens"] > 2048
    assert spec["mean_abs_logprob_diff"] <= 2 * floor["mean_abs_logprob_diff"], (
        spec,
        floor,
    )
    # A broken drafter accepts almost nothing (~1.0); Qwen3.5-0.8B gives ~1.7 here.
    assert mtp["acceptance_length"] > 1.3, mtp["acceptance_length"]
    # The drafter shares our embedding and lm_head; the weights the trainer pushes
    # into are unchanged.
    assert mtp["weight_names"] == off["weight_names"]
    assert mtp["has_lm_head"] and not off["has_lm_head"]
    # disable_eagle_block_drop keeps turn 2's prefix hits at the no-spec boundary:
    # one block per request, written partly by turn 1's speculative decode.
    assert mtp["turn_2_cached_blocks"] == off["turn_2_cached_blocks"] == 8
    # A block-boundary hit reuses the drafter's KV, whose last position saw the
    # other prompt's next token. That costs at most a little acceptance.
    divergent = mtp["divergent"]
    assert divergent["cached_tokens"] == divergent["shared_tokens"]
    assert (
        divergent["acceptance_length_hit"] >= 0.9 * divergent["acceptance_length_cold"]
    ), divergent


def _generate(engine, prompts, sampling_params, tag):
    rendered = engine.renderer.render_cmpl(
        [{"prompt_token_ids": prompt} for prompt in prompts]
    )
    for index, prompt in enumerate(rendered):
        engine.add_request(f"{tag}-{index}", prompt, sampling_params)
    outputs = {}
    while engine.has_unfinished_requests():
        for request in engine.step():
            if request.finished:
                outputs[request.request_id] = request
    return [outputs[f"{tag}-{index}"] for index in range(len(prompts))]


def _acceptance_length(engine, generate):
    """Run ``generate()``; also return its mean acceptance length (None without spec)."""

    names = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens")

    def counters():
        values = {m.name: m.value for m in engine.get_metrics() if m.name in names}
        return [values.get(name, 0) for name in names]

    before = counters()
    outputs = generate()
    drafts, accepted = (after - prior for after, prior in zip(counters(), before))
    return outputs, 1 + accepted / drafts if drafts else None


def _completions(prompts, outputs):
    return [
        {
            "prompt": prompt,
            "completion": list(output.outputs[0].token_ids),
            "logprobs": [
                next(iter(step.values())).logprob for step in output.outputs[0].logprobs
            ],
        }
        for prompt, output in zip(prompts, outputs)
    ]


def _score(engine, completions, tag):
    """Teacher-force completions and compare with the logprobs they returned."""
    outputs = _generate(
        engine,
        [item["prompt"] + item["completion"] for item in completions],
        SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0),
        tag,
    )
    diffs = [
        abs(returned - step[token].logprob)
        for item, output in zip(completions, outputs)
        for token, returned, step in zip(
            item["completion"],
            item["logprobs"],
            output.prompt_logprobs[len(item["prompt"]) :],
            strict=True,
        )
    ]
    return {"mean_abs_logprob_diff": sum(diffs) / len(diffs), "num_tokens": len(diffs)}


def run_engine(mode: str, tmp_path: Path) -> None:
    assert int(os.environ["WORLD_SIZE"]) == 1
    model = os.environ[MODEL_ENV]
    spec = mode == "mtp"
    model_config = build_model_config(
        "0.8B", seq_len=8192, attn_backend="varlen", parallelism=ParallelismConfig()
    )
    model_config.local_compile_regions = []
    registry.register_to_vllm(
        model_config,
        parallelism=registry.InferenceParallelismConfig(),
        checkpointer_config=CheckpointManager.Config(
            initial_load_in_hf=True, initial_load_path=model
        ),
        mtp=spec,
    )
    max_num_seqs = 8
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=model,
            config_format=registry.TORCHTITAN_CONFIG_FORMAT,
            dtype="bfloat16",
            worker_cls=registry.TORCHTITAN_WORKER_CLS,
            distributed_executor_backend="external_launcher",
            attention_config={"backend": "CUSTOM"},
            compilation_config=VLLMCudaGraphConfig(
                mode="FULL_DECODE_ONLY"
            ).get_vllm_compilation_config(
                max_num_seqs=max_num_seqs,
                expert_sequence_parallel_size=1,
                enable_sequence_parallel=False,
                num_speculative_tokens=NUM_SPECULATIVE_TOKENS if spec else 0,
            ),
            speculative_config=(
                {
                    "method": "mtp",
                    "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
                    "disable_eagle_block_drop": True,
                }
                if spec
                else None
            ),
            max_model_len=8192,
            max_num_seqs=max_num_seqs,
            # Prompt logprobs materialize [tokens, vocab] in fp32 per step.
            max_num_batched_tokens=2048,
            # PyTorch's FA2 paged-KV kernel before Hopper needs 256-token pages.
            **({} if has_cuda_capability(9, 0) else {"block_size": 256}),
            gpu_memory_utilization=0.3,
            enable_prefix_caching=True,
            prefix_cache_retention_interval=None,
            logprobs_mode="processed_logprobs",
            seed=0,
            disable_log_stats=False,
        )
    )
    block_size = engine.vllm_config.cache_config.block_size

    def weights(worker):
        wrapper = worker.model_runner.get_model()
        # Not state_dict(): its FSDP hooks reshard the unsharded parameters.
        names = sorted(name for name, _ in wrapper.model.named_parameters())
        return names, hasattr(wrapper, "lm_head")

    [(weight_names, has_lm_head)] = engine.collective_rpc(weights)
    tokenizer = engine.renderer.tokenizer
    document = tokenizer.encode(
        (Path(__file__).resolve().parents[4] / "README.md").read_text(),
        add_special_tokens=False,
    )
    assert len(document) >= 3 * 512 + 2 * block_size, (len(document), block_size)

    def ask(context, question):
        return context + tokenizer.encode(f"\n\n{question}", add_special_tokens=False)

    def sampled(max_tokens):
        return SamplingParams(
            temperature=1.0, max_tokens=max_tokens, ignore_eos=True, logprobs=0
        )

    turn_1 = [
        ask(document[64 * i : 64 * i + 260], "Summarize this text.") for i in range(8)
    ]
    # block_size + 99 reusable tokens per request: one block, whatever vLLM sizes it.
    max_tokens = block_size + 100 - len(turn_1[0])
    turn_1_outputs, acceptance_length = _acceptance_length(
        engine, lambda: _generate(engine, turn_1, sampled(max_tokens), "turn-1")
    )
    turn_2 = [
        ask(prompt + list(output.outputs[0].token_ids), "Now list three key terms.")
        for prompt, output in zip(turn_1, turn_1_outputs)
    ]
    turn_2_outputs = _generate(engine, turn_2, sampled(160), "turn-2")
    completions = _completions(turn_1, turn_1_outputs) + _completions(
        turn_2, turn_2_outputs
    )
    result = {
        "acceptance_length": acceptance_length,
        "weight_names": weight_names,
        "has_lm_head": has_lm_head,
        "turn_2_cached_blocks": sum(
            output.num_cached_tokens // block_size for output in turn_2_outputs
        ),
        "completions": completions,
    }

    if spec:
        # Pairs share whole blocks and differ from the next token on; the second of
        # each pair hits the shared blocks.
        greedy = SamplingParams(temperature=0.0, max_tokens=192, ignore_eos=True)
        shared = [document[512 * i : 512 * i + 2 * block_size] for i in range(4)]
        first = [prefix + tokenizer.encode("What is it about?") for prefix in shared]
        second = [prefix + tokenizer.encode("Who would use it?") for prefix in shared]
        engine.reset_prefix_cache()
        _generate(engine, first, greedy, "first")
        hit_outputs, hit = _acceptance_length(
            engine, lambda: _generate(engine, second, greedy, "second-hit")
        )
        engine.reset_prefix_cache()
        _, cold = _acceptance_length(
            engine, lambda: _generate(engine, second, greedy, "second-cold")
        )
        result["divergent"] = {
            "shared_tokens": sum(len(prefix) for prefix in shared),
            "cached_tokens": sum(output.num_cached_tokens for output in hit_outputs),
            "acceptance_length_hit": hit,
            "acceptance_length_cold": cold,
        }
    else:
        mtp = json.loads((tmp_path / "mtp.json").read_text())
        result["score"] = {
            "off": _score(engine, completions, "score-off"),
            "mtp": _score(engine, mtp["completions"], "score-mtp"),
        }

    (tmp_path / f"{mode}.json").write_text(json.dumps(result))
    engine.model_executor.shutdown()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_engine(sys.argv[1], Path(sys.argv[2]))
