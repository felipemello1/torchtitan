# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The session KV holder and the SimpleCPU connector inside vLLM's real `Scheduler`, with a fake
model runner."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torchtitan.rl.generator.vllm import simple_cpu_offload
from torchtitan.rl.generator.vllm.session_kv_holder import (
    cached_block_ids,
    VLLMSessionKVHolder,
)
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata
from vllm.v1.structured_output import StructuredOutputManager

BLOCK = 16
NUM_GPU_BLOCKS = 48
NUM_CPU_BLOCKS = 32


@pytest.fixture
def scheduler(tmp_path, monkeypatch) -> Scheduler:
    """vLLM's scheduler for a tiny local model: 1 attention + 1 GDN ("align") group, CPU offload."""
    # vLLM rejects a KV connector with expandable segments, which importing torchtitan.rl.train turns on.
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    init_none_hash(sha256)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 2,
        "vocab_size": 1024,
        "max_position_embeddings": 4096,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    attention = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    gdn = MambaSpec(
        block_size=BLOCK,
        shapes=((4, 8),),
        dtypes=(torch.float16,),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
    )
    cache_config = CacheConfig(
        block_size=BLOCK,
        cache_dtype="auto",
        enable_prefix_caching=True,
        mamba_cache_mode="align",
    )
    cache_config.num_gpu_blocks = NUM_GPU_BLOCKS
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="float16",
            skip_tokenizer_init=True,
            max_model_len=1024,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=64,
            max_model_len=1024,
            is_encoder_decoder=False,
            enable_chunked_prefill=True,
        ),
        cache_config=cache_config,
        device_config=DeviceConfig(device="cpu"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector",
            kv_role="kv_both",
            kv_connector_module_path="torchtitan.rl.generator.vllm.simple_cpu_offload",
            kv_connector_extra_config={
                "cpu_bytes_to_use_per_rank": NUM_CPU_BLOCKS * attention.page_size_bytes,
                "lazy_offload": True,
            },
        ),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_GPU_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=NUM_GPU_BLOCKS * attention.page_size_bytes,
                layers=["attn", "gdn"],
                layer_stride=0,
                block_stride=attention.page_size_bytes,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["gdn"], gdn),
        ],
        prefix_cache_retention_interval=None,
    )
    with patch.object(
        simple_cpu_offload, "get_tp_group", return_value=SimpleNamespace(world_size=1)
    ):
        return Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=BLOCK,
            log_stats=True,
        )


def _step(
    scheduler: Scheduler,
    holder: VLLMSessionKVHolder,
    loads_in_flight: set[str],
    abort_request_id: str | None = None,
) -> set[str]:
    """One engine step. The fake model runner samples token 7, finishes this step's CPU stores,
    and finishes the CPU loads of the step before, so loading blocks stay pinned for a step.
    `abort_request_id` is aborted at its first token, as `VLLMEngine.step` aborts a request whose
    logprob is not finite.

    Returns the requests whose CPU load starts this step.
    """
    output = scheduler.schedule()
    metadata = output.kv_connector_metadata
    new_loads = set(metadata.load_event_to_reqs.get(metadata.load_event, ()))
    completed_stores = {metadata.store_event: 1} if metadata.store_event >= 0 else {}
    req_ids = list(output.num_scheduled_tokens)
    # A request samples once its prompt is computed (schedule() already counted this step).
    sampled = [
        (
            [7]
            if scheduler.requests[req_id].num_computed_tokens
            >= scheduler.requests[req_id].num_prompt_tokens
            else []
        )
        for req_id in req_ids
    ]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            kv_connector_output=KVConnectorOutput(
                finished_recving=loads_in_flight or None,
                kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                    completed_stores
                ),
            ),
        ),
    )
    aborted_request_ids = set()
    request = scheduler.requests.get(abort_request_id)
    if request is not None and request.num_output_tokens > 0:
        scheduler.finish_requests(abort_request_id, RequestStatus.FINISHED_ABORTED)
        aborted_request_ids.add(abort_request_id)
    holder.after_step(aborted_request_ids)
    return new_loads


def test_fresh_prompt_and_full_gpu_hit_count_no_broken_chain(scheduler):
    # vLLM hashes a request's new full blocks before the connector's admission hook runs.
    connector = scheduler.connector
    holder = VLLMSessionKVHolder.Config().build(
        scheduler=scheduler, offload_connector=connector
    )
    block_hasher = get_request_block_hasher(BLOCK, sha256)
    loads_in_flight: set[str] = set()

    def run_turn(request_id: str, prompt: list[int]) -> Request:
        nonlocal loads_in_flight
        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt,
            sampling_params=SamplingParams(max_tokens=4, ignore_eos=True),
            pooling_params=None,
            block_hasher=block_hasher,
        )
        scheduler.add_request(request)
        holder.track(request_id, session_id="s0", group_id=0)
        while request_id in scheduler.requests:
            loads_in_flight = _step(scheduler, holder, loads_in_flight)
        return request

    # Turn 1: a fresh 100-token prompt. Turn 2 continues it; the holder keeps turn 1 on the GPU.
    turn1 = run_turn("s0/turn0", list(range(100, 200)))
    run_turn("s0/turn1", list(turn1.all_token_ids) + list(range(300, 348)))

    assert holder.num_reusable_prefix_tokens > 0 and holder.num_lost_prefix_tokens == 0
    assert connector.broken_chain_requests == {"hole": 0, "no_state": 0}
    assert connector.broken_chain_tokens == {"hole": 0, "no_state": 0}


def test_non_finite_abort_uncaches_the_sessions_kv_and_keeps_other_sessions(
    scheduler, monkeypatch
):
    # GRPO siblings s0 and s1 share a 32-token prompt; s2 has its own. The engine aborts s0's
    # second turn at its first token, after it loaded its history from the CPU.
    connector = scheduler.connector
    holder = VLLMSessionKVHolder.Config().build(
        scheduler=scheduler, offload_connector=connector
    )
    gpu_pool = scheduler.kv_cache_manager.block_pool
    cpu_pool = connector.scheduler_manager.cpu_block_pool
    block_hasher = get_request_block_hasher(BLOCK, sha256)
    loads_in_flight: set[str] = set()

    # Prefix tokens the GPU and CPU served each turn at admission (the holder pops them at prefill).
    cached_tokens = {}
    pop_cached_tokens = connector.pop_cached_tokens

    def recording_pop_cached_tokens(request_id: str) -> int | None:
        cached_tokens[request_id] = pop_cached_tokens(request_id)
        return cached_tokens[request_id]

    monkeypatch.setattr(connector, "pop_cached_tokens", recording_pop_cached_tokens)

    # The GPU and CPU copies of s1's and s2's KV just before and after the step that drops s0's.
    other_hashes: set[bytes] = set()
    other_copies = []
    after_step = holder.after_step

    def cached_other_copies() -> tuple[set[int], set[int]]:
        return (
            cached_block_ids(gpu_pool, other_hashes),
            cached_block_ids(cpu_pool, other_hashes),
        )

    def after_step_recording_other_copies(aborted_request_ids=()):
        if aborted_request_ids:
            other_copies.append(cached_other_copies())
        after_step(aborted_request_ids)
        if aborted_request_ids:
            other_copies.append(cached_other_copies())

    monkeypatch.setattr(holder, "after_step", after_step_recording_other_copies)

    def run_turn(request_id, session_id, group_id, prompt, abort=False) -> Request:
        nonlocal loads_in_flight
        request = Request(
            request_id=request_id,
            prompt_token_ids=list(prompt),
            sampling_params=SamplingParams(max_tokens=4, ignore_eos=True),
            pooling_params=None,
            block_hasher=block_hasher,
        )
        scheduler.add_request(request)
        holder.track(request_id, session_id=session_id, group_id=group_id)
        while request_id in scheduler.requests:
            loads_in_flight = _step(
                scheduler, holder, loads_in_flight, request_id if abort else None
            )
        return request

    shared = list(range(100, 132))
    s0_first = run_turn("s0/t0", "s0", 0, shared + list(range(200, 240)))
    s1_first = run_turn("s1/t0", "s1", 0, shared + list(range(400, 440)))
    s2_first = run_turn("s2/t0", "s2", 1, list(range(600, 672)))
    # A long prompt makes the holder evict the three sessions, and their KV goes to the CPU.
    run_turn("f/t0", "f", 2, list(range(800, 1300)))
    s0_only_hashes = set(s0_first.block_hashes) - set(s1_first.block_hashes)
    assert cached_block_ids(cpu_pool, s0_only_hashes)
    assert not cached_block_ids(gpu_pool, s0_only_hashes)

    other_hashes.update(s1_first.block_hashes, s2_first.block_hashes)
    aborted = run_turn(
        "s0/t1",
        "s0",
        0,
        list(s0_first.all_token_ids) + list(range(300, 348)),
        abort=True,
    )
    s0_only_hashes |= set(aborted.block_hashes) - set(s1_first.block_hashes)
    assert not cached_block_ids(gpu_pool, s0_only_hashes)
    assert not cached_block_ids(cpu_pool, s0_only_hashes)
    assert not holder.has_pending_work
    before, after = other_copies
    assert before == after and before[0] and before[1]

    # s0's next turn reuses at most the shared prompt; its sibling s1 reuses all of its last turn.
    run_turn("s0/t2", "s0", 0, list(aborted.prompt_token_ids) + list(range(500, 520)))
    assert cached_tokens["s0/t2"] <= len(shared)
    run_turn("s1/t1", "s1", 0, list(s1_first.all_token_ids) + list(range(900, 920)))
    assert cached_tokens["s1/t1"] == 64
    # s0's recompute is on purpose, not a lost prefix.
    assert holder.num_lost_prefix_tokens == 0
